#![feature(seek_stream_len)]
#![feature(new_uninit)]
//! Heightfield
#![warn(missing_docs)]

#[cfg(feature = "surf-obj")]
use base::math::Axis;
#[cfg(feature = "surf-gen")]
mod gen;
#[cfg(feature = "surf-gen")]
pub use gen::*;
pub mod dcel;
pub mod io;
pub mod subdivision;

#[cfg(feature = "embree")]
use embree::{BufferUsage, Device, Format, Geometry, GeometryKind};

use crate::{
    dcel::HalfEdgeMesh,
    subdivision::{curved, wiggle, Subdivision},
};
use base::{
    error::VgonioError,
    io::{
        CompressionScheme, FileEncoding, Header, HeaderMeta, ReadFileError, WriteFileError,
        WriteFileErrorKind,
    },
    math::{rcp_f32, sqr, Aabb, Vec3},
    range::RangeByStepCountInclusive,
    units::LengthUnit,
    Asset, Version,
};
use glam::{DVec3, Vec2};
use log::log;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};

/// Static variable used to generate height field name.
static mut MICRO_SURFACE_COUNTER: u32 = 0;

/// Helper function to generate a default name for height field instance.
fn gen_micro_surface_name() -> String {
    unsafe {
        let name = format!("micro_surface_{MICRO_SURFACE_COUNTER:03}");
        MICRO_SURFACE_COUNTER += 1;
        name
    }
}

/// Offset used when generating [`MicroSurfaceMesh`].
pub enum HeightOffset {
    /// Offset the height field by an arbitrary value.
    Arbitrary(f32),
    /// Offset the height field so that its minimum height is zero.
    Grounded,
    /// Offset the height field so that its median height is zero.
    Centered,
    /// Do not offset the height field.
    None,
}

impl HeightOffset {
    /// Evaluates the offset value according to the specified minimum and
    /// maximum height.
    pub fn eval(&self, min: f32, max: f32) -> f32 {
        match self {
            HeightOffset::Arbitrary(offset) => *offset,
            HeightOffset::Grounded => -min,
            HeightOffset::Centered => (min + max) * -0.5,
            HeightOffset::None => 0.0,
        }
    }
}

// TODO: support f64
/// Representation of the micro-surface.
#[cfg_attr(feature = "pybind", pyo3::pyclass)]
#[derive(Debug, Serialize, Deserialize)]
pub struct MicroSurface {
    /// Generated unique identifier.
    pub uuid: uuid::Uuid,

    /// User defined name for the height field.
    pub name: String,

    /// Location where the heightfield is loaded from.
    pub path: Option<PathBuf>,

    /// Number of sample points in vertical direction (first axis of
    /// alignment).
    pub rows: usize,

    /// Number of sample points in horizontal direction (second axis of
    /// alignment).
    pub cols: usize,

    /// The space between sample points in horizontal direction.
    pub du: f32,

    /// The space between sample points in vertical direction.
    pub dv: f32,

    /// The unit of the micro-surface (shared between du, dv and samples).
    pub unit: LengthUnit,

    /// Minimum height of the height field.
    pub min: f32,

    /// Maximum height of the height field.
    pub max: f32,

    /// The median height of the height field.
    pub median: f32,

    /// Height values of sample points on the height field (values are stored in
    /// row major order).
    pub samples: Vec<f32>,
}

impl Asset for MicroSurface {}

impl MicroSurface {
    /// Creates a micro-surface (height field) with specified height value.
    ///
    /// # Arguments
    ///
    /// * `cols` - the number of sample points in horizontal dimension
    /// * `rows` - the number of sample points in vertical dimension
    /// * `du` - horizontal spacing between samples points in micrometers
    /// * `dv` - vertical spacing between samples points in micrometers
    /// * `height` - the initial value of the height in micrometers
    ///
    /// # Examples
    ///
    /// ```
    /// # use base::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let height_field = MicroSurface::new(10, 10, 0.11, 0.11, 0.12, LengthUnit::UM);
    /// assert_eq!(height_field.samples_count(), 100);
    /// assert_eq!(height_field.cells_count(), 81);
    /// ```
    pub fn new(rows: usize, cols: usize, du: f32, dv: f32, height: f32, unit: LengthUnit) -> Self {
        assert!(cols > 1 && rows > 1);
        let mut samples = Vec::new();
        samples.resize(cols * rows, height);
        MicroSurface {
            uuid: uuid::Uuid::new_v4(),
            name: gen_micro_surface_name(),
            path: None,
            rows,
            cols,
            du,
            dv,
            unit,
            min: height,
            max: height,
            median: height,
            samples,
        }
    }

    /// Creates a micro-surface (height field) and sets its height values by
    /// using a function.
    ///
    /// # Arguments
    ///
    /// * `cols` - the number of sample points in dimension x
    /// * `rows` - the number of sample points in dimension y
    /// * `du` - horizontal spacing between samples points in micrometers
    /// * `dv` - vertical spacing between samples points in micrometers
    /// * `setter` - the setting function, this function will be invoked with
    ///   the row number and column number as parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// # use base::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let msurf = MicroSurface::new_by(4, 4, 0.1, 0.1, LengthUnit::UM, |row, col| {
    ///     (row + col) as f32
    /// });
    /// assert_eq!(msurf.samples_count(), 16);
    /// assert_eq!(msurf.max, 6.0);
    /// assert_eq!(msurf.min, 0.0);
    /// assert_eq!(msurf.samples[0], 0.0);
    /// assert_eq!(msurf.samples[2], 2.0);
    /// assert_eq!(msurf.sample_at(2, 3), 5.0);
    /// ```
    pub fn new_by<F>(
        rows: usize,
        cols: usize,
        du: f32,
        dv: f32,
        unit: LengthUnit,
        mut setter: F,
    ) -> MicroSurface
    where
        F: FnMut(usize, usize) -> f32,
    {
        assert!(cols > 1 && rows > 1);
        let mut samples = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                samples.push(setter(r, c));
            }
        }
        let max = *samples
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let min = *samples
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        MicroSurface {
            uuid: uuid::Uuid::new_v4(),
            name: gen_micro_surface_name(),
            path: None,
            cols,
            rows,
            du,
            dv,
            max,
            min,
            samples,
            median: (min + max) / 2.0,
            unit,
        }
    }

    /// Create a micro-surface from an array of elevation values.
    ///
    /// # Arguments
    ///
    /// * `cols` - number of columns (number of sample points in dimension x) in
    ///   the height field.
    /// * `rows` - number of rows (number of sample points in dimension y) in
    ///   the height field.
    /// * `du` - horizontal spacing between two samples
    /// * `dv` - vertical spacing between two samples
    /// * `samples` - array of elevation values of the height field.
    /// * `alignment` - axis alignment of height field
    ///
    /// # Examples
    ///
    /// ```
    /// # use base::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let height_field =
    ///     MicroSurface::from_samples(3, 3, (0.5, 0.5), LengthUnit::UM, &samples, None, None);
    /// assert_eq!(height_field.samples_count(), 9);
    /// assert_eq!(height_field.cells_count(), 4);
    /// assert_eq!(height_field.cols, 3);
    /// assert_eq!(height_field.rows, 3);
    /// ```
    pub fn from_samples<S: AsRef<[f32]>>(
        rows: usize,
        cols: usize,
        spacing: (f32, f32),
        unit: LengthUnit,
        samples: S,
        name: Option<String>,
        path: Option<PathBuf>,
    ) -> MicroSurface {
        debug_assert!(
            cols > 0 && rows > 0 && samples.as_ref().len() == cols * rows,
            "Samples count must be equal to cols * rows"
        );
        let (max, min) = samples
            .as_ref()
            .iter()
            .fold((f32::MIN, f32::MAX), |(max, min), x| {
                (f32::max(max, *x), f32::min(min, *x))
            });

        MicroSurface {
            uuid: uuid::Uuid::new_v4(),
            name: name.unwrap_or(gen_micro_surface_name()),
            path,
            rows,
            cols,
            du: spacing.0,
            dv: spacing.1,
            max,
            min,
            samples: samples.as_ref().to_owned(),
            median: min * 0.5 + max * 0.5,
            unit,
        }
    }

    /// Returns the filename of the micro-surface if it is loaded from a file.
    pub fn file_name(&self) -> Option<&str> {
        match self.path {
            Some(ref path) => path.file_name().and_then(|s| s.to_str()),
            None => None,
        }
    }

    /// Returns the file stem of the micro-surface if it is loaded from a file.
    pub fn file_stem(&self) -> Option<&str> {
        match self.path {
            Some(ref path) => path.file_stem().and_then(|s| s.to_str()),
            None => None,
        }
    }

    /// Returns the dimension of the surface (rows * du, cols * dv)
    /// # Examples
    ///
    /// ```
    /// # use base::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let msurf = MicroSurface::new(100, 100, 0.1, 0.1, 0.1, LengthUnit::UM);
    /// assert_eq!(msurf.dimension(), (10.0, 10.0));
    /// ```
    pub fn dimension(&self) -> (f32, f32) {
        (self.rows as f32 * self.du, self.cols as f32 * self.dv)
    }

    /// Returns the number of samples of height field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use base::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, (0.2, 0.2), LengthUnit::UM, samples, None, None);
    /// assert_eq!(msurf.samples_count(), 9);
    /// ```
    pub fn samples_count(&self) -> usize { self.cols * self.rows }

    /// Returns the number of cells of height field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use base::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, (0.2, 0.2), LengthUnit::UM, samples, None, None);
    /// assert_eq!(msurf.cells_count(), 4);
    /// ```
    pub fn cells_count(&self) -> usize {
        if self.cols == 0 || self.rows == 0 {
            0
        } else {
            (self.cols - 1) * (self.rows - 1)
        }
    }

    /// Returns the height value of a given sample.
    ///
    /// # Arguments
    ///
    /// * `col` - sample's position in dimension x
    /// * `row` - sample's position in dimension y
    ///
    /// # Examples
    ///
    /// ```
    /// # use base::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, (0.2, 0.2), LengthUnit::MM, samples, None, None);
    /// assert_eq!(msurf.sample_at(2, 2), 0.1);
    /// ```
    ///
    /// ```should_panic
    /// # use base::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, (0.2, 0.3), LengthUnit::MM, samples, None, None);
    /// let h = msurf.sample_at(4, 4);
    /// ```
    pub fn sample_at(&self, row: usize, col: usize) -> f32 {
        assert!(col < self.cols);
        assert!(row < self.rows);
        self.samples[row * self.cols + col]
    }

    /// Fill holes on the height field.
    /// TODO: bilinear interpolation
    pub fn repair(&mut self) {
        let mut idx = 0;
        for i in 0..self.cols * self.rows {
            if !self.samples[i].is_nan() {
                idx = i;
                break;
            }
        }

        for j in 0..self.rows {
            for i in 0..self.cols {
                if self.samples[j * self.cols + i].is_nan() {
                    // bilinear interpolation
                    // direction x
                    let mut prev_idx_x = i;
                    let mut next_idx_x = i;
                    for n in i..self.cols {
                        if !self.samples[j * self.cols + n].is_nan() {
                            next_idx_x = n;
                            break;
                        }
                    }

                    for n in 0..=i {
                        if !self.samples[j * self.cols + i - n].is_nan() {
                            prev_idx_x = i - n;
                            break;
                        }
                    }

                    let mut prev_x = self.samples[idx];
                    let mut next_x = self.samples[idx];

                    // prev not found
                    if prev_idx_x == i {
                        if i == 0 {
                            if j == 0 {
                                prev_x = self.samples[idx];
                            } else {
                                prev_x = self.samples[(j - 1) * self.cols + i];
                            }
                        }
                    } else {
                        prev_x = self.samples[j * self.cols + prev_idx_x];
                    }

                    // next not found
                    if next_idx_x == i {
                        if i == self.cols - 1 {
                            if j == 0 {
                                next_x = self.samples[idx]
                            } else {
                                next_x = self.samples[(j - 1) * self.cols + i];
                            }
                        }
                    } else {
                        next_x = self.samples[j * self.cols + next_idx_x]
                    }

                    let x = (next_x - prev_x) / (next_idx_x - prev_idx_x) as f32 + prev_x;

                    // direction y
                    let mut prev_idx_y = j;
                    let mut next_idx_y = j;
                    let prev_y;
                    let next_y;

                    for n in j..self.rows {
                        if !self.samples[n * self.cols + i].is_nan() {
                            next_idx_y = n;
                            break;
                        }
                    }

                    for n in 0..=j {
                        if !self.samples[(j - n) * self.cols + i].is_nan() {
                            prev_idx_y = j - n;
                            break;
                        }
                    }

                    // prev not found
                    if prev_idx_y == j {
                        prev_idx_y = 0;
                        prev_y = self.samples[idx];
                    } else {
                        prev_y = self.samples[prev_idx_y * self.cols + i];
                    }

                    // next not found
                    if next_idx_y == j {
                        next_idx_y = self.rows - 1;
                        next_y = self.samples[idx];
                    } else {
                        next_y = self.samples[next_idx_y * self.cols + i];
                    }

                    let y = (next_y - prev_y) / (next_idx_y - prev_idx_y + 1) as f32 + prev_y;

                    self.samples[j * self.cols + i] = (y + x) / 2.0;
                }
            }
        }
    }

    /// Computes the surface area of the height field.
    pub fn macro_area(&self) -> f32 {
        (self.cols - 1) as f32 * (self.rows - 1) as f32 * self.du * self.dv
    }

    /// Computes the root-mean-square height of the height field.
    pub fn rms_height(&self) -> f32 {
        let rcp_n = rcp_f32(self.samples.len() as f32);
        let mean = self.samples.iter().sum::<f32>() * rcp_n;
        self.samples
            .iter()
            .fold(0.0, |acc, x| acc + sqr(x - mean) * rcp_n)
            .sqrt()
    }

    /// Triangulate the heightfield into a [`MicroSurfaceMesh`].
    /// The triangulation is done on the XZ plane.
    pub fn as_micro_surface_mesh(
        &self,
        offset: HeightOffset,
        pattern: TriangulationPattern,
        subdivision: Subdivision,
    ) -> MicroSurfaceMesh {
        let height_offset = match offset {
            HeightOffset::Arbitrary(val) => val,
            HeightOffset::Centered => -0.5 * (self.min + self.max),
            HeightOffset::Grounded => -self.min,
            HeightOffset::None => 0.0,
        };
        let (verts, extent) = self.generate_vertices(height_offset);
        let tri_faces = regular_grid_triangulation(self.rows, self.cols, pattern);
        let num_faces = tri_faces.len() / 3;

        let mut facet_normals = vec![Vec3::ZERO; num_faces].into_boxed_slice();
        let mut areas = vec![0.0; num_faces].into_boxed_slice();
        let mut facet_total_area = 0.0;

        // Records the index of the facet that shares a vertex.
        let mut topology = vec![[-1; 6]; verts.len()];

        for i in 0..num_faces {
            let p0 = verts[tri_faces[i * 3] as usize];
            let p1 = verts[tri_faces[i * 3 + 1] as usize];
            let p2 = verts[tri_faces[i * 3 + 2] as usize];
            // Fill the topology
            // For each vertex of the triangle
            for j in 0..3 {
                let vert_topology = &mut topology[tri_faces[i * 3 + j] as usize];
                for k in 0..6 {
                    if vert_topology[k] == -1 {
                        vert_topology[k] = i as i32;
                        break;
                    }
                }
            }
            let cross = (p1 - p0).cross(p2 - p0);
            facet_normals[i] = cross.normalize();
            areas[i] = 0.5 * cross.length();
            facet_total_area += areas[i];
        }

        // Compute the vertex normals
        let mut vert_normals = Box::new_uninit_slice(verts.len());
        for i in 0..verts.len() {
            let mut normal = Vec3::ZERO;
            let vert_topology = &topology[i];
            for j in 0..6 {
                if vert_topology[j] != -1 {
                    normal += facet_normals[vert_topology[j] as usize];
                }
            }
            vert_normals[i].write(normal.normalize());
        }

        let mut mesh = MicroSurfaceMesh {
            uuid: uuid::Uuid::new_v4(),
            num_facets: num_faces,
            num_verts: verts.len(),
            bounds: extent,
            verts,
            facets: tri_faces,
            facet_normals,
            vert_normals: unsafe { vert_normals.assume_init() },
            facet_areas: areas,
            msurf: self.uuid,
            num_rows: self.rows,
            unit: self.unit,
            height_offset,
            facet_total_area,
            num_cols: self.cols,
        };
        log::debug!("Microfacet Area: {}", facet_total_area);

        match subdivision {
            Subdivision::Curved(lvl) => {
                log::debug!("Subdividing the mesh with level: {}", lvl);
                mesh.curved_smooth(lvl);
                log::debug!("Microfacet Area(subdivided): {}", mesh.facet_total_area);
            }
            Subdivision::Wiggly(lvl) => {
                log::debug!("Subdividing the mesh with level: {}", lvl);
                mesh.wiggly_smooth(lvl);
                log::debug!("Microfacet Area(subdivided): {}", mesh.facet_total_area);
            }
            _ => {}
        }

        mesh
    }

    /// Generate vertices from the height values.
    ///
    /// The vertices are generated following the order from left to right, top
    /// to bottom. The center of the heightfield is at the origin.
    ///
    /// The generated vertices are aligned to the xy plane of the right-handed,
    /// Z-up coordinate system.
    #[doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../misc/imgs/heightfield.svg"))]
    pub fn generate_vertices(&self, height_offset: f32) -> (Box<[Vec3]>, Aabb) {
        log::info!(
            "Generating height field vertices with {} rows and {} cols",
            self.rows,
            self.cols
        );
        let (rows, cols, half_rows, half_cols, du, dv) = (
            self.rows,
            self.cols,
            self.rows / 2,
            self.cols / 2,
            self.du,
            self.dv,
        );
        let mut positions = Box::new_uninit_slice(rows * cols);
        #[cfg(feature = "bench")]
        let t = std::time::Instant::now();

        let extent = positions
            .par_chunks_mut(1024)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut extent = Aabb::default();
                for (i, pos) in chunk.iter_mut().enumerate() {
                    let idx = chunk_idx * 1024 + i;
                    let c = idx % cols;
                    let r = idx / cols;

                    let p = Vec3::new(
                        (c as f32 - half_cols as f32) * du,
                        (half_rows as f32 - r as f32) * dv,
                        self.samples[idx] + height_offset,
                    );

                    pos.write(p);

                    for k in 0..3 {
                        if p[k] > extent.max[k] {
                            extent.max[k] = p[k];
                        }
                        if p[k] < extent.min[k] {
                            extent.min[k] = p[k];
                        }
                    }
                }
                extent
            })
            .reduce(Aabb::empty, Aabb::union);

        #[cfg(feature = "bench")]
        log::info!(
            "Height field vertices generated in {:?} ms",
            t.elapsed().as_millis()
        );

        (unsafe { positions.assume_init() }, extent)
    }

    /// Resize the heightfield.
    ///
    /// The samples will be taken from the top-left corner of the heightfield.
    /// TODO: add more options for different resizing methods.
    pub fn resize(&self, rows: usize, cols: usize) -> Self {
        if rows == self.rows && cols == self.cols {
            return Self::from_samples(
                self.rows,
                self.cols,
                (self.du, self.dv),
                self.unit,
                &self.samples,
                None,
                None,
            );
        }
        let mut ms = Self::new(rows, cols, self.du, self.dv, 0.0, self.unit);
        let min_rows = rows.min(self.rows);
        let min_cols = cols.min(self.cols);
        for r in 0..min_rows {
            for c in 0..min_cols {
                ms.samples[r * cols + c] = self.samples[r * self.cols + c];
            }
        }
        ms
    }

    /// Computes the rms slope of the height field in the x(horizontal)
    /// direction.
    pub fn rms_slope_x(&self) -> f32 {
        let slope_xs = (0..self.cols)
            .flat_map(|x| {
                (0..self.rows).map(move |y| {
                    let nx = (x + 1).min(self.cols - 1);
                    let dh = self.samples[y * self.cols + nx] - self.samples[y * self.cols + x];
                    dh / self.du
                })
            })
            .collect::<Vec<_>>();
        let rcp_n = rcp_f32(slope_xs.len() as f32);
        let mean = slope_xs.iter().sum::<f32>() * rcp_n;
        slope_xs
            .iter()
            .fold(0.0, |acc, x| acc + rcp_n * (x - mean) * (x - mean))
            .sqrt()
    }

    /// Computes the rms slope of the height field in the y(vertical) direction.
    pub fn rms_slope_y(&self) -> f32 {
        let slope_ys = (0..self.rows)
            .flat_map(|y| {
                (0..self.cols).map(move |x| {
                    let ny = (y + 1).min(self.rows - 1);
                    let dh = self.samples[ny * self.cols + x] - self.samples[y * self.cols + x];
                    dh / self.dv
                })
            })
            .collect::<Vec<_>>();
        let rcp_n = rcp_f32(slope_ys.len() as f32);
        let mean = slope_ys.iter().sum::<f32>() * rcp_n;
        slope_ys
            .iter()
            .fold(0.0, |acc, x| acc + rcp_n * (x - mean) * (x - mean))
            .sqrt()
    }
}

/// Triangulation pattern for grid triangulation.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TriangulationPattern {
    /// Triangulate from top to bottom, left to right.
    /// 0  <--  1
    /// | A  /  |
    /// |  /  B |
    /// 2  -->  3
    BottomLeftToTopRight,
    /// Triangulate from top to bottom, right to left.
    /// 0  <--  1
    /// |  \  B |
    /// | A  \  |
    /// 2  -->  3
    #[default]
    TopLeftToBottomRight,
}

/// Generate triangle indices for grid triangulation.
///
/// The grid is assumed to be a regular grid with `cols` columns and `rows`
/// rows. The triangles are generated in counter-clockwise order. The
/// triangulation starts from the first row, from left to right. Thus,
/// when using the indices, you need to provide the vertices in the same
/// order.
///
/// Triangle winding is counter-clockwise.
///
/// The Pattern is specified by `TriangulationPattern`.
///
/// # Returns
///
/// Box<[u32]>: An array of vertex indices forming triangles.
pub fn regular_grid_triangulation(
    rows: usize,
    cols: usize,
    pattern: TriangulationPattern,
) -> Box<[u32]> {
    let mut triangulate: Box<dyn FnMut(usize, usize, &mut usize, &mut [u32])> = match pattern {
        TriangulationPattern::BottomLeftToTopRight => Box::new(
            |i: usize, col: usize, tri: &mut usize, indices: &mut [u32]| {
                if col > 0 {
                    indices[*tri] = i as u32;
                    indices[*tri + 1] = (i - 1) as u32;
                    indices[*tri + 2] = (i + cols - 1) as u32;
                    indices[*tri + 3] = i as u32;
                    indices[*tri + 4] = (i + cols - 1) as u32;
                    indices[*tri + 5] = (i + cols) as u32;
                    *tri += 6;
                }
            },
        ),
        TriangulationPattern::TopLeftToBottomRight => Box::new(
            |i: usize, col: usize, tri: &mut usize, indices: &mut [u32]| {
                if col != cols - 1 {
                    indices[*tri] = i as u32;
                    indices[*tri + 1] = (i + cols) as u32;
                    indices[*tri + 2] = (i + cols + 1) as u32;
                    indices[*tri + 3] = i as u32;
                    indices[*tri + 4] = (i + cols + 1) as u32;
                    indices[*tri + 5] = (i + 1) as u32;
                    *tri += 6;
                }
            },
        ),
    };

    let mut indices = vec![0; 2 * (cols - 1) * (rows - 1) * 3].into_boxed_slice();
    let mut tri = 0;
    for i in 0..cols * rows {
        let row = i / cols;
        let col = i % cols;

        // last row
        if row == rows - 1 {
            continue;
        }

        triangulate(i, col, &mut tri, &mut indices);
    }

    indices
}

#[test]
fn regular_grid_triangulation_test() {
    let indices = regular_grid_triangulation(4, 3, TriangulationPattern::BottomLeftToTopRight);
    assert_eq!(indices.len(), 36);
    println!("{:?}", indices);
}

/// Triangle representation of the surface mesh.
///
/// Created from [`MicroSurface`](`crate::MicroSurface`) using
/// [`MicroSurface::as_micro_surface_mesh`](`crate::MicroSurface::as_micro_surface_mesh`),
/// and has the same length unit ([`Micrometres`](`crate::units::Micrometres`))
/// as the [`MicroSurface`](`crate::MicroSurface`).
///
/// By default, the generated mesh is located on XZ plane in right-handed Y up
/// coordinate system.
/// See [`MicroSurface::as_micro_surface_mesh`](`crate::MicroSurface::as_micro_surface_mesh`)
#[derive(Debug)]
pub struct MicroSurfaceMesh {
    /// Unique identifier, different from the [`MicroSurface`] uuid.
    pub uuid: uuid::Uuid,

    /// Uuid of the [`MicroSurface`] from which the mesh is generated.
    pub msurf: uuid::Uuid,

    /// Number of rows in the mesh.
    pub num_rows: usize,

    /// Number of columns in the mesh.
    pub num_cols: usize,

    /// Axis-aligned bounding box of the mesh.
    pub bounds: Aabb,

    /// Height offset applied to the original heightfield.
    pub height_offset: f32,

    /// Number of triangles in the mesh.
    pub num_facets: usize,

    /// Number of vertices in the mesh.
    pub num_verts: usize,

    /// Vertices of the mesh.
    pub verts: Box<[Vec3]>,

    /// Vertex indices forming the facets which are triangles.
    pub facets: Box<[u32]>,

    /// Normal vectors of each facet.
    pub facet_normals: Box<[Vec3]>,

    /// Normal vectors of each vertex (average of the normals of the facets
    /// sharing the vertex).
    pub vert_normals: Box<[Vec3]>,

    /// Surface area of each facet.
    pub facet_areas: Box<[f32]>,

    /// Total surface area of the triangles.
    pub facet_total_area: f32,

    /// Length unit of inherited from the
    /// [`MicroSurface`](`crate::MicroSurface`).
    pub unit: LengthUnit,
}

impl Asset for MicroSurfaceMesh {}

impl MicroSurfaceMesh {
    /// Returns the surface area of a facet.
    ///
    /// # Arguments
    ///
    /// * `facet` - Index of the facet.
    ///
    /// TODO(yang): unit of surface area
    pub fn facet_surface_area(&self, facet: usize) -> f32 { self.facet_areas[facet] }

    /// Calculate the macro surface area of the mesh.
    pub fn macro_surface_area(&self) -> f32 {
        (self.bounds.max.x - self.bounds.min.x) * (self.bounds.max.y - self.bounds.min.y)
    }

    /// Returns the center of a facet.
    pub fn center_of_facet(&self, facet: usize) -> Vec3 {
        let idx = facet * 3;
        let p0 = self.verts[self.facets[idx] as usize];
        let p1 = self.verts[self.facets[idx + 1] as usize];
        let p2 = self.verts[self.facets[idx + 2] as usize];
        (p0 + p1 + p2) / 3.0
    }

    /// Constructs an embree geometry from the `MicroSurfaceMesh`.
    #[cfg(feature = "embree")]
    pub fn as_embree_geometry<'g>(&'g self, device: &Device) -> Geometry<'g> {
        let mut geom = device
            .create_geometry::<'g>(GeometryKind::TRIANGLE)
            .unwrap();
        geom.set_new_buffer(BufferUsage::VERTEX, 0, Format::FLOAT3, 16, self.num_verts)
            .unwrap()
            .view_mut::<[f32; 4]>()
            .unwrap()
            .iter_mut()
            .zip(self.verts.iter())
            .for_each(|(vert, pos)| {
                vert[0] = pos.x;
                vert[1] = pos.y;
                vert[2] = pos.z;
                vert[3] = 1.0;
            });
        geom.set_new_buffer(BufferUsage::INDEX, 0, Format::UINT3, 12, self.num_facets)
            .unwrap()
            .view_mut::<u32>()
            .unwrap()
            .copy_from_slice(&self.facets);
        geom.commit();
        geom
    }

    /// Subdivide the mesh.
    pub fn subdivide(&mut self, opts: Subdivision) {
        if opts.level() == 0 {
            return;
        }

        let mut dcel = HalfEdgeMesh::new(Cow::Borrowed(&self.verts), &self.facets);
        let subdivision = TriangleUVSubdivision::new(opts.level());

        let subdivided = match opts {
            Subdivision::Wiggly(lvl) => {
                dcel.subdivide_by_uvs(&subdivision, wiggle::subdivide_triangle)
            }
            _ => {
                unreachable!("Only wiggly subdivision is supported for now!");
            }
        };
    }

    /// Smooth the mesh by subdividing the triangles into curved surfaces.
    pub fn curved_smooth(&mut self, level: u32) {
        if level == 0 {
            return;
        }

        let subdivision = TriangleUVSubdivision::new(level);
        let new_num_verts = self.num_facets * subdivision.n_pnts as usize;
        let mut new_verts = vec![Vec3::ZERO; new_num_verts].into_boxed_slice();
        let mut new_vert_normals = vec![Vec3::ZERO; new_num_verts].into_boxed_slice();
        let new_num_facets = self.num_facets * subdivision.n_tris as usize;
        let mut new_facets = vec![0u32; new_num_facets * 3].into_boxed_slice();
        let mut new_facet_normals = vec![Vec3::ZERO; new_num_facets].into_boxed_slice();
        let mut new_facet_areas = vec![0.0f64; new_num_facets].into_boxed_slice();

        const FACET_CHUNK_SIZE: usize = 128;
        // Iterate over the facets in chunks to subdivide in parallel
        self.facets
            .par_chunks(3 * FACET_CHUNK_SIZE)
            .zip(new_verts.par_chunks_mut(FACET_CHUNK_SIZE * subdivision.n_pnts as usize))
            .zip(new_vert_normals.par_chunks_mut(FACET_CHUNK_SIZE * subdivision.n_pnts as usize))
            .zip(new_facets.par_chunks_mut(FACET_CHUNK_SIZE * 3 * subdivision.n_tris as usize))
            .zip(new_facet_normals.par_chunks_mut(FACET_CHUNK_SIZE * subdivision.n_tris as usize))
            .zip(new_facet_areas.par_chunks_mut(FACET_CHUNK_SIZE * subdivision.n_tris as usize))
            .zip(self.facet_normals.par_chunks(FACET_CHUNK_SIZE))
            .enumerate()
            .for_each(
                |(
                    chunk_idx,
                    (
                        (
                            (
                                (
                                    ((facets_chunk, new_verts_chunk), new_vert_normals_chunk),
                                    new_facets_chunk,
                                ),
                                new_facets_normal_chunk,
                            ),
                            new_facets_area_chunk,
                        ),
                        facet_normals_chunk,
                    ),
                )| {
                    let mut new_verts_f64 =
                        vec![DVec3::ZERO; subdivision.n_pnts as usize].into_boxed_slice();
                    facets_chunk
                        .chunks(3)
                        .zip(new_verts_chunk.chunks_mut(subdivision.n_pnts as usize))
                        .zip(new_vert_normals_chunk.chunks_mut(subdivision.n_pnts as usize))
                        .zip(new_facets_chunk.chunks_mut(3 * subdivision.n_tris as usize))
                        .zip(new_facets_normal_chunk.chunks_mut(subdivision.n_tris as usize))
                        .zip(new_facets_area_chunk.chunks_mut(subdivision.n_tris as usize))
                        .zip(facet_normals_chunk.iter())
                        .enumerate()
                        .for_each(
                            |(
                                i,
                                (
                                    (
                                        (
                                            (((facet, new_verts), new_vert_normals), new_facets),
                                            new_facet_normals,
                                        ),
                                        new_area,
                                    ),
                                    facet_normal,
                                ),
                            )| {
                                let original_facet_normal = DVec3::new(
                                    facet_normal.x as f64,
                                    facet_normal.y as f64,
                                    facet_normal.z as f64,
                                );
                                let tri_verts = [
                                    self.verts[facet[0] as usize],
                                    self.verts[facet[1] as usize],
                                    self.verts[facet[2] as usize],
                                ];
                                let tri_norms = [
                                    self.vert_normals[facet[0] as usize],
                                    self.vert_normals[facet[1] as usize],
                                    self.vert_normals[facet[2] as usize],
                                ];
                                // Get interpolated points and normals
                                curved::subdivide_triangle(
                                    &tri_verts,
                                    &tri_norms,
                                    &subdivision.uvs,
                                    &mut new_verts_f64,
                                    new_vert_normals,
                                );
                                // Triangulate the new points
                                let vert_base =
                                    (chunk_idx * FACET_CHUNK_SIZE * subdivision.n_pnts as usize
                                        + i * subdivision.n_pnts as usize)
                                        as u32;
                                TriangleUVSubdivision::triangulate(level, vert_base, new_facets);
                                // Compute the per-facet normals and areas
                                for ((new_facet, new_normal), new_area) in new_facets
                                    .chunks_mut(3)
                                    .zip(new_facet_normals.iter_mut())
                                    .zip(new_area.iter_mut())
                                {
                                    let p0 =
                                        new_verts_f64[new_facet[0] as usize - vert_base as usize];
                                    let p1 =
                                        new_verts_f64[new_facet[1] as usize - vert_base as usize];
                                    let p2 =
                                        new_verts_f64[new_facet[2] as usize - vert_base as usize];
                                    let cross = (p1 - p0).cross(p2 - p0);
                                    *new_area = 0.5 * cross.length();
                                    let mut normal = cross.normalize();
                                    if normal.dot(original_facet_normal) < 0.0 {
                                        normal = -normal;
                                        *new_normal =
                                            (normal.x as f32, normal.y as f32, normal.z as f32)
                                                .into();
                                        new_facet.swap(1, 2);
                                    } else {
                                        *new_normal =
                                            (normal.x as f32, normal.y as f32, normal.z as f32)
                                                .into();
                                    }
                                }
                                new_verts_f64.iter().zip(new_verts.iter_mut()).for_each(
                                    |(vertf64, vert)| {
                                        vert.x = vertf64.x as f32;
                                        vert.y = vertf64.y as f32;
                                        vert.z = vertf64.z as f32;
                                    },
                                );
                            },
                        );
                },
            );
        self.facet_total_area = new_facet_areas.iter().sum::<f64>() as f32;
        self.facet_normals = new_facet_normals;
        self.facet_areas = new_facet_areas
            .iter()
            .map(|x| *x as f32)
            .collect::<Box<[f32]>>();
        self.facets = new_facets;
        self.verts = new_verts;
        self.num_facets = new_num_facets;
        self.num_verts = new_num_verts;
        self.vert_normals = new_vert_normals;
    }

    /// Smooth the mesh by subdividing the triangles into wiggly surfaces.
    pub fn wiggly_smooth(&mut self, level: u32) {
        log::debug!("Wiggly smoothing the mesh with level: {}", level);
        if level == 0 {
            return;
        }
        let subdivision = TriangleUVSubdivision::new(level);
        let subdivided = {
            let dcel = HalfEdgeMesh::new(Cow::Borrowed(&self.verts), &self.facets);
            dcel.subdivide_by_uvs(&subdivision, wiggle::subdivide_triangle)
        };
        println!("subdivided: {:?}", subdivided.darts);
        // let mut new_verts = subdivided.positions;
        // let mut new_facets = vec![u32::MAX; dcel.n_faces() * 3].into_boxed_slice();
        // let mut new_facets_normals = vec![Vec3::ZERO;
        // dcel.n_faces()].into_boxed_slice(); let mut new_verts_normals =
        // vec![Vec3::ZERO; dcel.n_verts()].into_boxed_slice();

        println!("faces of vert 3: {:?}", subdivided.faces_of_vert(3));
        println!("faces of vert 9: {:?}", subdivided.faces_of_vert(9));

        // self.facet_total_area = new_facet_areas.iter().sum::<f64>() as f32;
        // self.facet_normals = new_facet_normals;
        // self.facet_areas = new_facet_areas
        //     .iter()
        //     .map(|x| *x as f32)
        //     .collect::<Box<[f32]>>();
        // self.facets = new_facets;
        // self.verts = new_verts;
        // self.num_facets = new_num_facets;
        // self.num_verts = new_num_verts;
        // self.vert_normals = new_vert_normals;
    }
}

/// Helper struct for micro-surface mesh subdivision.
#[derive(Debug)]
struct TriangleUVSubdivision {
    /// Level of subdivision.
    level: u32,
    /// Number of vertices in total after subdivision.
    n_pnts: u32,
    /// Number of triangles in total after subdivision.
    n_tris: u32,
    /// Number of control points per edge.
    /// The original triangle has 2 control points per edge.
    n_ctrl_pnts: u32,
    /// UV coordinates for the points on the subdivided triangle.
    uvs: Box<[Vec2]>,
    /// Indices of the resulting triangles in the local index space (in terms of
    /// all the vertices inside the triangle).
    /// The subdivision starts from the base triangle, and the indices are
    /// generated in the order of the base triangle.
    /// The indices are generated in the order first by u then by v (v increases
    /// faster), which is the same order as the UV coordinates; thus, the base
    /// edge where u = 0 is the 2nd edge in the original triangle (1st one
    /// has v = 0).
    indices: Box<[u32]>,
    /// Records the edge where the new vertices are located.
    /// 0, 1, 2 means the vertex is the original triangle vertex.
    locations: Box<[VertLoc]>,
}

/// Location of the vertex in the subdivided triangle.
#[derive(Debug, Copy, Clone)]
pub enum VertLoc {
    /// Vertex of the original triangle.
    Vert(u32),
    /// Vertex on the edge formed by the original vertices:
    Edge(u32),
    /// Vertex inside the triangle.
    Inside,
}

impl VertLoc {
    /// Vertex 0 of the original triangle.
    pub const V0: Self = Self::Vert(0);
    /// Vertex 1 of the original triangle.
    pub const V1: Self = Self::Vert(1);
    /// Vertex 2 of the original triangle.
    pub const V2: Self = Self::Vert(2);
    /// Vertex on the edge formed by the original vertices 0 and 1.
    pub const E01: Self = Self::Edge(0);
    /// Vertex on the edge formed by the original vertices 2 and 0.
    pub const E12: Self = Self::Edge(1);
    /// Vertex on the edge formed by the original vertices 1 and 2.
    pub const E02: Self = Self::Edge(2);

    /// Returns true if the original vertex.
    pub fn is_vert(&self) -> bool { matches!(self, Self::Vert(_)) }

    /// Returns true if the vertex is on the edge.
    pub fn is_on_edge(&self) -> bool { matches!(self, Self::Edge(_)) }

    /// Returns true if the vertex is inside the triangle.
    pub fn edge_idx(&self) -> Option<u32> {
        match self {
            Self::Edge(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl TriangleUVSubdivision {
    pub fn new(level: u32) -> Self {
        let n_ctrl_pnts = level + 2;
        let n_pnts = Self::calc_n_pnts(level);
        let n_tris = (level + 1) * (level + 1);
        let uvs = Self::calc_pnts_uvs(n_ctrl_pnts, n_pnts);
        let indices = Self::calc_indices(n_ctrl_pnts, n_tris);
        let locations = Self::calc_locations(n_ctrl_pnts, n_pnts);
        log::debug!(
            "Triangle UV subdivision: level: {}, n_pnts: {}, n_tris: {}",
            level,
            n_pnts,
            n_tris
        );
        Self {
            level,
            n_pnts,
            n_tris,
            n_ctrl_pnts,
            uvs,
            indices,
            locations,
        }
    }

    /// Returns the number of vertices in the triangulation of a facet at the
    /// specified level.
    ///
    /// Level 0 has the original 3 vertices.
    const fn calc_n_pnts(level: u32) -> u32 {
        if level == 0 {
            3
        } else {
            Self::calc_n_pnts(level - 1) + level + 2
        }
    }

    /// Computes the UV coordinates for the points on the subdivided triangle.
    ///
    /// The UV coordinates are generated in the order first by u then by v (v
    /// increases faster).
    fn calc_pnts_uvs(n_ctrl_pnts_per_edge: u32, n_pnts_per_facet: u32) -> Box<[Vec2]> {
        let uv_vals = RangeByStepCountInclusive::new(0.0f32, 1.0, n_ctrl_pnts_per_edge as usize)
            .values()
            .map(|x| x.min(1.0))
            .collect::<Box<_>>();
        let mut uvs = Box::new_uninit_slice(n_pnts_per_facet as usize);
        let mut i = 0;
        for u in uv_vals.iter() {
            for v in uv_vals.iter() {
                if u + v <= 1.0 || (u + v - 1.0).abs() <= 1e-6 {
                    uvs[i].write(Vec2::new(*u, *v));
                    i += 1;
                }
            }
        }
        assert_eq!(
            i, n_pnts_per_facet as usize,
            "Number of UVs does not match the number of points"
        );
        unsafe { uvs.assume_init() }
    }

    /// Computes the indices subdivide the triangle into smaller triangles.
    ///
    /// Construct the new triangles from the base where u = 0, level by level
    /// until the last level.
    ///
    /// Index 0 matches the first vertex of the face,
    /// Index num_ctrl_pts - 1 matches the last vertex of the face (along which
    /// the u = 0). The index of the last vertex calculated is the last
    /// vertex of the face.
    ///
    /// Example:
    ///  num_ctrl_pnts = 3
    ///  level = 2 = num_ctrl_pnts - 1
    ///         5
    ///       /  \     <- level 1, 1 triangle, n_pnts_base = 2
    ///      4 -- 3
    ///     / \ /  \   <- level 0, 3 triangles, n_pnts_base = 3
    ///   2 -- 1 -- 0  <- base, u = 0
    ///
    ///  num_ctrl_pnts = 4
    ///  level = 3 = num_ctrl_pnts - 1
    ///         9
    ///       /  \        <- level 2, 1 triangle, n_pnts_base = 2, pnts_idx = 7
    ///      8 -- 7
    ///    /  \ /  \      <- level 1, 3 triangles, n_pnts_base = 3, pnts_idx = 4
    ///   6 -- 5 -- 4
    ///  /  \ /  \ /  \   <- level 0, 6 triangles, n_pnts_base = 4, pnts_idx = 0
    /// 3 -- 2 -- 1 -- 0  <- base, u = 0
    pub fn calc_indices(n_ctrl_pnts: u32, n_tris: u32) -> Box<[u32]> {
        let mut tris = vec![0; n_tris as usize * 3].into_boxed_slice();
        let mut tri_idx = 0;
        let mut pnts_idx = 0;
        for l in 0..n_ctrl_pnts - 1 {
            let n_pnts_base = n_ctrl_pnts - l;
            for i in 0..n_pnts_base - 1 {
                tris[tri_idx * 3] = i + pnts_idx;
                tris[tri_idx * 3 + 1] = i + n_pnts_base + pnts_idx;
                tris[tri_idx * 3 + 2] = i + 1 + pnts_idx;
                tri_idx += 1;
                if i < n_pnts_base - 2 {
                    tris[tri_idx * 3] = i + n_pnts_base + pnts_idx;
                    tris[tri_idx * 3 + 1] = i + 1 + n_pnts_base + pnts_idx;
                    tris[tri_idx * 3 + 2] = i + 1 + pnts_idx;
                    tri_idx += 1;
                }
            }
            pnts_idx += n_pnts_base;
        }
        tris
    }

    /// Computes the locations of the vertices in the subdivided triangle.
    pub fn calc_locations(n_ctrl_pnts: u32, n_pnts: u32) -> Box<[VertLoc]> {
        let mut locs = vec![VertLoc::Inside; n_pnts as usize].into_boxed_slice();
        let mut idx = 0;
        for l in 0..n_ctrl_pnts {
            let n_pnts_base = n_ctrl_pnts - l;
            for i in 0..n_pnts_base {
                if l == 0 {
                    if i == 0 {
                        locs[idx] = VertLoc::V0;
                    } else if i == n_pnts_base - 1 {
                        locs[idx] = VertLoc::V2;
                    } else {
                        locs[idx] = VertLoc::E02;
                    }
                } else if l == n_ctrl_pnts - 1 {
                    locs[idx] = VertLoc::V1;
                } else {
                    if i == 0 {
                        locs[idx] = VertLoc::E01;
                    } else if i == n_pnts_base - 1 {
                        locs[idx] = VertLoc::E12;
                    } else {
                        locs[idx] = VertLoc::Inside;
                    }
                }
                idx += 1;
            }
        }

        locs
    }

    // TODO: rewrite using pre-computed indices
    // TODO: check triangle winding order
    /// Triangulate a single facet into a set of triangles according to the
    /// specified level.
    ///
    /// # Arguments
    ///
    /// * `level` - Level of subdivision, 0 means no subdivision.
    /// * `base` - Base index of the first vertex of the facet.
    /// * `tris` - Array of vertex indices forming triangles.
    pub fn triangulate(level: u32, base: u32, tris: &mut [u32]) {
        let num_ctrl_pts = level + 2;
        let mut offset = 0;
        let mut tris_idx = 0;
        for l in 0..num_ctrl_pts {
            let nl = num_ctrl_pts - l;
            for i in 0..nl - 1 {
                if i < nl - 2 {
                    tris[tris_idx..tris_idx + 6].copy_from_slice(&[
                        // 1st
                        base + offset + i,
                        base + offset + i + 1,
                        base + offset + i + nl,
                        // 2nd
                        base + offset + i + 1,
                        base + offset + i + nl,
                        base + offset + i + nl + 1,
                    ]);
                    tris_idx += 6;
                } else {
                    tris[tris_idx..tris_idx + 3].copy_from_slice(&[
                        // 1st
                        base + offset + i,
                        base + offset + i + 1,
                        base + offset + i + nl,
                    ]);
                    tris_idx += 3;
                };
            }
            offset += nl;
        }
    }
}

/// Origin of the micro-geometry height field.
#[cfg_attr(feature = "pybind", pyo3::pyclass)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MicroSurfaceOrigin {
    /// Micro-geometry height field from the paper
    /// "Predicting Appearance from the Measured Micro-geometry of Metal
    /// Surfaces".
    Dong2015,
    /// Micro-geometry height field from a µsurf confocal microscope system.
    Usurf,
    /// Micro-geometry height field from OmniSurf3D.
    OmniSurf3D,
}

impl MicroSurface {
    #[rustfmt::skip]
    /// Creates micro-geometry height field by reading the samples stored in
    /// different file format. Supported formats are
    ///
    /// 1. Ascii Matrix file (plain text) coming from Predicting Appearance from
    ///    Measured Micro-geometry of Metal Surfaces.
    ///
    /// 2. Plain text data coming from a µsurf confocal microscope system.
    ///
    /// 3. Micro-surface height field file (binary format, ends with *.vgms).
    ///
    /// 4. Micro-surface height field cache file (binary format, ends with *.vgcc).
    pub fn read_from_file(
        filepath: &Path,
        origin: Option<MicroSurfaceOrigin>,
    ) -> Result<MicroSurface, VgonioError> {
        use crate::io;
        let file = File::open(filepath).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!("Failed to open micro-surface file: {}", filepath.display()))
        })?;
        let mut reader = BufReader::new(file);

        if let Some(origin) = origin {
            // If the origin is specified, call directly corresponding loading function.
            match origin {
                MicroSurfaceOrigin::Dong2015 => io::read_ascii_dong2015(&mut reader, filepath),
                MicroSurfaceOrigin::Usurf => io::read_ascii_usurf(&mut reader, filepath),
                MicroSurfaceOrigin::OmniSurf3D => io::read_omni_surf_3d(&mut reader, filepath),
            }
        } else {
            // Otherwise, try to figure out the file format by reading the first several bytes.
            let mut buf = [0_u8; 4];
            reader.read_exact(&mut buf).map_err(|err| {
                VgonioError::from_io_error(
                    err,
                    format!("Failed to read first 4 bytes of file: {}", filepath.display()))
            })?;
            reader.seek(std::io::SeekFrom::Start(0)).unwrap(); // Reset the cursor to the beginning of the file.
            match std::str::from_utf8(&buf).unwrap() {
                "Asci" => io::read_ascii_dong2015(&mut reader, filepath),
                "DATA" => io::read_ascii_usurf(&mut reader, filepath),
                "VGMS" => {
                    let (header, samples) = io::vgms::read(&mut reader)
                        .map_err(|err| VgonioError::from_read_file_error(ReadFileError {
                            path: filepath.to_owned().into_boxed_path(),
                            kind: err,
                        }, "Failed to read VGMS file."))?;
                    Ok(MicroSurface::from_samples(
                        header.extra.rows as usize,
                        header.extra.cols as usize,
                        (header.extra.du, header.extra.dv),
                        header.extra.unit,
                        samples,
                        filepath
                            .file_stem()
                            .and_then(|name| name.to_str().map(|name| name.to_owned())),
                        Some(filepath.to_owned()),
                    ))
                }
                "Omni" => io::read_omni_surf_3d(&mut reader, filepath),
                _ => Err(VgonioError::new("Unknown file format.", None))
            }
        }
            .map(|mut ms| {
                log::debug!("Loaded micro-surface from file: {}", filepath.display());
                log::debug!("- Resolution: {} x {}", ms.rows, ms.cols);
                log::debug!("- Spacing: {} x {}", ms.du, ms.dv);
                ms.repair();
                ms
            })
    }

    #[cfg(feature = "surf-obj")]
    /// Creates micro-geometry height field by reading the samples stored in
    /// Wavefront OBJ file.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the Wavefront OBJ file.
    /// * `axis` - Axis of the height of the micro-surface.
    /// * `unit` - Length unit of the micro-surface.
    pub fn read_from_wavefront(
        filepath: &Path,
        axis: Axis,
        unit: LengthUnit,
    ) -> Result<MicroSurface, VgonioError> {
        log::debug!(
            "Loading micro-surface from Wavefront OBJ file: {}",
            filepath.display()
        );
        let file = File::open(filepath).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!("Failed to open micro-surface file: {}", filepath.display()),
            )
        })?;
        let mut reader = BufReader::new(file);
        io::read_wavefront(&mut reader, filepath, axis, unit)
    }

    /// Save the micro-surface height field to a file.
    pub fn write_to_file(
        &self,
        filepath: &Path,
        encoding: FileEncoding,
        compression: CompressionScheme,
    ) -> Result<(), VgonioError> {
        let mut file = File::create(filepath).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!(
                    "Failed to create micro-surface file: {}",
                    filepath.display()
                ),
            )
        })?;
        use io::vgms::VgmsHeaderExt;

        let timestamp = {
            let mut timestamp = [0_u8; 32];
            timestamp.copy_from_slice(base::utils::iso_timestamp().as_bytes());
            timestamp
        };
        let header = Header {
            meta: HeaderMeta {
                version: Version::new(0, 1, 0),
                timestamp,
                length: 0,
                sample_size: 4,
                encoding,
                compression,
            },
            extra: VgmsHeaderExt {
                unit: self.unit,
                cols: self.cols as u32,
                rows: self.rows as u32,
                du: self.du,
                dv: self.dv,
            },
        };
        let mut writer = BufWriter::new(&mut file);
        io::vgms::write(&mut writer, header, &self.samples).map_err(|err| {
            VgonioError::from_write_file_error(
                WriteFileError {
                    path: filepath.to_owned().into_boxed_path(),
                    kind: err,
                },
                "Failed to write VGMS file.",
            )
        })?;
        writer.flush().map_err(|err| {
            VgonioError::from_write_file_error(
                WriteFileError {
                    path: filepath.to_owned().into_boxed_path(),
                    kind: WriteFileErrorKind::Write(err),
                },
                "Failed to flush VGMS file.",
            )
        })
    }
}

#[cfg(feature = "pybind")]
mod pybind {
    use crate::MicroSurface;
    use pyo3::prelude::*;

    #[pyfunction]
    fn sum_as_string(a: i32, b: i32) -> PyResult<String> { Ok((a + b).to_string()) }

    #[pymodule]
    fn vgonio_surf(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(sum_as_string, module)?)?;
        module.add_class::<MicroSurface>();
        Ok(())
    }
}
