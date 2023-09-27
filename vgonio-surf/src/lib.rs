#![feature(byte_slice_trim_ascii)]
//! Heightfield
#![warn(missing_docs)]

#[cfg(feature = "surf-obj")]
use vgcore::math::Axis;
#[cfg(feature = "surf-gen")]
mod gen;
#[cfg(feature = "surf-gen")]
pub use gen::*;
pub mod io;

#[cfg(feature = "embree")]
use embree::{BufferUsage, Device, Format, Geometry, GeometryKind};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek},
    path::{Path, PathBuf},
};
use vgcore::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding, ReadFileError, WriteFileError},
    math::{rcp_f32, Aabb, Vec3},
    units::LengthUnit,
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
    /// Offset the height field by a arbitrary value.
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
    /// # use vgcore::units::LengthUnit;
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
    /// # use vgcore::units::LengthUnit;
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

    /// Create a micro-surface from a array of elevation values.
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
    /// # use vgcore::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let height_field =
    ///     MicroSurface::from_samples(3, 3, 0.5, 0.5, LengthUnit::UM, &samples, None, None);
    /// assert_eq!(height_field.samples_count(), 9);
    /// assert_eq!(height_field.cells_count(), 4);
    /// assert_eq!(height_field.cols, 3);
    /// assert_eq!(height_field.rows, 3);
    /// ```
    pub fn from_samples<S: AsRef<[f32]>>(
        rows: usize,
        cols: usize,
        du: f32,
        dv: f32,
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
            du,
            dv,
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
    /// # use vgcore::units::LengthUnit;
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
    /// # use vgcore::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, 0.2, 0.2, LengthUnit::UM, samples, None, None);
    /// assert_eq!(msurf.samples_count(), 9);
    /// ```
    pub fn samples_count(&self) -> usize { self.cols * self.rows }

    /// Returns the number of cells of height field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgcore::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, 0.2, 0.2, LengthUnit::UM, samples, None, None);
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
    /// # use vgcore::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, 0.2, 0.2, LengthUnit::MM, samples, None, None);
    /// assert_eq!(msurf.sample_at(2, 2), 0.1);
    /// ```
    ///
    /// ```should_panic
    /// # use vgcore::units::LengthUnit;
    /// # use vgonio_surf::MicroSurface;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, 0.2, 0.3, LengthUnit::MM, samples, None, None);
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

    /// Computes the root mean square height of the height field.
    pub fn rms_height(&self) -> f32 {
        let rcp_n = rcp_f32(self.samples.len() as f32);
        self.samples
            .iter()
            .fold(0.0, |acc, x| acc + x * x * rcp_n)
            .sqrt()
    }

    /// Triangulate the heightfield into a [`MicroSurfaceMesh`].
    /// The triangulation is done in the XZ plane.
    pub fn as_micro_surface_mesh(
        &self,
        offset: HeightOffset,
        pattern: TriangulationPattern,
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

        let mut normals = vec![Vec3::ZERO; num_faces];
        let mut areas = vec![0.0; num_faces];
        let mut total_area = 0.0;

        for i in 0..num_faces {
            let p0 = verts[tri_faces[i * 3] as usize];
            let p1 = verts[tri_faces[i * 3 + 1] as usize];
            let p2 = verts[tri_faces[i * 3 + 2] as usize];
            let cross = (p1 - p0).cross(p2 - p0);
            normals[i] = cross.normalize();
            areas[i] = 0.5 * cross.length();
            total_area += areas[i];
        }

        MicroSurfaceMesh {
            uuid: uuid::Uuid::new_v4(),
            num_facets: num_faces,
            num_verts: verts.len(),
            bounds: extent,
            verts,
            facets: tri_faces,
            facet_normals: normals,
            facet_areas: areas,
            msurf: self.uuid,
            unit: self.unit,
            height_offset,
            facet_total_area: total_area,
        }
    }

    /// Generate vertices from the height values.
    ///
    /// The vertices are generated following the order from left to right, top
    /// to bottom. The center of the heightfield is at the origin.
    ///
    /// The generated vertices are aligned to the xy plane of the right-handed,
    /// Z-up coordinate system.
    #[doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../misc/imgs/heightfield.svg"))]
    pub fn generate_vertices(&self, height_offset: f32) -> (Vec<Vec3>, Aabb) {
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
        let mut positions: Vec<Vec3> = vec![Vec3::ZERO; rows * cols];

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

                    for k in 0..3 {
                        if p[k] > extent.max[k] {
                            extent.max[k] = p[k];
                        }
                        if p[k] < extent.min[k] {
                            extent.min[k] = p[k];
                        }
                    }

                    *pos = p;
                }
                extent
            })
            .reduce(Aabb::empty, Aabb::union);

        #[cfg(feature = "bench")]
        log::info!(
            "Height field vertices generated in {:?} ms",
            t.elapsed().as_millis()
        );

        (positions, extent)
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
                self.du,
                self.dv,
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
    #[default]
    BottomLeftToTopRight,
    /// Triangulate from top to bottom, right to left.
    /// 0  <--  1
    /// |  \  B |
    /// | A  \  |
    /// 2  -->  3
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
/// Pattern is specified by `TriangulationPattern`.
///
/// # Returns
///
/// Vec<u32>: An array of vertex indices forming triangles.
pub fn regular_grid_triangulation(
    rows: usize,
    cols: usize,
    pattern: TriangulationPattern,
) -> Vec<u32> {
    let mut triangulate: Box<dyn FnMut(usize, usize, &mut usize, &mut [u32])> = match pattern {
        TriangulationPattern::BottomLeftToTopRight => Box::new(
            |i: usize, col: usize, tri: &mut usize, indices: &mut [u32]| {
                if col == 0 {
                    indices[*tri] = i as u32;
                    indices[*tri + 1] = (i + cols) as u32;
                    indices[*tri + 2] = (i + 1) as u32;
                    *tri += 3;
                } else if col == cols - 1 {
                    indices[*tri] = i as u32;
                    indices[*tri + 1] = (i + cols - 1) as u32;
                    indices[*tri + 2] = (i + cols) as u32;
                    *tri += 3;
                } else {
                    indices[*tri] = i as u32;
                    indices[*tri + 1] = (i + cols - 1) as u32;
                    indices[*tri + 2] = (i + cols) as u32;
                    indices[*tri + 3] = i as u32;
                    indices[*tri + 4] = (i + cols) as u32;
                    indices[*tri + 5] = (i + 1) as u32;
                    *tri += 6;
                }
            },
        ),
        TriangulationPattern::TopLeftToBottomRight => Box::new(
            |i: usize, col: usize, tri: &mut usize, indices: &mut [u32]| {
                if col == 0 {
                    indices[*tri] = i as u32;
                    indices[*tri + 1] = (i + cols) as u32;
                    indices[*tri + 2] = (i + cols + 1) as u32;
                    *tri += 3;
                } else if col == cols - 1 {
                    indices[*tri] = i as u32;
                    indices[*tri + 1] = (i - 1) as u32;
                    indices[*tri + 2] = (i + cols) as u32;
                    *tri += 3;
                } else {
                    indices[*tri] = i as u32;
                    indices[*tri + 1] = (i - 1) as u32;
                    indices[*tri + 2] = (i + cols) as u32;
                    indices[*tri + 3] = i as u32;
                    indices[*tri + 4] = (i + cols) as u32;
                    indices[*tri + 5] = (i + cols + 1) as u32;
                    *tri += 6;
                }
            },
        ),
    };

    let mut indices: Vec<u32> = vec![0; 2 * (cols - 1) * (rows - 1) * 3];
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
/// Created from [`MicroSurface`](`crate::msurf::MicroSurface`) using
/// [`MicroSurface::as_micro_surface_mesh`](`crate::msurf::MicroSurface::as_micro_surface_mesh`),
/// and has the same length unit ([`Micrometres`](`crate::units::Micrometres`))
/// as the [`MicroSurface`](`crate::msurf::MicroSurface`).
///
/// By default, the generated mesh is located on XZ plane in right-handed Y up
/// coordinate system.
/// See [`MicroSurface::as_micro_surface_mesh`](`crate::msurf::MicroSurface::as_micro_surface_mesh`)
#[derive(Debug)]
pub struct MicroSurfaceMesh {
    /// Unique identifier, different from the [`MicroSurface`] uuid.
    pub uuid: uuid::Uuid,

    /// Uuid of the [`MicroSurface`] from which the mesh is generated.
    pub msurf: uuid::Uuid,

    /// Axis-aligned bounding box of the mesh.
    pub bounds: Aabb,

    /// Height offset applied to original heightfield.
    pub height_offset: f32,

    /// Number of triangles in the mesh.
    pub num_facets: usize,

    /// Number of vertices in the mesh.
    pub num_verts: usize,

    /// Vertices of the mesh.
    pub verts: Vec<Vec3>,

    /// Vertex indices forming the facets which are triangles.
    pub facets: Vec<u32>,

    /// Normal vectors of each facet.
    pub facet_normals: Vec<Vec3>,

    /// Surface area of each facet.
    pub facet_areas: Vec<f32>,

    /// Total surface area of the triangles.
    pub facet_total_area: f32,

    /// Length unit of inherited from the
    /// [`MicroSurface`](`crate::msurf::MicroSurface`).
    pub unit: LengthUnit,
}

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

    /// Constructs an embree geometry from the `MicroSurfaceMesh`.
    #[cfg(feature = "embree")]
    pub fn as_embree_geometry<'g>(&self, device: &Device) -> Geometry<'g> {
        let mut geom = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
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
}

/// Origin of the micro-geometry height field.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum MicroSurfaceOrigin {
    /// Micro-geometry height field from the paper
    /// "Predicting Appearance from the Measured Micro-geometry of Metal
    /// Surfaces".
    Dong2015,
    /// Micro-geometry height field from µsurf confocal microscope system.
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
    ///    Measured Microgeometry of Metal Surfaces.
    ///
    /// 2. Plain text data coming from µsurf confocal microscope system.
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
            // If origin is specified, call directly corresponding loading function.
            match origin {
                MicroSurfaceOrigin::Dong2015 => io::read_ascii_dong2015(&mut reader, filepath),
                MicroSurfaceOrigin::Usurf => io::read_ascii_usurf(&mut reader, filepath),
                MicroSurfaceOrigin::OmniSurf3D => io::read_omni_surf_3d(&mut reader, filepath),
            }
        } else {
            // Otherwise, try to figure out the file format by reading first several bytes.
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
                        header.rows as usize,
                        header.cols as usize,
                        header.du,
                        header.dv,
                        header.unit,
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
        let header = io::vgms::Header {
            rows: self.rows as u32,
            cols: self.cols as u32,
            du: self.du,
            dv: self.dv,
            unit: self.unit,
            sample_data_size: 4,
            encoding,
            compression,
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
        })
    }
}
