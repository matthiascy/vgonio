//! Heightfield

use crate::{app::cache::Asset, measure::rtc::isect::Aabb};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, path::PathBuf};

mod io;
mod mesh;

use crate::{
    app::gfx::RenderableMesh,
    units::{nm, um, Length, LengthUnit, Micrometres, Nanometres},
};
pub use mesh::MicroSurfaceMesh;

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

/// Alignment used when generating height field.
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum AxisAlignment {
    /// Aligned with XY plane.
    XY,

    /// Aligned with XZ plane.
    #[default]
    XZ,

    /// Aligned with YX plane.
    YX,

    /// Aligned with YZ plane.
    YZ,

    /// Aligned with ZX plane.
    ZX,

    /// Aligned with ZY plane.
    ZY,
}

/// Representation of the micro-surface.
///
/// All measurements are in micrometers.
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
    /// # use vgonio::msurf::{AxisAlignment, MicroSurface};
    /// # use vgonio::units::um;
    /// let height_field = MicroSurface::new(10, 10, um!(0.11), um!(0.11), um!(0.12));
    /// assert_eq!(height_field.samples_count(), 100);
    /// assert_eq!(height_field.cells_count(), 81);
    /// ```
    pub fn new(
        cols: usize,
        rows: usize,
        du: Micrometres,
        dv: Micrometres,
        height: Micrometres,
    ) -> Self {
        assert!(cols > 1 && rows > 1);
        let mut samples = Vec::new();
        let height = height.as_f32();
        samples.resize(cols * rows, height);
        MicroSurface {
            uuid: uuid::Uuid::new_v4(),
            name: gen_micro_surface_name(),
            path: None,
            rows,
            cols,
            du: du.as_f32(),
            dv: dv.as_f32(),
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
    /// # use vgonio::msurf::{AxisAlignment, MicroSurface};
    /// # use vgonio::units::um;
    /// let msurf = MicroSurface::new_by(4, 4, um!(0.1), um!(0.1), |row, col| um!((row + col) as f32));
    /// assert_eq!(msurf.samples_count(), 16);
    /// assert_eq!(msurf.max, 6.0);
    /// assert_eq!(msurf.min, 0.0);
    /// assert_eq!(msurf.samples[0], 0.0);
    /// assert_eq!(msurf.samples[2], 2.0);
    /// assert_eq!(msurf.sample_at(2, 3), um!(5.0));
    /// ```
    pub fn new_by<F>(
        cols: usize,
        rows: usize,
        du: Micrometres,
        dv: Micrometres,
        setter: F,
    ) -> MicroSurface
    where
        F: Fn(usize, usize) -> Micrometres,
    {
        assert!(cols > 1 && rows > 1);
        let mut samples = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                samples.push(setter(r, c).as_f32());
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
            du: du.as_f32(),
            dv: dv.as_f32(),
            max,
            min,
            samples,
            median: (min + max) / 2.0,
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
    /// # use vgonio::msurf::MicroSurface;
    /// # use vgonio::units::{um, UMicrometre};
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let height_field =
    ///     MicroSurface::from_samples::<UMicrometre>(3, 3, um!(0.5), um!(0.5), samples, None);
    /// assert_eq!(height_field.samples_count(), 9);
    /// assert_eq!(height_field.cells_count(), 4);
    /// assert_eq!(height_field.cols, 3);
    /// assert_eq!(height_field.rows, 3);
    /// ```
    pub fn from_samples<U: LengthUnit>(
        cols: usize,
        rows: usize,
        du: Micrometres,
        dv: Micrometres,
        samples: Vec<f32>,
        path: Option<PathBuf>,
    ) -> MicroSurface {
        debug_assert!(cols > 0 && rows > 0 && samples.len() == cols * rows);
        let to_micrometre_factor = U::FACTOR_TO_MICROMETRE;
        let max = samples.iter().fold(f32::MIN, |acc, x| f32::max(acc, *x)) * to_micrometre_factor;
        let min = samples.iter().fold(f32::MAX, |acc, x| f32::min(acc, *x)) * to_micrometre_factor;

        let samples = if to_micrometre_factor == 1.0 {
            samples
        } else {
            let mut _samples = samples;
            for s in _samples.iter_mut() {
                *s *= to_micrometre_factor;
            }
            _samples
        };

        MicroSurface {
            uuid: uuid::Uuid::new_v4(),
            name: gen_micro_surface_name(),
            path,
            rows,
            cols,
            du: du.as_f32(),
            dv: dv.as_f32(),
            max,
            min,
            samples,
            median: (min + max) / 2.0,
        }
    }

    /// Returns the dimension of the surface (rows * du, cols * dv)
    /// # Examples
    ///
    /// ```
    /// # use vgonio::msurf::MicroSurface;
    /// use vgonio::units::um;
    /// let msurf = MicroSurface::new(100, 100, um!(0.1), um!(0.1), um!(0.1));
    /// assert_eq!(msurf.dimension(), (um!(10.0), um!(10.0)));
    /// ```
    pub fn dimension(&self) -> (Micrometres, Micrometres) {
        (
            um!(self.rows as f32 * self.du),
            um!(self.cols as f32 * self.dv),
        )
    }

    /// Returns the number of samples of height field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgonio::msurf::MicroSurface;
    /// # use vgonio::units::{um, UMetre};
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf =
    ///     MicroSurface::from_samples::<UMetre>(3, 3, um!(0.2), um!(0.2), samples, Default::default());
    /// assert_eq!(msurf.samples_count(), 9);
    /// ```
    pub fn samples_count(&self) -> usize { self.cols * self.rows }

    /// Returns the number of cells of height field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgonio::msurf::MicroSurface;
    /// # use vgonio::units::{um, UMicrometre};
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples::<UMicrometre>(
    ///     3,
    ///     3,
    ///     um!(0.2),
    ///     um!(0.2),
    ///     samples,
    ///     Default::default(),
    /// );
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
    /// # use vgonio::msurf::MicroSurface;
    /// # use vgonio::units::{um, UMillimetre};
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples::<UMillimetre>(
    ///     3,
    ///     3,
    ///     um!(0.2),
    ///     um!(0.2),
    ///     samples,
    ///     Default::default(),
    /// );
    /// assert_eq!(msurf.sample_at(2, 2), um!(0.1 * 1000.0));
    /// ```
    ///
    /// ```should_panic
    /// # use vgonio::msurf::MicroSurface;
    /// # use vgonio::units::{um, UMillimetre};
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples::<UMillimetre>(
    ///     3,
    ///     3,
    ///     um!(0.2),
    ///     um!(0.3),
    ///     samples,
    ///     Default::default(),
    /// );
    /// let h = msurf.sample_at(4, 4);
    /// ```
    pub fn sample_at(&self, col: usize, row: usize) -> Micrometres {
        assert!(col < self.cols);
        assert!(row < self.rows);
        um!(self.samples[row * self.cols + col])
    }

    /// Fill holes on the height field.
    /// TODO: bilinear interpolation
    pub fn fill_holes(&mut self) {
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

    /// Triangulate the heightfield into a [`MicroSurfaceMesh`].
    pub fn as_micro_surface_mesh(&self, alignment: AxisAlignment) -> MicroSurfaceMesh {
        let (verts, extent) = self.generate_vertices(alignment);
        let tri_faces = grid_triangulation_regular(self.cols, self.rows);
        let num_faces = tri_faces.len() / 3;

        let mut normals = vec![Vec3::ZERO; num_faces];
        let mut areas = vec![0.0; num_faces];

        for i in 0..num_faces {
            let p0 = verts[tri_faces[i * 3] as usize];
            let p1 = verts[tri_faces[i * 3 + 1] as usize];
            let p2 = verts[tri_faces[i * 3 + 2] as usize];
            let cross = (p1 - p0).cross(p2 - p0);
            normals[i] = cross.normalize();
            areas[i] = 0.5 * cross.length();
        }

        MicroSurfaceMesh {
            uuid: uuid::Uuid::new_v4(),
            num_facets: num_faces,
            num_verts: verts.len(),
            extent,
            alignment,
            verts,
            facets: tri_faces,
            facet_normals: normals,
            facet_areas: areas,
            msurf: self.uuid,
        }
    }

    /// Triangulate the heightfield then convert it into a [`RenderableMesh`].
    pub fn as_renderable_mesh(
        &self,
        device: &wgpu::Device,
        alignment: AxisAlignment,
    ) -> RenderableMesh {
        RenderableMesh::from_micro_surface(device, self, alignment)
    }

    /// Generate vertices from the height values.
    ///
    /// The vertices are generated following the order from left to right, top
    /// to bottom.
    pub(crate) fn generate_vertices(&self, alignment: AxisAlignment) -> (Vec<Vec3>, Aabb) {
        log::info!(
            "Generating height field vertices with {:?} alignment",
            alignment
        );
        let (rows, cols, half_rows, half_cols, du, dv) = (
            self.rows,
            self.cols,
            self.rows / 2,
            self.cols / 2,
            self.du,
            self.dv,
        );
        let mut positions: Vec<Vec3> = Vec::with_capacity(rows * cols);
        let mut extent = Aabb::default();
        for r in 0..rows {
            for c in 0..cols {
                let u = (c as f32 - half_cols as f32) * du;
                let v = (r as f32 - half_rows as f32) * dv;
                let h = self.samples[r * cols + c];
                let p = match alignment {
                    AxisAlignment::XY => Vec3::new(u, v, h),
                    AxisAlignment::XZ => Vec3::new(u, h, v),
                    AxisAlignment::YX => Vec3::new(v, u, h),
                    AxisAlignment::YZ => Vec3::new(h, u, v),
                    AxisAlignment::ZX => Vec3::new(v, h, u),
                    AxisAlignment::ZY => Vec3::new(h, v, u),
                };
                for k in 0..3 {
                    if p[k] > extent.max[k] {
                        extent.max[k] = p[k];
                    }
                    if p[k] < extent.min[k] {
                        extent.min[k] = p[k];
                    }
                }
                positions.push(p);
            }
        }

        (positions, extent)
    }
}

/// Generate triangle indices for grid triangulation.
///
/// Triangle winding is counter-clockwise.
/// 0  <-- 1
/// |   /  |
/// |  /   |
/// 2  --> 3
///
/// # Returns
///
/// Vec<u32>: An array of vertex indices forming triangles.
pub(crate) fn grid_triangulation_regular(cols: usize, rows: usize) -> Vec<u32> {
    let mut indices: Vec<u32> = vec![0; 2 * (cols - 1) * (rows - 1) * 3];
    let mut tri = 0;
    for i in 0..cols * rows {
        let row = i / cols;
        let col = i % cols;

        // last row
        if row == rows - 1 {
            continue;
        }

        if col == 0 {
            indices[tri] = i as u32;
            indices[tri + 1] = (i + cols) as u32;
            indices[tri + 2] = (i + 1) as u32;
            tri += 3;
        } else if col == cols - 1 {
            indices[tri] = i as u32;
            indices[tri + 1] = (i + cols - 1) as u32;
            indices[tri + 2] = (i + cols) as u32;
            tri += 3;
        } else {
            indices[tri] = i as u32;
            indices[tri + 1] = (i + cols - 1) as u32;
            indices[tri + 2] = (i + cols) as u32;
            indices[tri + 3] = i as u32;
            indices[tri + 4] = (i + cols) as u32;
            indices[tri + 5] = (i + 1) as u32;
            tri += 6;
        }
    }

    indices
}
