//! Heightfield

use crate::{app::cache::Asset, measure::rtc::Aabb};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::{
    app::gfx::RenderableMesh,
    units::{um, Length, LengthUnit, Micrometres},
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
    /// let height_field = MicroSurface::from_samples(3, 3, um!(0.5), um!(0.5), &samples, None);
    /// assert_eq!(height_field.samples_count(), 9);
    /// assert_eq!(height_field.cells_count(), 4);
    /// assert_eq!(height_field.cols, 3);
    /// assert_eq!(height_field.rows, 3);
    /// ```
    pub fn from_samples<U: LengthUnit, S: AsRef<[f32]>>(
        cols: usize,
        rows: usize,
        du: Length<U>,
        dv: Length<U>,
        samples: S,
        path: Option<PathBuf>,
    ) -> MicroSurface {
        debug_assert!(
            cols > 0 && rows > 0 && samples.as_ref().len() == cols * rows,
            "Samples count must be equal to cols * rows"
        );
        let to_micrometre_factor = U::FACTOR_TO_MICROMETRE;
        let (min, max) = {
            let (max, min) = samples
                .as_ref()
                .iter()
                .fold((f32::MIN, f32::MAX), |(max, min), x| {
                    (f32::max(max, *x), f32::min(min, *x))
                });
            (min * to_micrometre_factor, max * to_micrometre_factor)
        };

        let samples = if to_micrometre_factor == 1.0 {
            samples.as_ref().to_owned()
        } else {
            samples
                .as_ref()
                .iter()
                .map(|x| x * to_micrometre_factor)
                .collect::<Vec<_>>()
        };

        MicroSurface {
            uuid: uuid::Uuid::new_v4(),
            name: gen_micro_surface_name(),
            path,
            rows,
            cols,
            du: du.as_f32() * to_micrometre_factor,
            dv: dv.as_f32() * to_micrometre_factor,
            max,
            min,
            samples,
            median: min * 0.5 + max * 0.5,
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
    /// # use vgonio::units::{m};
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, m!(0.2), m!(0.2), samples, None);
    /// assert_eq!(msurf.samples_count(), 9);
    /// ```
    pub fn samples_count(&self) -> usize { self.cols * self.rows }

    /// Returns the number of cells of height field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgonio::msurf::MicroSurface;
    /// # use vgonio::units::{um};
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, um!(0.2), um!(0.2), samples, None);
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
    /// # use vgonio::units::{mm, um};
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, mm!(0.2), mm!(0.2), samples, Default::default());
    /// assert_eq!(msurf.sample_at(2, 2), um!(0.1 * 1000.0));
    /// ```
    ///
    /// ```should_panic
    /// # use vgonio::msurf::MicroSurface;
    /// # use vgonio::units::{mm};
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let msurf = MicroSurface::from_samples(3, 3, mm!(0.2), mm!(0.3), samples, Default::default());
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
    /// The triangulation is done in the XZ plane.
    pub fn as_micro_surface_mesh(&self) -> MicroSurfaceMesh {
        let (verts, extent) = self.generate_vertices(AxisAlignment::XZ);
        let tri_faces = regular_grid_triangulation(self.cols, self.rows);
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
            bounds: extent,
            verts,
            facets: tri_faces,
            facet_normals: normals,
            facet_areas: areas,
            msurf: self.uuid,
        }
    }

    /// Triangulate the heightfield then convert it into a [`RenderableMesh`].
    pub fn as_renderable_mesh(&self, device: &wgpu::Device) -> RenderableMesh {
        RenderableMesh::from_micro_surface(device, self)
    }

    /// Generate vertices from the height values.
    ///
    /// The vertices are generated following the order from left to right, top
    /// to bottom. The vertices are also aligned to the given axis. The center
    /// of the heightfield is at the origin.
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
/// The grid is assumed to be a regular grid with `cols` columns and `rows`
/// rows. The triangles are generated in counter-clockwise order. The
/// triangulation starts from the first row, from left to right. Thus, the
/// when using the indices, you need to provide the vertices in the same
/// order.
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
pub(crate) fn regular_grid_triangulation(cols: usize, rows: usize) -> Vec<u32> {
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

#[test]
fn regular_grid_triangulation_test() {
    let indices = regular_grid_triangulation(4, 3);
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
    /// Unique identifier.
    pub uuid: uuid::Uuid,

    /// Uuid of the [`MicroSurface`] from which the mesh is generated.
    pub msurf: uuid::Uuid,

    /// Axis-aligned bounding box of the mesh.
    pub bounds: Aabb,

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
    ///
    /// REVIEW(yang): temporarily the surface mesh is generated on XZ plane in
    /// right-handed Y up coordinate system.
    /// TODO(yang): unit of surface area
    pub fn macro_surface_area(&self) -> f32 {
        (self.bounds.max.x - self.bounds.min.x) * (self.bounds.max.z - self.bounds.min.z)
    }
}

mod io {
    use crate::{error::Error, msurf::MicroSurface, units::um};
    use std::{
        fs::File,
        io::{BufRead, BufReader, Read},
        path::Path,
    };

    /// Origin of the micro-geometry height field.
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum MicroSurfaceOrigin {
        /// Micro-geometry height field from Predicting Appearance from Measured
        /// Microgeometry of Metal Surfaces.
        Dong2015,
        /// Micro-geometry height field from µsurf confocal microscope system.
        Usurf,
    }

    impl MicroSurface {
        /// Creates micro-geometry height field by reading the samples stored in
        /// different file format. Supported formats are
        ///
        /// 1. Ascii Matrix file (plain text) coming from
        ///    Predicting Appearance from Measured Microgeometry of Metal
        /// Surfaces. 2. Plain text data coming from µsurf confocal
        /// microscope system. 2. Micro-surface height field file
        /// (binary format, ends with *.dcms). 3. Micro-surface height
        /// field cache file (binary format, ends with *.dccc).
        pub fn load_from_file(
            path: &Path,
            origin: Option<MicroSurfaceOrigin>,
        ) -> Result<MicroSurface, Error> {
            let file = File::open(path)?;
            let mut reader = BufReader::new(file);

            if let Some(origin) = origin {
                // If origin is specified, call directly corresponding loading function.
                match origin {
                    MicroSurfaceOrigin::Dong2015 => read_ascii_dong2015(reader, true, path),
                    MicroSurfaceOrigin::Usurf => read_ascii_usurf(reader, true, path),
                }
            } else {
                // Otherwise, try to figure out the file format by reading first several bytes.
                let mut buf = [0_u8; 4];
                reader.read_exact(&mut buf)?;

                match std::str::from_utf8(&buf)? {
                    "Asci" => read_ascii_dong2015(reader, false, path),
                    "DATA" => read_ascii_usurf(reader, false, path),
                    // TODO: fix DCCC and DCMS
                    // "DCCC" => {
                    //     let header = {
                    //         let mut buf = [0_u8; 6];
                    //         reader.read_exact(&mut buf)?;
                    //         CacheHeader::new(buf)
                    //     };
                    //
                    //     if header.kind != CacheKind::HeightField {
                    //         Err(Error::FileError("Not a valid height field cache!"))
                    //     } else if header.binary {
                    //         Ok(bincode::deserialize_from(reader)?)
                    //     } else {
                    //         Ok(serde_yaml::from_reader(reader)?)
                    //     }
                    // }
                    // "DCMS" => {
                    //     let header = {
                    //         let mut buf = [0_u8; 23];
                    //         reader.read_exact(&mut buf)?;
                    //         MsHeader::new(buf)
                    //     };
                    //
                    //     let samples = if header.binary {
                    //         read_binary_samples(reader, (header.size / 4) as usize)
                    //     } else {
                    //         read_ascii_samples(reader)
                    //     };
                    //
                    //     Ok(MicroSurface::from_samples(
                    //         header.extent[0] as usize,
                    //         header.extent[1] as usize,
                    //         um!(header.spacing[0]),
                    //         um!(header.spacing[1]),
                    //         samples,
                    //         alignment.unwrap_or_default(),
                    //         Some(path.into()),
                    //     ))
                    // }
                    _ => Err(Error::UnrecognizedFile),
                }
            }
            .map(|mut ms| {
                ms.fill_holes();
                ms
            })
        }
    }

    /// Read micro-surface height field following the convention specified in
    /// the paper:
    ///
    /// [`Predicting Appearance from Measured Microgeometry of Metal Surfaces. Zhao Dong, Bruce Walter, Steve Marschner, and Donald P. Greenberg. 2016.`](https://dl.acm.org/doi/10.1145/2815618)
    ///
    /// Unit used during the measurement is micrometre.
    fn read_ascii_dong2015<R: BufRead>(
        mut reader: R,
        read_first_4_bytes: bool,
        path: &Path,
    ) -> Result<MicroSurface, Error> {
        if read_first_4_bytes {
            let mut buf = [0_u8; 4];
            reader.read_exact(&mut buf)?;

            if std::str::from_utf8(&buf)? != "Asci" {
                return Err(Error::UnrecognizedFile);
            }
        }

        let mut line = String::new();
        reader.read_line(&mut line)?;
        let (cols, rows, du, dv) = {
            let first_line = line.trim().split_ascii_whitespace().collect::<Vec<_>>();

            let cols = first_line[1].parse::<usize>().unwrap();
            let rows = first_line[2].parse::<usize>().unwrap();

            if first_line.len() == 3 {
                (cols, rows, 0.11, 0.11)
            } else if first_line.len() == 5 {
                let du = first_line[3].parse::<f32>().unwrap();
                let dv = first_line[4].parse::<f32>().unwrap();
                (cols, rows, du, dv)
            } else {
                panic!("Invalid first line: {line:?}");
            }
        };
        let samples = read_ascii_samples(reader);
        Ok(MicroSurface::from_samples(
            cols,
            rows,
            um!(du),
            um!(dv),
            samples,
            Some(path.into()),
        ))
    }

    /// Read micro-surface height field issued from µsurf confocal microscope.
    fn read_ascii_usurf<R: BufRead>(
        mut reader: R,
        read_first_4_bytes: bool,
        path: &Path,
    ) -> Result<MicroSurface, Error> {
        let mut line = String::new();
        reader.read_line(&mut line)?;

        if read_first_4_bytes && line.trim() != "DATA" {
            return Err(Error::UnrecognizedFile);
        }

        // Read horizontal coordinates
        reader.read_line(&mut line)?;
        let x_coords: Vec<f32> = line
            .trim()
            .split_ascii_whitespace()
            .map(|x_coord| x_coord.parse::<f32>().expect("Read f32 error!"))
            .collect();

        let (y_coords, values): (Vec<f32>, Vec<Vec<f32>>) = reader
            .lines()
            .map(|line| {
                let mut values = read_line_ascii_usurf(&line.unwrap());
                let head = values.remove(0);
                (head, values)
            })
            .unzip();

        // TODO: deal with case when coordinates are not uniform.
        let du = x_coords[1] - x_coords[0];
        let dv = y_coords[1] - y_coords[0];
        let samples: Vec<f32> = values.into_iter().flatten().collect();

        Ok(MicroSurface::from_samples(
            x_coords.len(),
            y_coords.len(),
            um!(du),
            um!(dv),
            samples,
            Some(path.into()),
        ))
    }

    /// Read a line of usurf file. Height values are separated by tab character.
    /// Consecutive tabs signifies that the height value at this point is
    /// missing.
    fn read_line_ascii_usurf(line: &str) -> Vec<f32> {
        assert!(line.is_ascii());
        line.chars()
            .enumerate()
            .filter_map(|(index, byte)| if byte == '\t' { Some(index) } else { None }) // find tab positions
            .scan((0, false), |(last, last_word_is_tab), curr| {
                // cut string into pieces: floating points string and tab character
                if *last != curr - 1 {
                    let val_str = if *last == 0 {
                        &line[*last..curr]
                    } else {
                        &line[(*last + 1)..curr]
                    };
                    *last = curr;
                    *last_word_is_tab = false;
                    Some(val_str)
                } else {
                    *last = curr;
                    *last_word_is_tab = true;
                    if *last_word_is_tab {
                        if curr != line.len() - 2 {
                            Some("\t")
                        } else {
                            Some("")
                        }
                    } else {
                        Some("")
                    }
                }
            })
            .filter_map(|s| {
                // parse float string into floating point value
                if s.is_empty() {
                    None
                } else if s == "\t" {
                    Some(f32::NAN)
                } else {
                    Some(s.parse::<f32>().unwrap())
                }
            })
            .collect()
    }

    /// Read sample values separated by whitespace line by line.
    fn read_ascii_samples<R: BufRead>(reader: R) -> Vec<f32> {
        reader
            .lines()
            .enumerate()
            .flat_map(|(n, line)| {
                let l = line.unwrap_or_else(|_| panic!("Bad line at {n}"));
                l.trim()
                    .split_ascii_whitespace()
                    .enumerate()
                    .map(|(i, x)| {
                        x.parse()
                            .unwrap_or_else(|_| panic!("Parse float error at line {n} pos {i}"))
                    })
                    .collect::<Vec<f32>>()
            })
            .collect()
    }

    fn read_binary_samples<R: Read>(mut reader: R, count: usize) -> Vec<f32> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let mut samples = vec![0.0; count];

        (0..count).for_each(|i| {
            samples[i] = reader.read_f32::<LittleEndian>().expect("read f32 error");
        });

        samples
    }

    #[cfg(test)]
    mod tests {
        use crate::msurf::io::read_line_ascii_usurf;

        #[test]
        #[rustfmt::skip]
        fn test_read_line_ascii_surf0() {
            let lines = [
                "0.00\t12.65\t\t12.63\t\t\t\t12.70\t12.73\t\t\t\t\t\t12.85\t\t\t\n",
                "0.00\t12.65\t\t\t12.63\t\t\t\t\t\t12.70\t12.73\t\t\t\t\t\t\t\t\t12.85\t\t\t\t\n",
                "0.00\t12.65\t\t\t\t12.63\t\t\t\t\t\t\t\t12.70\t12.73\t\t\t\t\t\t\t\t\t\t\t\t12.85\t\t\t\t\t\n",
            ];

            assert_eq!(read_line_ascii_usurf(lines[0]).len(), 16);
            assert_eq!(read_line_ascii_usurf(lines[1]).len(), 23);
            assert_eq!(read_line_ascii_usurf(lines[2]).len(), 30);
        }

        #[test]
        #[rustfmt::skip]
        fn test_read_line_ascii_surf1() {
            let lines = [
                "0.00\t12.65\t\t12.63\t12.70\t12.73\t\t12.85\t\n",
                "0.00\t12.65\t\t12.63\t\t\t\t12.70\t12.73\t\t\t\t\t\t12.85\t\t\t\n",
                "0.00\t12.65\t\t\t12.63\t\t\t\t\t\t12.70\t12.73\t\t\t\t\t\t\t\t\t12.85\t\t\t\t\n",
                "0.00\t12.65\t\t\t\t12.63\t\t\t\t\t\t\t\t12.70\t12.73\t\t\t\t\t\t\t\t\t\t\t\t12.85\t\t\t\t\t\n",
            ];

            fn _read_line(line: &str) -> Vec<&str> {
                let tabs = line
                    .chars()
                    .enumerate()
                    .filter_map(|(index, byte)| if byte == '\t' { Some(index) } else { None })
                    .collect::<Vec<usize>>();

                let pieces = tabs
                    .iter()
                    .scan((0, false), |(last, last_word_is_tab), curr| {
                        // cut string into pieces: floating points string and tab character
                        if *last != curr - 1 {
                            let val_str = if *last == 0 {
                                &line[*last..*curr]
                            } else {
                                &line[(*last + 1)..*curr]
                            };
                            *last = *curr;
                            *last_word_is_tab = false;
                            Some(val_str)
                        } else {
                            *last = *curr;
                            *last_word_is_tab = true;
                            if *last_word_is_tab {
                                if *curr != tabs[tabs.len() - 1] {
                                    Some(&"\t")
                                } else {
                                    Some(&"")
                                }
                            } else {
                                Some(&"")
                            }
                        }
                    })
                    .filter(|piece| !piece.is_empty())
                    .collect::<Vec<&str>>();
                pieces
            }

            let mut results = vec![];

            for &line in &lines {
                let pieces = _read_line(line);
                println!("pieces: {:?}", pieces);
                results.push(pieces.len());
            }

            assert_eq!(results[0], 8);
            assert_eq!(results[1], 16);
            assert_eq!(results[2], 23);
            assert_eq!(results[3], 30);
        }
    }
}
