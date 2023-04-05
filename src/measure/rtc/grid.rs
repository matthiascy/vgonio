//! Customised grid ray tracing for micro-surface measurements.

use crate::{
    app::{
        cache::Cache,
        cli::{BRIGHT_YELLOW, RESET},
    },
    common::ulp_eq,
    math::{close_enough, rcp},
    measure::{
        bsdf::SpectrumSampler,
        collector::CollectorPatches,
        emitter::EmitterSamples,
        measurement::{BsdfMeasurement, Radius},
        rtc::{
            embr::Trajectory,
            isect::{ray_tri_intersect_woop, RayTriIsect},
            Ray,
        },
        Collector, Emitter, RtcRecord, TrajectoryNode,
    },
    msurf::{AxisAlignment, MicroSurface, MicroSurfaceMesh},
    optics::{fresnel::reflect, ior::RefractiveIndex},
    units::{um, UMicrometre},
};
use cfg_if::cfg_if;
use glam::{IVec2, UVec2, Vec2, Vec3, Vec3Swizzles};
use rayon::prelude::*;
use std::time::Instant;

/// Measures the BSDF of the given micro-surface mesh.
pub fn measure_bsdf(
    desc: &BsdfMeasurement,
    surf: &MicroSurface,
    mesh: &MicroSurfaceMesh,
    samples: &EmitterSamples,
    patches: &CollectorPatches,
    cache: &Cache,
) {
    let radius = match desc.emitter.radius {
        // FIXME: max_extent() updated, thus 2.5 is not a good choice
        Radius::Auto(_) => um!(mesh.bounds.max_extent() * 2.5),
        Radius::Fixed(r) => r.in_micrometres(),
    };
    let max_bounces = desc.emitter.max_bounces;
    for pos in desc.emitter.meas_points() {
        println!(
            "      {BRIGHT_YELLOW}>{RESET} Emit rays from {}° {}°",
            pos.zenith.in_degrees().value(),
            pos.azimuth.in_degrees().value()
        );

        let emitted_rays = desc.emitter.emit_rays_with_radius(samples, pos, radius);
        let num_emitted_rays = emitted_rays.len();
    }
}

/// Base grid cell (the smallest unit of the grid).
#[derive(Debug, Copy, Clone)]
pub struct BaseCell {
    /// Horizontal grid coordinate of the cell.
    pub x: u32,
    /// Vertical grid coordinate of the cell.
    pub y: u32,
    /// Indices of 4 vertices of the cell, stored in counter-clockwise order.
    /// a --- d
    /// |     |
    /// b --- c
    /// The order is: a, b, c, d.
    pub verts: [u32; 4],
    /// Indices of 2 triangles of the cell.
    /// First triangle: a, b, d
    /// Second triangle: d, b, c
    pub tris: [u32; 2],
    /// Minimum height of the cell.
    pub min_height: f32,
    /// Maximum height of the cell.
    pub max_height: f32,
}

impl Default for BaseCell {
    fn default() -> Self {
        BaseCell {
            x: u32::MAX,
            y: u32::MAX,
            verts: [u32::MAX; 4],
            tris: [u32::MAX; 2],
            min_height: f32::MAX,
            max_height: f32::MAX,
        }
    }
}

/// Coarse grid cell.
#[derive(Debug, Copy, Clone)]
pub struct CoarseCell {
    /// Horizontal grid coordinate of the cell.
    x: u32,
    /// Vertical grid coordinate of the cell.
    y: u32,
    /// The min grid coordinates of the last level coarse cells or base cells.
    min: UVec2,
    /// The max grid coordinates of the last level coarse cells or base cells.
    max: UVec2,
    /// Minimum height of the cell.
    min_height: f32,
    /// Maximum height of the cell.
    max_height: f32,
}

impl Default for CoarseCell {
    fn default() -> Self {
        CoarseCell {
            x: u32::MAX,
            y: u32::MAX,
            min: UVec2::new(u32::MAX, u32::MAX),
            max: UVec2::new(u32::MAX, u32::MAX),
            min_height: f32::MAX,
            max_height: f32::MAX,
        }
    }
}

pub trait Cell: Sized {
    fn min_height(&self) -> f32;
    fn max_height(&self) -> f32;
}

impl Cell for BaseCell {
    fn min_height(&self) -> f32 { self.min_height }

    fn max_height(&self) -> f32 { self.max_height }
}

impl Cell for CoarseCell {
    fn min_height(&self) -> f32 { self.min_height }

    fn max_height(&self) -> f32 { self.max_height }
}

/// One level of the multi-level grid [`MultilevelGrid`].
///
/// The grid is built on top of the micro-surface mesh. It consists of a set of
/// grid cells of different coarseness.
///
///
/// The coordinates of the grid cells are defined in the grid space, which is
/// different from the world space. The grid space is defined as follows:
///
/// ```text
/// |- - - -> +horizontal
/// |
/// |
/// V
/// +vertical
/// ```
///
/// The finest grid coordinates lie in the range of [0, cols - 2] and [0, rows -
/// 2]. Knowing the grid coordinates, the corresponding vertices of the
/// micro-surface mesh can be found by the following formula:
///
/// ```text
/// v0 = (x, y)
/// v1 = (x + 1, y)
/// v2 = (x, y + 1)
/// ```
///
/// The top-left corner is used as the origin of the grid, for the reason that
/// vertices of the triangle mesh are generated following the order from left
/// to right, top to bottom.
#[doc = include_str!("../../../misc/imgs/grid.svg")]
///
/// The blue dots are the micro-surface samples, and the orange dots are the
/// grid cells.
///
/// The grid cells are stored in row-major order, i.e. the cells in the same
/// row are stored contiguously.
#[derive(Debug)]
pub struct Grid<C: Cell> {
    /// Width of the grid (max number of columns).
    cols: u32,

    /// Height of the grid (max number of rows).
    rows: u32,

    /// Size of a grid cell in the world space (in micrometres).
    /// Equals to the grid space cell size multiplied by the spacing between
    /// the micro-surface samples.
    world_space_cell_size: Vec2,

    /// Size of a grid cell in the grid space (number of base cells).
    /// For coarse level grid, the actual size of a cell may be smaller than
    /// this value due to the boundary of the grid. For the finest level grid,
    /// the size of a cell is always equal to this value. The first level of
    /// coarse grid is always MIN_COARSE_CELL_SIZE, the further levels are
    /// doubled in each level.
    grid_space_cell_size: u32,

    /// Whether the grid is coarse.
    is_coarse: bool,

    /// Cells of the grid stored in row-major order.
    cells: Vec<C>,
}

/// Grid traversal outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum GridTraversal {
    FromTopOrBottom([IVec2; 1]),
    TraversedBaseCells {
        /// Cells traversed by the ray.
        cells: Vec<IVec2>,
        /// Distances from the origin of the ray (entering cell of the ray)
        /// to each intersection point between the ray and cells.
        /// The first element is always 0, and the last element is the
        /// distance from the origin to the exit point of the ray.
        dists: Vec<f32>,
    },
    TraversedCoarseCells {
        /// Cells traversed by the ray.
        cells: Vec<IVec2>,
    },
    Cell(IVec2),
}

impl GridTraversal {
    #[cfg(debug_assertions)]
    pub fn traversed_cells(&self) -> Option<&[IVec2]> {
        match self {
            GridTraversal::FromTopOrBottom(cells) => Some(cells.as_ref()),
            GridTraversal::TraversedBaseCells { cells, .. } => Some(cells),
            GridTraversal::TraversedCoarseCells { cells } => Some(cells),
            _ => None,
        }
    }

    pub fn cell(&self) -> Option<IVec2> {
        match self {
            GridTraversal::Cell(cell) => Some(*cell),
            _ => None,
        }
    }
}

impl<C: Cell> Grid<C> {
    /// Returns the minimum and maximum height of the grid cells in the given
    /// region defined by the minimum and maximum grid coordinates.
    pub fn min_max_of_region(&self, min: UVec2, max: UVec2) -> (f32, f32) {
        let mut min_height = f32::MAX;
        let mut max_height = f32::MIN;
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let cell = &self.cells[y as usize * self.cols as usize + x as usize];
                min_height = min_height.min(cell.min_height());
                max_height = max_height.max(cell.max_height());
            }
        }
        (min_height, max_height)
    }

    /// Returns the cell at the given grid coordinates.
    #[track_caller]
    pub fn cell_at(&self, x: u32, y: u32) -> &C {
        debug_assert!(
            x < self.cols && y < self.rows,
            "Cell index out of bounds: ({}, {})",
            x,
            y
        );
        &self.cells[y as usize * self.cols as usize + x as usize]
    }

    /// Traverses the grid cells along the given ray to identify all the cells
    /// along the ray.
    ///
    /// Modified version of Digital Differential Analyzer (DDA) algorithm.
    ///
    /// # Arguments
    ///
    /// * `origin` - Origin of the grid in the world space (top-left corner).
    /// * `ray` - The ray to traverse the grid.
    pub fn traverse(&self, origin: Vec2, ray: &Ray) -> GridTraversal {
        log::debug!("Traversing the grid along the ray: {:?}", ray);
        let start_pos = ray.o.xz() - origin;
        log::debug!(
            "Start position in the world space relative to grid's origin: {:?}",
            start_pos
        );
        let start_cell = self.world_to_local(&origin, &ray.o.xz()).unwrap();
        println!("Start position in the grid space: {:?}", start_cell);
        let ray_dir = ray.d.xz();
        log::debug!("Ray direction in the grid space: {:?}", ray_dir);

        let is_parallel_to_grid_x = ulp_eq(ray_dir.y, 0.0);
        let is_parallel_to_grid_y = ulp_eq(ray_dir.x, 0.0);

        if is_parallel_to_grid_x && is_parallel_to_grid_y {
            // The ray is parallel to both the grid x and y axis; coming from
            // the top or bottom of the grid.
            log::debug!("The ray is parallel to both the grid x and y axis");
            return GridTraversal::FromTopOrBottom([start_cell]);
        }

        let mut travelled_dists = vec![0.0];
        let mut traversed_cells = vec![];

        // We are using the DDA algorithm to find all the cells that the ray traverses
        // and its intersection points. In case we are traversing on the coarse
        // grid we can assume that the size of the grid cell is always 1x1.
        log::debug!("The ray is not parallel to either the grid x or y axis");
        let cell_size = if self.is_coarse {
            Vec2::new(1.0, 1.0)
        } else {
            self.world_space_cell_size
        };

        // 1. Calculate the slope of the ray on the grid.
        let (m, m_rcp) = {
            if is_parallel_to_grid_x && !is_parallel_to_grid_y {
                log::debug!("The ray is parallel to the grid x axis");
                (1.0, 0.0)
            } else if !is_parallel_to_grid_x && is_parallel_to_grid_y {
                log::debug!("The ray is parallel to the grid y axis");
                (0.0, 1.0)
            } else {
                log::debug!("The ray is not parallel to either the grid x or y axis");
                let m = ray_dir.y * rcp(ray_dir.x);
                (m, rcp(m))
            }
        };
        log::debug!("Slope of the ray on the grid: {}, reciprocal: {}", m, m_rcp);
        // 2. Calculate the distance along each axis when moving one cell in the grid
        // space.
        let dl = Vec2::new(
            m.mul_add(m, 1.0).sqrt() * cell_size.x,
            m_rcp.mul_add(m_rcp, 1.0).sqrt() * cell_size.y,
        );
        log::debug!(
            "Distance along each axis when moving one cell in the grid space: {:?}",
            dl
        );
        let mut curr_cell = start_cell;
        let step_dir = IVec2::new(
            if ray_dir.x >= 0.0 { 1 } else { -1 },
            if ray_dir.y >= 0.0 { 1 } else { -1 },
        );

        // 3. Calculate the line length from the start position when moving one
        // unit along the x-axis and y-axis.
        let mut accumulated_line = Vec2::new(
            if ray_dir.x < 0.0 {
                (start_pos.x / self.world_space_cell_size.x - curr_cell.x as f32) * dl.x
            } else {
                ((curr_cell.x + 1) as f32 - start_pos.x / self.world_space_cell_size.x) * dl.x
            },
            if ray_dir.y < 0.0 {
                (start_pos.y / self.world_space_cell_size.y - curr_cell.y as f32) * dl.y
            } else {
                ((curr_cell.y + 1) as f32 - start_pos.y / self.world_space_cell_size.y) * dl.y
            },
        );

        log::debug!("Initial accumulated line length: {:?}", accumulated_line);

        // 4. Identify the traversed cells and intersection points
        loop {
            let reaching_left_or_right_edge = step_dir.x > 0 && curr_cell.x >= self.cols as i32
                || step_dir.x < 0 && curr_cell.x < 0;
            let reaching_top_or_bottom_edge = step_dir.y > 0 && curr_cell.y >= self.rows as i32
                || step_dir.y < 0 && curr_cell.y < 0;

            if reaching_left_or_right_edge || reaching_top_or_bottom_edge {
                break;
            }

            traversed_cells.push(curr_cell);

            if accumulated_line.x <= accumulated_line.y {
                // Move along the x-axis.
                travelled_dists.push(accumulated_line.x);
                curr_cell.x += step_dir.x;
                accumulated_line.x += dl.x;
            } else {
                // Move along the y-axis.
                travelled_dists.push(accumulated_line.y);
                curr_cell.y += step_dir.y;
                accumulated_line.y += dl.y;
            }
        }

        if self.is_coarse {
            GridTraversal::TraversedCoarseCells {
                cells: traversed_cells,
            }
        } else {
            GridTraversal::TraversedBaseCells {
                cells: traversed_cells,
                dists: travelled_dists,
            }
        }
    }

    /// Returns the grid coordinates of the cell that contains the given
    /// position in the world space.
    ///
    /// Returns `None` if the position is outside the grid.
    ///
    /// # Arguments
    ///
    /// * `origin` - The origin of the grid in the world space.
    /// * `pos` - The position in the world space.
    pub fn world_to_local(&self, origin: &Vec2, pos: &Vec2) -> Option<IVec2> {
        let x = {
            let x = (pos.x - origin.x) / self.world_space_cell_size.x;
            // If the position is exactly on the boundary of the cell, we
            // should use the cell on the left.
            if x.trunc() == self.cols as f32 && x.fract() == 0.0 {
                x as u32 - 1
            } else {
                x as u32
            }
        };
        let y = {
            let y = (pos.y - origin.y) / self.world_space_cell_size.y;
            // If the position is exactly on the boundary of the cell, we
            // should use the cell on the left.
            if y.trunc() == self.rows as f32 && y.fract() == 0.0 {
                y as u32 - 1
            } else {
                y as u32
            }
        };
        if x >= self.cols || y >= self.rows {
            None
        } else {
            debug_assert!(x > 0 && y > 0, "The position should be positive");
            Some(IVec2::new(x as i32, y as i32))
        }
    }

    /// Checks if the given ray may intersect with the cell at the given
    /// position in the grid space with the given entering and exiting
    /// distances.
    ///
    /// # Arguments
    ///
    /// * `ray` - The ray.
    /// * `pos` - The position of the cell in the grid space.
    /// * `entering` - The distance from the ray origin to the entering point.
    /// * `exiting` - The distance from the ray origin to the exiting point.
    pub fn may_intersect_cell(&self, ray: &Ray, pos: &IVec2, entering: f32, exiting: f32) -> bool {
        let entering_height = ray.o.y + entering * ray.d.y;
        let exiting_height = ray.o.y + exiting * ray.d.y;
        let cell = self.cell_at(pos.x as _, pos.y as _);
        let ge_max = entering_height >= cell.max_height() || exiting_height >= cell.max_height();
        let le_min = entering_height <= cell.min_height() || exiting_height <= cell.min_height();
        !ge_max && !le_min
    }
}

// TODO: rename
pub struct Isect {
    pub p: Vec3,
    pub n: Vec3,
}

impl Grid<BaseCell> {
    /// Checks if the given ray intersects with triangles within the cell at
    /// the given position in the grid space.
    pub fn intersects_cell_triangles(
        &self,
        ray: &Ray,
        pos: &IVec2,
        mesh: &MicroSurfaceMesh,
    ) -> Option<(u32, Isect)> {
        let cell = self.cell_at(pos.x as _, pos.y as _);
        let triangles = self.triangles_of_cell(pos, mesh);
        let isect0 = ray_tri_intersect_woop(ray, &triangles[0]);
        let isect1 = ray_tri_intersect_woop(ray, &triangles[1]);

        match (isect0, isect1) {
            (Some(isect0), Some(isect1)) => {
                let p0 = isect0.p;
                let p1 = isect1.p;
                if close_enough(&p0, &p1) {
                    // The ray intersects with two triangles on the shared edge.
                    // u of the first triangle is 0.0, and w of the second triangle
                    // is 0.0.
                    debug_assert!(ulp_eq(isect0.u, 0.0));
                    debug_assert!(ulp_eq((1.0 - isect1.u - isect1.v), 0.0));
                    Some((
                        cell.tris[0],
                        Isect {
                            n: (isect0.n + isect1.n).normalize(),
                            p: p0,
                        },
                    ))
                } else {
                    unreachable!("The ray should not intersect with two triangles")
                }
            }
            (Some(isect), None) => {
                // Intersects with the first triangle.
                // Check if the ray intersects with other edges, get the adjacent triangle
                // index and its normal.
                let self_idx = cell.tris[0];
                let opposite_idx = cell.tris[1];
                let adjacent = if ulp_eq(isect.u, 0.0) {
                    // u is 0.0, so the ray intersects with the edge p1-p2.
                    Some(opposite_idx)
                } else if ulp_eq(isect.v, 0.0) {
                    // v is 0.0, so the ray intersects with the edge p0-p2.
                    if self_idx < 2 * self.cols {
                        // The cell is on the top row.
                        None
                    } else {
                        // Return the triangle index of the cell one row above.
                        Some(self_idx - (self.cols - 1))
                    }
                } else if ulp_eq(isect.u + isect.v, 1.0) {
                    // u + v is 1.0, so the ray intersects with the edge p0-p1.
                    if self_idx % 2 * self.cols == 0 {
                        // The cell is on the left column.
                        None
                    } else {
                        // Return the triangle index of the cell one column to the left.
                        Some(self_idx - 1)
                    }
                } else {
                    None
                };
                let n = if let Some(adjacent) = adjacent {
                    (isect.n + mesh.facet_normals[adjacent as usize]).normalize()
                } else {
                    isect.n
                };
                Some((cell.tris[0], Isect { p: isect.p, n }))
            }
            (None, Some(isect)) => {
                // Intersects with the second triangle.
                // Check if the ray intersects with other edges, get the adjacent triangle
                // index and its normal.
                let self_idx = cell.tris[1];
                let opposite_idx = cell.tris[0];
                let adjacent = if ulp_eq(isect.u, 0.0) {
                    // u is 0.0, so the ray intersects with the edge p1-p2.
                    if self_idx >= (2 * self.cols) * (self.rows - 1) {
                        // The cell is on the bottom row.
                        None
                    } else {
                        // Return the triangle index of the cell one row below.
                        Some(self_idx + (self.cols - 1))
                    }
                } else if ulp_eq(isect.v, 0.0) {
                    // v is 0.0, so the ray intersects with the edge p0-p2.
                    if self_idx % (2 * self.cols) == 2 * self.cols - 1 {
                        // The cell is on the right column.
                        None
                    } else {
                        // Return the triangle index of the cell one column to the left.
                        Some(self_idx + 1)
                    }
                } else if ulp_eq(isect.u + isect.v, 1.0) {
                    // u + v is 1.0, so the ray intersects with the edge p0-p1.
                    Some(opposite_idx)
                } else {
                    None
                };
                let n = if let Some(adjacent) = adjacent {
                    (isect.n + mesh.facet_normals[adjacent as usize]).normalize()
                } else {
                    isect.n
                };
                Some((cell.tris[0], Isect { p: isect.p, n }))
            }
            (None, None) => None,
        }
    }

    /// Returns the triangles of the cell at the given position in the grid
    /// space.
    pub fn triangles_of_cell(&self, pos: &IVec2, mesh: &MicroSurfaceMesh) -> [[Vec3; 3]; 2] {
        let cell = self.cell_at(pos.x as _, pos.y as _);
        let [v0, v1, v2, v3] = cell.verts;
        [
            [
                mesh.verts[v0 as usize],
                mesh.verts[v1 as usize],
                mesh.verts[v3 as usize],
            ],
            [
                mesh.verts[v3 as usize],
                mesh.verts[v1 as usize],
                mesh.verts[v2 as usize],
            ],
        ]
    }

    /// Traces the given ray through the grid and returns the intersection
    /// point with the closest triangle.
    pub fn trace(&self, origin: Vec2, ray: &Ray, mesh: &MicroSurfaceMesh) -> Option<(u32, Isect)> {
        log::debug!("Traversing the grid along the ray: {:?}", ray);
        let start_pos = ray.o.xz() - origin;
        log::debug!(
            "Start position in the world space relative to grid's origin: {:?}",
            start_pos
        );
        let start_cell = self.world_to_local(&origin, &ray.o.xz()).unwrap();
        println!("Start position in the grid space: {:?}", start_cell);
        let ray_dir = ray.d.xz();
        log::debug!("Ray direction in the grid space: {:?}", ray_dir);

        let is_parallel_to_grid_x = ulp_eq(ray_dir.y, 0.0);
        let is_parallel_to_grid_y = ulp_eq(ray_dir.x, 0.0);

        if is_parallel_to_grid_x && is_parallel_to_grid_y {
            // The ray is parallel to both the grid x and y axis; coming from
            // the top or bottom of the grid.
            log::debug!("The ray is parallel to both the grid x and y axis");
            return self.intersects_cell_triangles(ray, &start_cell, mesh);
        }

        let mut entering_dist = 0.0;
        let mut exiting_dist = 0.0;

        // We are using the DDA algorithm to find all the cells that the ray
        // traverses and its intersection points. In case we are traversing
        // on the coarse grid we can assume that the size of the grid cell is
        // always 1x1.
        log::debug!("The ray is not parallel to either the grid x or y axis");
        let cell_size = if self.is_coarse {
            Vec2::new(1.0, 1.0)
        } else {
            self.world_space_cell_size
        };

        // 1. Calculate the slope of the ray on the grid.
        let (m, m_rcp) = {
            if is_parallel_to_grid_x && !is_parallel_to_grid_y {
                log::debug!("The ray is parallel to the grid x axis");
                (1.0, 0.0)
            } else if !is_parallel_to_grid_x && is_parallel_to_grid_y {
                log::debug!("The ray is parallel to the grid y axis");
                (0.0, 1.0)
            } else {
                log::debug!("The ray is not parallel to either the grid x or y axis");
                let m = ray_dir.y * rcp(ray_dir.x);
                (m, rcp(m))
            }
        };
        log::debug!("Slope of the ray on the grid: {}, reciprocal: {}", m, m_rcp);

        // 2. Calculate the distance along each axis when moving one cell in the grid
        // space.
        let dl = Vec2::new(
            m.mul_add(m, 1.0).sqrt() * cell_size.x,
            m_rcp.mul_add(m_rcp, 1.0).sqrt() * cell_size.y,
        );
        log::debug!(
            "Distance along each axis when moving one cell in the grid space: {:?}",
            dl
        );
        let mut curr_cell = start_cell;
        let step_dir = IVec2::new(
            if ray_dir.x >= 0.0 { 1 } else { -1 },
            if ray_dir.y >= 0.0 { 1 } else { -1 },
        );

        // 3. Calculate the line length from the start position when moving one
        // unit along the x-axis and y-axis.
        let mut accumulated_line = Vec2::new(
            if ray_dir.x < 0.0 {
                (start_pos.x / self.world_space_cell_size.x - curr_cell.x as f32) * dl.x
            } else {
                ((curr_cell.x + 1) as f32 - start_pos.x / self.world_space_cell_size.x) * dl.x
            },
            if ray_dir.y < 0.0 {
                (start_pos.y / self.world_space_cell_size.y - curr_cell.y as f32) * dl.y
            } else {
                ((curr_cell.y + 1) as f32 - start_pos.y / self.world_space_cell_size.y) * dl.y
            },
        );
        exiting_dist = accumulated_line.x.min(accumulated_line.y);

        log::debug!("Initial accumulated line length: {:?}", accumulated_line);

        // 4. Identify the traversed cells and intersection points
        loop {
            let reaching_left_or_right_edge = step_dir.x > 0 && curr_cell.x >= self.cols as i32
                || step_dir.x < 0 && curr_cell.x < 0;
            let reaching_top_or_bottom_edge = step_dir.y > 0 && curr_cell.y >= self.rows as i32
                || step_dir.y < 0 && curr_cell.y < 0;

            if reaching_left_or_right_edge || reaching_top_or_bottom_edge {
                break;
            }

            // Perform the pre-check to see if the cell may intersect the ray.
            if self.may_intersect_cell(ray, &curr_cell, entering_dist, exiting_dist) {
                // Perform the actual intersection test.
                log::debug!("Cell {:?} may intersect the ray", curr_cell);
                if let Some((tri_idx, isect)) =
                    self.intersects_cell_triangles(ray, &curr_cell, mesh)
                {
                    log::debug!("Cell {:?} intersects the ray", curr_cell);
                    return Some((tri_idx, isect));
                }
            }

            // Move to the next cell.
            if accumulated_line.x <= accumulated_line.y {
                // Move along the x-axis.
                entering_dist = accumulated_line.x;
                curr_cell.x += step_dir.x;
                accumulated_line.x += dl.x;
            } else {
                // Move along the y-axis.
                entering_dist = accumulated_line.y;
                curr_cell.y += step_dir.y;
                accumulated_line.y += dl.y;
            }
            exiting_dist = accumulated_line.x.min(accumulated_line.y);
        }

        None
    }
}

/// Multilevel grid.
pub struct MultilevelGrid<'ms> {
    /// The heightfield where the grid is defined.
    surf: &'ms MicroSurface,

    /// Corresponding `TriangleMesh` of the surface.
    mesh: &'ms MicroSurfaceMesh,

    /// Minimum size of a coarse grid cell (number of base cells).
    min_coarse_cell_size: u32,

    /// Number of levels of the coarse grid.
    level: u32,

    /// Origin of the grid in the world space (top-left corner).
    origin: Vec2,

    /// The finest grid.
    base: Grid<BaseCell>,

    /// The different levels of the coarse grid, from finest to coarsest (the
    /// size of the grid cells are doubled in each level).
    coarse: Vec<Grid<CoarseCell>>,
}

impl<'ms> MultilevelGrid<'ms> {
    /// Creates a new grid ray tracing object.
    pub fn new(
        surf: &'ms MicroSurface,
        mesh: &'ms MicroSurfaceMesh,
        min_coarse_cell_size: u32,
    ) -> Self {
        let base = Self::build_base_grid(surf);
        let (level, coarse) = {
            let n = surf.cols.max(surf.rows) / min_coarse_cell_size as usize;
            if n == 0 {
                (0, vec![])
            } else {
                let num_levels = (n as f32).log2().floor() as u32;
                (
                    num_levels,
                    (0..num_levels)
                        .into_iter()
                        .into_par_iter()
                        .map(|level| Self::build_coarse_grid(&base, level, min_coarse_cell_size))
                        .collect::<Vec<_>>(),
                )
            }
        };

        #[cfg(debug_assertions)]
        {
            // Check order of generated coarse grids.
            for i in 0..coarse.len() - 1 {
                let s0 = coarse[i].grid_space_cell_size;
                let s1 = coarse[i + 1].grid_space_cell_size;
                assert!(s1 > s0);
            }
        }

        MultilevelGrid {
            surf,
            mesh,
            min_coarse_cell_size,
            level,
            origin: Vec2::new(mesh.bounds.min.x, mesh.bounds.min.z),
            base,
            coarse,
        }
    }

    /// Builds the base (finest) grid.
    fn build_base_grid(surf: &MicroSurface) -> Grid<BaseCell> {
        let surf_width = surf.cols;
        let grid_width = surf.cols - 1;
        let grid_height = surf.rows - 1;
        let mut base_cells = vec![BaseCell::default(); (surf.cols - 1) * (surf.rows - 1)];
        base_cells
            .par_chunks_mut(512)
            .enumerate()
            .for_each(|(i, chunk)| {
                for (j, cell) in chunk.iter_mut().enumerate() {
                    let idx = i * 512 + j;
                    let x = idx % grid_width;
                    let y = idx / grid_width;
                    let verts = [
                        (surf_width * y + x) as u32,
                        (surf_width * (y + 1) + x) as u32,
                        (surf_width * (y + 1) + x + 1) as u32,
                        (surf_width * y + x + 1) as u32,
                    ];
                    let tris = [(idx * 2) as u32, (idx * 2 + 1) as u32];
                    let (min_height, max_height) =
                        verts.iter().fold((f32::MAX, f32::MIN), |(min, max), vert| {
                            let height = surf.samples[*vert as usize];
                            (min.min(height), max.max(height))
                        });
                    *cell = BaseCell {
                        x: x as u32,
                        y: y as u32,
                        verts,
                        tris,
                        min_height,
                        max_height,
                    };
                }
            });

        Grid {
            cols: grid_width as u32,
            rows: grid_height as u32,
            world_space_cell_size: Vec2::new(surf.du, surf.dv),
            grid_space_cell_size: 1,
            is_coarse: false,
            cells: base_cells,
        }
    }

    /// Builds a coarse grid.
    ///
    /// # Arguments
    ///
    /// * `base` - The finest grid.
    /// * `level` - The current level of the coarse grid.
    fn build_coarse_grid(
        base: &Grid<BaseCell>,
        level: u32,
        min_coarse_cell_size: u32,
    ) -> Grid<CoarseCell> {
        // Size of a coarse grid cell in the base grid space.
        let cell_size_in_base = min_coarse_cell_size * (1 << level);
        // Number of coarse grid cells in each direction.
        let cols = (base.cols as f32 / cell_size_in_base as f32).ceil() as u32;
        let rows = (base.rows as f32 / cell_size_in_base as f32).ceil() as u32;
        let mut cells = vec![CoarseCell::default(); (cols * rows) as usize];
        // Size of a coarse grid cell compared to the last level.
        let (relative_cell_size, max_cell_x, max_cell_y, prev_cell_size) = if level == 0 {
            (min_coarse_cell_size, base.cols - 1, base.rows - 1, 1)
        } else {
            let prev_cell_size = min_coarse_cell_size * (1 << (level - 1));
            let prev_cols = (base.cols as f32 / prev_cell_size as f32).ceil() as u32;
            let prev_rows = (base.rows as f32 / prev_cell_size as f32).ceil() as u32;
            (2, prev_cols - 1, prev_rows - 1, prev_cell_size)
        };
        cells
            .par_chunks_mut(256)
            .enumerate()
            .for_each(|(i, chunk)| {
                for (j, cell) in chunk.iter_mut().enumerate() {
                    let idx = (i * 256 + j) as u32;
                    let x = idx % cols;
                    let y = idx / cols;
                    let min = UVec2::new(x * relative_cell_size, y * relative_cell_size);
                    let max = UVec2::new(
                        ((x + 1) * relative_cell_size - 1).min(max_cell_x),
                        ((y + 1) * relative_cell_size - 1).min(max_cell_y),
                    );
                    let min_in_base = min * prev_cell_size;
                    let max_in_base = max * prev_cell_size;
                    let (min_height, max_height) = base.min_max_of_region(min_in_base, max_in_base);

                    *cell = CoarseCell {
                        x,
                        y,
                        min,
                        max,
                        min_height,
                        max_height,
                    };
                }
            });

        Grid {
            cols,
            rows,
            world_space_cell_size: Vec2::new(
                base.world_space_cell_size.x * cell_size_in_base as f32,
                base.world_space_cell_size.y * cell_size_in_base as f32,
            ),
            grid_space_cell_size: cell_size_in_base,
            is_coarse: true,
            cells,
        }
    }

    /// Returns the number of levels.
    pub fn level(&self) -> u32 { self.level }

    /// Returns the base grid.
    pub fn base(&self) -> &Grid<BaseCell> { &self.base }

    /// Returns the coarse grid at the given level.
    /// The finest grid is at level 0.
    pub fn coarse(&self, level: usize) -> &Grid<CoarseCell> { &self.coarse[level] }

    // /// Traces a ray through the grid.
    // pub fn trace(&self, ray: &Ray) -> Trajectory { if self.coarse.len() == 0 {} }
}

#[cfg(test)]
mod tests {
    use crate::{
        common::ulp_eq,
        measure::rtc::{
            grid::{GridTraversal, MultilevelGrid},
            Ray,
        },
        msurf::{AxisAlignment, MicroSurface},
        units::{um, UMicrometre},
    };
    use glam::{IVec2, UVec2, Vec3, Vec3Swizzles};
    use proptest::proptest;

    #[test]
    fn multilevel_grid_creation() {
        let surf = MicroSurface::from_samples::<UMicrometre>(
            9,
            9,
            um!(0.5),
            um!(0.5),
            vec![0.0; 81],
            None,
        );
        let mesh = surf.as_micro_surface_mesh();
        let grid = MultilevelGrid::new(&surf, &mesh, 2);
        assert_eq!(grid.level(), 2);
        assert_eq!(grid.coarse.len(), 2);
        assert_eq!(grid.base().cols, 8);
        assert_eq!(grid.base().rows, 8);
        assert_eq!(grid.base().grid_space_cell_size, 1);
        assert_eq!(grid.coarse(0).cols, 4);
        assert_eq!(grid.coarse(0).rows, 4);
        assert_eq!(grid.coarse(0).grid_space_cell_size, 2);
        assert_eq!(grid.coarse(1).cols, 2);
        assert_eq!(grid.coarse(1).rows, 2);
        assert_eq!(grid.coarse(1).grid_space_cell_size, 4);

        for cell in grid.base.cells {
            let vert_indices = cell.verts;
            let tri0 = cell.tris[0] as usize;
            let tri0_vert_indices = &mesh.facets[tri0 * 3..tri0 * 3 + 3];
            let tri1 = cell.tris[1] as usize;
            let tri1_vert_indices = &mesh.facets[tri1 * 3..tri1 * 3 + 3];
            assert_eq!(
                tri0_vert_indices,
                [vert_indices[0], vert_indices[1], vert_indices[3]]
            );
            assert_eq!(
                tri1_vert_indices,
                [vert_indices[3], vert_indices[1], vert_indices[2]]
            );
        }
    }

    #[test]
    #[rustfmt::skip]
    fn multilevel_grid_region() {
        let cols = 10;
        let rows = 8;

        let surf = MicroSurface::from_samples::<UMicrometre>(
            cols,
            rows,
            um!(0.5),
            um!(0.5),
            vec![
                0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
                0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 
                0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            ],
            None,
        );
        let mesh = surf.as_micro_surface_mesh();
        let grid = MultilevelGrid::new(&surf, &mesh, 2);
        let base = grid.base();
        assert_eq!(grid.level(), 2);
        assert_eq!(grid.coarse.len(), 2);
        assert_eq!(base.cols, cols as u32 - 1);
        assert_eq!(base.rows, rows as u32 - 1);
        assert_eq!(base.grid_space_cell_size, 1);

        let coarse0 = grid.coarse(0);
        let coarse1 = grid.coarse(1);

        assert_eq!(coarse0.cols, 5);
        assert_eq!(coarse0.rows, 4);
        assert_eq!(coarse0.grid_space_cell_size, 2);
        assert_eq!(coarse1.cols, 3);
        assert_eq!(coarse1.rows, 2);
        assert_eq!(coarse1.grid_space_cell_size, 4);
        
        for y in 0..base.rows {
            for x in 0..base.cols {
                let cell = base.cell_at(x, y);
                assert!(ulp_eq(cell.min_height, (x + y) as f32 * 0.1), "base min height at x={}, y={}", x, y);
                assert!(ulp_eq(cell.max_height, (x + y) as f32 * 0.1 + 0.2), "base max height at x={}, y={}", x, y);
            }
        }

        for y in 0..coarse0.rows {
            for x in 0..coarse0.cols {
                let cell = coarse0.cell_at(x, y);
                assert_eq!(
                    cell.min,
                    UVec2::new(grid.min_coarse_cell_size * x, grid.min_coarse_cell_size * y),
                    "coarse level 0 min at x={}, y={}",
                    x,
                    y
                );
                assert_eq!(
                    cell.max,
                    UVec2::new(
                        (grid.min_coarse_cell_size * (x + 1) - 1).min(base.cols - 1),
                        (grid.min_coarse_cell_size * (y + 1) - 1).min(base.rows - 1)
                    ),
                    "coarse level 0 max at x={}, y={}",
                    x,
                    y
                );
                assert!(ulp_eq(cell.min_height, (cell.min.x + cell.min.y) as f32 * 0.1));
                assert!(ulp_eq(cell.max_height, (cell.max.x + cell.max.y) as f32 * 0.1 + 0.2));
            }
        }
        
        for y in 0..coarse1.rows {
            for x in 0..coarse1.cols {
                let cell = coarse1.cell_at(x, y);
                assert_eq!(
                    cell.min,
                    UVec2::new(2 * x, 2 * y),
                    "coarse level 1 min at x={}, y={}",
                    x,
                    y
                );
                assert_eq!(
                    cell.max,
                    UVec2::new(
                        (2 * (x + 1) - 1).min(coarse0.cols - 1),
                        (2 * (y + 1) - 1).min(coarse0.rows - 1)
                    ),
                    "coarse level 1 max at x={}, y={}",
                    x,
                    y
                );
                assert!(ulp_eq(cell.min_height, (cell.min.x + cell.min.y) as f32 * 2.0 * 0.1));
                assert!(ulp_eq(cell.max_height, (cell.max.x + cell.max.y) as f32 * 2.0 * 0.1 + 0.2));
            }
        }
    }

    #[test]
    fn grid_traverse() {
        let surf = MicroSurface::new(10, 10, um!(1.0), um!(1.0), um!(0.0));
        let mesh = surf.as_micro_surface_mesh();
        let grid = MultilevelGrid::new(&surf, &mesh, 2);
        let base = grid.base();
        println!("level: {}", grid.level());
        let coarse0 = grid.coarse(0);
        let coarse1 = grid.coarse(1);
        assert_eq!(coarse1.cols, 3);
        assert_eq!(coarse1.rows, 3);

        {
            let ray_slope_0_5 = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, -0.1, 1.0));
            let ray_slope_1 = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, -0.1, 1.0));
            let ray_slope_2 = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, -0.1, 2.0));
            {
                let traversal_base = base.traverse(mesh.bounds.min.xz(), &ray_slope_1);
                let traversal_lvl0 = coarse0.traverse(mesh.bounds.min.xz(), &ray_slope_1);
                let traversal_lvl1 = coarse1.traverse(mesh.bounds.min.xz(), &ray_slope_1);
                assert_eq!(
                    traversal_base.traversed_cells().unwrap(),
                    vec![
                        IVec2::new(5, 5),
                        IVec2::new(6, 5),
                        IVec2::new(6, 6),
                        IVec2::new(7, 6),
                        IVec2::new(7, 7),
                        IVec2::new(8, 7),
                        IVec2::new(8, 8),
                    ]
                );
                assert_eq!(
                    traversal_lvl0,
                    GridTraversal::TraversedCoarseCells {
                        cells: vec![
                            IVec2::new(2, 2),
                            IVec2::new(3, 2),
                            IVec2::new(3, 3),
                            IVec2::new(4, 3),
                            IVec2::new(4, 4)
                        ]
                    }
                );
                assert_eq!(
                    traversal_lvl1,
                    GridTraversal::TraversedCoarseCells {
                        cells: vec![IVec2::new(1, 1), IVec2::new(2, 1), IVec2::new(2, 2)]
                    }
                );
            }
            {
                let traversal_base = base.traverse(mesh.bounds.min.xz(), &ray_slope_0_5);
                let traversal_lvl0 = coarse0.traverse(mesh.bounds.min.xz(), &ray_slope_0_5);
                let traversal_lvl1 = coarse1.traverse(mesh.bounds.min.xz(), &ray_slope_0_5);
                assert_eq!(
                    traversal_base.traversed_cells().unwrap(),
                    vec![
                        IVec2::new(5, 5),
                        IVec2::new(6, 5),
                        IVec2::new(7, 5),
                        IVec2::new(7, 6),
                        IVec2::new(8, 6),
                    ]
                );
                assert_eq!(
                    traversal_lvl0,
                    GridTraversal::TraversedCoarseCells {
                        cells: vec![
                            IVec2::new(2, 2),
                            IVec2::new(3, 2),
                            IVec2::new(3, 3),
                            IVec2::new(4, 3),
                        ]
                    }
                );
                assert_eq!(
                    traversal_lvl1,
                    GridTraversal::TraversedCoarseCells {
                        cells: vec![IVec2::new(1, 1), IVec2::new(2, 1), IVec2::new(2, 2)]
                    }
                );
            }
        }

        {
            let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(-1.0, -0.1, -1.0));
            let traversal_lvl0 = coarse0.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl0,
                GridTraversal::TraversedCoarseCells {
                    cells: vec![
                        IVec2::new(2, 2),
                        IVec2::new(1, 2),
                        IVec2::new(1, 1),
                        IVec2::new(0, 1),
                        IVec2::new(0, 0)
                    ]
                }
            );

            let traversal_lvl1 = coarse1.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl1,
                GridTraversal::TraversedCoarseCells {
                    cells: vec![IVec2::new(1, 1), IVec2::new(0, 1), IVec2::new(0, 0)]
                }
            );
        }

        {
            let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, -0.1, -1.0));
            let traversal_lvl0 = coarse0.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl0,
                GridTraversal::TraversedCoarseCells {
                    cells: vec![
                        IVec2::new(2, 2),
                        IVec2::new(3, 2),
                        IVec2::new(3, 1),
                        IVec2::new(4, 1),
                        IVec2::new(4, 0)
                    ]
                }
            );

            let traversal_lvl1 = coarse1.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl1,
                GridTraversal::TraversedCoarseCells {
                    cells: vec![IVec2::new(1, 1), IVec2::new(1, 0), IVec2::new(2, 0)]
                }
            );
        }

        {
            let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(-1.0, -0.1, 1.0));
            let traversal_lvl0 = coarse0.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl0,
                GridTraversal::TraversedCoarseCells {
                    cells: vec![
                        IVec2::new(2, 2),
                        IVec2::new(1, 2),
                        IVec2::new(1, 3),
                        IVec2::new(0, 3),
                        IVec2::new(0, 4)
                    ]
                }
            );

            let traversal_lvl1 = coarse1.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl1,
                GridTraversal::TraversedCoarseCells {
                    cells: vec![IVec2::new(1, 1), IVec2::new(0, 1), IVec2::new(0, 2)]
                }
            );
        }
    }
}

//
// pub fn trace_one_ray_dbg(
//     &self,
//     ray: Ray,
//     max_bounces: u32,
//     curr_bounces: u32,
//     last_prim: Option<u32>,
//     output: &mut Vec<Ray>,
// ) {
//     log::debug!("[{curr_bounces}]");
//     log::debug!("Trace {:?}", ray);
//     output.push(ray);
//
//     let entering_point = if !self.contains(&self.world_to_grid_3d(ray.o)) {
//         log::debug!(
//             "  - ray origin is outside of the grid, test if it intersects
// with the bounding \              box"
//         );
//         // The ray origin is outside the grid, tests first with the bounding
// box.         self.mesh
//             .extent
//             .ray_intersects(ray, f32::EPSILON, f32::INFINITY)
//             .map(|point| {
//                 log::debug!("  - intersected with bounding box: {:?}",
// point);                 // Displace the hit point backwards along the ray
// direction.                 point - ray.d * 0.001
//             })
//     } else {
//         Some(ray.o)
//     };
//     log::debug!("  - entering point: {:?}", entering_point);
//
//     if let Some(start) = entering_point {
//         match self.traverse(start.xz(), ray.d.xz()) {
//             GridTraversal::FromTopOrBottom(cell) => {
//                 log::debug!("  - [top/bot] traversed cells: {:?}", cell);
//                 // The ray coming from the top or bottom of the grid,
// calculate the intersection                 // with the grid surface.
//                 let intersections = self.intersect_with_cell(ray, cell);
//                 log::debug!("  - [top/bot] intersections: {:?}",
// intersections);                 match intersections.len() {
//                     0 => {}
//                     1 | 2 => {
//                         log::debug!("  --> intersected with triangle {:?}",
// intersections[0].0);                         let p = intersections[0].1.p;
//                         let n = intersections[0].1.n;
//                         log::debug!("    n: {:?}", n);
//                         log::debug!("    p: {:?}", p);
//                         let d = reflect(ray.d.into(), n.into()).normalize();
//                         log::debug!("    r: {:?}", d);
//                         self.trace_one_ray_dbg(
//                             Ray::new(p, d.into()),
//                             max_bounces,
//                             curr_bounces + 1,
//                             Some(intersections[0].0),
//                             output,
//                         );
//                     }
//                     _ => {
//                         unreachable!(
//                             "Can't have more than 2 intersections with one
// cell of the grid."                         );
//                     }
//                 }
//             }
//             GridTraversal::Traversed { cells, dists } => {
//                 let displaced_ray = Ray::new(start, ray.d);
//                 log::debug!("      -- traversed cells: {:?}", cells);
//                 log::debug!("      -- traversed dists: {:?}", dists);
//                 // For the reason that the entering point is displaced
// backwards along the ray                 // direction, here we will skip the
// first cells if it is                 // outside of the grid.
//                 let first = cells.iter().position(|c|
// self.contains(c)).unwrap();                 let ray_dists = {
//                     let cos = ray.d.dot(Vec3::Y).abs();
//                     let sin = (1.0 - cos * cos).sqrt();
//                     dists
//                         .iter()
//                         .map(|d| displaced_ray.o.y + *d / sin *
// displaced_ray.d.y)                         .collect::<Vec<_>>()
//                 };
//                 log::debug!("      -- ray dists:       {:?}", ray_dists);
//
//                 let mut intersections = vec![None; cells.len()];
//                 for i in first..cells.len() {
//                     let cell = &cells[i];
//                     let entering = ray_dists[i];
//                     let exiting = ray_dists[i + 1];
//                     if self.intersections_happened_at(*cell, entering,
// exiting) {                         log::debug!("        ✓ intersected");
//                         log::debug!("          -> intersect with triangles in
// cell");                         let prim_intersections = match last_prim {
//                             Some(last_prim) => {
//                                 log::debug!("             has last prim");
//                                 self.intersect_with_cell(displaced_ray,
// cells[i])                                     .into_iter()
//                                     .filter(|(index, info, _)| {
//                                         log::debug!(
//                                             "            isect test with tri:
// {:?}, \                                              last_prim: {:?}",
//                                             index,
//                                             last_prim
//                                         );
//                                         *index != last_prim
//                                     })
//                                     .collect::<Vec<_>>()
//                             }
//                             None => {
//                                 log::debug!("             no last prim");
//                                 self.intersect_with_cell(displaced_ray,
// cells[i])                             }
//                         };
//                         log::debug!("           intersections: {:?}",
// prim_intersections);                         match prim_intersections.len() {
//                             0 => {
//                                 intersections[i] = None;
//                             }
//                             1 => {
//                                 let p = prim_intersections[0].1.p;
//                                 let prim = prim_intersections[0].0;
//                                 log::debug!("  - p: {:?}", p);
//                                 // Avoid backface hitting.
//                                 if ray.d.dot(prim_intersections[0].1.n) < 0.0
// {                                     let d =
//                                         reflect(ray.d.into(),
// prim_intersections[0].1.n.into());
// intersections[i] = Some((p, d, prim))                                 }
//                             }
//                             2 => {
//                                 // When the ray is intersected with two
// triangles inside of a                                 // cell, check if they
// are                                 // the same.
//                                 let prim0 = &prim_intersections[0];
//                                 let prim1 = &prim_intersections[1];
//                                 match (prim0.2, prim1.2) {
//                                     (Some(adj_0), Some(adj_1)) => {
//                                         if prim0.0 == adj_1 && prim1.0 ==
// adj_0 {                                             let p = prim0.1.p;
//                                             let n = prim0.1.n;
//                                             let prim = prim0.0;
//                                             log::debug!("    n: {:?}", n);
//                                             log::debug!("    p: {:?}", p);
//                                             if ray.d.dot(prim0.1.n) < 0.0 {
//                                                 let d = reflect(ray.d.into(),
// n.into());                                                 log::debug!("
// r: {:?}", d);
// intersections[i] = Some((p, d, prim))
// }                                         }
//                                     }
//                                     _ => {
//                                         panic!(
//                                             "Intersected with two triangles
// but they are not \                                              the same! {},
// {}",                                             prim_intersections[0].1.p,
//                                             prim_intersections[1].1.p
//                                         );
//                                     }
//                                 }
//                             }
//                             _ => {
//                                 unreachable!(
//                                     "Can't have more than 2 intersections
// with one cell of \                                      the grid."
//                                 );
//                             }
//                         }
//                     }
//                 }
//
//                 if let Some(Some((p, d, prim))) =
// intersections.iter().find(|i| i.is_some()) {
// self.trace_one_ray_dbg(                         Ray::new(*p, (*d).into()),
//                         max_bounces,
//                         curr_bounces + 1,
//                         Some(*prim),
//                         output,
//                     );
//                 }
//             }
//         }
//     } else {
//         log::debug!("  - no starting point");
//     }
// }
//
// pub fn trace_one_ray(
//     &self,
//     ray: Ray,
//     max_bounces: u32,
//     ior_t: &[RefractiveIndex],
// ) -> Option<RtcRecord> {
//     // self.trace_one_ray_dbg(ray, max_bounces, 0, None, &mut output);
//     // output
//     todo!()
// }
//
// fn trace_one_ray_inner(
//     &self,
//     ray: Ray,
//     max_bounces: u32,
//     curr_bounces: u32,
//     prev_prim: Option<u32>,
//     trajectory: &mut Vec<TrajectoryNode>,
// ) {
//     log::debug!("--- current bounce {} ----", curr_bounces);
//     trajectory.push(TrajectoryNode { ray, cos: 0.0 });
//     log::debug!("push ray: {:?} | len: {:?}", ray, trajectory.len());
//
//     if curr_bounces >= max_bounces {
//         log::debug!("  > bounce limit reached");
//         return;
//     }
//
//     // todo: hierachical grid
//
//     let entering_point = if !self.contains(&self.world_to_grid_3d(ray.o)) {
//         log::debug!(
//             "  > entering point outside of grid -- test if it intersects with
// the bounding box"         );
//         self.mesh
//             .extent
//             .ray_intersects(ray, f32::EPSILON, f32::INFINITY)
//             .map(|point| {
//                 log::debug!("    - intersected with bounding box: {:?}",
// point);                 // Displaces the hit point backwards along the ray
// direction.                 point - ray.d * 0.001
//             })
//     } else {
//         Some(ray.o)
//     };
//     log::debug!("  > entering point: {:?}", entering_point);
//
//     if let Some(start) = entering_point {
//         match self.traverse(start.xz(), ray.d.xz()) {
//             GridTraversal::FromTopOrBottom(_) => {}
//             GridTraversal::Traversed { .. } => {}
//         }
//     }
// }
// }

// fn compute_normal(pts: &[Vec3; 3]) -> Vec3 { (pts[1] - pts[0]).cross(pts[2] -
// pts[0]).normalize() }
