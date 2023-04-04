//! Customised grid ray tracing for micro-surface measurements.

use crate::{
    app::{
        cache::Cache,
        cli::{BRIGHT_YELLOW, RESET},
    },
    common::ulp_eq,
    math::rcp,
    measure::{
        bsdf::SpectrumSampler,
        collector::CollectorPatches,
        emitter::EmitterSamples,
        measurement::{BsdfMeasurement, Radius},
        rtc::{
            isect::{ray_tri_intersect_woop, RayTriIsect},
            Ray,
        },
        Collector, Emitter, RtcRecord, TrajectoryNode,
    },
    msurf::{AxisAlignment, MicroSurface, MicroSurfaceMesh},
    optics::{fresnel::reflect, ior::RefractiveIndex},
    units::{um, UMicrometre},
};
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
    /// a --- c
    /// |     |
    /// b --- d
    /// The order is: a, b, c, d.
    pub verts: [u32; 4],
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
#[derive(Debug, Clone)]
pub enum GridTraversal {
    FromTopOrBottom(IVec2),
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
    /// and the corresponding intersection.
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
        let start_cell = self.world_to_local(&origin, &start_pos).unwrap();
        println!("Start position in the grid space: {:?}", start_cell);
        let ray_dir = ray.d.xz();
        log::debug!("Ray direction in the grid space: {:?}", ray_dir);

        let is_parallel_to_grid_x = ulp_eq(ray_dir.y, 0.0);
        let is_parallel_to_grid_y = ulp_eq(ray_dir.x, 0.0);

        if is_parallel_to_grid_x && is_parallel_to_grid_y {
            // The ray is parallel to both the grid x and y axis; coming from
            // the top or bottom of the grid.
            log::debug!("The ray is parallel to both the grid x and y axis");
            return GridTraversal::FromTopOrBottom(start_cell);
        }

        let mut travelled_dists = vec![0.0];
        let mut traversed_cells = vec![start_cell];

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
            let reaching_left_or_right_edge = step_dir.x > 0
                && curr_cell.x > (self.cols - 1) as i32
                || step_dir.x < 0 && curr_cell.x < 0;
            let reaching_top_or_bottom_edge = step_dir.y > 0
                && curr_cell.y > (self.rows - 1) as i32
                || step_dir.y < 0 && curr_cell.y < 0;
            if accumulated_line.x <= accumulated_line.y {
                if reaching_left_or_right_edge {
                    break;
                }
                // Move along the x-axis.
                travelled_dists.push(accumulated_line.x);
                curr_cell.x += step_dir.x;
                accumulated_line.x += dl.x;
            } else {
                if reaching_top_or_bottom_edge {
                    break;
                }
                // Move along the y-axis.
                travelled_dists.push(accumulated_line.y);
                curr_cell.y += step_dir.y;
                accumulated_line.y += dl.y;
            }

            traversed_cells.push(curr_cell);
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
    pub fn world_to_local(&self, origin: &Vec2, pos: &Vec2) -> Option<IVec2> {
        let x = {
            let x = (pos.x - origin.x) / self.world_space_cell_size.x;
            // If the position is exactly on the boundary of the cell, we
            // should use the cell on the left.
            if x.fract() == 0.0 {
                x as u32 - 1
            } else {
                x as u32
            }
        };
        let y = {
            let y = (pos.y - origin.y) / self.world_space_cell_size.y;
            // If the position is exactly on the boundary of the cell, we
            // should use the cell on the left.
            if y.fract() == 0.0 {
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
                    let (min_height, max_height) =
                        verts.iter().fold((f32::MAX, f32::MIN), |(min, max), vert| {
                            let height = surf.samples[*vert as usize];
                            (min.min(height), max.max(height))
                        });
                    *cell = BaseCell {
                        x: x as u32,
                        y: y as u32,
                        verts,
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
}

#[cfg(test)]
mod tests {
    use crate::{
        common::ulp_eq,
        measure::rtc::{grid::MultilevelGrid, Ray},
        msurf::{AxisAlignment, MicroSurface},
        units::{um, UMicrometre},
    };
    use glam::{UVec2, Vec3, Vec3Swizzles};
    use proptest::proptest;

    #[test]
    fn test_multilevel_grid_creation() {
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
    }

    #[test]
    #[rustfmt::skip]
    fn test_multilevel_grid_region() {
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
        // todo: test traversal
        let surf = MicroSurface::new(10, 10, um!(1.0), um!(1.0), um!(0.0));
        let mesh = surf.as_micro_surface_mesh();
        let grid = MultilevelGrid::new(&surf, &mesh, 2);
        let base = grid.base();
        println!("level: {}", grid.level());
        let coarse0 = grid.coarse(0);
        let coarse1 = grid.coarse(1);
        let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, -0.1, -1.0));
        let mut traversal = coarse1.traverse(mesh.bounds.min.xz(), &ray);
        println!("traversal: {:?}", traversal);
    }
}

// /// Convert a world space position into a grid space position (coordinates
// /// of the cell) relative to the origin of the surface (left-top corner).
// pub fn world_to_grid_3d(&self, world_pos: Vec3) -> IVec2 {
//     // TODO: deal with the case where the grid is aligned with different
// world axes.     let (x, y) = match self.mesh.alignment {
//         AxisAlignment::XY => (world_pos.x, world_pos.y),
//         AxisAlignment::XZ => (world_pos.x, world_pos.z),
//         AxisAlignment::YX => (world_pos.y, world_pos.x),
//         AxisAlignment::YZ => (world_pos.y, world_pos.z),
//         AxisAlignment::ZX => (world_pos.z, world_pos.x),
//         AxisAlignment::ZY => (world_pos.z, world_pos.y),
//     };
//
//     // TODO:
//     IVec2::new(
//         ((x - self.origin.x) / self.surf.du) as i32,
//         ((y - self.origin.y) / self.surf.dv) as i32,
//     )
// }
//
// /// Convert a world space position into a grid space position.
// pub fn world_to_grid_2d(&self, world_pos: Vec2) -> IVec2 {
//     IVec2::new(
//         ((world_pos.x - self.origin.x) / self.surf.du) as i32,
//         ((world_pos.y - self.origin.y) / self.surf.dv) as i32,
//     )
// }
//
// /// Obtain the two triangles (with its corresponding index) contained within
// /// the cell.
// fn triangles_at(&self, pos: IVec2) -> [(u32, [Vec3; 3]); 2] {
//     assert!(self.contains(&pos), "the position is out of the grid.");
//     let cell = (self.surf.cols - 1) * pos.y as usize + pos.x as usize;
//     let pts = &self.mesh.facets[cell * 6..cell * 6 + 6];
//     log::debug!(
//         "             - cell: {:?}, tris: {:?}, pts {:?}",
//         cell,
//         [cell * 2, cell * 2 + 1],
//         pts
//     );
//     [
//         (
//             (cell * 2) as u32,
//             [
//                 self.mesh.verts[pts[0] as usize],
//                 self.mesh.verts[pts[1] as usize],
//                 self.mesh.verts[pts[2] as usize],
//             ],
//         ),
//         (
//             (cell * 2 + 1) as u32,
//             [
//                 self.mesh.verts[pts[3] as usize],
//                 self.mesh.verts[pts[4] as usize],
//                 self.mesh.verts[pts[5] as usize],
//             ],
//         ),
//     ]
// }
//
// /// Get the four altitudes associated with the cell at the given
// /// coordinates.
// fn altitudes_of_cell(&self, pos: IVec2) -> [f32; 4] {
//     assert!(self.contains(&pos), "the position is out of the grid.");
//
//     let x = pos.x as usize;
//     let y = pos.y as usize;
//     [
//         self.surf.sample_at(x, y).as_f32(),
//         self.surf.sample_at(x, y + 1).as_f32(),
//         self.surf.sample_at(x + 1, y + 1).as_f32(),
//         self.surf.sample_at(x + 1, y).as_f32(),
//     ]
// }
//
// /// Given the entering and exiting altitude of a ray, test if the ray
// /// intersects the cell of the height field at the given position.
// fn intersections_happened_at(&self, cell: IVec2, entering: f32, exiting: f32)
// -> bool {     let altitudes = self.altitudes_of_cell(cell);
//     let (min, max) = {
//         altitudes
//             .iter()
//             .fold((f32::MAX, f32::MIN), |(min, max), height| {
//                 (min.min(*height), max.max(*height))
//             })
//     };
//     let condition = (entering >= min && exiting <= max) || (entering >= max
// && exiting <= min);     log::debug!(
//         "      -> intersecting with cell [{}]: {:?}, altitudes: {:?}, min:
// {:?}, max: {:?}, \          entering & existing: {}, {}",
//         condition,
//         cell,
//         altitudes,
//         min,
//         max,
//         entering,
//         exiting
//     );
//     condition
// }

// /// Calculate the intersection test result of the ray with the triangles
// /// inside of a cell.
// ///
// /// # Returns
// ///
// /// A vector of intersection information [`RayTriInt`] with corresponding
// /// triangle index.
// fn intersect_with_cell(&self, ray: Ray, cell: IVec2) -> Vec<(u32,
// RayTriIsect, Option<u32>)> {     let tris = self.triangles_at(cell);
//     tris.iter()
//         .filter_map(|(index, pts)| {
//             log::debug!("               - isect test with tri: {:?}", index);
//             ray_tri_intersect_woop(ray, pts)
//                 // ray_tri_intersect_moller_trumbore(ray, pts)
//                 .map(|isect| {
//                     let tris_per_row = (self.surf.cols - 1) * 2;
//                     let cells_per_row = self.surf.cols - 1;
//                     let tri_index = *index as usize;
//                     let is_tri_index_odd = tri_index % 2 != 0;
//                     let cell_index = tri_index / 2;
//                     let cell_row = cell_index / cells_per_row;
//                     let cell_col = cell_index % cells_per_row;
//                     let is_first_col = cell_col == 0;
//                     let is_last_col = cell_col == cells_per_row - 1;
//                     let is_first_row = cell_row == 0;
//                     let is_last_row = cell_row == self.surf.rows - 1;
//                     let adjacent: Option<(usize, Vec3)> = if isect.u.abs() <
// 2.0 * f32::EPSILON                     {
//                         // u == 0, intersection happens on the first edge of
// triangle                         // If the triangle index is odd or it's not
// located in the cell of the                         // 1st column
//                         if is_tri_index_odd || !is_first_col {
//                             log::debug!(
//                                 "              adjacent triangle of {} is
// {}",                                 tri_index,
//                                 tri_index - 1
//                             );
//                             Some((tri_index - 1,
// self.mesh.facet_normals[tri_index - 1]))                         } else {
//                             None
//                         }
//                     } else if isect.v.abs() < 2.0 * f32::EPSILON {
//                         // v == 0, intersection happens on the second edge of
// triangle                         if !is_tri_index_odd && !is_first_row {
//                             log::debug!(
//                                 "              adjacent triangle of {} is
// {}",                                 tri_index,
//                                 tri_index - (tris_per_row - 1)
//                             );
//                             Some((
//                                 tri_index - (tris_per_row - 1),
//                                 (self.mesh.facet_normals[tri_index -
// (tris_per_row - 1)]),                             ))
//                         } else if is_tri_index_odd && !is_last_col {
//                             log::debug!(
//                                 "              adjacent triangle of {} is
// {}",                                 tri_index,
//                                 tri_index + 1
//                             );
//                             Some((tri_index + 1,
// self.mesh.facet_normals[tri_index + 1]))                         } else {
//                             None
//                         }
//                     } else if (isect.u + isect.v - 1.0).abs() < f32::EPSILON
// {                         // u + v == 1, intersection happens on the third
// edge of triangle                         if !is_tri_index_odd {
//                             log::debug!(
//                                 "              adjacent triangle of {} is
// {}",                                 tri_index,
//                                 tri_index + 1
//                             );
//                             Some((tri_index + 1,
// self.mesh.facet_normals[tri_index + 1]))                         } else if
// is_tri_index_odd && !is_last_row {                             log::debug!(
//                                 "              adjacent triangle of {} is
// {}",                                 tri_index,
//                                 tri_index + (tris_per_row - 1)
//                             );
//                             Some((
//                                 tri_index + (tris_per_row - 1),
//                                 self.mesh.facet_normals[tri_index +
// (tris_per_row - 1)],                             ))
//                         } else {
//                             None
//                         }
//                     } else {
//                         None
//                     };
//                     match adjacent {
//                         None => (*index, isect, None),
//                         Some((adj_tri, adj_n)) => {
//                             let avg_n = (isect.n + adj_n).normalize();
//                             log::debug!(
//                                 "              -- hitting shared edge, use
// averaged normal: \                                  {:?}",
//                                 avg_n
//                             );
//                             (
//                                 *index,
//                                 RayTriIsect { n: avg_n, ..isect },
//                                 Some(adj_tri as u32),
//                             )
//                         }
//                     }
//                 })
//         })
//         .collect::<Vec<_>>()
// }
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
