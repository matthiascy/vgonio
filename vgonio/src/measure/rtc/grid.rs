//! Customised grid ray tracing for micro-surface measurements.

// TODO: verification

use crate::{
    app::{
        cache::Cache,
        cli::{BRIGHT_YELLOW, RESET},
    },
    measure::{
        bsdf::MeasuredBsdfData,
        collector::CollectorPatches,
        emitter::EmitterSamples,
        measurement::BsdfMeasurementParams,
        rtc,
        rtc::{Hit, LastHit, Ray, RayTrajectory, RayTrajectoryNode, MAX_RAY_STREAM_SIZE},
        Emitter,
    },
    msurf::{MicroSurface, MicroSurfaceMesh},
    optics::fresnel,
};
use rayon::prelude::*;
use std::time::Instant;
use vgcore::{
    math,
    math::{IVec2, UVec2, Vec2, Vec3, Vec3A, Vec3Swizzles},
};

/// Extra data associated with a ray stream.
///
/// A ray stream is a chunk of rays that are emitted from the same emitter.
#[derive(Debug, Clone)]
pub struct RayStreamData {
    /// The last hit of each ray in the stream.
    last_hit: Vec<LastHit>,
    /// The trajectory of each ray in the stream. The trajectory is a list of
    trajectory: Vec<RayTrajectory>,
}

/// Measures the BSDF of the given micro-surface mesh.
pub fn measure_bsdf(
    params: &BsdfMeasurementParams,
    surf: &MicroSurface,
    mesh: &MicroSurfaceMesh,
    samples: &EmitterSamples,
    patches: &CollectorPatches,
    cache: &Cache,
) -> MeasuredBsdfData {
    // Unify the units of the micro-surface and emitter radius by converting
    // to micrometres.
    let orbit_radius = params.emitter.estimate_orbit_radius(mesh);
    let disk_radius = params.emitter.estimate_disk_radius(mesh);
    let max_bounces = params.emitter.max_bounces;
    let grid = MultilevelGrid::new(surf, mesh, 64);
    let mut data = vec![];
    log::debug!("mesh extent: {:?}", mesh.bounds);
    log::debug!("emitter orbit radius: {}", orbit_radius);
    log::debug!("emitter disk radius: {:?}", disk_radius);

    for pos in params.emitter.measurement_points() {
        println!(
            "      {BRIGHT_YELLOW}>{RESET} Emit rays from {}° {}°",
            pos.zenith.in_degrees().value(),
            pos.azimuth.in_degrees().value()
        );
        let t = Instant::now();
        let emitted_rays = Emitter::emit_rays(samples, pos, orbit_radius, disk_radius);
        let num_emitted_rays = emitted_rays.len();
        let elapsed = t.elapsed();

        log::debug!(
            "emitted {} rays with direction {} from position {}° {}° in {:?} secs.",
            num_emitted_rays,
            emitted_rays[0].dir,
            pos.zenith.in_degrees().value(),
            pos.azimuth.in_degrees().value(),
            elapsed.as_secs_f64(),
        );

        let num_streams = (num_emitted_rays + MAX_RAY_STREAM_SIZE - 1) / MAX_RAY_STREAM_SIZE;
        let stream_size = if num_streams == 1 {
            num_emitted_rays
        } else {
            MAX_RAY_STREAM_SIZE
        };

        let mut stream_data = vec![
            RayStreamData {
                last_hit: vec![
                    LastHit {
                        geom_id: u32::MAX,
                        prim_id: u32::MAX,
                        normal: Vec3A::ZERO,
                    };
                    stream_size
                ],
                trajectory: vec![
                    RayTrajectory(Vec::with_capacity(max_bounces as usize));
                    stream_size
                ],
            };
            num_streams
        ];

        emitted_rays
            .chunks(MAX_RAY_STREAM_SIZE)
            .zip(stream_data.iter_mut())
            .enumerate()
            //.par_chunks(MAX_RAY_STREAM_SIZE).zip(stream_data.par_iter_mut())
            .for_each(|(i, (rays, data))| {
                let chunk_size = rays.len();
                let mut validities = vec![true; chunk_size];
                let mut rays = rays.to_owned();
                let mut hits = vec![Hit::default(); chunk_size];
                let mut bounces = 0;
                let mut num_active_rays = chunk_size;

                while bounces < max_bounces && num_active_rays > 0 {
                    log::trace!(
                        "stream {}, bounces: {}, num_active_rays: {}",
                        i,
                        bounces,
                        num_active_rays
                    );

                    for (ray, hit) in rays.iter().zip(hits.iter_mut()) {
                        grid.trace(ray, hit);
                    }

                    for (i, (ray, (is_valid, hit))) in rays
                        .iter_mut()
                        .zip(validities.iter_mut().zip(hits.iter_mut()))
                        .enumerate()
                    {
                        if !*is_valid {
                            continue;
                        }

                        if !hit.is_valid() {
                            *is_valid = false;
                            num_active_rays -= 1;
                            continue;
                        }

                        if hit.prim_id == data.last_hit[i].prim_id {
                            // Hit the same primitive as the last hit.
                            log::trace!("self-intersection: nudging the ray origin");
                            let traj_node = data.trajectory[i].last_mut().unwrap();
                            traj_node.org += data.last_hit[i].normal * 1e-6;
                            continue;
                        } else {
                            // Hit a different primitive.
                            // Record the cos of the last hit.
                            let last_node = data.trajectory[i].last_mut().unwrap();
                            last_node.cos = Some(hit.normal.dot(ray.dir));
                            let reflected_dir = fresnel::reflect(ray.dir.into(), hit.normal.into());
                            data.trajectory[i].push(RayTrajectoryNode {
                                org: hit.point.into(),
                                dir: reflected_dir,
                                cos: None,
                            });
                            // Update the last hit.
                            let last_hit = &mut data.last_hit[i];
                            last_hit.geom_id = hit.geom_id;
                            last_hit.prim_id = hit.prim_id;
                            last_hit.normal = hit.normal.into();

                            // Update the ray and hit.
                            ray.org = hit.point;
                            ray.dir = reflected_dir.into();
                            hit.invalidate();
                        }
                    }

                    bounces += 1;
                }

                log::trace!(
                    "------------ result {}, active rays {}\n {:?} | {:?}\n{:?}",
                    bounces,
                    num_active_rays,
                    data.trajectory,
                    data.trajectory.len() - 1,
                    validities
                );
            });
        // Extract the trajectory of each ray.
        let trajectories = stream_data
            .into_iter()
            .flat_map(|data| data.trajectory)
            .collect::<Vec<_>>();
        data.push(
            params
                .collector
                .collect(params, mesh, pos, &trajectories, patches, cache),
        );
    }

    MeasuredBsdfData {
        params: *params,
        samples: data,
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
    /// The min grid coordinates of the last level (one level finer) coarse
    /// cells or base cells, inclusive.
    min: UVec2,
    /// The max grid coordinates of the last level (one level finer) coarse
    /// cells or base cells, inclusive.
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

/// Represents a grid cell of any level.
pub trait Cell: Sized {
    /// Minimum height of the cell.
    fn min_height(&self) -> f32;
    /// Maximum height of the cell.
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
#[doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../misc/imgs/grid.svg"))]
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
pub struct GridTraversal {
    /// Whether the cells are coarse.
    pub is_coarse: bool,
    /// Whether the ray is coming vertically.
    pub is_coming_vertically: bool,
    /// Cells traversed by the ray.
    pub cells: Vec<IVec2>,
    /// Distances from the origin of the ray (entering cell of the ray)
    /// to each intersection point between the ray and cells.
    /// The first element is always 0, and the last element is the
    /// distance from the origin to the exit point of the ray.
    /// The distance is in the world space.
    dists: Vec<f32>,
}

impl GridTraversal {
    /// Returns an iterator over the cells traversed by the ray.
    /// Each cell is represented by its grid coordinates and the
    /// entering and exiting distances of the ray to the cell.
    ///
    /// In the case of a ray coming vertically, the iterator will
    /// return only one cell with the entering and exiting distances
    /// set to INFINITY.
    pub fn iter(&self) -> Box<dyn Iterator<Item = (IVec2, f32, f32)> + '_> {
        debug_assert!(
            !self.is_coming_vertically,
            "For ray coming vertically, use directly the cell."
        );

        if self.is_coming_vertically {
            return Box::new(
                self.cells
                    .iter()
                    .map(|c| (*c, f32::INFINITY, f32::INFINITY)),
            );
        }

        Box::new(self.cells.iter().enumerate().map(|(i, c)| {
            let entering = self.dists[i];
            let exiting = self.dists[i + 1];
            (*c, entering, exiting)
        }))
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
    /// along the ray path inside a given region.
    ///
    /// Modified version of Digital Differential Analyzer (DDA) algorithm.
    ///
    /// # Arguments
    ///
    /// * `origin` - Origin of the grid in the world space (top-left corner).
    /// * `ray` - The ray to traverse the grid.
    /// * `min` - Minimum grid coordinates (current level) of the region,
    ///   inclusive.
    /// * `max` - Maximum grid coordinates (current level) of the region,
    ///   inclusive.
    pub fn traverse(&self, origin: Vec2, ray: &Ray) -> GridTraversal {
        log::debug!("Traversing the grid along the ray: {:?}", ray);
        let start_pos = ray.org.xz() - origin;
        log::debug!(
            "Start position in the world space relative to grid's origin: {:?}",
            start_pos
        );
        let start_cell = self.world_to_local(&origin, &ray.org.xz()).unwrap();
        println!("Start position in the grid space: {:?}", start_cell);
        let ray_dir = ray.dir.xz();
        log::debug!("Ray direction in the grid space: {:?}", ray_dir);

        let is_parallel_to_grid_x = math::ulp_eq(ray_dir.y, 0.0);
        let is_parallel_to_grid_y = math::ulp_eq(ray_dir.x, 0.0);

        if is_parallel_to_grid_x && is_parallel_to_grid_y {
            // The ray is parallel to both the grid x and y axis; coming from
            // the top or bottom of the grid.
            log::debug!("(traversing) The ray is parallel to both the grid x and y axis");
            return GridTraversal {
                is_coarse: self.is_coarse,
                is_coming_vertically: true,
                cells: vec![start_cell],
                dists: vec![],
            };
        }

        let mut travelled_dists = vec![0.0];
        let mut traversed_cells = vec![];

        // We are using the DDA algorithm to find all the cells that the ray traverses
        // and its intersection points. In case we are traversing on the coarse
        // grid we can assume that the size of the grid cell is always
        // (1.0, mesh.dv/mesh.du).
        log::debug!("The ray is not parallel to either the grid x or y axis");
        let cell_size = self.world_space_cell_size;

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
                let m = ray_dir.y * math::rcp(ray_dir.x);
                (m, math::rcp(m))
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

        GridTraversal {
            is_coarse: self.is_coarse,
            is_coming_vertically: false,
            cells: traversed_cells,
            dists: travelled_dists,
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
        let entering_height = ray.org.y + entering * ray.dir.y;
        let exiting_height = ray.org.y + exiting * ray.dir.y;
        let cell = self.cell_at(pos.x as _, pos.y as _);
        let ge_max = entering_height >= cell.max_height() || exiting_height >= cell.max_height();
        let le_min = entering_height <= cell.min_height() || exiting_height <= cell.min_height();
        !ge_max && !le_min
    }
}

impl Grid<BaseCell> {
    /// Checks if the given ray intersects with triangles within the cell at
    /// the given position in the grid space.
    pub fn intersects_cell_triangles(
        &self,
        ray: &Ray,
        pos: &IVec2,
        mesh: &MicroSurfaceMesh,
        hit: &mut Hit,
    ) {
        #[cfg(not(test))]
        use log::debug;
        #[cfg(test)]
        use std::println as debug;

        debug!("Intersecting with cell at {:?}", pos);
        let cell = self.cell_at(pos.x as _, pos.y as _);
        debug!("  - triangle indices: {:?}", cell.tris);
        let triangles = self.triangles_of_cell(pos, mesh);
        debug!("  - triangles: {:?}", triangles);
        let isect0 = rtc::ray_tri_intersect_woop(ray, &triangles[0], f32::MAX);
        let isect1 = rtc::ray_tri_intersect_woop(ray, &triangles[1], f32::MAX);
        debug!("  - isect0: {:?}", isect0);
        debug!("  - isect1: {:?}", isect1);
        match (isect0, isect1) {
            (Some(isect0), Some(isect1)) => {
                let p0 = isect0.p;
                let p1 = isect1.p;
                if math::close_enough(&p0, &p1) {
                    // The ray intersects with two triangles on the shared edge.
                    // u of the first triangle is 0.0, and w of the second triangle
                    // is 0.0.
                    debug_assert!(math::ulp_eq(isect0.u, 0.0));
                    debug_assert!(math::ulp_eq(1.0 - isect1.u - isect1.v, 0.0));
                    *hit = Hit {
                        normal: (isect0.n + isect1.n).normalize(),
                        point: p0,
                        u: isect0.u,
                        v: isect0.v,
                        geom_id: 0,
                        prim_id: cell.tris[0],
                    };
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
                let adjacent = if math::ulp_eq(isect.u, 0.0) {
                    // u is 0.0, so the ray intersects with the edge p1-p2.
                    Some(opposite_idx)
                } else if math::ulp_eq(isect.v, 0.0) {
                    // v is 0.0, so the ray intersects with the edge p0-p2.
                    if self_idx < 2 * self.cols {
                        // The cell is on the top row.
                        None
                    } else {
                        // Return the triangle index of the cell one row above.
                        Some(self_idx - (self.cols - 1))
                    }
                } else if math::ulp_eq(isect.u + isect.v, 1.0) {
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
                *hit = Hit {
                    normal: n,
                    point: isect.p,
                    u: isect.u,
                    v: isect.v,
                    geom_id: 0,
                    prim_id: cell.tris[0],
                };
            }
            (None, Some(isect)) => {
                // Intersects with the second triangle.
                // Check if the ray intersects with other edges, get the adjacent triangle
                // index and its normal.
                let self_idx = cell.tris[1];
                let opposite_idx = cell.tris[0];
                let adjacent = if math::ulp_eq(isect.u, 0.0) {
                    // u is 0.0, so the ray intersects with the edge p1-p2.
                    if self_idx >= (2 * self.cols) * (self.rows - 1) {
                        // The cell is on the bottom row.
                        None
                    } else {
                        // Return the triangle index of the cell one row below.
                        Some(self_idx + (self.cols - 1))
                    }
                } else if math::ulp_eq(isect.v, 0.0) {
                    // v is 0.0, so the ray intersects with the edge p0-p2.
                    if self_idx % (2 * self.cols) == 2 * self.cols - 1 {
                        // The cell is on the right column.
                        None
                    } else {
                        // Return the triangle index of the cell one column to the left.
                        Some(self_idx + 1)
                    }
                } else if math::ulp_eq(isect.u + isect.v, 1.0) {
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
                *hit = Hit {
                    normal: n,
                    point: isect.p,
                    u: isect.u,
                    v: isect.v,
                    geom_id: 0,
                    prim_id: cell.tris[1],
                };
            }
            (None, None) => {
                hit.geom_id = u32::MAX;
                hit.prim_id = u32::MAX;
            }
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

    /// Trace the ray on the grid knowing the origin position of the grid in the
    /// world space.
    pub fn trace_with_origin(
        &self,
        origin: Vec2,
        ray: &Ray,
        mesh: &MicroSurfaceMesh,
        hit: &mut Hit,
    ) {
        #[cfg(not(test))]
        use log::debug;
        #[cfg(test)]
        use std::println as debug;

        debug!("Tracing the ray {:?} on the grid", ray);
        let start_pos = ray.org.xz() - origin;
        debug!(
            "Relative start position {:?} in the world space, grid's origin: {:?}",
            start_pos, origin
        );
        let start_cell = self.world_to_local(&origin, &ray.org.xz()).unwrap();
        debug!("Start position in the grid space: {:?}", start_cell);
        let ray_dir = ray.dir.xz();
        debug!("Ray direction in the grid space: {:?}", ray_dir);
        self.trace(start_cell, start_pos, ray, mesh, hit);
    }

    /// Traces the given ray through the grid over a region and returns the
    /// intersection point with the closest triangle.
    ///
    /// The ray direction is assumed to be on the grid plane in the world space.
    ///
    /// # Arguments
    ///
    /// * `start_cell` - The cell where the ray starts.
    /// * `start_pos` - The position of the ray origin in the grid space.
    /// * `ray` - The ray to trace.
    /// * `mesh` - The mesh to trace the ray against.
    /// * `hit` - The intersection point with the closest triangle.
    /// * `min` - The minimum cell position in the grid space, inclusive.
    /// * `max` - The maximum cell position in the grid space, inclusive.
    pub fn trace(
        &self,
        start_cell: IVec2,
        start_pos: Vec2,
        ray: &Ray,
        mesh: &MicroSurfaceMesh,
        hit: &mut Hit,
    ) {
        #[cfg(not(test))]
        use log::debug;
        #[cfg(test)]
        use std::println as debug;

        let ray_dir = ray.dir.xz();
        let is_parallel_to_grid_x = math::ulp_eq(ray_dir.y, 0.0);
        let is_parallel_to_grid_y = math::ulp_eq(ray_dir.x, 0.0);

        if is_parallel_to_grid_x && is_parallel_to_grid_y {
            // The ray is parallel to both the grid x and y axis; coming from
            // the top or bottom of the grid.
            debug!("(tracing) The ray is parallel to both the grid x and y axis");
            return self.intersects_cell_triangles(ray, &start_cell, mesh, hit);
        }

        let mut entering_dist = 0.0;
        let mut exiting_dist = 0.0;

        // We are using the DDA algorithm to find all the cells that the ray
        // traverses and its intersection points. In case we are traversing
        // on the coarse grid we can assume that the size of the grid cell is
        // always 1x1.
        let cell_size = if self.is_coarse {
            Vec2::new(1.0, 1.0)
        } else {
            self.world_space_cell_size
        };
        debug!("Cell size: {:?}", cell_size);

        // 1. Calculate the slope of the ray on the grid.
        let (m, m_rcp) = {
            if is_parallel_to_grid_x && !is_parallel_to_grid_y {
                debug!("The ray is parallel to the grid x axis");
                (1.0, 0.0)
            } else if !is_parallel_to_grid_x && is_parallel_to_grid_y {
                debug!("The ray is parallel to the grid y axis");
                (0.0, 1.0)
            } else {
                debug!("The ray is not parallel to either the grid x or y axis");
                let m = ray_dir.y * math::rcp(ray_dir.x);
                (m, math::rcp(m))
            }
        };
        debug!("Slope of the ray on the grid: {}, reciprocal: {}", m, m_rcp);

        // 2. Calculate the distance along each axis when moving one cell in the grid
        // space.
        let dl = Vec2::new(
            m.mul_add(m, 1.0).sqrt() * cell_size.x,
            m_rcp.mul_add(m_rcp, 1.0).sqrt() * cell_size.y,
        );
        debug!(
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

        debug!("Initial accumulated line length: {:?}", accumulated_line);

        // 4. Identify the traversed cells and intersection points
        loop {
            debug!("Current cell: {:?}", curr_cell);
            let reaching_left_or_right_edge = step_dir.x > 0 && curr_cell.x >= self.cols as i32
                || step_dir.x < 0 && curr_cell.x < 0;
            let reaching_top_or_bottom_edge = step_dir.y > 0 && curr_cell.y >= self.rows as i32
                || step_dir.y < 0 && curr_cell.y < 0;

            if reaching_left_or_right_edge || reaching_top_or_bottom_edge {
                break;
            }

            // Perform the pre-check to see if the cell may intersect the ray.
            if self.may_intersect_cell(ray, &curr_cell, entering_dist, exiting_dist) {
                // Perform the actual intersection test. The ray origin is displaced by the
                // entering distance.
                debug!("Cell {:?} may intersect the ray", curr_cell);
                self.intersects_cell_triangles(ray, &curr_cell, mesh, hit);
                if hit.is_valid() {
                    debug!("Cell {:?} intersects the ray", curr_cell);
                    return;
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
                        .into_par_iter()
                        .map(|level| Self::build_coarse_grid(&base, level, min_coarse_cell_size))
                        .collect::<Vec<_>>(),
                )
            }
        };

        #[cfg(debug_assertions)]
        {
            if !coarse.is_empty() {
                // Check order of generated coarse grids.
                for i in 0..coarse.len() - 1 {
                    let s0 = coarse[i].grid_space_cell_size;
                    let s1 = coarse[i + 1].grid_space_cell_size;
                    assert!(s1 > s0);
                }
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

    /// Traces a ray through the grid.
    pub fn trace(&self, ray: &Ray, hit: &mut Hit) {
        fn trace_coarse(multi_grid: &MultilevelGrid, level: u32, ray: Ray, hit: &mut Hit) {
            let grid = multi_grid.coarse(level as usize);
            for (_, entering_dist) in grid
                .traverse(multi_grid.mesh.bounds.min.xz(), &ray)
                .iter()
                .filter_map(|(cell, entering, exiting)| {
                    if grid.may_intersect_cell(&ray, &cell, entering, exiting) {
                        Some((grid.cell_at(cell.x as _, cell.y as _), entering))
                    } else {
                        None
                    }
                })
            {
                let displaced_ray = Ray::new(ray.org + ray.dir * entering_dist, ray.dir);
                if level != 0 {
                    trace_coarse(multi_grid, level - 1, displaced_ray, hit);
                } else {
                    multi_grid.base.trace_with_origin(
                        multi_grid.mesh.bounds.min.xz(),
                        &displaced_ray,
                        multi_grid.mesh,
                        hit,
                    );
                };

                if hit.is_valid() {
                    return;
                }
            }
        }

        hit.invalidate();

        match ray.intersects_aabb(&self.mesh.bounds) {
            None => {}
            Some(isect) => {
                // The ray is emitted from the micro surface, which means that the
                // ray has been reflected. We don't need to trace the coarse grid.
                if isect.is_inside() || self.level == 0 || ray.dir.x == 0.0 && ray.dir.z == 0.0 {
                    self.base
                        .trace_with_origin(self.mesh.bounds.min.xz(), ray, self.mesh, hit);
                } else {
                    // Trace the coarse grid.
                    let init_level = self.level - 1;
                    trace_coarse(self, init_level, *ray, hit);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        measure::rtc::{grid::MultilevelGrid, Hit, Ray},
        msurf::{AxisAlignment, MicroSurface},
        ulp_eq,
        units::{um, LengthUnit, UMicrometre},
    };
    use vgcore::math::{IVec2, UVec2, Vec3, Vec3Swizzles};

    #[test]
    fn multilevel_grid_creation() {
        let surf =
            MicroSurface::from_samples(9, 9, 0.5, 0.5, LengthUnit::UM, &vec![0.0; 81], None, None);
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

        let surf = MicroSurface::from_samples(
            rows,
            cols,
            0.5,
            0.5,
            LengthUnit::UM,
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
            None
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
        
        for y in 0..coarse0.rows {
            for x in 0..coarse0.cols {
                let cell = coarse0.cell_at(x, y);
                assert_eq!(cell.min, UVec2::new(x * 2, y * 2));
                assert_eq!(cell.max, UVec2::new(((x+1) * 2 - 1).min(base.cols - 1), 
                                                ((y+1) * 2 - 1).min(base.rows - 1)));
            }
        }
        
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
        let surf = MicroSurface::new(10, 10, 1.0, 1.0, 0.0, LengthUnit::UM);
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
                assert!(!traversal_base.is_coarse);
                assert_eq!(
                    traversal_base.cells,
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
                    traversal_lvl0.cells,
                    vec![
                        IVec2::new(2, 2),
                        IVec2::new(3, 2),
                        IVec2::new(3, 3),
                        IVec2::new(4, 3),
                        IVec2::new(4, 4)
                    ]
                );
                assert!(traversal_lvl0.is_coarse);

                assert_eq!(
                    traversal_lvl1.cells,
                    vec![IVec2::new(1, 1), IVec2::new(2, 1), IVec2::new(2, 2)],
                );
                assert!(traversal_lvl1.is_coarse);
            }
            {
                let traversal_base = base.traverse(mesh.bounds.min.xz(), &ray_slope_0_5);
                let traversal_lvl0 = coarse0.traverse(mesh.bounds.min.xz(), &ray_slope_0_5);
                let traversal_lvl1 = coarse1.traverse(mesh.bounds.min.xz(), &ray_slope_0_5);
                assert!(!traversal_base.is_coarse);
                assert_eq!(
                    traversal_base.cells,
                    vec![
                        IVec2::new(5, 5),
                        IVec2::new(6, 5),
                        IVec2::new(7, 5),
                        IVec2::new(7, 6),
                        IVec2::new(8, 6),
                    ]
                );
                assert_eq!(
                    traversal_lvl0.cells,
                    vec![
                        IVec2::new(2, 2),
                        IVec2::new(3, 2),
                        IVec2::new(3, 3),
                        IVec2::new(4, 3),
                    ]
                );
                assert_eq!(
                    traversal_lvl1.cells,
                    vec![IVec2::new(1, 1), IVec2::new(2, 1), IVec2::new(2, 2)]
                );
            }
        }

        {
            let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(-1.0, -0.1, -1.0));
            let traversal_lvl0 = coarse0.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl0.cells,
                vec![
                    IVec2::new(2, 2),
                    IVec2::new(1, 2),
                    IVec2::new(1, 1),
                    IVec2::new(0, 1),
                    IVec2::new(0, 0)
                ]
            );

            let traversal_lvl1 = coarse1.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl1.cells,
                vec![IVec2::new(1, 1), IVec2::new(0, 1), IVec2::new(0, 0)]
            );
        }

        {
            let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, -0.1, -1.0));
            let traversal_lvl0 = coarse0.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl0.cells,
                vec![
                    IVec2::new(2, 2),
                    IVec2::new(3, 2),
                    IVec2::new(3, 1),
                    IVec2::new(4, 1),
                    IVec2::new(4, 0)
                ]
            );

            let traversal_lvl1 = coarse1.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl1.cells,
                vec![IVec2::new(1, 1), IVec2::new(1, 0), IVec2::new(2, 0)]
            );
        }

        {
            let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(-1.0, -0.1, 1.0));
            let traversal_lvl0 = coarse0.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl0.cells,
                vec![
                    IVec2::new(2, 2),
                    IVec2::new(1, 2),
                    IVec2::new(1, 3),
                    IVec2::new(0, 3),
                    IVec2::new(0, 4)
                ]
            );

            let traversal_lvl1 = coarse1.traverse(mesh.bounds.min.xz(), &ray);
            assert_eq!(
                traversal_lvl1.cells,
                vec![IVec2::new(1, 1), IVec2::new(0, 1), IVec2::new(0, 2)]
            );
        }
    }

    #[test]
    fn grid_trace() {
        // todo: check triangle intersections
        #[rustfmt::skip]
            let surf = MicroSurface::from_samples(
            5,
            5,
            1.0,
            1.0,
            LengthUnit::UM,
            [
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 2.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            None,
            None,
        );
        let mesh = surf.as_micro_surface_mesh();
        let grid = MultilevelGrid::new(&surf, &mesh, 2);
        let base = grid.base();

        let mut hit_base = Hit::default();
        let ray = Ray::new(Vec3::new(-3.0, 0.2, -3.0), Vec3::new(1.0, 0.0, 1.0));
        {
            base.trace_with_origin(mesh.bounds.min.xz(), &ray, &mesh, &mut hit_base);
            assert!(hit_base.is_valid());
            assert_eq!(hit_base.prim_id, 11);
        }

        let mut hit = Hit::default();
        {
            grid.trace(&ray, &mut hit);
        }
        {
            assert!(hit.is_valid());
            assert_eq!(hit.prim_id, 11);
        }

        assert_eq!(hit_base, hit);
    }
}
