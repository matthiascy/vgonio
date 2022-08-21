use crate::{
    acq::{scattering::reflect, Ray},
    htfld::{AxisAlignment, Heightfield},
    isect::{ray_tri_intersect_moller_trumbore, ray_tri_intersect_woop, RayTriIsect},
    mesh::TriangleMesh,
};
use glam::{IVec2, Vec2, Vec3, Vec3Swizzles};
use std::rc::Rc;

/// Helper structure for grid ray tracing.
///
/// The top-left corner is used as the origin of the grid, for the reason that
/// vertices of the triangle mesh are generated following the order from left
/// to right, top to bottom.
///
/// The geometrical center of the grid is located at the center of the scene.
///
/// TODO: deal with the case where the grid is not aligned with the world axes.
/// TODO: deal with axis alignment of the heightfield, currently XY alignment is
/// assumed. (with its own transformation matrix).
#[derive(Debug)]
pub struct GridRayTracing {
    /// The heightfield where the grid is defined.
    surface: Rc<Heightfield>,

    /// Corresponding `TriangleMesh` of the surface.
    surface_mesh: Rc<TriangleMesh>,

    /// Minimum coordinates of the grid.
    pub min: IVec2,

    /// Maximum coordinates of the grid: number of cols or rows - 2.
    pub max: IVec2,

    /// The origin x and y coordinates of the grid in the world space.
    pub origin: Vec2,
}

impl GridRayTracing {
    pub fn new(surface: Rc<Heightfield>, mesh: Rc<TriangleMesh>) -> Self {
        let max = IVec2::new(surface.cols as i32 - 2, surface.rows as i32 - 2);
        let origin = Vec2::new(
            -surface.du * (surface.cols / 2) as f32,
            -surface.dv * (surface.rows / 2) as f32,
        );
        GridRayTracing {
            surface,
            surface_mesh: mesh,
            min: IVec2::ZERO,
            max,
            origin,
        }
    }

    /// Check if the given cell position is inside the grid.
    fn contains(&self, cell_pos: &IVec2) -> bool {
        cell_pos.x >= self.min.x
            && cell_pos.x <= self.max.x
            && cell_pos.y >= self.min.y
            && cell_pos.y <= self.max.y
    }

    /// Convert a world space position into a grid space position (coordinates
    /// of the cell) relative to the origin of the surface (left-top corner).
    pub fn world_to_grid_3d(&self, world_pos: Vec3) -> IVec2 {
        // TODO: deal with the case where the grid is aligned with different world axes.
        let (x, y) = match self.surface.alignment {
            AxisAlignment::XY => (world_pos.x, world_pos.y),
            AxisAlignment::XZ => (world_pos.x, world_pos.z),
            AxisAlignment::YX => (world_pos.y, world_pos.x),
            AxisAlignment::YZ => (world_pos.y, world_pos.z),
            AxisAlignment::ZX => (world_pos.z, world_pos.x),
            AxisAlignment::ZY => (world_pos.z, world_pos.y),
        };

        IVec2::new(
            ((x - self.origin.x) / self.surface.du) as i32,
            ((y - self.origin.y) / self.surface.dv) as i32,
        )
    }

    /// Convert a world space position into a grid space position.
    pub fn world_to_grid_2d(&self, world_pos: Vec2) -> IVec2 {
        IVec2::new(
            ((world_pos.x - self.origin.x) / self.surface.du) as i32,
            ((world_pos.y - self.origin.y) / self.surface.dv) as i32,
        )
    }

    /// Obtain the two triangles (with its corresponding index) contained within
    /// the cell.
    fn triangles_at(&self, pos: IVec2) -> [(u32, [Vec3; 3]); 2] {
        assert!(
            pos.x >= self.min.x
                && pos.y >= self.min.y
                && pos.x <= self.max.x
                && pos.y <= self.max.y,
            "The position is out of the grid."
        );
        let cell = (self.surface.cols - 1) * pos.y as usize + pos.x as usize;
        let pts = &self.surface_mesh.faces[cell * 6..cell * 6 + 6];
        log::debug!(
            "             - cell: {:?}, tris: {:?}, pts {:?}",
            cell,
            [cell * 2, cell * 2 + 1],
            pts
        );
        [
            (
                (cell * 2) as u32,
                [
                    self.surface_mesh.verts[pts[0] as usize],
                    self.surface_mesh.verts[pts[1] as usize],
                    self.surface_mesh.verts[pts[2] as usize],
                ],
            ),
            (
                (cell * 2 + 1) as u32,
                [
                    self.surface_mesh.verts[pts[3] as usize],
                    self.surface_mesh.verts[pts[4] as usize],
                    self.surface_mesh.verts[pts[5] as usize],
                ],
            ),
        ]
    }

    /// Get the four altitudes associated with the cell at the given
    /// coordinates.
    fn altitudes_of_cell(&self, pos: IVec2) -> [f32; 4] {
        assert!(
            pos.x >= self.min.x
                && pos.y >= self.min.y
                && pos.x <= self.max.x
                && pos.y <= self.max.y,
            "The position is out of the grid."
        );

        let x = pos.x as usize;
        let y = pos.y as usize;
        [
            self.surface.sample_at(x, y),
            self.surface.sample_at(x, y + 1),
            self.surface.sample_at(x + 1, y + 1),
            self.surface.sample_at(x + 1, y),
        ]
    }

    /// Given the entering and exiting altitude of a ray, test if the ray
    /// intersects the cell of the height field at the given position.
    fn intersections_happened_at(&self, cell: IVec2, entering: f32, exiting: f32) -> bool {
        let altitudes = self.altitudes_of_cell(cell);
        let (min, max) = {
            altitudes
                .iter()
                .fold((f32::MAX, f32::MIN), |(min, max), height| {
                    (min.min(*height), max.max(*height))
                })
        };
        let condition = (entering >= min && exiting <= max) || (entering >= max && exiting <= min);
        log::debug!(
            "      -> intersecting with cell [{}]: {:?}, altitudes: {:?}, min: {:?}, max: {:?}, \
             entering & existing: {}, {}",
            condition,
            cell,
            altitudes,
            min,
            max,
            entering,
            exiting
        );
        condition
    }

    /// Modified version of Digital Differential Analyzer (DDA) Algorithm.
    ///
    /// Traverse the grid to identify all cells and corresponding intersection
    /// that are traversed by the ray.
    ///
    /// # Arguments
    ///
    /// * `ray_org` - The starting position (in world space) of the ray.
    /// * `ray_dir` - The direction of the ray in world space.
    pub fn traverse(&self, ray_org_world: Vec2, ray_dir: Vec2) -> GridTraversal {
        log::debug!(
            "    Grid origin: {:?}\n       min: {:?}\n       max: {:?}",
            self.origin,
            self.min,
            self.max
        );
        log::debug!(
            "    Traverse the grid with the ray: o {:?}, d: {:?}",
            ray_org_world,
            ray_dir.normalize()
        );
        // Relocate the ray origin to the position relative to the origin of the grid.
        let ray_org_grid = ray_org_world - self.origin;
        let ray_org_cell = self.world_to_grid_2d(ray_org_world);
        log::debug!("      - ray origin cell: {:?}", ray_org_cell);
        log::debug!("      - ray origin grid: {:?}", ray_org_grid);

        let is_parallel_to_x_axis = f32::abs(ray_dir.y - 0.0) < f32::EPSILON;
        let is_parallel_to_y_axis = f32::abs(ray_dir.x - 0.0) < f32::EPSILON;

        if is_parallel_to_x_axis && is_parallel_to_y_axis {
            // The ray is parallel to both axes, which means it comes from the top or bottom
            // of the grid.
            log::debug!("      -> parallel to both axes");
            return GridTraversal::FromTopOrBottom(ray_org_cell);
        }

        let ray_dir = ray_dir.normalize();
        if is_parallel_to_x_axis && !is_parallel_to_y_axis {
            log::debug!("      -> parallel to X axis");
            let (dir, initial_dist, num_cells) = if ray_dir.x < 0.0 {
                (
                    -1,
                    ray_org_grid.x - ray_org_cell.x as f32 * self.surface.du,
                    ray_org_cell.x - self.min.x + 1,
                )
            } else {
                (
                    1,
                    (ray_org_cell.x + 1) as f32 * self.surface.du - ray_org_grid.x,
                    self.max.x - ray_org_cell.x + 1,
                )
            };
            // March along the ray direction until the ray hits the grid boundary.
            let mut cells = vec![IVec2::ZERO; num_cells as usize];
            let mut dists = vec![0.0; num_cells as usize + 1];
            for i in 0..num_cells {
                cells[i as usize] = ray_org_cell + IVec2::new(dir * i, 0);
                dists[i as usize + 1] = initial_dist + (i as f32 * self.surface.du);
            }
            GridTraversal::Traversed { cells, dists }
        } else if is_parallel_to_y_axis && !is_parallel_to_x_axis {
            log::debug!("      -> parallel to Y axis");
            let (dir, initial_dist, num_cells) = if ray_dir.y < 0.0 {
                (
                    -1,
                    ray_org_grid.y - ray_org_cell.y as f32 * self.surface.dv,
                    ray_org_cell.y - self.min.y + 1,
                )
            } else {
                (
                    1,
                    (ray_org_cell.y + 1) as f32 * self.surface.dv - ray_org_grid.y,
                    self.max.y - ray_org_cell.y + 1,
                )
            };
            let mut cells = vec![IVec2::ZERO; num_cells as usize];
            let mut dists = vec![0.0; num_cells as usize + 1];
            // March along the ray direction until the ray hits the grid boundary.
            for i in 0..num_cells {
                cells[i as usize] = ray_org_cell + IVec2::new(0, dir * i);
                dists[i as usize + 1] = initial_dist + (i as f32 * self.surface.dv);
            }
            GridTraversal::Traversed { cells, dists }
        } else {
            // The ray is not parallel to either x or y axis.
            let m = ray_dir.y / ray_dir.x; // The slope of the ray on the grid, dy/dx.
            let m_recip = m.recip(); // The reciprocal of the slope of the ray on the grid, dy/dx.

            log::debug!("    Slope: {:?}, Reciprocal: {:?}", m, m_recip);

            // Calculate the distance along the direction of the ray when moving
            // a unit distance (size of the cell) along the x-axis and y-axis.
            let unit = Vec2::new(
                (1.0 + m * m).sqrt() * self.surface.du,
                (1.0 + m_recip * m_recip).sqrt() * self.surface.dv,
            )
            .abs();

            let mut curr_cell = ray_org_cell;
            log::debug!("  - starting cell: {:?}", curr_cell);
            log::debug!("  - unit dist: {:?}", unit);

            let step_dir = IVec2::new(
                if ray_dir.x >= 0.0 { 1 } else { -1 },
                if ray_dir.y >= 0.0 { 1 } else { -1 },
            );

            // Accumulated line length when moving along the x-axis and y-axis.
            let mut accumulated = Vec2::new(
                if ray_dir.x < 0.0 {
                    //(ray_org_grid.x - curr_cell.x as f32 * self.surface.du) * unit.x
                    (ray_org_grid.x / self.surface.du - curr_cell.x as f32) * unit.x
                } else {
                    //((curr_cell.x + 1) as f32 * self.surface.du - ray_org_grid.x) * unit.x
                    ((curr_cell.x + 1) as f32 - ray_org_grid.x / self.surface.du) * unit.x
                },
                if ray_dir.y < 0.0 {
                    // (ray_org_grid.y - curr_cell.y as f32 * self.surface.dv) * unit.y
                    (ray_org_grid.y / self.surface.dv - curr_cell.y as f32) * unit.y
                } else {
                    // ((curr_cell.y + 1) as f32 * self.surface.dv - ray_org_grid.y) * unit.y
                    ((curr_cell.y + 1) as f32 - ray_org_grid.y / self.surface.dv) * unit.y
                },
            );

            log::debug!("  - accumulated: {:?}", accumulated);

            let mut distances = vec![0.0];
            let mut traversed = vec![curr_cell];

            loop {
                if accumulated.x <= accumulated.y {
                    // avoid min - 1, max + 1
                    if (step_dir.x > 0 && curr_cell.x == self.max.x)
                        || (step_dir.x < 0 && curr_cell.x == self.min.x)
                    {
                        break;
                    }
                    distances.push(accumulated.x);
                    curr_cell.x += step_dir.x;
                    accumulated.x += unit.x;
                } else {
                    if (step_dir.y > 0 && curr_cell.y == self.max.y)
                        || (step_dir.y < 0 && curr_cell.y == self.min.y)
                    {
                        break;
                    }
                    distances.push(accumulated.y);
                    curr_cell.y += step_dir.y;
                    accumulated.y += unit.y;
                };

                traversed.push(IVec2::new(curr_cell.x, curr_cell.y));

                if (curr_cell.x < self.min.x && curr_cell.x > self.max.x)
                    && (curr_cell.y < self.min.y && curr_cell.y > self.max.y)
                {
                    break;
                }
            }

            // Push distance to the existing point of the last cell.
            if accumulated.x <= accumulated.y {
                distances.push(accumulated.x);
            } else {
                distances.push(accumulated.y);
            }

            GridTraversal::Traversed {
                cells: traversed,
                dists: distances,
            }
        }
    }

    /// Calculate the intersection test result of the ray with the triangles
    /// inside of a cell.
    ///
    /// # Returns
    ///
    /// A vector of intersection information [`RayTriInt`] with corresponding
    /// triangle index.
    fn intersect_with_cell(&self, ray: Ray, cell: IVec2) -> Vec<(u32, RayTriIsect, Option<u32>)> {
        let tris = self.triangles_at(cell);
        tris.iter()
            .filter_map(|(index, pts)| {
                log::debug!("               - isect test with tri: {:?}", index);
                ray_tri_intersect_woop(ray, pts)
                    // ray_tri_intersect_moller_trumbore(ray, pts)
                    .map(|isect| {
                        let tris_per_row = (self.surface.cols - 1) * 2;
                        let cells_per_row = self.surface.cols - 1;
                        let tri_index = *index as usize;
                        let is_tri_index_odd = tri_index % 2 != 0;
                        let cell_index = tri_index / 2;
                        let cell_row = cell_index / cells_per_row;
                        let cell_col = cell_index % cells_per_row;
                        let is_first_col = cell_col == 0;
                        let is_last_col = cell_col == cells_per_row - 1;
                        let is_first_row = cell_row == 0;
                        let is_last_row = cell_row == self.surface.rows - 1;
                        let adjacent: Option<(usize, Vec3)> = if isect.u.abs() < 2.0 * f32::EPSILON
                        {
                            // u == 0, intersection happens on the first edge of triangle
                            // If the triangle index is odd or it's not located in the cell of the
                            // 1st column
                            if is_tri_index_odd || !is_first_col {
                                log::debug!(
                                    "              adjacent triangle of {} is {}",
                                    tri_index,
                                    tri_index - 1
                                );
                                Some((tri_index - 1, self.surface_mesh.normals[tri_index - 1]))
                            } else {
                                None
                            }
                        } else if isect.v.abs() < 2.0 * f32::EPSILON {
                            // v == 0, intersection happens on the second edge of triangle
                            if !is_tri_index_odd && !is_first_row {
                                log::debug!(
                                    "              adjacent triangle of {} is {}",
                                    tri_index,
                                    tri_index - (tris_per_row - 1)
                                );
                                Some((
                                    tri_index - (tris_per_row - 1),
                                    (self.surface_mesh.normals[tri_index - (tris_per_row - 1)]),
                                ))
                            } else if is_tri_index_odd && !is_last_col {
                                log::debug!(
                                    "              adjacent triangle of {} is {}",
                                    tri_index,
                                    tri_index + 1
                                );
                                Some((tri_index + 1, self.surface_mesh.normals[tri_index + 1]))
                            } else {
                                None
                            }
                        } else if (isect.u + isect.v - 1.0).abs() < f32::EPSILON {
                            // u + v == 1, intersection happens on the third edge of triangle
                            if !is_tri_index_odd {
                                log::debug!(
                                    "              adjacent triangle of {} is {}",
                                    tri_index,
                                    tri_index + 1
                                );
                                Some((tri_index + 1, self.surface_mesh.normals[tri_index + 1]))
                            } else if is_tri_index_odd && !is_last_row {
                                log::debug!(
                                    "              adjacent triangle of {} is {}",
                                    tri_index,
                                    tri_index + (tris_per_row - 1)
                                );
                                Some((
                                    tri_index + (tris_per_row - 1),
                                    self.surface_mesh.normals[tri_index + (tris_per_row - 1)],
                                ))
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        match adjacent {
                            None => (*index, isect, None),
                            Some((adj_tri, adj_n)) => {
                                let avg_n = (isect.n + adj_n).normalize();
                                log::debug!(
                                    "              -- hitting shared edge, use averaged normal: \
                                     {:?}",
                                    avg_n
                                );
                                (
                                    *index,
                                    RayTriIsect { n: avg_n, ..isect },
                                    Some(adj_tri as u32),
                                )
                            }
                        }
                    })
            })
            .collect::<Vec<_>>()
    }

    pub fn trace_one_ray_dbg(
        &self,
        ray: Ray,
        max_bounces: u32,
        curr_bounces: u32,
        last_prim: Option<u32>,
        output: &mut Vec<Ray>,
    ) {
        log::debug!("[{curr_bounces}]");
        log::debug!("Trace {:?}", ray);
        output.push(ray);

        let entering_point = if !self.contains(&self.world_to_grid_3d(ray.o)) {
            log::debug!(
                "  - ray origin is outside of the grid, test if it intersects with the bounding \
                 box"
            );
            // The ray origin is outside the grid, tests first with the bounding box.
            self.surface_mesh
                .extent
                .intersect_with_ray(ray, f32::EPSILON, f32::INFINITY)
                .map(|point| {
                    log::debug!("  - intersected with bounding box: {:?}", point);
                    // Displace the hit point backwards along the ray direction.
                    point - ray.d * 0.001
                })
        } else {
            Some(ray.o)
        };
        log::debug!("  - entering point: {:?}", entering_point);

        if let Some(start) = entering_point {
            match self.traverse(start.xz(), ray.d.xz()) {
                GridTraversal::FromTopOrBottom(cell) => {
                    log::debug!("  - [top/bot] traversed cells: {:?}", cell);
                    // The ray coming from the top or bottom of the grid, calculate the intersection
                    // with the grid surface.
                    let intersections = self.intersect_with_cell(ray, cell);
                    log::debug!("  - [top/bot] intersections: {:?}", intersections);
                    match intersections.len() {
                        0 => {}
                        1 | 2 => {
                            log::debug!("  --> intersected with triangle {:?}", intersections[0].0);
                            let p = intersections[0].1.p;
                            let n = intersections[0].1.n;
                            log::debug!("    n: {:?}", n);
                            log::debug!("    p: {:?}", p);
                            let d = reflect(ray.d, n).normalize();
                            log::debug!("    r: {:?}", d);
                            self.trace_one_ray_dbg(
                                Ray::new(p, d),
                                max_bounces,
                                curr_bounces + 1,
                                Some(intersections[0].0),
                                output,
                            );
                        }
                        _ => {
                            unreachable!(
                                "Can't have more than 2 intersections with one cell of the grid."
                            );
                        }
                    }
                }
                GridTraversal::Traversed { cells, dists } => {
                    let displaced_ray = Ray::new(start, ray.d);
                    log::debug!("      -- traversed cells: {:?}", cells);
                    log::debug!("      -- traversed dists: {:?}", dists);
                    // For the reason that the entering point is displaced backwards along the ray
                    // direction, here we will skip the first cells if it is
                    // outside of the grid.
                    let first = cells.iter().position(|c| self.contains(c)).unwrap();
                    let ray_dists = {
                        let cos = ray.d.dot(Vec3::Y).abs();
                        let sin = (1.0 - cos * cos).sqrt();
                        dists
                            .iter()
                            .map(|d| displaced_ray.o.y + *d / sin * displaced_ray.d.y)
                            .collect::<Vec<_>>()
                    };
                    log::debug!("      -- ray dists:       {:?}", ray_dists);

                    let mut intersections = vec![None; cells.len()];
                    for i in first..cells.len() {
                        let cell = &cells[i];
                        let entering = ray_dists[i];
                        let exiting = ray_dists[i + 1];
                        if self.intersections_happened_at(*cell, entering, exiting) {
                            log::debug!("        ✓ intersected");
                            log::debug!("          -> intersect with triangles in cell");
                            let prim_intersections = match last_prim {
                                Some(last_prim) => {
                                    log::debug!("             has last prim");
                                    self.intersect_with_cell(displaced_ray, cells[i])
                                        .into_iter()
                                        .filter(|(index, info, _)| {
                                            log::debug!(
                                                "            isect test with tri: {:?}, \
                                                 last_prim: {:?}",
                                                index,
                                                last_prim
                                            );
                                            *index != last_prim
                                        })
                                        .collect::<Vec<_>>()
                                }
                                None => {
                                    log::debug!("             no last prim");
                                    self.intersect_with_cell(displaced_ray, cells[i])
                                }
                            };
                            log::debug!("           intersections: {:?}", prim_intersections);
                            match prim_intersections.len() {
                                0 => {
                                    intersections[i] = None;
                                }
                                1 => {
                                    let p = prim_intersections[0].1.p;
                                    let prim = prim_intersections[0].0;
                                    log::debug!("  - p: {:?}", p);
                                    // Avoid backface hitting.
                                    if ray.d.dot(prim_intersections[0].1.n) < 0.0 {
                                        let d = reflect(ray.d, prim_intersections[0].1.n);
                                        intersections[i] = Some((p, d, prim))
                                    }
                                }
                                2 => {
                                    // When the ray is intersected with two triangles inside of a
                                    // cell, check if they are
                                    // the same.
                                    let prim0 = &prim_intersections[0];
                                    let prim1 = &prim_intersections[1];
                                    match (prim0.2, prim1.2) {
                                        (Some(adj_0), Some(adj_1)) => {
                                            if prim0.0 == adj_1 && prim1.0 == adj_0 {
                                                let p = prim0.1.p;
                                                let n = prim0.1.n;
                                                let prim = prim0.0;
                                                log::debug!("    n: {:?}", n);
                                                log::debug!("    p: {:?}", p);
                                                if ray.d.dot(prim0.1.n) < 0.0 {
                                                    let d = reflect(ray.d, n);
                                                    log::debug!("    r: {:?}", d);
                                                    intersections[i] = Some((p, d, prim))
                                                }
                                            }
                                        }
                                        _ => {
                                            panic!(
                                                "Intersected with two triangles but they are not \
                                                 the same! {}, {}",
                                                prim_intersections[0].1.p,
                                                prim_intersections[1].1.p
                                            );
                                        }
                                    }
                                }
                                _ => {
                                    unreachable!(
                                        "Can't have more than 2 intersections with one cell of \
                                         the grid."
                                    );
                                }
                            }
                        }
                    }

                    if let Some(Some((p, d, prim))) = intersections.iter().find(|i| i.is_some()) {
                        self.trace_one_ray_dbg(
                            Ray::new(*p, *d),
                            max_bounces,
                            curr_bounces + 1,
                            Some(*prim),
                            output,
                        );
                    }
                }
            }
        } else {
            log::debug!("  - no starting point");
        }
    }
}

fn compute_normal(pts: &[Vec3; 3]) -> Vec3 { (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalize() }

/// Grid traversal outcome.
#[derive(Debug)]
pub enum GridTraversal {
    FromTopOrBottom(IVec2),

    Traversed {
        /// Cells traversed by the ray.
        cells: Vec<IVec2>,
        /// Distances from the origin of the ray (entering cell of the ray) to
        /// each intersection point between the ray and cells.
        dists: Vec<f32>,
    },
}

#[test]
fn test_grid_traversal() {
    use crate::mesh::TriangulationMethod;
    let heightfield = Rc::new(Heightfield::new(6, 6, 1.0, 1.0, 2.0, AxisAlignment::XY));
    let triangle_mesh = Rc::new(heightfield.triangulate(TriangulationMethod::Regular));
    let grid = GridRayTracing::new(heightfield.clone(), triangle_mesh.clone());
    let result = grid.traverse(Vec2::new(-3.5, -3.5), Vec2::new(1.0, 1.0));
    println!("{:?}", result);
}
