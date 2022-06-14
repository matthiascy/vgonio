use crate::acq::ray::{reflect, Ray};
use crate::acq::tracing::IntersectRecord;
use crate::htfld::{AxisAlignment, Heightfield};
use crate::isect::{isect_ray_tri, RayTriInt};
use crate::mesh::TriangleMesh;
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
    mesh: Rc<TriangleMesh>,

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
            mesh,
            min: IVec2::ZERO,
            max,
            origin,
        }
    }

    /// Check if the given cell position is inside the grid.
    fn inside(&self, cell_pos: IVec2) -> bool {
        cell_pos.x >= self.min.x && cell_pos.x <= self.max.x && cell_pos.y >= self.min.y && cell_pos.y <= self.max.y
    }

    // pub fn trace_ray(&self, ray: Ray) -> Option<IntersectRecord> {
    //     let starting_point = if !self.inside(self.world_to_grid(ray.o)) {
    //         // If the ray origin is outside the grid, first check if it
    // intersects         // with the surface bounding box.
    //         self.mesh
    //             .extent
    //             .intersect_with_ray(ray, f32::NEG_INFINITY, f32::INFINITY)
    //             .map(|isect_point| {
    //                 log::debug!("Ray origin outside the grid, intersecting with
    // the bounding box.");                 isect_point - ray.d * 0.01
    //             })
    //     } else {
    //         // If the ray origin is inside the grid, use the ray origin.
    //         Some(ray.o)
    //     };
    //     log::debug!("Starting point: {:?}", starting_point);
    //
    //     if let Some(starting_point) = starting_point {
    //         // Traverse the grid in x, y coordinates, until the ray exits the
    // grid and         // identify all traversed cells.
    //         let GridTraversal {
    //             traversed,
    //             distances,
    //         } = self.traverse(starting_point.xz(), ray.d.xz());
    //
    //         let mut record = None;
    //
    //         log::debug!("traversed: {:?}", traversed);
    //
    //         // Iterate over the traversed cells and find the closest
    // intersection.         for (i, cell) in
    // traversed.iter().enumerate().filter(|(_, cell)| {             cell.x >=
    // self.min.x - 1                 && cell.y >= self.min.y - 1
    //                 && cell.x <= self.max.x
    //                 && cell.y <= self.max.y
    //         }) {
    //             if cell.x == self.min.x - 1
    //                 || cell.y == self.min.y - 1
    //                 || cell.x == self.max.x
    //                 || cell.y == self.max.y
    //             {
    //                 // Skip the cell outside the grid, since it is the starting
    // point.                 continue;
    //             }
    //
    //             // Calculate the two ray endpoints at the cell boundaries.
    //             let entering = ray.o + distances[i] * ray.d;
    //             let exiting = ray.o + distances[i + 1] * ray.d;
    //
    //             if self.intersected_with_cell(*cell, entering.z, exiting.z) {
    //                 // Calculate the intersection point of the ray with the
    //                 // two triangles inside of the cell.
    //                 let tris = self.triangles_at(*cell);
    //                 let isect0 = isect_ray_tri(ray, &tris[0]);
    //                 let isect1 = isect_ray_tri(ray, &tris[1]);
    //
    //                 match (isect0, isect1) {
    //                     (None, None) => {
    //                         continue;
    //                     }
    //                     (Some((_, u, v)), None) => {
    //                         record = Some(IntersectRecord {
    //                             ray,
    //                             geom_id: 0,
    //                             prim_id: 0,
    //                             hit_point: (1.0 - u - v) * tris[0][0]
    //                                 + u * tris[0][1]
    //                                 + v * tris[0][2],
    //                             normal: compute_normal(&tris[0]),
    //                             nudged_times: 0,
    //                         });
    //                         break;
    //                     }
    //                     (None, Some((_, u, v))) => {
    //                         record = Some(IntersectRecord {
    //                             ray,
    //                             geom_id: 0,
    //                             prim_id: 0,
    //                             hit_point: (1.0 - u - v) * tris[1][0]
    //                                 + u * tris[1][1]
    //                                 + v * tris[1][2],
    //                             normal: compute_normal(&tris[1]),
    //                             nudged_times: 0,
    //                         });
    //                         break;
    //                     }
    //                     (Some((t0, u, v)), Some((t1, _, _))) => {
    //                         // The ray hit the shared edge of two triangles.
    //                         if (t0.abs() - t1.abs()).abs() < f32::EPSILON {
    //                             record = Some(IntersectRecord {
    //                                 ray,
    //                                 geom_id: 0,
    //                                 prim_id: 0,
    //                                 hit_point: (1.0 - u - v) * tris[0][0]
    //                                     + u * tris[0][1]
    //                                     + v * tris[0][2],
    //                                 normal: compute_normal(&tris[0]),
    //                                 nudged_times: 0,
    //                             });
    //                             break;
    //                         } else {
    //                             log::error!("The ray hit two triangles at the
    // same time.");                             continue;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         record
    //     } else {
    //         None
    //     }
    // }

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
            pos.x >= self.min.x && pos.y >= self.min.y && pos.x <= self.max.x && pos.y <= self.max.y,
            "The position is out of the grid."
        );
        let cell = ((self.surface.cols - 1) * pos.y as usize + pos.x as usize);
        let pts = &self.mesh.faces[cell * 6..cell * 6 + 6];
        log::debug!("cell: {:?}, tris: {:?}, pts {:?}", cell, [cell * 2, cell * 2 + 1], pts);
        [
            (
                (cell * 2) as u32,
                [
                    self.mesh.verts[pts[0] as usize],
                    self.mesh.verts[pts[1] as usize],
                    self.mesh.verts[pts[2] as usize],
                ],
            ),
            (
                (cell * 2 + 1) as u32,
                [
                    self.mesh.verts[pts[3] as usize],
                    self.mesh.verts[pts[4] as usize],
                    self.mesh.verts[pts[5] as usize],
                ],
            ),
        ]
    }

    /// Get the four altitudes associated with the cell at the given
    /// coordinates.
    fn altitudes_of_cell(&self, pos: IVec2) -> [f32; 4] {
        assert!(
            pos.x >= self.min.x && pos.y >= self.min.y && pos.x <= self.max.x && pos.y <= self.max.y,
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
    fn intersected_with_cell(&self, cell: IVec2, entering: f32, exiting: f32) -> bool {
        let altitudes = self.altitudes_of_cell(cell);
        let (min, max) = {
            let (min, max) = (f32::MAX, f32::MIN);
            altitudes
                .iter()
                .fold((min, max), |(min, max), height| (min.min(*height), max.max(*height)))
        };

        // The ray is either entering the cell beneath the surface or
        // entering and exiting the cell above the surface.
        //            a > max             b > max
        // max - - - - - - - - - - - - - - - - - -
        //      min < a < max       min < b < max
        // min - - - - - - - - - - - - - - - - - -
        //            a < min             b < min
        !(entering < min || (entering > max && exiting > max))
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
            "    Grid origin: {:?}\
             \n       min: {:?}\
             \n       max: {:?}",
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
        log::debug!("    Ray origin cell: {:?}", ray_org_cell);
        log::debug!("    Ray origin grid: {:?}", ray_org_grid);

        let is_parallel_to_x_axis = f32::abs(ray_dir.y - 0.0) < f32::EPSILON;
        let is_parallel_to_y_axis = f32::abs(ray_dir.x - 0.0) < f32::EPSILON;

        if is_parallel_to_x_axis && is_parallel_to_y_axis {
            // The ray is parallel to both axes, which means it comes from the top or bottom
            // of the grid.
            return GridTraversal::FromTopOrBottom(ray_org_grid.floor().as_ivec2());
        }

        let ray_dir = ray_dir.normalize();
        if is_parallel_to_x_axis && !is_parallel_to_y_axis {
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
            let cells_dists: (Vec<IVec2>, Vec<f32>) = (0..num_cells)
                .map(|i| {
                    let cell = ray_org_cell + IVec2::new(dir * i, 0);
                    let dist = initial_dist + (i as f32 * self.surface.du);
                    (cell, dist)
                })
                .unzip();

            GridTraversal::Traversed {
                cells: cells_dists.0,
                dists: cells_dists.1,
            }
        } else if is_parallel_to_y_axis && !is_parallel_to_x_axis {
            let (dir, initial_dist, count) = if ray_dir.y < 0.0 {
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
            // March along the ray direction until the ray hits the grid boundary.
            let cells_dists: (Vec<IVec2>, Vec<f32>) = (0..count)
                .map(|i| {
                    let cell = ray_org_cell + IVec2::new(0, dir * i);
                    let dist = initial_dist + (i as f32 * self.surface.dv);
                    (cell, dist)
                })
                .unzip();

            GridTraversal::Traversed {
                cells: cells_dists.0,
                dists: cells_dists.1,
            }
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
                    (ray_org_grid.x - curr_cell.x as f32 * self.surface.du) * unit.x
                } else {
                    ((curr_cell.x + 1) as f32 * self.surface.du - ray_org_grid.x) * unit.x
                },
                if ray_dir.y < 0.0 {
                    (ray_org_grid.y - curr_cell.y as f32 * self.surface.dv) * unit.y
                } else {
                    ((curr_cell.y + 1) as f32 * self.surface.dv - ray_org_grid.y) * unit.y
                },
            );

            log::debug!("  - accumulated: {:?}", accumulated);

            let mut distances = if accumulated.x > accumulated.y {
                vec![accumulated.y]
            } else {
                vec![accumulated.x]
            };

            let mut traversed = vec![curr_cell];

            loop {
                let distance = if accumulated.x < accumulated.y {
                    // avoid min - 1, max + 1
                    if (step_dir.x > 0 && curr_cell.x == self.max.x) || (step_dir.x < 0 && curr_cell.x == self.min.x) {
                        break;
                    }
                    curr_cell.x += step_dir.x;
                    accumulated.x += unit.x;
                    accumulated.x
                } else {
                    if (step_dir.y > 0 && curr_cell.y == self.max.y) || (step_dir.y < 0 && curr_cell.y == self.min.y) {
                        break;
                    }
                    curr_cell.y += step_dir.y;
                    accumulated.y += unit.y;
                    accumulated.y
                };

                distances.push(distance);
                traversed.push(IVec2::new(curr_cell.x, curr_cell.y));

                if (curr_cell.x < self.min.x && curr_cell.x > self.max.x)
                    && (curr_cell.y < self.min.y && curr_cell.y > self.max.y)
                {
                    break;
                }
            }

            GridTraversal::Traversed {
                cells: traversed,
                dists: distances,
            }
        }
    }

    /// Calculate the intersection test result of the ray with the triangles inside of a cell.
    ///
    /// # Returns
    ///
    /// A vector of intersection information [`RayTriInt`] with corresponding triangle index.
    fn intersect_with_cell(&self, ray: Ray, cell: IVec2) -> Vec<(u32, RayTriInt)> {
        let tris = self.triangles_at(cell);
        tris
            .iter()
            .filter_map(|(index, pts)| {
                log::debug!("    tri: {:?}", index);
                isect_ray_tri(ray, pts).map(|isect| (*index, isect))
            })
            .collect::<Vec<_>>()
    }

    pub fn trace_one_ray_dbg(&self, ray: Ray, max_bounces: u32, curr_bounces: u32, last_prim: Option<u32>, output: &mut Vec<Ray>) {
        log::debug!("[{curr_bounces}]");
        let starting_point = if !self.inside(self.world_to_grid_3d(ray.o)) {
            // The ray origin is outside the grid, tests first with the bounding box.
            self.mesh
                .extent
                .intersect_with_ray(ray, f32::EPSILON, f32::INFINITY)
                .map(|point| {
                    log::debug!("  - intersected with bounding box: {:?}", point);
                    // Displace the hit point backwards along the ray direction.
                    point - ray.d * 0.01
                })
        } else {
            Some(ray.o)
        };
        log::debug!("  - starting point: {:?}", starting_point);

        output.push(ray);

        if let Some(start) = starting_point {
            match self.traverse(start.xz(), ray.d.xz()) {
                GridTraversal::FromTopOrBottom(cell) => {
                    log::debug!("  - traversed cells: {:?}", cell);
                    // The ray coming from the top or bottom of the grid, calculate the intersection
                    // with the grid surface.
                    let intersections = self.intersect_with_cell(ray, cell);
                    match intersections.len() {
                        0 => {
                            return;
                        }
                        1 => {
                            let p = intersections[0].1.p;
                            let n = intersections[0].1.n;
                            log::debug!("  - intersection point: {:?}", p);
                            let d = reflect(ray.d, n);
                            self.trace_one_ray_dbg(Ray::new(p, d), max_bounces, curr_bounces + 1, Some(intersections[0].0), output);
                        }
                        2 => {
                            // When the ray is intersected with two triangles inside of a cell,
                            // check if they are the same.
                            if intersections[0].1.p != intersections[1].1.p {
                                panic!("Intersected with two triangles but they are not the same!");
                            }
                        }
                        _ => {
                            unreachable!("Can't have more than 2 intersections with one cell of the grid.");
                        }
                    }
                }
                GridTraversal::Traversed { cells, dists } => {
                    let dists = {
                        let mut dists_ = vec![0.0];
                        dists_.extend_from_slice(&dists);
                        dists_
                    };

                    // Iterate over the cells and the distances along the ray.
                    for i in 0..cells.len() {
                        let entering = dists[i] * ray.d.y;
                        let exiting = dists[i + 1] * ray.d.y;
                        if self.intersected_with_cell(cells[i], entering, exiting) {
                            let intersections = match last_prim {
                                Some(last_prim) => {
                                    self.intersect_with_cell(ray, cells[i]).into_iter().filter(|(index, info)| {
                                        *index != last_prim
                                    }).collect::<Vec<_>>()
                                }
                                None => {
                                    self.intersect_with_cell(ray, cells[i])
                                }
                            };

                            match intersections.len() {
                                0 => {
                                    return;
                                }
                                1 => {
                                    let p = intersections[0].1.p;
                                    log::debug!("  - intersection point: {:?}", p);
                                    let d = reflect(ray.d, intersections[0].1.n);
                                    self.trace_one_ray_dbg(Ray::new(p, d), max_bounces, curr_bounces + 1, Some(intersections[0].0), output);
                                }
                                _ => {
                                    panic!("Intersected with multiple triangles inside of a cell!");
                                }
                            }
                        }
                    }

                    log::debug!("  - traversed cells: {:?}", cells);
                    log::debug!("  - traversed dists: {:?}", dists);
                }
            }
        } else {
            log::debug!("  - no starting point");
        }
    }
}

fn compute_normal(pts: &[Vec3; 3]) -> Vec3 {
    (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalize()
}

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
