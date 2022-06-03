use crate::acq::bxdf::IntersectRecord;
use crate::acq::ray::{Ray, RayTraceRecord, Scattering};
use crate::htfld::{AxisAlignment, Heightfield};
use crate::isect::isect_ray_tri;
use crate::mesh::TriangleMesh;
use glam::{IVec2, Vec2, Vec3, Vec3Swizzles};

/// Helper structure for grid ray tracing.
///
/// The top-left corner is used as the origin of the grid, for the reason that
/// vertices of the triangle mesh are generated following the order from left
/// to right, top to bottom.
///
/// TODO: deal with the case where the grid is not aligned with the world axes.
/// TODO: deal with axis alignment of the heightfield, currently XY alignment is
/// assumed. (with its own transformation matrix).
#[derive(Debug)]
pub struct GridRayTracing<'a> {
    /// The heightfield where the grid is defined.
    surface: &'a Heightfield,

    /// Corresponding `TriangleMesh` of the surface.
    mesh: &'a TriangleMesh,

    /// Minimum coordinates of the grid.
    pub min: IVec2,

    /// Maximum coordinates of the grid.
    pub max: IVec2,

    /// The origin x and y coordinates of the grid in the world space.
    pub origin: Vec2,
}

impl<'a> GridRayTracing<'a> {
    pub fn new(surface: &'a Heightfield, mesh: &'a TriangleMesh) -> Self {
        GridRayTracing {
            surface,
            mesh,
            min: IVec2::ZERO,
            max: IVec2::new(surface.cols as i32 - 1, surface.rows as i32 - 1),
            origin: Vec2::new(
                -surface.du * (surface.cols / 2) as f32,
                -surface.dv * (surface.rows / 2) as f32,
            ),
        }
    }

    /// Check if the given point is inside the grid.
    fn inside(&self, pos: IVec2) -> bool {
        pos.x >= self.min.x && pos.x <= self.max.x && pos.y >= self.min.y && pos.y <= self.max.y
    }

    pub fn trace_ray(&self, ray: Ray) -> Option<IntersectRecord> {
        let starting_point = if !self.inside(self.world_to_grid(ray.o)) {
            // If the ray origin is outside the grid, first check if it intersects
            // with the surface bounding box.
            self.mesh
                .extent
                .intersect_with_ray(ray, f32::NEG_INFINITY, f32::INFINITY)
                .map(|isect_point| {
                    log::debug!("Ray origin outside the grid, intersecting with the bounding box.");
                    isect_point - ray.d * 0.01
                })
        } else {
            // If the ray origin is inside the grid, use the ray origin.
            Some(ray.o)
        };
        log::debug!("Starting point: {:?}", starting_point);

        if let Some(starting_point) = starting_point {
            // Traverse the grid in x, y coordinates, until the ray exits the grid and
            // identify all traversed cells.
            let GridTraversal {
                traversed,
                distances,
            } = self.traverse(starting_point.xz(), ray.d.xz());

            let mut record = None;

            log::debug!("traversed: {:?}", traversed);

            // Iterate over the traversed cells and find the closest intersection.
            for (i, cell) in traversed.iter().enumerate().filter(|(_, cell)| {
                cell.x >= self.min.x - 1
                    && cell.y >= self.min.y - 1
                    && cell.x <= self.max.x
                    && cell.y <= self.max.y
            }) {
                if cell.x == self.min.x - 1
                    || cell.y == self.min.y - 1
                    || cell.x == self.max.x
                    || cell.y == self.max.y
                {
                    // Skip the cell outside the grid, since it is the starting point.
                    continue;
                }

                // Calculate the two ray endpoints at the cell boundaries.
                let entering = ray.o + distances[i] * ray.d;
                let exiting = ray.o + distances[i + 1] * ray.d;

                if self.intersected_with_cell(*cell, entering.z, exiting.z) {
                    // Calculate the intersection point of the ray with the
                    // two triangles inside of the cell.
                    let tris = self.triangles_at(*cell);
                    let isect0 = isect_ray_tri(ray, &tris[0]);
                    let isect1 = isect_ray_tri(ray, &tris[1]);

                    match (isect0, isect1) {
                        (None, None) => {
                            continue;
                        }
                        (Some((_, u, v)), None) => {
                            record = Some(IntersectRecord {
                                ray,
                                geom_id: 0,
                                prim_id: 0,
                                hit_point: (1.0 - u - v) * tris[0][0]
                                    + u * tris[0][1]
                                    + v * tris[0][2],
                                normal: compute_normal(&tris[0]),
                                nudged_times: 0,
                            });
                            break;
                        }
                        (None, Some((_, u, v))) => {
                            record = Some(IntersectRecord {
                                ray,
                                geom_id: 0,
                                prim_id: 0,
                                hit_point: (1.0 - u - v) * tris[1][0]
                                    + u * tris[1][1]
                                    + v * tris[1][2],
                                normal: compute_normal(&tris[1]),
                                nudged_times: 0,
                            });
                            break;
                        }
                        (Some((t0, u, v)), Some((t1, _, _))) => {
                            // The ray hit the shared edge of two triangles.
                            if (t0.abs() - t1.abs()).abs() < f32::EPSILON {
                                record = Some(IntersectRecord {
                                    ray,
                                    geom_id: 0,
                                    prim_id: 0,
                                    hit_point: (1.0 - u - v) * tris[0][0]
                                        + u * tris[0][1]
                                        + v * tris[0][2],
                                    normal: compute_normal(&tris[0]),
                                    nudged_times: 0,
                                });
                                break;
                            } else {
                                log::error!("The ray hit two triangles at the same time.");
                                continue;
                            }
                        }
                    }
                }
            }
            record
        } else {
            None
        }
    }

    /// Convert a world space position into a grid space position (coordinates
    /// of the cell) relative to the origin of the surface (left-top corner).
    pub fn world_to_grid(&self, world_pos: Vec3) -> IVec2 {
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

    /// Obtain the two triangles contained within the cell.
    fn triangles_at(&self, pos: IVec2) -> [[Vec3; 3]; 2] {
        assert!(
            pos.x < self.min.x || pos.y < self.min.y || pos.x > self.max.x || pos.y > self.max.y,
            "The position is out of the grid."
        );
        let idx = (self.surface.cols * pos.y as usize + pos.x as usize) * 2 * 3;
        [
            [
                self.mesh.verts[idx],
                self.mesh.verts[idx + 1],
                self.mesh.verts[idx + 2],
            ],
            [
                self.mesh.verts[idx + 3],
                self.mesh.verts[idx + 4],
                self.mesh.verts[idx + 5],
            ],
        ]
    }

    /// Get the four altitudes associated with the cell at the given
    /// coordinates.
    fn samples_at(&self, pos: IVec2) -> [f32; 4] {
        assert!(
            pos.x < self.min.x || pos.y < self.min.y || pos.x > self.max.x || pos.y > self.max.y,
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
        let altitudes = self.samples_at(cell);
        let (min, max) = {
            let (min, max) = (f32::MAX, f32::MIN);
            altitudes.iter().fold((min, max), |(min, max), height| {
                (min.min(*height), max.max(*height))
            })
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
    /// * `ray_dir` - The direction of the ray.
    pub fn traverse(&self, ray_org: Vec2, ray_dir: Vec2) -> GridTraversal {
        let ray_dir = ray_dir.normalize();
        // Relocate the ray origin to the position relative to the origin of the grid.
        let ray_org = ray_org - self.origin;
        log::debug!("Grid traversal - ray orig: {:?}", ray_org);

        // Calculate dy/dx -- slope of the ray on the grid.
        let m = ray_dir.y / ray_dir.x;
        let m_recip = m.recip();

        // Calculate the distance along the direction of the ray when moving
        // a unit distance (1) along the x-axis and y-axis.
        let unit_dist = Vec2::new((1.0 + m * m).sqrt(), (1.0 + m_recip * m_recip).sqrt());

        let mut current = ray_org.floor().as_ivec2();
        log::debug!("  - starting position: {:?}", current);

        let step_dir = IVec2::new(
            if ray_dir.x >= 0.0 { 1 } else { -1 },
            if ray_dir.y >= 0.0 { 1 } else { -1 },
        );

        // Accumulated line length when moving along the x-axis and y-axis.
        let mut accumulated = Vec2::new(
            if ray_dir.x < 0.0 {
                (ray_org.x - current.x as f32) * unit_dist.x
            } else {
                ((current.x + 1) as f32 - ray_org.x) * unit_dist.x
            },
            if ray_dir.y < 0.0 {
                (ray_org.y - current.y as f32) * unit_dist.y
            } else {
                ((current.y + 1) as f32 - ray_org.y) * unit_dist.y
            },
        );

        let mut distances = if accumulated.x > accumulated.y {
            vec![accumulated.y]
        } else {
            vec![accumulated.x]
        };

        let mut traversed = vec![current];

        while (current.x >= self.min.x - 1 && current.x <= self.max.x + 1)
            && (current.y >= self.min.y - 1 && current.y <= self.max.y + 1)
        {
            let distance = if accumulated.x < accumulated.y {
                current.x += step_dir.x;
                accumulated.x += unit_dist.x;
                accumulated.x
            } else {
                current.y += step_dir.y;
                accumulated.y += unit_dist.y;
                accumulated.y
            };

            distances.push(distance);
            traversed.push(IVec2::new(current.x, current.y));
        }

        GridTraversal {
            traversed,
            distances,
        }
    }
}

fn compute_normal(pts: &[Vec3; 3]) -> Vec3 {
    (pts[1] - pts[0]).cross(pts[2] - pts[0]).normalize()
}

/// Grid traversal outcome.
#[derive(Debug)]
pub struct GridTraversal {
    /// Cells traversed by the ray.
    pub traversed: Vec<IVec2>,

    /// Distances from the origin of the ray (entering cell of the ray) to each
    /// intersection point between the ray and cells.
    pub distances: Vec<f32>,
}

#[test]
fn test_grid_traversal() {
    use crate::mesh::TriangulationMethod;
    let heightfield = Heightfield::new(6, 6, 1.0, 1.0, 2.0, AxisAlignment::XY);
    let triangle_mesh = heightfield.triangulate(TriangulationMethod::Regular);
    let grid = GridRayTracing::new(&heightfield, &triangle_mesh);
    let result = grid.traverse(Vec2::new(-3.5, -3.5), Vec2::new(1.0, 1.0));
    println!("{:?}", result);
}
