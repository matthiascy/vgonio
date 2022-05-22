use crate::acq::ray::{Ray, RayTraceRecord};
use crate::htfld::{AxisAlignment, Heightfield};
use crate::mesh::TriangleMesh;
use glam::{IVec2, Mat4, UVec2, Vec2, Vec3, Vec3Swizzles};

/// Helper structure for grid ray tracing.
///
/// The top-left corner is used as the origin of the grid, for the reason that
/// vertices of the triangle mesh are generated following the order from left
/// to right, top to bottom.
///
/// TODO: deal with the case where the grid is not aligned with the world axes.
/// TODO: deal with axis alignment of the heightfield, currently XY alignment is assumed.
/// (with its own transformation matrix).
#[derive(Debug)]
pub struct RayTracingGrid<'a> {
    /// The heightfield where the grid is defined.
    surface: &'a Heightfield,

    /// Corresponding `TriangleMesh` of the surface.
    mesh: &'a TriangleMesh,

    /// Minimum coordinates of the grid.
    min: IVec2,

    /// Maximum coordinates of the grid.
    max: IVec2,

    /// The origin x and y coordinates of the grid in the world space.
    origin: Vec2,
}

impl<'a> RayTracingGrid<'a> {
    pub fn new(surface: &'a Heightfield, mesh: &'a TriangleMesh) -> Self {
        RayTracingGrid {
            surface,
            mesh,
            min: IVec2::ZERO,
            max: IVec2::new(surface.cols as i32 - 1, surface.rows as i32 - 1),
            origin: Vec2::new(-surface.du * (surface.cols / 2) as f32, -surface.dv * (surface.rows / 2) as f32)
        }
    }

    pub fn trace_one_ray(&self, ray: Ray) -> Option<RayTraceRecord> {
        // Calculate the intersection point of the ray with the bounding box of the
        // surface.
        self.mesh
            .extent
            .intersect_with_ray(ray, f32::NEG_INFINITY, f32::INFINITY).map(|isect_point| {
            // Displace the intersection backwards along the ray direction.
            let isect_point = isect_point - ray.d * 0.01;

            // Traverse the grid in x, y coordinates, until the ray exits the grid and identify
            // all traversed cells.
            let GridTraversal {
                traversed, distances
            } = self.traverse(isect_point.xy(), ray.d.xy());

            RayTraceRecord {
                initial: ray,
                current: ray,
                bounces: 0,
            }
        })
    }

    /// Convert a world space position into a grid space position (coordinates
    /// of the cell) relative to the origin of the surface (left-top corner).
    pub fn world_to_grid(&self, world_pos: Vec3) -> IVec2 {
        // TODO: deal with the case where the grid is aligned with different world axes.
        let (x, y) = match self.surface.alignment {
            AxisAlignment::XY => {
                (world_pos.x, world_pos.y)
            }
            AxisAlignment::XZ => {
                (world_pos.x, world_pos.z)
            }
            AxisAlignment::YX => {
                (world_pos.y, world_pos.x)
            }
            AxisAlignment::YZ => {
                (world_pos.y, world_pos.z)
            }
            AxisAlignment::ZX => {
                (world_pos.z, world_pos.x)
            }
            AxisAlignment::ZY => {
                (world_pos.z, world_pos.y)
            }
        };

        IVec2::new(
            ((x - self.origin.x) / self.surface.du) as i32,
            ((y - self.origin.y) / self.surface.dv) as i32,
        )
    }

    /// Traverse the grid to identify all cells and corresponding intersection that are traversed by the ray.
    /// Based on the DDA algorithm.
    pub fn traverse(&self, ray_org: Vec2, ray_dir: Vec2) -> GridTraversal {
        let ray_dir = ray_dir.normalize();
        // Calculate dy/dx -- slope of the ray on the grid.
        let m = ray_dir.y / ray_dir.x;
        let m_recip = m.recip();

        // Calculate the distance along the direction of the ray when moving
        // a unit distance (1) along the x-axis and y-axis.
        let unit_dist = Vec2::new((1.0 + m * m).sqrt(), (1.0 + m_recip * m_recip).sqrt());

        let mut current = ray_org.as_ivec2();

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
            }
        );

        let mut distances = if accumulated.x > accumulated.y {
            vec![accumulated.y]
        } else {
            vec![accumulated.x]
        };

        let mut traversed = vec![current];

        while (current.x >= self.min.x && current.x <= self.max.x)
            && (current.y >= self.min.y && current.y <= self.max.y)
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
            distances
        }
    }
}

/// Grid traversal outcome.
pub struct GridTraversal {
    /// Cells traversed by the ray.
    pub traversed: Vec<IVec2>,

    /// Distances from the origin of the ray (entering cell of the ray) to each intersection point between the ray and cells.
    pub distances: Vec<f32>,
}

/// Digital Differential Analyzer (DDA) Algorithm
pub fn dda(
    ray_start: Vec2,
    ray_dir: Vec2,
    min_cell_x: i32,
    min_cell_y: i32,
    max_cell_x: i32,
    max_cell_y: i32,
) -> (Vec<IVec2>, Vec<Vec2>) {
    let ray_dir = ray_dir.normalize();
    // Calculate dy/dx, slope the line
    let m = ray_dir.y / ray_dir.x;
    let m_recip = 1.0 / m;

    // Calculate the unit step size along the direction of the line when moving
    // a unit distance along the x-axis and y-axis.
    let unit_step_size = Vec2::new((1.0 + m * m).sqrt(), (1.0 + m_recip * m_recip).sqrt());

    // Current cell position.
    let mut curr = ray_start.as_ivec2();

    // Accumulated line length when moving along the x-axis and y-axis.
    let mut walk_dist = Vec2::ZERO;

    // Determine the direction that we are going to walk along the line.
    let step_dir = IVec2::new(
        if ray_dir.x < 0.0 { -1 } else { 1 },
        if ray_dir.y < 0.0 { -1 } else { 1 },
    );

    // Initialise the accumulated line length.
    walk_dist.x = if ray_dir.x < 0.0 {
        (ray_start.x - curr.x as f32) * unit_step_size.x
    } else {
        ((curr.x + 1) as f32 - ray_start.x) * unit_step_size.x
    };

    walk_dist.y = if ray_dir.y < 0.0 {
        (ray_start.y - curr.y as f32) * unit_step_size.y
    } else {
        ((curr.y + 1) as f32 - ray_start.y) * unit_step_size.y
    };

    let mut isects = if walk_dist.x > walk_dist.y {
        vec![ray_start + ray_dir * walk_dist.y]
    } else {
        vec![ray_start + ray_dir * walk_dist.x]
    };
    let mut visited = vec![curr];

    while (curr.x >= min_cell_x && curr.x <= max_cell_x)
        && (curr.y >= min_cell_y && curr.y <= max_cell_y)
    {
        let length = if walk_dist.x < walk_dist.y {
            curr.x += step_dir.x;
            walk_dist.x += unit_step_size.x;
            walk_dist.x
        } else {
            curr.y += step_dir.y;
            walk_dist.y += unit_step_size.y;
            walk_dist.y
        };

        isects.push(ray_start + ray_dir * length);
        visited.push(IVec2::new(curr.x, curr.y));
    }

    (visited, isects)
}

#[test]
fn test_dda() {
    let (cells, intersections) = dda(Vec2::new(-0.5, -0.5), Vec2::new(7.0, 2.0), -1, -1, 7, 7);

    println!("cells: {:?}", cells);
    println!("intersections: {:?}", intersections);
}

fn test_grid_traversal() {
    let grid = RayTracingGrid::new(&)
}
