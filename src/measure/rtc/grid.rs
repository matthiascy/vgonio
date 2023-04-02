//! Customised grid ray tracing for micro-surface measurements.

use crate::{
    app::{
        cache::Cache,
        cli::{BRIGHT_YELLOW, RESET},
    },
    measure::{
        bsdf::SpectrumSampler,
        collector::CollectorPatches,
        emitter::EmitterSamples,
        measurement::{BsdfMeasurement, Radius},
        rtc::{
            isect::{ray_tri_intersect_woop, Aabb, RayTriIsect},
            Ray,
        },
        Collector, Emitter, RtcRecord, TrajectoryNode,
    },
    msurf::{AxisAlignment, MicroSurface, MicroSurfaceMesh},
    optics::{fresnel::reflect, ior::RefractiveIndex},
    units::um,
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

/// Axis at which the kd-tree is split.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum Axis {
    Horizontal,
    Vertical,
}

impl Axis {
    /// Returns the next axis to split.
    fn flip(&self) -> Self {
        match self {
            Axis::Horizontal => Axis::Vertical,
            Axis::Vertical => Axis::Horizontal,
        }
    }
}

/// Evenly split kd-tree for accelerating grid ray tracing.
///
/// The kd-tree is built from the grid of the micro-surface mesh. Each node of
/// the kd-tree is either a leaf node or a branch node. A leaf node contains a
/// part of the rectangular grid of the micro-surface mesh, and a branch node
/// contains two child nodes. The splitting axis of the branch node is
/// determined by the axis of the parent node, and is flipped for each level of
/// the tree.
#[derive(Debug)]
struct KdTree<'ms> {
    surf: &'ms MicroSurface,
    mesh: &'ms MicroSurfaceMesh,
    root: u32,
    nodes: Vec<Node>,
}

#[derive(Debug, Copy, Clone)]
enum Node {
    /// Leaf node of the kd-tree.
    /// The leaf node contains a part of the rectangular grid of the
    /// micro-surface mesh (represented by lower-right and upper-left grid
    /// coordinates), , the minimum and maximum height of the grid cells and
    /// the axis-aligned bounding box of the node.
    Leaf {
        /// Minimum grid coordinate of the node.
        min: UVec2,
        /// Maximum grid coordinate of the node.
        max: UVec2,
        /// Minimum height of the node.
        min_height: f32,
        /// Maximum height of the node.
        max_height: f32,
    },
    /// Branch node of the kd-tree.
    /// The branch node contains two child nodes, and the axis at which the
    /// node is split. The splitting axis of the branch node is determined by
    /// the axis of the parent node, and is flipped for each level of the tree.
    /// The axis index is the coordinate of the axis in the grid space of the
    /// dimension specified by the splitting axis.
    Branch {
        /// Axis at which the node is split.
        axis: Axis,
        /// Minimum grid coordinate contained in the node.
        min: UVec2,
        /// Maximum grid coordinate contained in the node.
        max: UVec2,
        /// Minimum height of the node.
        min_height: f32,
        /// Maximum height of the node.
        max_height: f32,
        /// Index of the first child node.
        child0: NodeIndex,
        /// Index of the second child node.
        child1: NodeIndex,
        /// Axis-aligned bounding box of the node.
        bounds: Aabb,
    },
}

impl Node {
    pub fn is_leaf(&self) -> bool { matches!(self, Node::Leaf { .. }) }

    pub fn is_branch(&self) -> bool { matches!(self, Node::Branch { .. }) }

    /// Returns the minimum grid coordinate of the node.
    pub fn min(&self) -> UVec2 {
        match self {
            Node::Leaf { min, .. } => *min,
            Node::Branch { min, .. } => *min,
        }
    }

    /// Returns the maximum grid coordinate of the node.
    pub fn max(&self) -> UVec2 {
        match self {
            Node::Leaf { max, .. } => *max,
            Node::Branch { max, .. } => *max,
        }
    }

    pub fn axis(&self) -> Option<Axis> {
        match self {
            Node::Leaf { .. } => None,
            Node::Branch { axis, .. } => Some(*axis),
        }
    }

    pub fn set_child0(&mut self, child0: NodeIndex) {
        match self {
            Node::Leaf { .. } => panic!("Leaf node has no children"),
            Node::Branch { child0: c0, .. } => *c0 = child0,
        }
    }

    pub fn set_child1(&mut self, child1: NodeIndex) {
        match self {
            Node::Leaf { .. } => panic!("Leaf node has no children"),
            Node::Branch { child1: c1, .. } => *c1 = child1,
        }
    }
}

/// Index of a node in the kd-tree.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct NodeIndex(u32);

impl NodeIndex {
    fn none() -> Self { Self(u32::MAX) }

    fn some(index: u32) -> Self { Self(index) }

    fn is_none(&self) -> bool { self.0 == u32::MAX }

    fn is_some(&self) -> bool { self.0 != u32::MAX }

    fn unwrap(self) -> u32 { self.0 }
}

impl<'ms> KdTree<'ms> {
    /// Maximum number of grid cells in a leaf node in each dimension.
    pub const MAX_LEAF_CELL_DIM: u32 = 32;

    pub fn new(grid: &Grid<'ms>) -> Self {
        let mut nodes = Vec::new();
        let root = Self::build_branch(
            &mut nodes,
            Axis::Horizontal,
            UVec2::new(0, 0),
            UVec2::new(grid.width - 1, grid.height - 1),
            grid.surf.min,
            grid.surf.max,
            grid.mesh.bounds,
        );
        Self::build(grid, &mut nodes);
        Self {
            surf: grid.surf,
            mesh: grid.mesh,
            root,
            nodes,
        }
    }

    /// Builds the kd-tree.
    fn build(grid: &Grid<'ms>, nodes: &mut Vec<Node>) { Self::build_recursive(grid, 0, nodes); }

    fn build_recursive(grid: &Grid<'ms>, parent: u32, nodes: &mut Vec<Node>) {
        if nodes[parent as usize].is_leaf() {
            return;
        }

        let (min, max, axis) = {
            let parent = &nodes[parent as usize];
            (parent.min(), parent.max(), parent.axis().unwrap())
        };

        let ((min0, max0), (min1, max1)) = match axis {
            Axis::Horizontal => {
                let min0 = UVec2::new(min.x, min.y);
                let max0 = UVec2::new(max.x, max.y / 2);
                let min1 = UVec2::new(min.x, max.y / 2 + 1);
                let max1 = UVec2::new(max.x, max.y);
                ((min0, max0), (min1, max1))
            }
            Axis::Vertical => {
                let min0 = UVec2::new(min.x, min.y);
                let max0 = UVec2::new(max.x / 2, max.y);
                let min1 = UVec2::new(max.x / 2 + 1, min.y);
                let max1 = UVec2::new(max.x, max.y);
                ((min0, max0), (min1, max1))
            }
        };

        {
            // Creation of the first child node.
            let grid_size = max0 - min0;
            let (min_height, max_height) = grid.min_max_of_region(min0, max0);
            // If the grid size is smaller than the maximum leaf size, create a
            // leaf node.
            if grid_size.x <= Self::MAX_LEAF_CELL_DIM && grid_size.y <= Self::MAX_LEAF_CELL_DIM {
                let index = Self::build_leaf(nodes, min0, max0, min_height, max_height);
                nodes[parent as usize].set_child0(NodeIndex::some(index));
            } else {
                // Otherwise, create a branch node.
                let split_axis = if grid_size.x > grid_size.y {
                    Axis::Horizontal
                } else {
                    Axis::Vertical
                };
                let index = Self::build_branch(
                    nodes,
                    split_axis,
                    min0,
                    max0,
                    min_height,
                    max_height,
                    grid.bounds_of_region(min0, max0),
                );
                nodes[parent as usize].set_child0(NodeIndex::some(index));
                Self::build_recursive(grid, index, nodes);
            }
        }

        {
            // Creation of the second child node.
            let grid_size = max1 - min1;
            let (min_height, max_height) = grid.min_max_of_region(min1, max1);
            // If the grid size is smaller than the maximum leaf size, create a
            // leaf node.
            if grid_size.x <= Self::MAX_LEAF_CELL_DIM && grid_size.y <= Self::MAX_LEAF_CELL_DIM {
                let index = Self::build_leaf(nodes, min1, max1, min_height, max_height);
                nodes[parent as usize].set_child1(NodeIndex::some(index));
            } else {
                // Otherwise, create a branch node.
                let split_axis = if grid_size.x > grid_size.y {
                    Axis::Horizontal
                } else {
                    Axis::Vertical
                };
                let index = Self::build_branch(
                    nodes,
                    split_axis,
                    min1,
                    max1,
                    min_height,
                    max_height,
                    grid.bounds_of_region(min1, max1),
                );
                nodes[parent as usize].set_child1(NodeIndex::some(index));
                Self::build_recursive(grid, index, nodes);
            }
        }
    }

    fn build_leaf(
        nodes: &mut Vec<Node>,
        min: UVec2,
        max: UVec2,
        min_height: f32,
        max_height: f32,
    ) -> u32 {
        let node_index = nodes.len() as u32;
        nodes.push(Node::Leaf {
            min,
            max,
            min_height,
            max_height,
        });
        node_index
    }

    fn build_branch(
        nodes: &mut Vec<Node>,
        axis: Axis,
        min: UVec2,
        max: UVec2,
        min_height: f32,
        max_height: f32,
        bounds: Aabb,
    ) -> u32 {
        let node_index = nodes.len() as u32;
        nodes.push(Node::Branch {
            axis,
            min,
            max,
            min_height,
            max_height,
            child0: NodeIndex::none(),
            child1: NodeIndex::none(),
            bounds,
        });
        node_index
    }
}

/// Helper structure for grid ray tracing.
///
/// The grid is built on top of the micro-surface, and the grid cells are
/// represented by two triangles as defined in the `MicroSurfaceMesh`.
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
/// The grid coordinates lie in the range of [0, cols - 2] and [0, rows - 2].
/// Knowing the grid coordinates, the corresponding vertices of the
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
#[derive(Debug)]
pub struct Grid<'ms> {
    /// The heightfield where the grid is defined.
    surf: &'ms MicroSurface,

    /// Corresponding `TriangleMesh` of the surface.
    mesh: &'ms MicroSurfaceMesh,

    /// Width of the grid (max number of columns).
    width: u32,

    /// Height of the grid (max number of rows).
    height: u32,

    /// Cells of the grid stored in row-major order.
    cells: Vec<Cell>,
}

/// A grid cell.
#[derive(Debug, Copy, Clone)]
pub struct Cell {
    /// Horizontal grid coordinate of the cell.
    pub x: u32,
    /// Vertical grid coordinate of the cell.
    pub y: u32,
    /// Index of 4 vertices of the cell, stored in counter-clockwise order.
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

impl Default for Cell {
    fn default() -> Self {
        Cell {
            x: u32::MAX,
            y: u32::MAX,
            verts: [u32::MAX; 4],
            min_height: f32::MAX,
            max_height: f32::MAX,
        }
    }
}

impl<'ms> Grid<'ms> {
    /// Creates a new grid ray tracing object.
    pub fn new(surface: &'ms MicroSurface, mesh: &'ms MicroSurfaceMesh) -> Self {
        let surf_width = surface.cols;
        let grid_width = surface.cols - 1;
        let grid_height = surface.rows - 1;
        let mut cells = vec![Cell::default(); (surface.cols - 1) * (surface.rows - 1)];
        cells
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
                            let height = surface.samples[*vert as usize];
                            (min.min(height), max.max(height))
                        });
                    *cell = Cell {
                        x: x as u32,
                        y: y as u32,
                        verts,
                        min_height,
                        max_height,
                    };
                }
            });

        Grid {
            surf: surface,
            mesh,
            width: grid_width as u32,
            height: grid_height as u32,
            cells,
        }
    }

    /// Returns the minimum and maximum height of the grid cells in the given
    /// region defined by the minimum and maximum grid coordinates.
    pub fn min_max_of_region(&self, min: UVec2, max: UVec2) -> (f32, f32) {
        let mut min_height = f32::MAX;
        let mut max_height = f32::MIN;
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let cell = &self.cells[y as usize * self.width as usize + x as usize];
                min_height = min_height.min(cell.min_height);
                max_height = max_height.max(cell.max_height);
            }
        }
        (min_height, max_height)
    }

    /// Returns the bounding box of the grid cells in the given region defined
    /// by the minimum and maximum grid coordinates.
    pub fn bounds_of_region(&self, min: UVec2, max: UVec2) -> Aabb {
        let (min_height, max_height) = self.min_max_of_region(min, max);
        let min_vert = self.mesh.verts[self.cell_at(min.x, min.y).verts[0] as usize];
        let max_vert = self.mesh.verts[self.cell_at(max.x, max.y).verts[2] as usize];
        // TODO: handedness
        let min_horizontal = min_vert.x.min(max_vert.x);
        let max_horizontal = min_vert.x.max(max_vert.x);
        let min_vertical = min_vert.z.min(max_vert.z);
        let max_vertical = min_vert.z.max(max_vert.z);
        let min = Vec3::new(min_horizontal as f32, min_height, min_vertical);
        let max = Vec3::new(max_horizontal as f32, max_height, max_vertical);
        Aabb::new(min, max)
    }

    /// Returns the cell at the given grid coordinates.
    pub fn cell_at(&self, x: u32, y: u32) -> &Cell {
        debug_assert!(
            x < self.width && y < self.height,
            "Cell index out of bounds: ({}, {})",
            x,
            y
        );
        &self.cells[y as usize * self.width as usize + x as usize]
    }
}

struct GridRT<'ms> {
    grid: Grid<'ms>,
    accel: KdTree<'ms>,
}

impl<'ms> GridRT<'ms> {
    pub fn new(surface: &'ms MicroSurface, mesh: &'ms MicroSurfaceMesh) -> Self {
        let grid = Grid::new(surface, mesh);
        let accel = KdTree::new(&grid);
        GridRT { grid, accel }
    }

    pub fn traverse(&self, ray: &Ray) {}
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
//
// /// Modified version of Digital Differential Analyzer (DDA) Algorithm.
// ///
// /// Traverse the grid to identify all cells and corresponding intersection
// /// that are traversed by the ray.
// ///
// /// # Arguments
// ///
// /// * `ray_org` - The starting position (in world space) of the ray.
// /// * `ray_dir` - The direction of the ray in world space.
// pub fn traverse(&self, ray_org_world: Vec2, ray_dir: Vec2) -> GridTraversal {
//     log::debug!(
//         "    Grid origin: {:?}\n       min: {:?}\n       max: {:?}",
//         self.origin,
//         self.min,
//         self.max_coord
//     );
//     log::debug!(
//         "    Traverse the grid with the ray: o {:?}, d: {:?}",
//         ray_org_world,
//         ray_dir.normalize()
//     );
//     // Relocate the ray origin to the position relative to the origin of the
// grid.     let ray_org_grid = ray_org_world - self.origin;
//     let ray_org_cell = self.world_to_grid_2d(ray_org_world);
//     log::debug!("      - ray origin cell: {:?}", ray_org_cell);
//     log::debug!("      - ray origin grid: {:?}", ray_org_grid);
//
//     let is_parallel_to_x_axis = f32::abs(ray_dir.y - 0.0) < f32::EPSILON;
//     let is_parallel_to_y_axis = f32::abs(ray_dir.x - 0.0) < f32::EPSILON;
//
//     if is_parallel_to_x_axis && is_parallel_to_y_axis {
//         // The ray is parallel to both axes, which means it comes from the
// top or bottom         // of the grid.
//         log::debug!("      -> parallel to both axes");
//         return GridTraversal::FromTopOrBottom(ray_org_cell);
//     }
//
//     let ray_dir = ray_dir.normalize();
//     if is_parallel_to_x_axis && !is_parallel_to_y_axis {
//         log::debug!("      -> parallel to X axis");
//         let (dir, initial_dist, num_cells) = if ray_dir.x < 0.0 {
//             (
//                 -1,
//                 ray_org_grid.x - ray_org_cell.x as f32 * self.surf.du,
//                 ray_org_cell.x - self.min.x + 1,
//             )
//         } else {
//             (
//                 1,
//                 (ray_org_cell.x + 1) as f32 * self.surf.du - ray_org_grid.x,
//                 self.max_coord.x - ray_org_cell.x + 1,
//             )
//         };
//         // March along the ray direction until the ray hits the grid
// boundary.         let mut cells = vec![IVec2::ZERO; num_cells as usize];
//         let mut dists = vec![0.0; num_cells as usize + 1];
//         for i in 0..num_cells {
//             cells[i as usize] = ray_org_cell + IVec2::new(dir * i, 0);
//             dists[i as usize + 1] = initial_dist + (i as f32 * self.surf.du);
//         }
//         GridTraversal::Traversed { cells, dists }
//     } else if is_parallel_to_y_axis && !is_parallel_to_x_axis {
//         log::debug!("      -> parallel to Y axis");
//         let (dir, initial_dist, num_cells) = if ray_dir.y < 0.0 {
//             (
//                 -1,
//                 ray_org_grid.y - ray_org_cell.y as f32 * self.surf.dv,
//                 ray_org_cell.y - self.min.y + 1,
//             )
//         } else {
//             (
//                 1,
//                 (ray_org_cell.y + 1) as f32 * self.surf.dv - ray_org_grid.y,
//                 self.max_coord.y - ray_org_cell.y + 1,
//             )
//         };
//         let mut cells = vec![IVec2::ZERO; num_cells as usize];
//         let mut dists = vec![0.0; num_cells as usize + 1];
//         // March along the ray direction until the ray hits the grid
// boundary.         for i in 0..num_cells {
//             cells[i as usize] = ray_org_cell + IVec2::new(0, dir * i);
//             dists[i as usize + 1] = initial_dist + (i as f32 * self.surf.dv);
//         }
//         GridTraversal::Traversed { cells, dists }
//     } else {
//         // The ray is not parallel to either x or y axis.
//         let m = ray_dir.y / ray_dir.x; // The slope of the ray on the grid,
// dy/dx.         let m_recip = m.recip(); // The reciprocal of the slope of
// the ray on the grid, dy/dx.
//
//         log::debug!("    Slope: {:?}, Reciprocal: {:?}", m, m_recip);
//
//         // Calculate the distance along the direction of the ray when moving
//         // a unit distance (size of the cell) along the x-axis and y-axis.
//         let unit = Vec2::new(
//             (1.0 + m * m).sqrt() * self.surf.du,
//             (1.0 + m_recip * m_recip).sqrt() * self.surf.dv,
//         )
//         .abs();
//
//         let mut curr_cell = ray_org_cell;
//         log::debug!("  - starting cell: {:?}", curr_cell);
//         log::debug!("  - unit dist: {:?}", unit);
//
//         let step_dir = IVec2::new(
//             if ray_dir.x >= 0.0 { 1 } else { -1 },
//             if ray_dir.y >= 0.0 { 1 } else { -1 },
//         );
//
//         // Accumulated line length when moving along the x-axis and y-axis.
//         let mut accumulated = Vec2::new(
//             if ray_dir.x < 0.0 {
//                 //(ray_org_grid.x - curr_cell.x as f32 * self.surface.du) *
// unit.x                 (ray_org_grid.x / self.surf.du - curr_cell.x as
// f32) * unit.x             } else {
//                 //((curr_cell.x + 1) as f32 * self.surface.du -
// ray_org_grid.x) * unit.x                 ((curr_cell.x + 1) as f32 -
// ray_org_grid.x / self.surf.du) * unit.x             },
//             if ray_dir.y < 0.0 {
//                 // (ray_org_grid.y - curr_cell.y as f32 * self.surface.dv) *
// unit.y                 (ray_org_grid.y / self.surf.dv - curr_cell.y as
// f32) * unit.y             } else {
//                 // ((curr_cell.y + 1) as f32 * self.surface.dv -
// ray_org_grid.y) * unit.y                 ((curr_cell.y + 1) as f32 -
// ray_org_grid.y / self.surf.dv) * unit.y             },
//         );
//
//         log::debug!("  - accumulated: {:?}", accumulated);
//
//         let mut distances = vec![0.0];
//         let mut traversed = vec![curr_cell];
//
//         loop {
//             if accumulated.x <= accumulated.y {
//                 // avoid min - 1, max + 1
//                 if (step_dir.x > 0 && curr_cell.x == self.max_coord.x)
//                     || (step_dir.x < 0 && curr_cell.x == self.min.x)
//                 {
//                     break;
//                 }
//                 distances.push(accumulated.x);
//                 curr_cell.x += step_dir.x;
//                 accumulated.x += unit.x;
//             } else {
//                 if (step_dir.y > 0 && curr_cell.y == self.max_coord.y)
//                     || (step_dir.y < 0 && curr_cell.y == self.min.y)
//                 {
//                     break;
//                 }
//                 distances.push(accumulated.y);
//                 curr_cell.y += step_dir.y;
//                 accumulated.y += unit.y;
//             };
//
//             traversed.push(IVec2::new(curr_cell.x, curr_cell.y));
//
//             if (curr_cell.x < self.min.x && curr_cell.x > self.max_coord.x)
//                 && (curr_cell.y < self.min.y && curr_cell.y >
// self.max_coord.y)             {
//                 break;
//             }
//         }
//
//         // Push distance to the existing point of the last cell.
//         if accumulated.x <= accumulated.y {
//             distances.push(accumulated.x);
//         } else {
//             distances.push(accumulated.y);
//         }
//
//         GridTraversal::Traversed {
//             cells: traversed,
//             dists: distances,
//         }
//     }
//}
//
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

// #[test]
// fn test_grid_traversal() {
//     let heightfield = MicroSurface::new(6, 6, 1.0, 1.0, 2.0,
// AxisAlignment::XY);     let triangle_mesh = heightfield.triangulate();
//     let grid = GridRT::new(&heightfield, &triangle_mesh);
//     let result = grid.traverse(Vec2::new(-3.5, -3.5), Vec2::new(1.0, 1.0));
//     println!("{:?}", result);
// }
