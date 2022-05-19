use glam::{IVec2, Vec2};

/// Digital Differential Analyzer (DDA) Algorithm
pub fn dda(ray_start: Vec2, ray_dir: Vec2, min_cell_x: i32, min_cell_y: i32, max_cell_x: i32, max_cell_y: i32) -> (Vec<IVec2>, Vec<Vec2>) {
    let ray_dir = ray_dir.normalize();
    // Calculate dy/dx, slope the line
    let m = ray_dir.y / ray_dir.x;
    let m_recip = 1.0 / m;

    // Calculate the unit step size along the direction of the line when moving
    // a unit distance along the x-axis and y-axis.
    let unit_step_size = Vec2::new(
        (1.0 + m * m).sqrt(),
        (1.0 + m_recip * m_recip).sqrt(),
    );

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

    let mut isects = if walk_dist.x > walk_dist.y { vec![ray_start + ray_dir * walk_dist.y] } else { vec![ray_start + ray_dir * walk_dist.x] };
    let mut visited = vec![curr];

    while (curr.x >= min_cell_x && curr.x <= max_cell_x) && (curr.y >= min_cell_y && curr.y <= max_cell_y) {
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
    let (cells, intersections) = dda(
        Vec2::new(0.0, 3.0),
        Vec2::new(10.0, -2.0),
        0,
        0,
        10,
        10,
    );

    println!("cells: {:?}", cells);
    println!("intersections: {:?}", intersections);
}
