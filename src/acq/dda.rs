use glam::{IVec2, UVec2, Vec2};

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

    // Truncate the floating point values to integers.
    let mut curr_cell = ray_start.as_ivec2();

    // Accumulated line length when moving along the x-axis and y-axis.
    let mut line_length = Vec2::ZERO;

    // Determine the direction that we are going to walk along the line.
    let step_dir = IVec2::new(
        if ray_dir.x < 0.0 { -1 } else { 1 },
        if ray_dir.y < 0.0 { -1 } else { 1 },
    );

    // Initialise the accumulated line length.
    line_length.x = if ray_dir.x < 0.0 {
        (ray_start.x - curr_cell.x as f32) * unit_step_size.x
    } else {
        ((curr_cell.x + 1) as f32 - ray_start.x) * unit_step_size.x
    };

    line_length.y = if ray_dir.y < 0.0 {
        (ray_start.y - curr_cell.y as f32) * unit_step_size.y
    } else {
        ((curr_cell.y + 1) as f32 - ray_start.y) * unit_step_size.y
    };

    let mut intersections = if line_length.x > line_length.y { vec![ray_start + ray_dir * line_length.y] } else { vec![ray_start + ray_dir * line_length.x] };
    let mut grid_cells = vec![curr_cell];

    while (curr_cell.x >= min_cell_x && curr_cell.x < max_cell_x) && (curr_cell.y >= min_cell_y && curr_cell.y < max_cell_y) {
        let length = if line_length.x < line_length.y {
            curr_cell.x += step_dir.x;
            line_length.x += unit_step_size.x;
            line_length.x
        } else {
            curr_cell.y += step_dir.y;
            line_length.y += unit_step_size.y;
            line_length.y
        };

        intersections.push(ray_start + ray_dir * length);
        grid_cells.push(IVec2::new(curr_cell.x, curr_cell.y));
    }

    (grid_cells, intersections)
}

#[test]
fn test_dda() {
    let (cells, intersections) = dda(
        Vec2::new(0.25, 0.5),
        Vec2::new(1.0, 2.0),
        0,
        0,
        4,
        4,
    );

    println!("cells: {:?}", cells);
    println!("intersections: {:?}", intersections);
}
