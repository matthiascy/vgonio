struct Uniforms {
    view_mat: mat4x4<f32>,
    proj_mat: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    grid_line_color: vec4<f32>,
}

var<private> GRID_PLANE: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(-1.0, -1.0, 0.0), vec3<f32>(-1.0, 1.0, 0.0),
    vec3<f32>(-1.0, -1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, -1.0, 0.0)
);

fn unproject(p: vec3<f32>) -> vec3<f32> {
    let unprojected = uniforms.view_inv * uniforms.proj_inv * vec4<f32>(p, 1.0);
    return unprojected.xyz / unprojected.w;
}

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) near_point: vec3<f32>,
    @location(1) far_point: vec3<f32>,
    @location(2) view_mat_c0: vec4<f32>,
    @location(3) view_mat_c1: vec4<f32>,
    @location(4) view_mat_c2: vec4<f32>,
    @location(5) view_mat_c3: vec4<f32>,
    @location(6) proj_mat_c0: vec4<f32>,
    @location(7) proj_mat_c1: vec4<f32>,
    @location(8) proj_mat_c2: vec4<f32>,
    @location(9) proj_mat_c3: vec4<f32>,
    @location(10) grid_line_color: vec4<f32>,
}

//vec4 checker_board(vec2 p, float scale) {
//    float c = float((int(round(p.x * 5.0)) + int(round(p.y * 5.0))) % 2);
//    return vec4(vec3(c), 1.0);
//}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VOut {
    var vout: VOut;

    let p = GRID_PLANE[vertex_index];

    vout.near_point = unproject(p);
    vout.far_point = unproject(vec3<f32>(p.xy, 1.0));

    vout.view_mat_c0 = uniforms.view_mat[0];
    vout.view_mat_c1 = uniforms.view_mat[1];
    vout.view_mat_c2 = uniforms.view_mat[2];
    vout.view_mat_c3 = uniforms.view_mat[3];

    vout.proj_mat_c0 = uniforms.proj_mat[0];
    vout.proj_mat_c1 = uniforms.proj_mat[1];
    vout.proj_mat_c2 = uniforms.proj_mat[2];
    vout.proj_mat_c3 = uniforms.proj_mat[3];

    vout.grid_line_color = uniforms.grid_line_color;
    vout.position = vec4<f32>(p, 1.0);

    return vout;
}

const CAMERA_NEAR: f32 = 0.1;
const CAMERA_FAR: f32 = 100.0;

fn grid(frag_pos: vec3<f32>, scale: f32, show_axis: bool, is_dark_mode: bool, fading: f32, grid_line_color: vec4<f32>) -> vec4<f32> {
    let coord: vec2<f32> = frag_pos.xy * scale; // use the scale variable to set the distance between the lines
    let derivative: vec2<f32> = fwidth(coord);

    // Compute anti-aliased world-space grid lines
    let grid: vec2<f32> = abs(fract(coord - 0.5) - 0.5) / derivative;
    let line: f32 = min(grid.x, grid.y);

    var color: vec4<f32> = vec4<f32>(grid_line_color.xyz, 1.0 - min(line, 1.0));

    let derivative_scaled: vec2<f32> = derivative * 10.0;

    if (show_axis) {
        let min_axis_x: f32 = min(derivative_scaled.x, 1.0);
        let min_axis_y: f32 = min(derivative_scaled.y, 1.0);
        // y axis
        if (frag_pos.x > -0.1 * min_axis_x && frag_pos.x < 0.1 * min_axis_x) {
            color = vec4<f32>(0.0, 1.0, 0.28, 1.0);
        }

        // x axis
        if (frag_pos.y > -0.1 * min_axis_y && frag_pos.y < 0.1 * min_axis_y) {
            color = vec4<f32>(1.0, 0.0, 0.28, 1.0);
        }
    }

    if (is_dark_mode) {
        return color * fading * 0.6;
    } else {
        return color * fading * 0.8;
    }
}

fn compute_linear_depth(clip_space_pos: vec4<f32>) -> f32 {
    // remap to [-1.0, 1.0]
    let clip_space_depth: f32 = (clip_space_pos.z / clip_space_pos.w) * 2.0 - 1.0;
    let linear_depth: f32 = (2.0 * CAMERA_NEAR * CAMERA_FAR) / (CAMERA_NEAR + CAMERA_FAR - clip_space_depth * (CAMERA_FAR - CAMERA_NEAR));
    return linear_depth / CAMERA_FAR;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(vin: VOut) -> FragOutput {
    var frag_output: FragOutput;

    // shoot the ray from near point to far point.
    // the ray is defined as: ray = near_point + t * (far_point - near_point)
    // where t is the distance from near point to the intersection point with the ground
    // NOTE: the unprojected points are in the world space (right-handed Z-up coordinate system)
    let t: f32 = vin.near_point.z / (vin.near_point.z - vin.far_point.z);

    // intersection point with the ground to get the pixel
    let frag_pos: vec3<f32> = vin.near_point + t * (vin.far_point - vin.near_point);

    let view_mat = mat4x4<f32>(
        vin.view_mat_c0,
        vin.view_mat_c1,
        vin.view_mat_c2,
        vin.view_mat_c3
    );

    let proj_mat = mat4x4<f32>(
        vin.proj_mat_c0,
        vin.proj_mat_c1,
        vin.proj_mat_c2,
        vin.proj_mat_c3
    );

    let clip_pos = proj_mat * view_mat * vec4<f32>(frag_pos.xyz, 1.0);
    let grid_line_color = vin.grid_line_color;

    let linear_depth = compute_linear_depth(clip_pos);
    let fading = 1.0 - smoothstep(0.0, abs(0.5 - linear_depth), linear_depth);
    // let fading = max(0.0, (0.5 - linear_depth));

    var factor = 0.0;
    if (t > 0.0) {
        factor = 1.0;
    }

    frag_output.color = grid(frag_pos, 1.0, true, grid_line_color.w == 1.0, fading, grid_line_color) * factor;
    frag_output.depth = clip_pos.z / clip_pos.w;

    return frag_output;
}