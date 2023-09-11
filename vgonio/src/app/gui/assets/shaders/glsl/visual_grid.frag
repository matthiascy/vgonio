#version 450

const float camera_near = 0.1;
const float camera_far = 100.0;

layout(location = 0) in vec3 near_point;
layout(location = 1) in vec3 far_point;
layout(location = 2) in vec4 view_mat_c0;
layout(location = 3) in vec4 view_mat_c1;
layout(location = 4) in vec4 view_mat_c2;
layout(location = 5) in vec4 view_mat_c3;
layout(location = 6) in vec4 proj_mat_c0;
layout(location = 7) in vec4 proj_mat_c1;
layout(location = 8) in vec4 proj_mat_c2;
layout(location = 9) in vec4 proj_mat_c3;
layout(location = 10) in vec4 grid_line_color;

layout(location = 0) out vec4 color;

//vec4 checker_board(vec2 p, float scale) {
//    float c = float((int(round(p.x * 5.0)) + int(round(p.y * 5.0))) % 2);
//    return vec4(vec3(c), 1.0);
//}

vec4 grid(vec3 frag_pos, float scale, bool show_axis, bool is_dark_mode, float fading) {
    vec2 coord = frag_pos.xy * scale; // use the scale variable to set the distance between the lines
    vec2 derivative = fwidth(coord);
    // Compute anti-aliased world-space grid lines
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    float line = min(grid.x, grid.y);

    vec4 color = vec4(grid_line_color.xyz, 1.0 - min(line, 1.0));

    vec2 derivative_scaled = derivative * 10.0;

    if (show_axis) {
        float min_axis_x = min(derivative_scaled.x, 1.0);
        float min_axis_y = min(derivative_scaled.y, 1.0);
        // y axis
        if (frag_pos.x > -0.1 * min_axis_x && frag_pos.x < 0.1 * min_axis_x) {
            color = vec4(0.0, 1.0, 0.28, 1.0);
        }

        // x axis
        if (frag_pos.y > -0.1 * min_axis_y && frag_pos.y < 0.1 * min_axis_y) {
            color = vec4(1.0, 0.0, 0.28, 1.0);
        }
    }

    if (is_dark_mode) {
        return color * fading;
    } else {
        return color * fading * 1.2;
    }
}

float compute_linear_depth(vec4 clip_space_pos) {
    // remap to [-1.0, 1.0]
    float clip_space_depth = (clip_space_pos.z / clip_space_pos.w) * 2.0 - 1.0;
    float linear_depth = (2.0 * camera_near * camera_far) / (camera_near + camera_far - clip_space_depth * (camera_far - camera_near));
    return linear_depth / camera_far;
}

void main() {
    // shoot the ray from near point to far point.
    // the ray is defined as: ray = near_point + t * (far_point - near_point)
    // where t is the distance from near point to the intersection point with the ground
    // NOTE: the unprojected points are in the world space (right-handed Z-up coordinate system)
    float t = near_point.z / (near_point.z - far_point.z);

    // intersection point with the ground to get the pixel
    vec3 frag_pos = near_point + t * (far_point - near_point);

    mat4 view_mat = mat4(view_mat_c0, view_mat_c1, view_mat_c2, view_mat_c3);
    mat4 proj_mat = mat4(proj_mat_c0, proj_mat_c1, proj_mat_c2, proj_mat_c3);
    vec4 clip_space_pos = proj_mat * view_mat * vec4(frag_pos.xyz, 1.0);

    gl_FragDepth = clip_space_pos.z / clip_space_pos.w;

    float linear_depth = compute_linear_depth(clip_space_pos);
    float fading = max(0.0, (0.5 - linear_depth));

    color = grid(frag_pos, 1.0, true, grid_line_color.w == 1.0, fading) * float(t > 0.0);
}