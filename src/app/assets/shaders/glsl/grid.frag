#version 450

const float camera_near = 0.1;
const float camera_far = 100.0;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 near;
layout(location = 2) in vec3 far;
layout(location = 3) in mat4 view;
layout(location = 7) in mat4 proj;

layout(location = 0) out vec4 color;

vec4 checker_board(vec2 p, float scale) {
    float c = float((int(round(p.x * 5.0)) + int(round(p.y * 5.0))) % 2);
    return vec4(vec3(c), 1.0);
}

vec4 grid(vec3 frag_pos, float scale) {
    vec2 coord = frag_pos.xz * scale; // use the scale variable to set the distance between the lines

    // Compute anti-aliased world-space grid lines
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    float line = min(grid.x, grid.y);
    float min_x = min(derivative.x, 1.0);
    float min_z = min(derivative.y, 1.0);

    vec4 color = vec4(0.4, 0.4, 0.4, 1.0 - min(line, 1.0));

    // z axis
    if (frag_pos.x > -0.2 * min_x && frag_pos.x < 0.2 * min_x) {
        color.z = 0.8;
    }

    // x axis
    if(frag_pos.z > -0.2 * min_z && frag_pos.z < 0.2 * min_z) {
        color.x = 0.8;
    }

    return color;
}

float compute_depth(vec4 clip_space_pos) {
    return clip_space_pos.z / clip_space_pos.w;
}

float compute_linear_depth(vec4 clip_space_pos) {
    // remap to [-1.0, 1.0]
    float clip_space_depth = (clip_space_pos.z / clip_space_pos.w) * 2.0 - 1.0;
    float linear_depth = (2.0 * camera_near * camera_far) / (camera_near + camera_far - clip_space_depth * (camera_far - camera_near));
    return linear_depth / camera_far;
}

void main() {
    // shoot the ray from near point to far point.
    float t = near.y / (near.y - far.y);

    // intersection point with the ground to get the pixel
    vec3 frag_pos = near + t * (far - near);
    vec4 clip_space_pos = proj * view * vec4(frag_pos, 1.0);
    float linear_depth = compute_linear_depth(clip_space_pos);
    float fading = max(0.0, (0.2 - linear_depth));

//    color = checker_board(frag_pos.xz, 1.0) * float(t > 0.0);
    color = grid(frag_pos, 10.0) * float(t > 0.0);
    color.a *= fading;

    gl_FragDepth = compute_depth(clip_space_pos);
}