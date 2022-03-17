#version 450

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 near;
layout(location = 2) in vec3 far;

layout(location = 0) out vec4 color;

vec4 checker_board(vec2 p, float scale) {
    float c = float((int(round(p.x * 5.0)) + int(round(p.y * 5.0))) % 2);
    return vec4(vec3(c), 1.0);
}

void main() {
    // shoot the ray from near point to far point.
    float t = near.y / (near.y - far.y);

    // intersection point with the ground to get the pixel
    vec3 frag_pos = near + t * (far - near);

    color = checker_board(frag_pos.xz, 1.0) * float(t > 0.0);
}