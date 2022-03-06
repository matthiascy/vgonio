#version 450

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_tex_coord;

layout(location = 0) out vec2 v_tex_coord;

void main() {
    gl_Position = vec4(a_position, 1.0);
    v_tex_coord = a_tex_coord;
}