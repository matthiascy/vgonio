#version 450

layout(set = 0, binding = 0) Uniforms {
    mat4 proj_view;
    float lowest;
    float highest;
    float span;
    float scale;
} uniforms;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} constants;