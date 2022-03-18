#version 450

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 view;
    mat4 proj;
    mat4 view_inv;
    mat4 proj_inv;
} uniforms;

const vec3 grid_plane[6] = vec3[](
    vec3(1.0, 1.0, 0.0), vec3(-1.0, -1.0, 0.0), vec3(-1.0, 1.0, 0.0),
    vec3(-1.0, -1.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(1.0, -1.0, 0.0)
);

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec3 near;
layout(location = 2) out vec3 far;
layout(location = 3) out mat4 view;
layout(location = 7) out mat4 proj;

vec3 unproject(vec3 p) {
    vec4 unprojected = uniforms.view_inv * uniforms.proj_inv * vec4(p, 1.0);
    return unprojected.xyz / unprojected.w;
}

void main() {
    vec3 p = grid_plane[gl_VertexIndex];
    view = uniforms.view;
    proj = uniforms.proj;
    v_position = p;
    near = unproject(p);
    far = unproject(vec3(p.xy, 1.0));
    gl_Position = vec4(p, 1.0);
}