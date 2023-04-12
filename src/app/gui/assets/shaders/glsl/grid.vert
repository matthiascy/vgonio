#version 450

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 view_mat;
    mat4 proj_mat;
    mat4 view_inv;
    mat4 proj_inv;
    vec4 grid_line_color;
} uniforms;

const vec3 grid_plane[6] = vec3[](
    vec3(1.0, 1.0, 0.0), vec3(-1.0, -1.0, 0.0), vec3(-1.0, 1.0, 0.0),
    vec3(-1.0, -1.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(1.0, -1.0, 0.0)
);

layout(location = 0) out vec3 near_point;
layout(location = 1) out vec3 far_point;
layout(location = 2) out vec4 view_mat_c0;
layout(location = 3) out vec4 view_mat_c1;
layout(location = 4) out vec4 view_mat_c2;
layout(location = 5) out vec4 view_mat_c3;
layout(location = 6) out vec4 proj_mat_c0;
layout(location = 7) out vec4 proj_mat_c1;
layout(location = 8) out vec4 proj_mat_c2;
layout(location = 9) out vec4 proj_mat_c3;
layout(location = 10) out vec4 grid_line_color;

vec3 unproject(vec3 p) {
    vec4 unprojected = uniforms.view_inv * uniforms.proj_inv * vec4(p, 1.0);
    return unprojected.xyz / unprojected.w;
}

void main() {
    vec3 p = grid_plane[gl_VertexIndex];
    near_point = unproject(p);
    far_point = unproject(vec3(p.xy, 1.0));

    view_mat_c0 = uniforms.view_mat[0];
    view_mat_c1 = uniforms.view_mat[1];
    view_mat_c2 = uniforms.view_mat[2];
    view_mat_c3 = uniforms.view_mat[3];

    proj_mat_c0 = uniforms.proj_mat[0];
    proj_mat_c1 = uniforms.proj_mat[1];
    proj_mat_c2 = uniforms.proj_mat[2];
    proj_mat_c3 = uniforms.proj_mat[3];

    grid_line_color = uniforms.grid_line_color;

    gl_Position = vec4(p, 1.0);
}