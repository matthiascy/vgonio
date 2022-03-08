// Vertex shader
struct CameraUniform {
    view_matrix: mat4x4<f32>;
    proj_matrix: mat4x4<f32>;
    model_matrix: mat4x4<f32>;
};

[[group(1), binding(0)]] var<uniform> camera: CameraUniform;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] tex_coord: vec2<f32>;
};

struct InstacingInput {
    [[location(5)]] col_0: vec4<f32>;
    [[location(6)]] col_1: vec4<f32>;
    [[location(7)]] col_2: vec4<f32>;
    [[location(8)]] col_3: vec4<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coord: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(vertex: VertexInput, instance: InstacingInput) -> VertexOutput {
    let model_matrix = mat4x4<f32>(instance.col_0, instance.col_1, instance.col_2, instance.col_3);
    var out: VertexOutput;
    out.clip_position = camera.proj_matrix * camera.view_matrix * camera.model_matrix * model_matrix * vec4<f32>(vertex.position, 1.0);
    out.tex_coord = vertex.tex_coord;
    return out;
}

[[group(0), binding(0)]] var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]] var s_diffuse: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    // return vec4<f32>(in.color, 1.0);
    return textureSample(t_diffuse, s_diffuse, in.tex_coord);
}