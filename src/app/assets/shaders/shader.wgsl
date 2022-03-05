// Vertex shader

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    // [[location(1)]] color: vec3<f32>;
    [[location(1)]] tex_coord: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    // [[location(0)]] color: vec3<f32>;
    [[location(0)]] tex_coord: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(vertex.position, 1.0);
    // out.color = vertex.color;
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