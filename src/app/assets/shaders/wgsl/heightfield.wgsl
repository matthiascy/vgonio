struct Uniforms {
    model: mat4x4<f32>;
    view: mat4x4<f32>;
    proj: mat4x4<f32>;
    // min, max, max - min, scale,
    info: vec4<f32>;
};

struct VOut {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] color: vec3<f32>;
};

[[group(0), binding(0)]] var<uniform> uniforms: Uniforms;

[[stage(vertex)]]
fn vs_main([[location(0)]] position: vec3<f32>) ->  VOut {
    var out: VOut;
    let scaled = position * uniforms.info.w;
    out.position = uniforms.proj * uniforms.view * uniforms.model * vec4<f32>(scaled, 1.0);
    let c: f32 = scaled.y / (uniforms.info.z * uniforms.info.w) + 0.5;
    out.color = vec3<f32>(c, 1.0 - c, 0.025);
    return out;
}

[[stage(fragment)]]
fn fs_main(in: VOut) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}