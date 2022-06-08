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
    var vout: VOut;
    let scale = uniforms.info.w;
    let scaled = scale * position;
    vout.position = uniforms.proj * uniforms.view * uniforms.model * vec4<f32>(scaled, 1.0);

    let lowest = uniforms.info.x * scale;
    let span = uniforms.info.z * scale;

    let c: f32 = (scaled.y - lowest) / span;
    vout.color = vec3<f32>(c, 1.0 - c, 0.025);
    return vout;
}

[[stage(fragment)]]
fn fs_main(vin: VOut) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(vin.color, 0.7);
}