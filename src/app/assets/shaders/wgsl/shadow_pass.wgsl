struct Uniforms {
    model: mat4x4<f32>;
    light_space_matrix: mat4x4<f32>;
};

[[group(0), binding(0)]] var<uniform> uniforms: Uniforms;

// Bake depth map
[[stage(vertex)]]
fn vs_main([[location(0)]] pos: vec3<f32>) -> [[builtin(position)]] vec4<f32> {
    return uniforms.light_space_matrix * uniforms.model * vec4<f32>(pos, 1.0);
}

// Calculating projected area while ignoring masking and shadowing.
//[[stage(fragment)]]
//fn fs_shadowing_masking([[builtin(position)]] clip_position: vec4<f32>) -> [[location(0)]] vec4<f32> {
//    return vec4<f32>(0.00048828125, 0.00048828125, 0.00048828125, 1.0);
//}