struct Uniforms {
    light_space_matrix: mat4x4<f32>;
    model_matrix: mat4x4<f32>;
};

[[group(0), binding(0)]] var<uniform> uniforms: Uniforms;

// Bake depth map
[[stage(vertex)]]
fn vs_main([[location(0)]] position: vec3<f32>) -> [[builtin(position)]] vec4<f32> {
    return uniforms.light_space_matrix * uniforms.model_matrix * vec4<f32>(position, 1.0);
}