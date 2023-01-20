@group(0) @binding(0)
var<uniform> proj_view_matrix : mat4x4<f32>;

@vertex
fn vs_main_common(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return proj_view_matrix * vec4<f32>(position, 1.0);
}

@group(0) @binding(1)
var depth_texture : texture_depth_2d;
@group(0) @binding(2)
var depth_sampler : sampler_comparison;

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    //let depth = textureSample(depth_texture, depth_sampler, gl_FragCoord.xy);
    // TODO: discard if depth is too far away
    return vec4<f32>(1.0 / 1024.0, 0.0, 0.0, 0.0);
}
