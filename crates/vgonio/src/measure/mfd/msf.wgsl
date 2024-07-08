struct Uniforms {
    proj_view_matrix: mat4x4<f32>,
    meas_point_index: vec2<u32>,
    meas_point_per_depth_map: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_depth_pass(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.proj_view_matrix * vec4<f32>(position, 1.0);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

// WGPU's NDC coordinates are y-up, x-right but the texture is y-down,
// x-right so we need to flip the y coordinate if we want to sample
// correctly the texture generated by the depth pass.
const tex_coord_flip: vec2<f32> = vec2<f32>(0.5, -0.5);

@vertex
fn vs_render_pass(@location(0) position: vec3<f32>) -> VertexOutput {
    var output: VertexOutput;
    let clip_pos = uniforms.proj_view_matrix * vec4<f32>(position, 1.0);
    output.position = clip_pos;
    output.ndc = clip_pos / clip_pos.w;
    output.uv = output.ndc.xy * tex_coord_flip + vec2<f32>(0.5, 0.5);
    return output;
}

@group(1) @binding(0)
var depth_map : texture_depth_2d_array;
@group(1) @binding(1)
var depth_sampler : sampler_comparison;

struct FragOutput {
    @location(0) visible_area: vec4<f32>,
    @location(1) total_area: vec4<f32>,
}

@fragment
fn fs_render_pass(vert: VertexOutput) -> FragOutput {
    var output: FragOutput;

    // Texture coordinates calculated in vertex shader.
    let uv = vert.uv;

    let index = i32(uniforms.meas_point_index.x) % i32(uniforms.meas_point_per_depth_map.x);

    // Test if the fragment is in front of the depth texture.
    let depth_cmp = textureSampleCompare(depth_map, depth_sampler, vert.uv, index, vert.ndc.z - 0.000001);

    if (depth_cmp > 0.0) {
        // RGB10A2_UNORM
        output.visible_area = vec4<f32>(1.0 / 1024.0, 0.0, 0.0, 1.0);

        // RGBA8_UNORM
        // output.visible_area = vec4<f32>(1.0 / 256.0, 0.0, 0.0, 1.0);
    } else {
        output.visible_area = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // RGB10A2_UNORM
    output.total_area = vec4<f32>(1.0 / 1024.0, 0.0, 0.0, 1.0);

    // RGBA8_UNORM
    // output.total_area = vec4<f32>(1.0 / 256.0, 0.0, 0.0, 1.0);

    return output;
}
