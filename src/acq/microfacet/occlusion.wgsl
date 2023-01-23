@group(0) @binding(0)
var<uniform> proj_view_matrix : mat4x4<f32>;

@vertex
fn vs_depth_pass(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return proj_view_matrix * vec4<f32>(position, 1.0);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_render_pass(@location(0) position: vec3<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.clip_pos = proj_view_matrix * vec4<f32>(position, 1.0);
    let ndc_pos = output.clip_pos;
    output.uv = ndc_pos.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return output;
}

@group(0) @binding(1)
var depth_texture : texture_depth_2d;
@group(0) @binding(2)
var depth_sampler : sampler_comparison;

@fragment
fn fs_render_pass(vert: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = vert.clip_pos / vert.clip_pos.w;
    //let depth = textureSampleCompare(depth_texture, depth_sampler, vert.uv, ndc.z);
    let frag_uv = vert.clip_pos.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (frag_uv.y < 0.0) {
        return vec4<f32>(0.0, 0.0, 1.0, 1.0);
    } else {
        return vec4<f32>(frag_uv.x, frag_uv.y, 0.0, 1.0);
    }
    //return vec4<f32>(vert.uv.x, vert.uv.y, 0.0, 1.0);
    //return vec4<f32>(0.0, 0.0, depth, 1.0);
    //return vec4<f32>(1.0 / 1024.0, 0.0, 0.0, 0.0);
    //return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
