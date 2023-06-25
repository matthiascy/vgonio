struct VertexOutput {
    @location(0) texcoord: vec2<f32>,
    @location(1) color: vec4<f32>, // gama 0-1
    @builtin(position) position: vec4<f32>,
}

struct Locals {
    screen_size: vec2<f32>,
    // Uniform buffers need to be at least 16 bytes in WebGL.
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> locals: Locals;

// 0 - 1 linear from 0 - 1 sRGB gamma
fn linear_from_gamma_rgb(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let lower = srgb / vec3<f32>(12.92);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

// 0 - 1 sRGB gamma from 0 - 1 linear
fn gamma_from_linear_rgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let lower = linear * vec3<f32>(12.92);
    let higher = vec3<f32>(1.055) * pow(linear, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(higher, lower, cutoff);
}

// 0 - 1 sRGBA gamma from 0 - 1 linear
fn gamma_from_linear_rgba(linear: vec4<f32>) -> vec4<f32> {
    let rgb = gamma_from_linear_rgb(linear.rgb);
    return vec4<f32>(rgb, linear.a);
}

// [u8; 4] srgb as u32 -> [r, g, b, a] in 0 - 1 linear
fn unpack_color(color: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(color & 255u),
        f32((color >> 8u) & 255u),
        f32((color >> 16u) & 255u),
        f32((color >> 24u) & 255u)
    ) / 255.0;
}

fn position_from_screen(screen_pos: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(
        screen_pos.x * 2.0 / locals.screen_size.x - 1.0,
        screen_pos.y * -2.0 / locals.screen_size.y + 1.0,
        0.0,
        1.0,
    );
}

@vertex
fn vs_main(
    @location(0) a_pos: vec2<f32>,
    @location(1) a_texcoord: vec2<f32>,
    @location(2) a_color: u32,
) -> VertexOutput {
    var output: VertexOutput;
    output.position = position_from_screen(a_pos);
    output.texcoord = a_texcoord;
    output.color = unpack_color(a_color);
    return output;
}

struct FOutput {
    @location(0) color0: vec4<f32>,
    @location(1) color1: vec4<f32>,
}

@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var spl: sampler;

@fragment
fn fs_main_linear_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    // Always have an sRGB aware texture at the moment.
    let color_linear = textureSample(tex, spl, in.texcoord);
    let color_gamma = gamma_from_linear_rgba(color_linear);
    let output_color_gamma = color_gamma * in.color;

    return vec4<f32>(linear_from_gamma_rgb(output_color_gamma.rgb), output_color_gamma.a);
}

@fragment
fn fs_main_gamma_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    // Always have an sRGB aware texture at the moment.
    let color_linear = textureSample(tex, spl, in.texcoord);
    let color_gamma = gamma_from_linear_rgba(color_linear);
    let output_color_gamma = color_gamma * in.color;

    return output_color_gamma;
}
