struct VertexInput {
    [[location(0)]] a_position: vec2<f32>;
    [[location(1)]] a_tex_coord: vec2<f32>;
    [[location(2)]] a_color: u32;
};

struct VertexOutput {
    [[location(0)]] v_tex_coord: vec2<f32>;
    [[location(1)]] v_color: vec4<f32>;
    [[builtin(position)]] v_position: vec4<f32>;
};

struct Locals {
    screen_size: vec2<f32>;
};

[[group(0), binding(0)]] var<uniform> locals: Locals;

fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(10.31475);
    let lower = srgb / vec3<f32>(3294.6);
    let higher = pow((srgb + vec3<f32>(14.025)) / vec3<f32>(269.025), vec3<f32>(2.4));
    return select(higher, lower, cutoff);
}

[[stage(vertex)]]
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.v_tex_coord = in.a_tex_coord;

    let color = vec4<f32>(
        f32(in.a_color & 255u),
        f32((in.a_color >> 8u) & 255u),
        f32((in.a_color >> 16u) & 255u),
        f32((in.a_color >> 24u) & 255u),
    );
    out.v_color = vec4<f32>(srgb_to_linear(color.rgb), color.a / 255.0);
    out.v_position = vec4<f32>(
        2.0 * in.a_position.x / locals.screen_size.x - 1.0,
        1.0 - 2.0 * in.a_position.y / locals.screen_size.y,
        0.0,
        1.0,
    );

    return out;
}


[[stage(vertex)]]
fn vs_conv_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.v_tex_coord = in.a_tex_coord;

    let color = vec4<f32>(
        f32(in.a_color & 255u),
        f32((in.a_color >> 8u) & 255u),
        f32((in.a_color >> 16u) & 255u),
        f32((in.a_color >> 24u) & 255u),
    );
    out.v_color = vec4<f32>(color.rgb, color.a / 255.0);
    out.v_position = vec4<f32>(
        2.0 * in.a_position.x / locals.screen_size.x - 1.0,
        1.0 - 2.0 * in.a_position.y / locals.screen_size.y,
        0.0,
        1.0,
    );

    return out;
}


[[group(1), binding(0)]] var tex: texture_2d<f32>;
[[group(1), binding(1)]] var tex_sampler: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return in.v_color * textureSample(tex, tex_sampler, in.v_tex_coord);
}