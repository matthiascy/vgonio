const WGSL_PRESENT: &str = r#"
var coords: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0)
);

@vertex
fn vertex_main([[builtin(vertex_index)]] index: u32) -> [[builtin(position)]] vec4<f32> {
    return vec4<f32>(coords[index], 0.0, 1.0);
}

@group(0) @binding(0)
var tex: texture_2d<f32>;

@fragment
fn fragment_main([[builtin(position)]] pos: vec4<f32>) -> [[location(0)]] vec4<f32> {
    return textureLoad(tex, vec2<i32>(pos.xy), 0);
}
"#;

pub struct Display {}
