struct Globals {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

struct Locals {
    model: mat4x4<f32>,
    lowest: f32,
    highest: f32,
    span: f32,
    scale: f32,
}

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> locals: Locals;

@vertex
fn vs_main(@location(0) position: vec3<f32>) ->  VOut {
    var vout: VOut;
    let scaled = locals.scale * position;
    vout.position = globals.proj * globals.view * locals.model * vec4<f32>(scaled, 1.0);

    let lowest = locals.lowest * locals.scale;
    let span = locals.span * locals.scale;

    let c: f32 = (scaled.z - lowest) / span;
    vout.color = vec3<f32>(c * 0.75, (1.0 - c) * 0.8, 0.125);
    return vout;
}

@fragment
fn fs_main(vin: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(vin.color, 0.7);
}