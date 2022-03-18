struct Uniforms {
    view: mat4x4<f32>;
    proj: mat4x4<f32>;
    view_inv: mat4x4<f32>;
    proj_inv: mat4x4<f32>;
};

struct VOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] v_position: vec3<f32>;
    [[location(1)]] near: vec3<f32>;
    [[location(2)]] far: vec3<f32>;
};

[[group(0), binding(0)]] var<uniform> uniforms: Uniforms;

fn unproject(p: vec3<f32>) -> vec3<f32> {
    let unprojected: vec4<f32> = uniforms.view_inv * uniforms.proj_inv * vec4<f32>(p, 1.0);
    return unprojected.xyz / unprojected.w;
}

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] idx: u32) -> VOutput {
    var out: VOutput;

    var vertices = array<vec3<f32>, 6>(
        vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(-1.0, -1.0, 0.0), vec3<f32>(-1.0, 1.0, 0.0),
        vec3<f32>(-1.0, -1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, -1.0, 0.0)
    );

    let p = vertices[idx];

    out.v_position = p;
    out.clip_position = vec4<f32>(p, 1.0);

    out.near = unproject(p);
    out.far = unproject(vec3<f32>(p.xy, 1.0));

    return out;
}

fn checkerboard(r: vec2<f32>, scale: f32) -> f32 {
   return f32((i32(round(r.x * 5.0)) + i32(round(r.y * 5.0))) % 2);
}

/// Draw grid on XZ plane.
//fn grid(pixel: vec3<f32>, scale: f32) -> vec4<f32> {
//    let coord: vec2<f32> = pixel.xz * scale;
//    let d: vec2<f32> = fwdith(coord);
//    let grid: vec2<f32> = abs(fract(coord - 0.5) - 0.5) / d;
//    let line: f32 = min(grid.x, grid.y);
//    let min_x = min(d.x, 1);
//    let min_z = min(d.z, 1);
//    let color: vec4<f32> = vec4(0.2, 0.2, 0.2, 1.0 - min(line, 1.0));

    // z axis
//    if pixel.x > -0.1 * min_x && pixel.x < 0.1 * min_x {
//        color.z = 1.0;
//    }

    // x axis
//    if pixel.z > -0.1 * min_z && pixel.z < 0.1 * min_z {
//        color.x = 1.0;
//    }

//    return color;
//}

struct FOutput {
    [[location(0)]] color: vec4<f32>;
};

[[stage(fragment)]]
fn fs_main(in: VOutput) -> FOutput {
    var out: FOutput;

    // shoot the ray from near point to far point.
    let t = in.near.y / (in.near.y - in.far.y);

    // intersection point with the ground to get the pixel
    let pixel = in.near + t * (in.far - in.near);

    // generate the checker pattern
    let c = checkerboard(pixel.xz, 1.0) * 0.3 + checkerboard(pixel.xz, 10.0) * 0.2 + checkerboard(pixel.xz, 100.0) * 0.1 + 0.1;
    out.color = vec4<f32>(vec3<f32>(c, c, c), 0.1) * f32(t > 0.0);

    //out.color = grid(pixel, 10) * f32(t > 0.0);

    return out;
}