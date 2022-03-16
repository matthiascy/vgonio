struct CameraUniform {
    view: mat4x4<f32>;
    proj: mat4x4<f32>;
    view_inv: mat4x4<f32>;
    proj_inv: mat4x4<f32>;
    model: mat4x4<f32>;
};

struct VOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] v_position: vec3<f32>;
    [[location(1)]] near: vec3<f32>;
    [[location(2)]] far: vec3<f32>;
};

[[group(0), binding(0)]] var<uniform> camera: CameraUniform;

fn unproject(p: vec3<f32>, view: mat4x4<f32>, proj: mat4x4<f32>) -> vec3<f32> {
    let unprojected: vec4<f32> = camera.view_inv * camera.proj_inv * vec4<f32>(p, 1.0);
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

    out.near = unproject(p, camera.view, camera.proj);
    out.far = unproject(vec3<f32>(p.xy, 1.0), camera.view, camera.proj);

    return out;
}

fn checkerboard(r: vec2<f32>, scale: f32) -> f32 {
   return f32((i32(round(r.x * 5.0)) + i32(round(r.y * 5.0))) % 2);
}

struct FOutput {
    [[location(0)]] color: vec4<f32>;
};

[[stage(fragment)]]
fn fs_main(in: VOutput) -> FOutput {
    // shoot the ray from near point to far point.
    let t = in.near.y / (in.near.y - in.far.y);

    // intersection point with the ground
    let r = in.near + t * (in.far - in.near);

    // generate the checker pattern
    let c = checkerboard(r.xz, 1.0) * 0.3 + checkerboard(r.xz, 10.0) * 0.2 + checkerboard(r.xz, 100.0) * 0.1 + 0.1;

    var out: FOutput;
    out.color = vec4<f32>(vec3<f32>(c, c, c), 0.1) * f32(t > 0.0);

    return out;
}