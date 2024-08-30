//! Smooth surface triangulation taken from "Curved PN Triangles" by A. Vlachos,
//! J. Peters, C. Boyd, and J. Mitchell

use base::math::cbr;
use glam::{DVec3, Vec2, Vec3};

/// Queries the sub-triangulated points of a triangle using the PN triangle
/// algorithm.
///
/// # Arguments
///
/// * `vs` - The vertices of the triangle.
/// * `ns` - The normals of triangle vertices.
/// * `uvs` - The uv coordinates of the desired interpolation points on the
///   triangle.
/// * `ovs` - The output points of the sub-triangulation.
/// * `ons` - The output normals of triangulated points.
pub fn subdivide_triangle(
    vs: &[Vec3],
    uvs: &[Vec2],
    ns: Option<&[Vec3]>,
    ovs: &mut [DVec3],
    ons: Option<&mut [Vec3]>,
) {
    let ns = ns.unwrap();
    let ons = ons.unwrap();
    debug_assert!(vs.len() >= 3, "The input vertices must be a triangle.");
    debug_assert!(ns.len() >= 3, "The input normals must be a triangle.");
    debug_assert!(
        uvs.len() == ovs.len(),
        "The input uvs must be the same count as the output points."
    );
    debug_assert!(
        uvs.len() == ons.len(),
        "The input uvs must be the same count as the output normals."
    );
    let vs: [DVec3; 3] = [vs[0].into(), vs[1].into(), vs[2].into()];
    let ns: [DVec3; 3] = [ns[0].into(), ns[1].into(), ns[2].into()];
    let mut ws = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                ws[i][j] = (vs[j] - vs[i]).dot(ns[i]);
            }
        }
    }
    let b210 = (2.0 * vs[0] + vs[1] - ws[0][1] * ns[0]) / 3.0;
    let b120 = (2.0 * vs[1] + vs[0] - ws[1][0] * ns[1]) / 3.0;
    let b021 = (2.0 * vs[1] + vs[2] - ws[1][2] * ns[1]) / 3.0;
    let b012 = (2.0 * vs[2] + vs[1] - ws[2][1] * ns[2]) / 3.0;
    let b102 = (2.0 * vs[2] + vs[0] - ws[2][0] * ns[2]) / 3.0;
    let b201 = (2.0 * vs[0] + vs[2] - ws[0][2] * ns[0]) / 3.0;
    let e = (b210 + b120 + b021 + b012 + b102 + b201) / 6.0;
    let v = (vs[0] + vs[1] + vs[2]) / 3.0;
    let b111 = e + (e - v) * 0.5;

    let v01 = 2.0 * (vs[1] - vs[0]).dot(ns[0] + ns[1]) / (vs[1] - vs[0]).length_squared();
    let v12 = 2.0 * (vs[2] - vs[1]).dot(ns[1] + ns[2]) / (vs[2] - vs[1]).length_squared();
    let v20 = 2.0 * (vs[0] - vs[2]).dot(ns[2] + ns[0]) / (vs[0] - vs[2]).length_squared();
    let h110 = ns[0] + ns[1] - v01 * (vs[1] - vs[0]);
    let h011 = ns[1] + ns[2] - v12 * (vs[2] - vs[1]);
    let h101 = ns[2] + ns[0] - v20 * (vs[0] - vs[2]);
    let n110 = h110.normalize();
    let n011 = h011.normalize();
    let n101 = h101.normalize();

    ovs.iter_mut()
        .zip(ons.iter_mut())
        .zip(uvs.iter())
        .for_each(|((ov, on), uv)| {
            let (u, v) = (uv.x as f64, uv.y as f64);
            let w = (1.0 - u - v).max(0.0);
            let w2 = w * w;
            let u2 = u * u;
            let v2 = v * v;
            let w3 = cbr(w);
            let u3 = cbr(u);
            let v3 = cbr(v);
            *ov = vs[0] * w3
                + vs[1] * u3
                + vs[2] * v3
                + b210 * 3.0 * w2 * u
                + b120 * 3.0 * w * u2
                + b201 * 3.0 * w2 * v
                + b021 * 3.0 * u2 * v
                + b102 * 3.0 * w * v2
                + b012 * 3.0 * u * v2
                + b111 * 6.0 * w * u * v;
            let n =
                ns[0] * w2 + ns[1] * u2 + ns[2] * v2 + n110 * w * u + n011 * u * v + n101 * w * v;
            *on = Vec3::new(n.x as f32, n.y as f32, n.z as f32);
        });
}
