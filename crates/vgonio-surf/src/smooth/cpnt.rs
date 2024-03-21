//! Smooth surface triangulation taken from "Curved PN Triangles" by A. Vlachos,
//! J. Peters, C. Boyd, and J. Mitchell

use base::math::cbr;
use glam::{Vec2, Vec3};
use std::mem::MaybeUninit;

/// Queries the sub-triangulated points of a triangle using the PN triangle
/// algorithm.
///
/// # Arguments
///
/// * `Vs` - The vertices of the triangle.
/// * `ns` - The normals of the triangle.
/// * `Uvs` - The uv coordinates of the desired interpolation points on the
///   triangle.
/// * `ps` - The output points of the sub-triangulation.
pub fn subdivide_triangle(
    vs: &[Vec3],
    ns: &[Vec3],
    uvs: &[Vec2],
    ovs: &mut [Vec3],
    ons: &mut [Vec3],
) {
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
    let b300 = vs[0];
    let b030 = vs[1];
    let b003 = vs[2];
    let mut ws = [[Vec3::ZERO; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                let pj_pi = vs[j] - vs[i];
                ws[i][j] = pj_pi * ns[i];
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
    let v = (b300 + b030 + b003) / 3.0;
    let b111 = e + (e - v) * 0.5;

    let n200 = ns[0];
    let n020 = ns[1];
    let n002 = ns[2];
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
            let (u, v) = (uv.x, uv.y);
            let w = (1.0 - u - v).max(0.0);
            let w2 = w * w;
            let u2 = u * u;
            let v2 = v * v;
            *ov = b300 * cbr(w)
                + b030 * cbr(u)
                + b003 * cbr(v)
                + b210 * 3.0 * w2 * u
                + b120 * 3.0 * w * u2
                + b201 * 3.0 * w2 * v
                + b021 * 3.0 * u2 * v
                + b102 * 3.0 * w * v2
                + b012 * 3.0 * u * v2
                + b111 * 6.0 * w * u * v;
            *on = n200 * w2 + n020 * u2 + n002 * v2 + n110 * w * u + n011 * u * v + n101 * w * v;
        });
}
