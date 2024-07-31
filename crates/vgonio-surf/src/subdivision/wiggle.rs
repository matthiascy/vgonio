use crate::dcel::HalfEdgeMesh;
use glam::{DVec3, Vec2, Vec3};
use rand::distributions::{Distribution, Uniform};

/// Subdivide a triangle by adding points on the edges and in the center.
///
/// # Arguments
///
/// * `vs` - The vertices of the triangle.
/// * `uvs` - The uv coordinates of the desired interpolation points on the
///  triangle.
/// * `ovs` - The output points of the sub-triangulation.
pub fn subdivide_triangle(vs: &[Vec3], uvs: &[Vec2], ovs: &mut [DVec3]) {
    debug_assert!(vs.len() >= 3, "The input vertices must be a triangle.");
    debug_assert!(
        uvs.len() == ovs.len(),
        "The input uvs must be the same count as the output points."
    );
    let vs: [DVec3; 3] = [vs[0].into(), vs[1].into(), vs[2].into()];
    let min_z = vs[0].z.min(vs[1].z).min(vs[2].z);
    let max_z = vs[0].z.max(vs[1].z).max(vs[2].z);
    let z_dist = max_z - min_z;
    let uniform_dist = Uniform::new(-0.5, 0.5);
    let mut rng = rand::thread_rng();

    ovs.iter_mut().zip(uvs.iter()).for_each(|(ov, uv)| {
        let (u, v) = (uv.x as f64, uv.y as f64);
        let w = (1.0 - u - v).max(0.0);

        // Interpolate the z coordinate.
        *ov = w * vs[0] + u * vs[1] + v * vs[2];

        // Wiggle the z coordinate of points on the edges excluding the
        // original vertices.
        if !((u - 1.0).abs() < f64::EPSILON
            || (v - 1.0).abs() < f64::EPSILON
            || (w - 1.0).abs() < f64::EPSILON)
        {
            ov.z += uniform_dist.sample(&mut rng) * z_dist;
        }
    });
}
