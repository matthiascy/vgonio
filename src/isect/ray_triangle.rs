use glam::Vec3;
use crate::acq::ray::Ray;
use crate::isect::RayTriInt;

pub const TOLERANCE: f32 = f32::EPSILON * 2.0;

/// Permutes the given vector according to the given permutation.
///
/// A permutation is a vector of three indices that map to the original vector.
///
/// Analogous to the vector swizzle operation.
pub fn permute(v: Vec3, x: u32, y: u32, z: u32) -> Vec3 {
    Vec3::new(v[x as usize], v[y as usize], v[z as usize])
}

/// Returns the index of the largest component of the given vector.
pub fn max_axis(v: Vec3) -> u32 {
    if v.x > v.y && v.x > v.z {
        0
    } else if v.y > v.z {
        1
    } else {
        2
    }
}

/// Modified Möller-Trumbore ray-triangle intersection algorithm.
///
/// As an improvement over the original algorithm (1), the algorithm is implemented in a way that
/// some factors are precalculated and the calculations are differently factorized to allow
/// precalculating the cross product e1 x e2 which is similar to the algorithm in (2).
///
/// # Algorithm
/// Möller and Trumbore solve the ray-triangle intersection problem by directly solving a linear
/// system of equations using Cramer's rule and by evaluating determinants using scalar triple
/// products.
/// Assuming that `P` is the point of intersection. It can be expressed:
///   + by barycentric coordinates $(u, v, w)$: $P = wA + uB + vC = A + u(B - A) + v(C - A)$
///   + or by ray parameter t: $P = O + tD$
///
/// Moller and Trumbore use the barycentric coordinates to evaluate the intersection point. The
/// system of equations is defined as follows:
///
/// $$O - A = \begin{bmatrix}-D & B-A & C-A\end{bmatrix} \begin{bmatrix}t \\\ u \\\ v\end{bmatrix}$$
///
/// with $$E0 = B - A,  E1 = C - A, O - A = T$$
///
/// $$T = \begin{bmatrix}-D & E0 & E1\end{bmatrix} \begin{bmatrix}t \\\ u \\\ v\end{bmatrix}$$
///
/// Apply Cramer's rule,
/// $$\begin{vmatrix}A & B & C\end{vmatrix} = -(A \times C) \cdot B = -(C \times B) \cdot A$$, we have
///
/// $$det = \begin{vmatrix}-D & E0 & E1\end{vmatrix} = -(-D \times E1) \cdot E0 = (D \times E1) \cdot E0$$
/// $$det_t = \begin{vmatrix}T & E0 & E1\end{vmatrix} = -(T \times E1) \cdot E0 = -(T \times E1) \cdot E0 = (T \times E0) \cdot E1$$
/// $$det_u = \begin{vmatrix}-D & T & E1\end{vmatrix} = -(-D \times E1) \cdot T = (D \times E1) \cdot T$$
/// $$det_v = \begin{vmatrix}-D & E0 & T\end{vmatrix} = -(T \times E0) \cdot -D = (T \times E0) \cdot D$$
///
/// and finally we have $$t = \frac{det_t}{det}, u = \frac{det_u}{det}, v = \frac{det_v}{det}$$.
///
/// # References
///
/// (1) Tomas Möller & Ben Trumbore (1997) Fast, Minimum Storage Ray-Triangle Intersection,
/// Journal of Graphics Tools, 2:1, 21-28, DOI: 10.1080/10867651.1997.10487468
///
/// (2) A. Kensler and P. Shirley, "Optimizing Ray-Triangle Intersection via Automated Search,"
/// 2006 IEEE Symposium on Interactive Ray Tracing, 2006, pp. 33-38, doi: 10.1109/RT.2006.280212.
///
/// # Arguments
///
/// * `ray`: The ray to intersect with the triangle.
///
/// * `triangle`: The triangle (array of 3 points) to intersect with the ray.
///
/// # Returns
///
/// [`RayTriInt`] if the ray intersects the triangle, otherwise `None`.
pub fn ray_tri_intersect_moller_trumbore(ray: Ray, triangle: &[Vec3; 3]) -> Option<RayTriInt> {
    let ray_d = ray.d.as_dvec3();
    let ray_o = ray.o.as_dvec3();
    let p0 = triangle[0].as_dvec3();
    let p1 = triangle[1].as_dvec3();
    let p2 = triangle[2].as_dvec3();

    let e0 = p1 - p0;
    let e1 = p2 - p0;

    let d_cross_e1 = ray_d.cross(e1); // D x E1

    let det = d_cross_e1.dot(e0) as f32;

    // If the determinant is zero, the ray misses the triangle.
    if det.abs() < f32::EPSILON {
        return None;
    }

    let inv_det = 1.0 / det;
    let tvec = ray_o - p0; // O - A

    let u = d_cross_e1.dot(tvec) as f32 * inv_det; // (D x E1) . T / det
    log::debug!("                 => u = {}", u);
    if !(-TOLERANCE..=1.0 + TOLERANCE).contains(&u) {
        log::debug!("                 => break u");
        return None;
    }

    let tvec_cross_e0 = tvec.cross(e0); // (T x E0)
    let v = tvec_cross_e0.dot(ray_d) as f32 * inv_det; // (T x E0) . D / det
    log::debug!("                 => v = {}", v);
    if v < -TOLERANCE || u + v > 1.0 + TOLERANCE {
        log::debug!("                 => break v");
        return None;
    }

    let t = tvec_cross_e0.dot(e1) as f32 * inv_det; // (T x E0) . E1 / det
    log::debug!("                 => t = {}", t);

    if t > f32::EPSILON {
        let n = e0.cross(e1).normalize();
        let p = (1.0 - u - v) as f64 * p0 + u as f64 * p1 + v as f64 * p2;
        log::debug!(
            "                   => ray/tri test, t = {}, u = {}, v = {}, n = {}, p = {}",
            t,
            u,
            v,
            n,
            p
        );
        Some(RayTriInt {
            t,
            u,
            v,
            n: n.as_vec3(),
            p: p.as_vec3(),
        })
    } else {
        None
    }
}

// TODO: precompute shear coefficients.
/// Sven Woop's watertight ray-triangle intersection algorithm.
///
/// # Algorithm
///
/// The algorithm operates in two stages. First an affine transformation is applied to the *ray*
/// and the *vertices* of the triangle to simplify the intersection problem. In the second stage,
/// the simplified problem is accurately solved using 2D edge tests with a double precision fallback.
///
/// ## 1. Affine transformation
///
/// The affine transformation transforms the ray such that its origin is at $(0, 0, 0)$ in the
/// transformed coordinate system, and its direction is along the $+z$ axis. Triangle vertices are also
/// transformed into this coordinate system before the intersection test. The transformation is defined
/// by a translation matrix followed by a shearing and scaling matrix.
///
/// Note that this transformation is *only* dependent on the ray (i.e. it does not depend on whatever
/// triangle is being intersected).
///
/// ### Translation
///
/// The translation matrix places the ray origin at the origin of the coordinate system, is defined
/// by the following:
///
/// $$T = \begin{bmatrix}1 & 0 & 0 & -o_x \\\0 & 1 & 0 & -o_y \\\0 & 0 & 1 & -o_z \\\0 & 0 & 0 & 1\end{bmatrix}$$
///
/// This transformation doesn't need to be explicitly applied to the ray, but will be applied to the
/// vertices of the triangle.
///
/// ### Axis Permutation
///
/// Make sure that the $z$ component of the ray direction has the largest absolute value. This can
/// be achieved by renaming the $x$, $y$ and $z$ dimensions in a *winding preserving* way.
///
/// ### Shearing
///
/// The shearing transformation aligns the ray direction with the $+z$ axis:
///
/// $$S = \begin{bmatrix}1 & 0 & -d_x & 0 \\\0 & 1 & -d_y/d_z & 0 \\\0 & 0 & 1/d_z & 0 \\\0 & 0 & 0 & 1\end{bmatrix}$$
///
/// ## 2. Intersection test
///
/// With the triangle vertices transformed to this coordinate system, the intersection test now is
/// to find if the ray starting from the origin and traveling along the $+z$ axis intersects the
/// transformed triangle. Because of the way the coordinate system was constructed, this problem is
/// equivalent to the 2D problem of determining if the $x$, $y$ coordinates $(0, 0)$ are inside the
/// $xy$ projection of the triangle.
///
/// Recall that in 2D, the area of a triangle formed by two vectors is defined as:
///
/// $$A = \vec{a}\times\vec{b} = a_xb_y - b_xa_y$$.
///
/// Given two triangle vertices, then we can define the directed edge function $e$ as the function
/// that gives twice the are of the triangle given by $p0$, $p1$ and a given point $p$
///
/// $$e(p) = (p1_x - p0_x)(p_y - p0_y) - (p1_y - p0_y)(p_x - p0_x)$$
///
/// Then we have the following equations:
///
/// $$e0 = p1_xp2_y - p2_xp1_y$$
/// $$e1 = p2_xp0_y - p2_yp0_x$$
/// $$e2 = p0_xp1_y - p0_yp1_x$$
///
/// The value of the edge function is positive if the point is on the right side of the edge, and
/// negative for points to the left of the line.
///
/// For points $A'$, $B'$ and $C'$ we have
///
/// $$A' = S \cdot T \cdot A = S \cdot (A - O)$$
/// $$B' = S \cdot T \cdot B = S \cdot (B - O)$$
/// $$C' = S \cdot T \cdot C = S \cdot (C - O)$$
///
/// $E0 = A - O$, $E1 = B - O$, $E2 = C - O$.
///
/// $$U = C'_x * B'_y - C'_y * B'_x$$
/// $$V = A'_x * C'_y - A'_y * C'_x$$
/// $$W = B'_x * A'_y - B'_y * A'_x$$
///
/// # References
/// (1) Sven Woop, Carsten Benthin, and Ingo Wald, Watertight Ray/Triangle Intersection, Journal of
/// Computer Graphics Techniques (JCGT), vol. 2, no. 1, 65-82, 2013
pub fn ray_tri_intersect_woop(ray: &Ray, triangle: &[Vec3; 3]) -> Option<RayTriInt> {
    // Transform the triangle vertices into the ray coordinate system.
    let p0 = triangle[0] - ray.o; // A'
    let p1 = triangle[1] - ray.o; // B'
    let p2 = triangle[2] - ray.o; // C'

    // Permutation in a winding preserving way.
    let kz = max_axis(ray.d.abs());
    let kx = (kz + 1) % 3;
    let ky = (kz + 2) % 3;

    let d = permute(ray.d, kx, ky, kz);
    let mut p0t = permute(p0, kx, ky, kz);
    let mut p1t = permute(p1, kx, ky, kz);
    let mut p2t = permute(p2, kx, ky, kz);

    // Shearing.
    let sx = -d.x / d.z;
    let sy = -d.y / d.z;
    let sz = 1.0 / d.z;

    p0t.x += sx * p0t.z;
    p0t.y += sy * p0t.z;
    p1t.x += sx * p1t.z;
    p1t.y += sy * p1t.z;
    p2t.x += sx * p2t.z;
    p2t.y += sy * p2t.z;

    // Intersection test.
    let mut e0 = p1t.x * p2t.y - p1t.y * p2t.x;
    let mut e1 = p2t.x * p0t.y - p2t.y * p0t.x;
    let mut e2 = p0t.x * p1t.y - p0t.y * p1t.x;

    if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
        e0 = (p1t.x as f64 * p2t.y as f64 - p1t.y as f64 * p2t.x as f64) as f32;
        e1 = (p2t.x as f64 * p0t.y as f64- p2t.y as f64 * p0t.x as f64) as f32;
        e2 = (p0t.x as f64 * p1t.y as f64 - p0t.y as f64 * p1t.x as f64) as f32;
    }

    // TODO: In the rare case that any of the edge function values is exactly zero, reevaluated using
    // double precision.
    if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
        return None;
    }

    let det = e0 + e1 + e2;

    // Parallel to the triangle.
    if det == 0.0 {
        return None;
    }

    // Compute scaled hit distance to triangle and test against ray tMin and tMax.
    p0t.z *= sz;
    p1t.z *= sz;
    p2t.z *= sz;
    let t_scaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if det < 0.0 && (t_scaled >= 0.0 || t_scaled < ray.t_max * det) {
        return None;
    } else if det > 0.0 && (t_scaled <= 0.0 || t_scaled > ray.t_max * det) {
        return None;
    }

    // Compute barycentric coordinates and t value for triangle intersection.
    let inv_det = 1.0 / det;
    let u = e0 * det;
    let v = e1 * det;
    let w = e2 * det;
    let t = t_scaled * inv_det;

    todo!()
}