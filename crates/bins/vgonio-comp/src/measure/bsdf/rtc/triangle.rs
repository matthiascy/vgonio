use crate::measure::bsdf::rtc::{Ray, RayTriIsect};
use vgcore::{
    math,
    math::{gamma, Vec3},
};

const TOLERANCE: f32 = f32::EPSILON * 2.0;

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

#[rustfmt::skip]
/// Modified Möller-Trumbore ray-triangle intersection algorithm.
///
/// As an improvement over the original algorithm (1), the algorithm is
/// implemented in a way that some factors are precalculated and the
/// calculations are differently factorized to allow precalculating the 
/// cross-product e1 x e2 which is similar to the algorithm in (2).
///
/// # Algorithm
/// Möller and Trumbore solve the ray-triangle intersection problem by directly
/// solving a linear system of equations using Cramer's rule and by evaluating
/// determinants using scalar triple products.
/// Assuming that `P` is the point of intersection. It can be expressed:
///   + by barycentric coordinates $(u, v, w)$:
///
///     $P = wA + uB + vC = A + u(B - A) + v(C - A) = (1-u-v)A + uB + vC$
///   
///   + or by ray parameter t:
///
///     $P = O + tD$
///
/// Moller and Trumbore use the barycentric coordinates to evaluate the
/// intersection point. The system of equations is defined as follows:
///
/// $O - A = \begin{bmatrix}-D & B-A & C-A\end{bmatrix} \begin{bmatrix}t \\\ u \\\ v\end{bmatrix}$
///
/// with $E0 = B - A, E1 = C - A, O - A = T$.
///
/// $T = \begin{bmatrix}-D & E0 & E1\end{bmatrix} \begin{bmatrix}t \\\ u \\\ v\end{bmatrix}$
///
/// Apply Cramer's rule.
///
/// $\begin{vmatrix}A & B & C\end{vmatrix} = -(A \times C) \cdot B = -(C \times B) \cdot A$
///
/// then, we have
///
/// $det = \begin{vmatrix}-D & E0 & E1\end{vmatrix} = -(-D \times E1) \cdot E0 = (D \times E1) \cdot E0$ 
/// 
/// $det_t = \begin{vmatrix}T & E0 & E1\end{vmatrix} = -(T \times E1) \cdot E0 = -(T \times E1) \cdot E0 = (T \times E0) \cdot E1$ 
/// 
/// $det_u = \begin{vmatrix}-D & T & E1\end{vmatrix} = -(-D \times E1) \cdot T = (D \times E1) \cdot T$ 
/// 
/// $det_v = \begin{vmatrix}-D & E0 & T\end{vmatrix} = -(T \times E0) \cdot -D = (T \times E0) \cdot D$
///
/// and finally we have 
/// 
/// $$t = \frac{det_t}{det}, u = \frac{det_u}{det}, v = \frac{det_v}{det}$$
///
/// # References
///
/// (1) Tomas Möller & Ben Trumbore (1997) Fast, Minimum Storage Ray-Triangle
/// Intersection, Journal of Graphics Tools, 2:1, 21-28, DOI:
/// 10.1080/10867651.1997.10487468
///
/// (2) A. Kensler and P. Shirley, "Optimizing Ray-Triangle Intersection via
/// Automated Search," 2006 IEEE Symposium on Interactive Ray Tracing, 2006, pp.
/// 33-38, doi: 10.1109/RT.2006.280212.
///
/// # Arguments
///
/// * `ray`: The ray to intersect with the triangle.
///
/// * `triangle`: The triangle (array of 3 points) to intersect with the ray.
///
/// # Returns
///
/// [`RayTriIsect`] if the ray intersects the triangle, otherwise `None`.
pub fn ray_tri_intersect_moller_trumbore(ray: &Ray, triangle: &[Vec3; 3]) -> Option<RayTriIsect> {
    let ray_d = ray.dir.as_dvec3();
    let ray_o = ray.org.as_dvec3();
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

    let inv_det = math::rcp_f32(det); 
    let tvec = ray_o - p0; // O - A

    let u = d_cross_e1.dot(tvec) as f32 * inv_det; // (D x E1) . T / det
    if !(-TOLERANCE..=1.0 + TOLERANCE).contains(&u) {
        return None;
    }

    let tvec_cross_e0 = tvec.cross(e0); // (T x E0)
    let v = tvec_cross_e0.dot(ray_d) as f32 * inv_det; // (T x E0) . D / det
    if v < -TOLERANCE || u + v > 1.0 + TOLERANCE {
        log::debug!("                 => break v");
        return None;
    }

    let t = tvec_cross_e0.dot(e1) as f32 * inv_det; // (T x E0) . E1 / det

    if t > f32::EPSILON {
        let n = e0.cross(e1).normalize();
        let p = (1.0 - u - v) as f64 * p0 + u as f64 * p1 + v as f64 * p2;
        Some(RayTriIsect {
            u,
            v,
            n: n.as_vec3(),
            p: p.as_vec3(),
        })
    } else {
        None
    }
}

/// Sven Woop's watertight ray-triangle intersection algorithm.
///
/// # Algorithm
///
/// The algorithm operates in two stages. First an affine transformation is
/// applied to the *ray* and the *vertices* of the triangle to simplify the
/// intersection problem. In the second stage, the simplified problem is
/// accurately solved using 2D edge tests with a double precision fallback.
///
/// ## 1. Affine transformation
///
/// The affine transformation transforms the ray such that its origin is at $(0,
/// 0, 0)$ in the transformed coordinate system, and its direction is along the
/// $+z$ axis. Triangle vertices are also transformed into this coordinate
/// system before the intersection test. The transformation is defined
/// by a translation matrix followed by a shearing and scaling matrix.
///
/// Note that this transformation is *only* dependent on the ray (i.e. it does
/// not depend on whatever triangle is being intersected).
///
/// ### Translation
///
/// The translation matrix places the ray origin at the origin of the coordinate
/// system, is defined by the following:
///
/// $$T = \begin{bmatrix}1 & 0 & 0 & -o_x \\\0 & 1 & 0 & -o_y \\\0 & 0 & 1 &
/// -o_z \\\0 & 0 & 0 & 1\end{bmatrix}$$
///
/// This transformation doesn't need to be explicitly applied to the ray, but
/// will be applied to the vertices of the triangle.
///
/// ### Axis Permutation
///
/// Make sure that the $z$ component of the ray direction has the largest
/// absolute value. This can be achieved by renaming the $x$, $y$ and $z$
/// dimensions in a *winding preserving* way.
///
/// ### Shearing
///
/// The shearing transformation aligns the ray direction with the $+z$ axis:
///
/// $$S = \begin{bmatrix}1 & 0 & -d_x & 0 \\\0 & 1 & -d_y/d_z & 0 \\\0 & 0 &
/// 1/d_z & 0 \\\0 & 0 & 0 & 1\end{bmatrix}$$
///
/// ## 2. Intersection test
///
/// With the triangle vertices transformed to this coordinate system, the
/// intersection test now is to find if the ray starting from the origin and
/// traveling along the $+z$ axis intersects the transformed triangle. Because
/// of the way the coordinate system was constructed, this problem is equivalent
/// to the 2D problem of determining if the $x$, $y$ coordinates $(0, 0)$ are
/// inside the $xy$ projection of the triangle.
///
/// Recall that in 2D, the area of a triangle formed by two vectors is defined
/// as:
///
/// $$Area = \frac{1}{2}\vec{a}\times\vec{b} = \frac{1}{2}(a_xb_y - b_xa_y)$$.
///
/// Given two triangle vertices, then we can define the directed edge function
/// $e$ as the function that gives twice the are of the triangle given by $p0$,
/// $p1$ and a given point $p$
///
/// $$e(p) = (p1_x - p0_x)(p_y - p0_y) - (p1_y - p0_y)(p_x - p0_x)$$
///
/// Then we have the following equations:
///
/// $$e0 = p1_xp2_y - p2_xp1_y \quad (p1, p2)$$
/// $$e1 = p2_xp0_y - p2_yp0_x \quad (p2, p0)$$
/// $$e2 = p0_xp1_y - p0_yp1_x \quad (p0, p1)$$
///
/// The value of the edge function is positive if the point is on the right side
/// of the edge, and negative for points to the left of the line.
///
/// Now we have the scaled barycentric coordinates, $u'=e0$, $v'=e1$ and
/// $w'=e2$.
///
/// If $u'<0$, $v'<0$ and $w'<0$, the ray misses the triangle.
///
/// The determinant of the system of equations is $det = u' + v' + w'$. If
/// $det=0$, the ray is co-planar to the triangle and therefore misses. This
/// guarantees later safe divisions.
///
/// Then we can calculate the scaled hit distance $t'$ by interpolating the
/// z-values of the transformed vertices:
///
/// $$t' = u'p0_z + v'p1_z + w'p2_z$$.
///
/// As this distance is calculated using non-normalized barycentric coordinates,
/// the actual distance still requires division by $det$. To defer this costly
/// division util actually required, we first compute a scaled depth test by
/// rejecting the triangle if the hit is either before the ray $$t' <= 0$$ or
/// behind and already-found hit ($t_{max}$) $$t' > det \cdot t_{hit}$$.
///
/// If the sign of the scaled distance and the sign of the interpolated t value
/// are different, the the final t value will certainly be negative and thus not
/// a valid intersection.
///
/// The check for $t < t_{max}$ can be equivalent performed in two ways:
///
/// $$t' < t_{max} \cdot det \quad \text{if} \quad u' + v' + w' > 0$$
/// $$t' > t_{max} \cdot det \quad \text{otherwise}$$
///
/// # References
/// (1) Sven Woop, Carsten Benthin, and Ingo Wald, Watertight Ray/Triangle
/// Intersection, Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 1,
/// 65-82, 2013
pub fn ray_tri_intersect_woop(ray: &Ray, triangle: &[Vec3; 3], tmax: f32) -> Option<RayTriIsect> {
    // Transform the triangle vertices into the ray coordinate system.
    let mut p0t = triangle[0] - ray.org; // A'
    let mut p1t = triangle[1] - ray.org; // B'
    let mut p2t = triangle[2] - ray.org; // C'

    // Calculate the dimension where the ray direction has the largest change.
    let kz = max_axis(ray.dir.abs());
    let kx = if kz + 1 == 3 { 0 } else { kz + 1 };
    let ky = if kx + 1 == 3 { 0 } else { kx + 1 };

    // Permutation in a winding preserving way.
    let d = permute(ray.dir, kx, ky, kz);
    p0t = permute(p0t, kx, ky, kz);
    p1t = permute(p1t, kx, ky, kz);
    p2t = permute(p2t, kx, ky, kz);

    // Perform Shear and scale of vertices.
    let rcp_dz = math::rcp_f32(d.z);
    let sx = -d.x * rcp_dz;
    let sy = -d.y * rcp_dz;
    let sz = rcp_dz;

    p0t.x += sx * p0t.z;
    p0t.y += sy * p0t.z;
    p1t.x += sx * p1t.z;
    p1t.y += sy * p1t.z;
    p2t.x += sx * p2t.z;
    p2t.y += sy * p2t.z;

    // Calculate scaled barycentric coordinates.
    let mut e0 = p1t.x * p2t.y - p1t.y * p2t.x;
    let mut e1 = p2t.x * p0t.y - p2t.y * p0t.x;
    let mut e2 = p0t.x * p1t.y - p0t.y * p1t.x;

    if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
        e0 = (p1t.x as f64 * p2t.y as f64 - p1t.y as f64 * p2t.x as f64) as f32;
        e1 = (p2t.x as f64 * p0t.y as f64 - p2t.y as f64 * p0t.x as f64) as f32;
        e2 = (p0t.x as f64 * p1t.y as f64 - p0t.y as f64 * p1t.x as f64) as f32;
    }

    // Perform edge tests.
    if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
        println!("case 1");
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
    // Check if the scaled distance value t is in the range [t_min, t_max].
    if (det < 0.0 && (t_scaled >= 0.0 || t_scaled < tmax * det))
        || (det > 0.0 && (t_scaled <= 0.0 || t_scaled > tmax * det))
    {
        return None;
    }

    // Compute barycentric coordinates and t value for triangle intersection.
    let inv_det = math::rcp_f32(det);
    let b0 = e0 * inv_det;
    let b1 = e1 * inv_det;
    let b2 = e2 * inv_det;
    let t = t_scaled * inv_det;

    // Ensure that computed triangle t is conservatively greater than zero
    let max_zt = p0t.z.abs().max(p1t.z.abs()).max(p2t.z);
    // Compute delta_z term for triangle t error bounds
    let delta_z = gamma(3) * max_zt;
    // Compute delta_x and delta_y terms for triangle t error bounds
    let max_xt = p0t.x.abs().max(p1t.x.abs()).max(p2t.x.abs());
    let max_yt = p0t.y.abs().max(p1t.y.abs()).max(p2t.y.abs());
    let delta_x = gamma(5) * (max_xt + max_zt);
    let delta_y = gamma(5) * (max_yt + max_zt);
    // Compute delta_e term for triangle t error bounds
    let delta_e = 2.0 * (gamma(2) * max_xt * max_yt + delta_y * max_xt + delta_x * max_yt);
    // Compute delta_t term for triangle t error bounds and check t
    let max_e = e0.abs().max(e1.abs()).max(e2.abs());
    let delta_t =
        3.0 * (gamma(3) * max_e * max_zt + delta_e * max_zt + delta_z * max_e) * inv_det.abs();

    if t <= delta_t {
        return None;
    }

    let n = (triangle[1] - triangle[0])
        .cross(triangle[2] - triangle[0])
        .normalize();

    // Compute error bounds for triangle intersection
    let abs_sum = (b0 * triangle[0]).abs() + (b1 * triangle[1]).abs() + (b2 * triangle[2]).abs();
    let p_err = gamma(7) * abs_sum;
    let p = b0 * triangle[0] + b1 * triangle[1] + b2 * triangle[2];

    Some(RayTriIsect::new(p, p_err, n, b1, b2))
}

#[cfg(test)]
mod tests {
    use super::ray_tri_intersect_woop;
    use crate::measure::bsdf::rtc::{ray_tri_intersect_moller_trumbore, Ray};
    use base::math::Vec3;

    #[test]
    fn test_ray_tri_intersection_woop() {
        let rays = [
            Ray::new(Vec3::new(0.0, 1.0, -10.0), Vec3::new(0.0, 0.0, 0.5)),
            Ray::new(Vec3::new(-1.0, -1.0, -10.0), Vec3::new(0.0, 0.0, 0.2)),
            Ray::new(Vec3::new(1.0, -1.0, -10.0), Vec3::new(0.0, 0.0, 1.0)),
            Ray::new(Vec3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.5)),
            Ray::new(Vec3::new(0.0, -1.0, -10.0), Vec3::new(0.0, 0.0, 0.5)),
            Ray::new(Vec3::new(0.5, 0.0, -10.0), Vec3::new(0.0, 0.0, 0.2)),
            Ray::new(Vec3::new(-0.5, 0.0, -10.0), Vec3::new(0.0, 0.0, 1.0)),
        ];
        let triangle = [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        for ray in rays {
            let isect = ray_tri_intersect_woop(&ray, &triangle, f32::INFINITY);
            // println!("{:?}", isect);
            assert!(isect.is_some());
        }

        {
            let ray = Ray::new(Vec3::new(-3.0, -3.0, 0.5), Vec3::new(1.0, 1.0, 0.0));
            let triangle = [
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ];
            let isect_woop = ray_tri_intersect_woop(&ray, &triangle, f32::INFINITY);
            let isect_moeller = ray_tri_intersect_moller_trumbore(&ray, &triangle);
            // println!("woop: {:?}", isect_woop);
            // print!("moller: {:?}", isect_moeller);
            assert!(isect_woop.is_some());
            assert!(isect_moeller.is_some());
        }

        {
            println!("last test");
            let ray = Ray::new(Vec3::new(-3.0, 0.5, -3.0), Vec3::new(1.0, 0.0, 1.0));
            let triangle = [
                Vec3::new(0.0, 0.0, -1.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ];
            let isect_woop = ray_tri_intersect_woop(&ray, &triangle, f32::INFINITY);
            println!("woop: {:?}", isect_woop);
            let isect = ray_tri_intersect_moller_trumbore(&ray, &triangle);
            println!("moller: {:?}", isect);
            assert!(isect_woop.is_some());
            assert!(isect.is_some());
        }
    }
}
