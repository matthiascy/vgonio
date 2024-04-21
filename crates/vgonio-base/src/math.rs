//! Math utilities.

use crate::units::{rad, radians, Angle, AngleUnit, Radians};
use cfg_if::cfg_if;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Add, Mul},
};

mod aabb;
mod axis;

pub use aabb::*;
pub use axis::*;

pub use glam::*;
// TODO: remove this
use num_traits::Float;

// TODO: self-contained math library, tailored for specific use cases

/// Trait for converting from one primitive numeric type to another.
#[const_trait]
pub trait NumericCast<T> {
    /// Casts `self` to `T`.
    fn cast(&self) -> T;
}

macro impl_as_primitive($t0:ty as $($t1:ty),*) {
$(
        impl const NumericCast<$t1> for $t0 {
            fn cast(&self) -> $t1 {
                *self as $t1
            }
        }
    )*
}

impl_as_primitive!(f32 as f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(f64 as f64, f32, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(i32 as i32, f32, f64, u32, i64, u64, i128, u128);
impl_as_primitive!(u32 as u32, f32, f64, i32, i64, u64, i128, u128);
impl_as_primitive!(i64 as i64, f32, f64, i32, u32, u64, i128, u128);
impl_as_primitive!(u64 as u64, f32, f64, i32, u32, i64, i128, u128);
impl_as_primitive!(i128 as i128, f32, f64, i32, u32, i64, u64, u128);
impl_as_primitive!(u128 as u128, f32, f64, i32, u32, i64, u64, i128);
impl_as_primitive!(u8 as u8, f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(i8 as i8, f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(u16 as u16, f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(i16 as i16, f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(usize as usize, f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(isize as isize, f32, f64, i32, u32, i64, u64, i128, u128);

/// Machine epsilon for double precision floating point numbers.
pub const MACHINE_EPSILON_F64: f64 = f64::EPSILON * 0.5;

/// Machine epsilon for single precision floating point numbers.
pub const MACHINE_EPSILON_F32: f32 = f32::EPSILON * 0.5;

/// Compute the conservative bounding of $(1 \pm \epsilon_{m})^n$ for a given
/// $n$.
pub const fn gamma(n: u32) -> f32 {
    (n as f32 * MACHINE_EPSILON_F32) / (1.0 - n as f32 * MACHINE_EPSILON_F32)
}

/// Identity matrix.
pub const IDENTITY_MAT4: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Equality test of two floating point numbers.
///
/// todo: specify the error bound.
///
/// # Arguments
///
/// * `a`: The first number.
/// * `b`: The second number.
///
/// returns: bool
pub fn ulp_eq(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let a_abs = a.abs();
    let b_abs = b.abs();
    if a == b {
        true
    } else if a == 0.0 || b == 0.0 || a_abs < f32::MIN_POSITIVE || b_abs < f32::MIN_POSITIVE {
        diff < (f32::MIN_POSITIVE * f32::EPSILON)
    } else {
        (diff / f32::min(a_abs + b_abs, f32::MAX)) < f32::EPSILON
    }
}

/// Spherical coordinate in radians.
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct Sph3 {
    /// Radius of the sphere.
    pub rho: f32,
    /// Zenith angle (polar angle) in radians. 0 is the zenith, pi is the
    /// nadir. The zenith angle is the angle between the positive z-axis and
    /// the point on the sphere. The zenith angle is always between 0 and pi.
    /// 0 ~ pi/2 is the upper hemisphere, pi/2 ~ pi is the lower hemisphere.
    pub theta: Radians,
    /// Azimuth angle (azimuthal angle) in radians. It is always between 0
    /// and 2pi: 0 is the positive x-axis, pi/2 is the positive y-axis, pi is
    /// the negative x-axis, 3pi/2 is the negative y-axis.
    pub phi: Radians,
}

impl Sph3 {
    /// Create a new spherical coordinate.
    pub fn new(radius: f32, zenith: Radians, azimuth: Radians) -> Self {
        Self {
            rho: radius,
            theta: zenith,
            phi: azimuth,
        }
    }

    /// Create a new spherical coordinate with radius 1.
    pub fn unit(zenith: Radians, azimuth: Radians) -> Self {
        Self {
            rho: 1.0,
            theta: zenith,
            phi: azimuth,
        }
    }

    /// Convert to a cartesian coordinate.
    pub fn to_cartesian(&self) -> Vec3 { sph_to_cart(self.theta, self.phi) * self.rho }

    /// Convert from a cartesian coordinate.
    pub fn from_cartesian(cartesian: Vec3) -> Self { cart_to_sph(cartesian) }
}

impl Debug for Sph3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{ ρ: {}, θ: {}, φ: {} }}",
            self.rho, self.theta, self.phi
        )
    }
}

impl Display for Sph3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{ ρ: {}, θ: {}, φ: {} }}",
            self.rho,
            self.theta.in_degrees().prettified(),
            self.phi.in_degrees().prettified()
        )
    }
}

/// Spherical coordinate in radians.
///
/// This is a version of [`Sph3`] with radius fixed to 1.
#[derive(Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sph2 {
    /// Zenith angle (polar angle) in radians.
    pub theta: Radians,
    /// Azimuth angle (azimuthal angle) in radians.
    pub phi: Radians,
}

impl Sph2 {
    /// Create a new spherical coordinate.
    pub const fn new(zenith: Radians, azimuth: Radians) -> Self {
        Self {
            theta: zenith,
            phi: azimuth,
        }
    }

    /// Create a new coordinate with zenith and azimuth angles set to 0.
    pub const fn zero() -> Self {
        Self {
            theta: Radians::ZERO,
            phi: Radians::ZERO,
        }
    }

    /// Test if the two spherical coordinates are approximately equal.
    pub fn approx_eq(&self, other: &Self) -> bool {
        approx::abs_diff_eq!(self.theta.value, other.theta.value)
            && approx::abs_diff_eq!(self.phi.value, other.phi.value)
    }

    /// Convert to a cartesian coordinate.
    pub fn to_cartesian(&self) -> Vec3 { sph_to_cart(self.theta, self.phi) }

    /// Returns true if the zenith angle and azimuth angle are both positive.
    pub fn is_positive(&self) -> bool { self.theta.is_positive() && self.phi.is_positive() }

    /// Convert from a cartesian coordinate.
    #[track_caller]
    pub fn from_cartesian(cartesian: Vec3) -> Self {
        debug_assert!(
            approx::ulps_eq!(cartesian.length(), 1.0, epsilon = 1.0e-6),
            "expected 1.0, got {}",
            cartesian.length()
        );
        let sph = cart_to_sph(cartesian);
        Self {
            theta: sph.theta,
            phi: sph.phi,
        }
    }
}

impl Debug for Sph2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ θ: {}, φ: {} }}", self.theta, self.phi)
    }
}

impl Add for Sph2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            theta: self.theta + rhs.theta,
            phi: self.phi + rhs.phi,
        }
    }
}

impl Mul<f32> for Sph2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            theta: self.theta * rhs,
            phi: self.phi * rhs,
        }
    }
}

impl Display for Sph2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{ θ: {}, φ: {} }}",
            self.theta.in_degrees().prettified(),
            self.phi.in_degrees().prettified()
        )
    }
}

/// Conversion from spherical coordinate system to cartesian coordinate system.
///
/// # Arguments
///
/// * `zenith` - polar angle
/// * `azimuth` - azimuthal angle
pub fn sph_to_cart(zenith: Radians, azimuth: Radians) -> Vec3 {
    Vec3::new(
        zenith.sin() * azimuth.cos(),
        zenith.sin() * azimuth.sin(),
        zenith.cos(),
    )
}

/// Conversion from cartesian coordinate system to spherical coordinate system.
///
/// # Arguments
///
/// * `v` - vector in cartesian coordinate system
///
/// # Returns
///
/// Spherical coordinate in radians.
pub fn cart_to_sph(v: Vec3) -> Sph3 {
    let rho = v.length();
    let mut theta = rad!((v.z * rcp_f32(rho)).acos());
    let mut phi = rad!(v.y.atan2(v.x));
    if theta < radians!(0.0) {
        theta += Radians::PI;
    }
    if phi < radians!(0.0) {
        phi += Radians::TAU;
    }
    Sph3::new(rho, theta, phi)
}

/// Compute the circular distance between two angles.
///
/// The circular distance is the shortest distance between two angles on a
/// circle.
#[inline(always)]
pub fn circular_angle_dist<A: AngleUnit>(a: Angle<A>, b: Angle<A>) -> Angle<A> {
    let diff = (a - b).abs();
    if diff > Angle::<A>::PI {
        Angle::<A>::TAU - diff
    } else {
        diff
    }
}

/// Comptues the barycentric coordinates of the given point projected onto the
/// triangle defined by the given vertices. The barycentric coordinates are
/// clamped to the range [0, 1].
pub fn projected_barycentric_coords(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> (f32, f32, f32) {
    let v0 = DVec3::from(v0);
    let e0 = DVec3::from(v1) - v0;
    let e1 = DVec3::from(v2) - v0;
    // Project the point onto the triangle plane
    let pp = {
        let p = DVec3::from(p);
        let n = e0.cross(e1).normalize();
        let v = p - v0;
        p - n * n.dot(v)
    };
    let e = pp - v0;
    let d00 = e0.dot(e0);
    let d01 = e0.dot(e1);
    let d11 = e1.dot(e1);
    let d20 = e.dot(e0);
    let d21 = e.dot(e1);
    let inv_denom = rcp_f64(d00 * d11 - d01 * d01);
    let v = ((d11 * d20 - d01 * d21) * inv_denom).clamp(0.0, 1.0);
    if v == 1.0 {
        return (0.0, 1.0, 0.0);
    }
    let w = ((d00 * d21 - d01 * d20) * inv_denom).clamp(0.0, 1.0);
    if w == 1.0 {
        return (0.0, 0.0, 1.0);
    }
    ((1.0 - v - w).clamp(0.0, 1.0) as f32, v as f32, w as f32)
}

// TODO: generalise rcp to f32 and f64

/// Returns the accurate reciprocal of the given value.
///
/// Newton-Raphson iteration is used to compute the reciprocal.
#[inline(always)]
pub fn rcp_f32(x: f32) -> f32 {
    // Intel's intrinsic will give us NaN if x is 0.0 or -0.0
    if x == 0.0 {
        return f32::INFINITY * x.signum();
    }

    cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            use std::arch::x86_64::{_mm_cvtss_f32, _mm_mul_ss, _mm_rcp_ss, _mm_set_ss, _mm_sub_ss};
            unsafe {
                let a = _mm_set_ss(x);
                let r = if is_x86_feature_detected!("avx512vl") {
                    use std::arch::x86_64::_mm_rcp14_ss;
                    _mm_rcp14_ss(_mm_set_ss(0.0), a) // error is less than 2^-14
                } else {
                    _mm_rcp_ss(a) // error is less than 1.5 * 2^-12
                };

                if is_x86_feature_detected!("fma") {
                    use std::arch::x86_64::_mm_fnmadd_ss;
                    _mm_cvtss_f32(_mm_mul_ss(r, _mm_fnmadd_ss(r, a, _mm_set_ss(2.0))))
                } else {
                    _mm_cvtss_f32(_mm_mul_ss(r, _mm_sub_ss(_mm_set_ss(2.0), _mm_mul_ss(r, a))))
                }
            }
        } else {
            1.0 / x
        }
    }
}

/// Returns the accurate reciprocal of the double precision floating point.
pub fn rcp_f64(x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY * x.signum();
    }

    cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            use std::arch::x86_64::{_mm_cvtsd_f64, _mm_set_sd, _mm_sub_sd, _mm_mul_sd};
            unsafe {
                if is_x86_feature_detected!("avx512f") {
                    use std::arch::x86_64::_mm_rcp14_sd;
                    let a = _mm_set_sd(x);
                    let r = _mm_rcp14_sd(_mm_set_sd(0.0), a); // error is less than 2^-14
                    if is_x86_feature_detected!("fma") {
                        use std::arch::x86_64::_mm_fnmadd_sd;
                        _mm_cvtsd_f64(_mm_mul_sd(r, _mm_fnmadd_sd(r, a, _mm_set_sd(2.0))))
                    } else {
                        _mm_cvtsd_f64(_mm_mul_sd(r, _mm_sub_sd(_mm_set_sd(2.0), _mm_mul_sd(r, a))))
                    }
                } else {
                    // There is no _mm_rcp_sd intrinsic, so we use _mm_rcp_ss instead.
                    rcp_f32(x as f32) as f64
                }
            }
        } else {
            1.0 / x
        }
    }
}

/// Returns the square of the given value.
#[inline(always)]
pub fn sqr<F: Float>(x: F) -> F { x * x }

/// Returns the cube of the given value.
#[inline(always)]
pub fn cbr<F: Float>(x: F) -> F { x * x * x }

/// Returns the accurate reciprocal square root of the given value.
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            use std::arch::x86_64::{_mm_cvtss_f32, _mm_mul_ss, _mm_rsqrt_ss, _mm_set_ss, _mm_add_ss};
            unsafe {
                let a = _mm_set_ss(x);

                let r = if is_x86_feature_detected!("avx512vl") {
                    use std::arch::x86_64::_mm_rsqrt14_ss;
                    _mm_rsqrt14_ss(_mm_set_ss(0.0), a) // relative error is less than 2^-14
                } else {
                    _mm_rsqrt_ss(a) // error is less than 1.5 * 2^-12
                };

                _mm_cvtss_f32(_mm_add_ss(
                    _mm_mul_ss(_mm_set_ss(1.5), r),
                    _mm_mul_ss(
                        _mm_mul_ss(_mm_mul_ss(a, _mm_set_ss(-0.5)), r),
                        _mm_mul_ss(r, r),
                    ),
                ))
            }
        } else {
            1.0 / x.sqrt()
        }
    }
}

/// Returns the zenith angle of the given vector in radians in the range
/// [0, pi].
pub fn theta(v: &Vec3) -> Radians {
    debug_assert!(
        approx::ulps_eq!(v.length(), 1.0, epsilon = 1.0e-6),
        "Vec3 must be normalized, got {}",
        v.length()
    );
    let zenith = rad!((v.z * rcp_f32(v.length())).acos());
    if zenith < radians!(0.0) {
        zenith + Radians::PI
    } else {
        zenith
    }
}

/// Returns the cosine of the zenith angle of the given vector.
/// The vector must be normalized.
#[inline(always)]
pub fn cos_theta(v: &Vec3) -> f32 {
    debug_assert!(
        approx::ulps_eq!(v.length(), 1.0, epsilon = 1.0e-6),
        "Vec3 must be normalized, got {}",
        v.length()
    );
    v.z
}

/// Returns the square of the cosine of the zenith angle of the given vector.
/// The vector must be normalized.
#[inline(always)]
pub fn cos_theta2(v: &Vec3) -> f32 {
    debug_assert!(
        approx::ulps_eq!(v.length(), 1.0, epsilon = 1.0e-6),
        "Vec3 must be normalized, got {}",
        v.length()
    );
    sqr(v.z)
}

/// Returns the square of the sine of the zenith angle of the given vector.
pub fn sin_theta2(v: &Vec3) -> f32 { (1.0 - cos_theta2(v)).max(0.0) }

/// Returns the sine of the zenith angle of the given vector.
pub fn sin_theta(v: &Vec3) -> f32 { sin_theta2(v).sqrt() }

/// Returns the azimuth angle of the given vector in radians in the range
pub fn tan_theta(v: &Vec3) -> f32 { sin_theta(v) / cos_theta(v) }

/// Returns the square of the tangent of the zenith angle of the given vector.
pub fn tan_theta2(v: &Vec3) -> f32 { sin_theta2(v) / cos_theta2(v) }

/// Returns the cosine of the azimuth angle of the given vector.
pub fn cos_phi(v: &Vec3) -> f32 {
    let sin_theta = sin_theta(v);
    if sin_theta == 0.0 {
        1.0
    } else {
        (v.x / sin_theta).clamp(-1.0, 1.0)
    }
}

/// Returns the square of the cosine of the azimuth angle of the given vector.
pub fn sin_phi(v: &Vec3) -> f32 {
    let sin_theta = sin_theta(v);
    if sin_theta == 0.0 {
        0.0
    } else {
        (v.y / sin_theta).clamp(-1.0, 1.0)
    }
}

/// Returns the fused multiply-subtract of the given values.
///
/// This is equivalent to `a * b - c`. However, this function may fall back to
/// a non-fused multiply-subtract on some platforms.
#[inline(always)]
pub fn msub(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(target_arch = "x86_64") && cfg!(target_feature = "fma") {
        use std::arch::x86_64::{_mm_cvtss_f32, _mm_fmsub_ss, _mm_set_ss};
        unsafe { _mm_cvtss_f32(_mm_fmsub_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c))) }
    } else {
        a * b - c
    }
}

/// Returns the fused multiply-add of the given values.
///
/// This is equivalent to `a * b + c`. However, this function may fall back to
/// a non-fused multiply-add on some platforms.
#[inline(always)]
pub fn madd(a: f32, b: f32, c: f32) -> f32 {
    cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            use std::arch::x86_64::{_mm_cvtss_f32, _mm_fmadd_ss, _mm_set_ss};
            unsafe { _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c))) }
        } else {
            a * b + c
        }
    }
}

/// Returns the fused negated multiply-subtract of the given values.
///
/// This is equivalent to `-a * b - c`. However, this function may fall back to
/// a non-fused negative multiply-subtract on some platforms.
#[inline(always)]
pub fn nmsub(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(target_arch = "x86_64") && cfg!(target_feature = "fma") {
        use std::arch::x86_64::{_mm_cvtss_f32, _mm_fnmsub_ss, _mm_set_ss};
        unsafe { _mm_cvtss_f32(_mm_fnmsub_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c))) }
    } else {
        -a * b - c
    }
}

/// Returns the fused negated multiply-add of the given values.
///
/// This is equivalent to `-a * b + c`. However, this function may fall back to
/// a non-fused negative multiply-add on some platforms.
#[inline(always)]
pub fn nmadd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(target_arch = "x86_64") && cfg!(target_feature = "fma") {
        use std::arch::x86_64::{_mm_cvtss_f32, _mm_fnmadd_ss, _mm_set_ss};
        unsafe { _mm_cvtss_f32(_mm_fnmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c))) }
    } else {
        -a * b + c
    }
}

/// Quadratic equation can have 0, 1 or 2 real solutions.
#[derive(Debug, Copy, Clone)]
pub enum QuadraticSolution {
    /// No real solutions.
    None,
    /// One real solution.
    One(f32),
    /// Two real solutions.
    Two(f32, f32),
}

impl PartialEq for QuadraticSolution {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (QuadraticSolution::None, QuadraticSolution::None) => true,
            (QuadraticSolution::One(x), QuadraticSolution::One(y)) => ulp_eq(*x, *y),
            (QuadraticSolution::Two(x1, x2), QuadraticSolution::Two(y1, y2)) => {
                (ulp_eq(*x1, *y1) && ulp_eq(*x2, *y2)) || (ulp_eq(*x1, *y2) && ulp_eq(*x2, *y1))
            }
            _ => false,
        }
    }
}

// TODO: implement approx for QuadraticSolution
// TODO: extend this to support complex solutions

/// Solves the quadratic equation `a * x^2 + b * x + c = 0`.
#[inline(always)]
pub fn solve_quadratic(a: f32, b: f32, c: f32) -> QuadraticSolution {
    let discriminant = b * b - 4.0 * a * c;
    let rcp_2a = 0.5 * rcp_f32(a);
    if discriminant < 0.0 {
        QuadraticSolution::None
    } else if discriminant == 0.0 {
        QuadraticSolution::One(-b * rcp_2a)
    } else {
        let discriminant = discriminant.sqrt();
        let p = (-b + discriminant) * rcp_2a;
        let q = (-b - discriminant) * rcp_2a;
        QuadraticSolution::Two(p.min(q), p.max(q))
    }
}

/// Checks if the given values are close enough to each other.
/// TODO: add a tolerance parameter or error bound
pub fn close_enough(a: &Vec3, b: &Vec3) -> bool {
    ulp_eq(a.x, b.x) && ulp_eq(a.y, b.y) && ulp_eq(a.z, b.z)
}

/// Generates a parametric hemisphere with the given theta and phi steps.
///
/// The generated vertices are at the exact theta and phi values.
///
/// Returns a tuple of the vertices and indices of the triangulated hemisphere.
pub fn generate_triangulated_hemisphere(
    theta_end: Radians,
    theta_steps: u32,
    phi_steps: u32,
) -> (Vec<Vec3>, Vec<UVec3>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let theta_step_size = theta_end.value / theta_steps as f32;
    let phi_step_size = Radians::TWO_PI / phi_steps as f32;

    // Generate top vertex
    vertices.push(sph_to_cart(Radians::ZERO, Radians::ZERO));

    for i in 1..=theta_steps {
        let theta = theta_step_size * i as f32;
        for j in 0..phi_steps {
            let phi = phi_step_size * j as f32;
            vertices.push(sph_to_cart(rad!(theta), phi));
        }
    }

    let offset = 1;

    // Generate indices top strip
    for i in 0..phi_steps {
        let i1 = i + offset;
        let i2 = i1 % phi_steps + offset;
        indices.push(UVec3::new(0, i1, i2));
    }

    // Generate indices for the rest of the cells, each cell has max 2 triangles
    for i in 1..theta_steps {
        for j in 0..phi_steps {
            let i0 = (i - 1) * phi_steps + j + offset;
            let i1 = i * phi_steps + j + offset;
            let i2 = i * phi_steps + (j + 1) % phi_steps + offset;
            let i3 = (i - 1) * phi_steps + (j + 1) % phi_steps + offset;
            indices.push(UVec3::new(i0, i2, i1));
            indices.push(UVec3::new(i0, i3, i2));
        }
    }

    (vertices, indices)
}

/// Generates a parametric hemisphere with the given theta and phi steps.
///
/// This is different from `generate_parametric_hemisphere_triangles` in that
/// the vertices are generated to make sure that the points with exact theta
/// and phi values are located on the center of each cell. In other words,
/// the vertices are generated around the center of each cell where the center
/// is defined by the theta and phi values.
///
/// The output is a tuple of the vertices and indices of the lines that connect
/// the vertices.
pub fn generate_parametric_hemisphere_cells(
    theta_steps: u32,
    phi_steps: u32,
) -> (Vec<Vec3>, Vec<UVec2>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let theta_step_size = Radians::HALF_PI / theta_steps as f32;
    let phi_step_size = Radians::TWO_PI / phi_steps as f32;

    // Generate top vertex
    vertices.push(sph_to_cart(Radians::ZERO, Radians::ZERO));

    for i in 1..=theta_steps {
        let theta = theta_step_size * i as f32;
        for j in 0..phi_steps {
            let phi = phi_step_size * j as f32;
            vertices.push(sph_to_cart(theta, phi));
        }
    }

    let offset = 1;

    // Generate indices top strip
    for i in 0..phi_steps {
        let i1 = i + offset;
        let i2 = i1 % phi_steps + offset;
        indices.push(UVec2::new(0, i1));
        indices.push(UVec2::new(i1, i2));
    }

    // Generate indices for the rest of the cells, each cell has max 2 triangles
    for i in 1..theta_steps {
        for j in 0..phi_steps {
            let i0 = (i - 1) * phi_steps + j + offset;
            let i1 = i * phi_steps + j + offset;
            let i2 = i * phi_steps + (j + 1) % phi_steps + offset;
            let i3 = (i - 1) * phi_steps + (j + 1) % phi_steps + offset;
            indices.push(UVec2::new(i0, i1));
            indices.push(UVec2::new(i1, i2));
            indices.push(UVec2::new(i2, i3));
            indices.push(UVec2::new(i3, i0));
        }
    }

    (vertices, indices)
}

/// Calculates the size of a buffer that is aligned to the given alignment.
pub fn calc_aligned_size(size: u32, alignment: u32) -> u32 {
    let mask = alignment - 1;
    (size + mask) & !mask
}

#[cfg(test)]
mod tests {
    use crate::{
        math::{
            cart_to_sph, circular_angle_dist, madd, msub, nmadd, nmsub, rcp_f32, rsqrt,
            solve_quadratic, sph_to_cart, ulp_eq, QuadraticSolution, MACHINE_EPSILON_F32,
        },
        units::{degrees, radians},
    };
    use glam::Vec3;

    #[test]
    fn test_ulp_eq() {
        assert!(ulp_eq(0.0, 0.0));
        assert!(ulp_eq(1.0, 1.0 + MACHINE_EPSILON_F32));
        assert!(ulp_eq(1.0, 1.0 + 1e-7 * 0.5));
        assert!(ulp_eq(1.0, 1.0 - 1e-7 * 0.5));
        assert!(!ulp_eq(1.0, 1.0 + 1e-6));
        assert!(!ulp_eq(1.0, 1.0 - 1e-6));
    }

    #[test]
    fn test_circular_angle_distance() {
        let a = degrees!(0.0);
        for i in 0..=360 {
            let b = degrees!(i as f32);
            if (a - b).abs() < degrees!(180.0) {
                assert!(ulp_eq(circular_angle_dist(a, b).value, (a - b).abs().value));
            } else {
                assert!(ulp_eq(
                    circular_angle_dist(a, b).value,
                    (degrees!(360.0) - (a - b).abs()).value
                ));
            }
        }
    }

    // TODO: improve accuracy
    #[test]
    fn spherical_cartesian_conversion() {
        println!("{:?}", sph_to_cart(radians!(0.0), radians!(0.0)));
        println!("{:?}", cart_to_sph(Vec3::new(0.0, 1.0, 0.0)));

        let r = 1.0;
        let zenith = radians!(0.0);
        let azimuth = radians!(0.0);
        let v = sph_to_cart(zenith, azimuth) * r;
        let sph3 = cart_to_sph(v);
        assert!(ulp_eq(r, sph3.rho));
        assert!(ulp_eq(zenith.value, sph3.theta.value));
        assert!(ulp_eq(azimuth.value, sph3.phi.value));

        let r = 2.0;
        let zenith = degrees!(45.0).into();
        let azimuth = degrees!(120.0).into();
        let v = sph_to_cart(zenith, azimuth) * r;
        let sph3 = cart_to_sph(v);
        assert!(ulp_eq(r, sph3.rho));
        assert!(ulp_eq(zenith.value, sph3.theta.value));
        assert!(ulp_eq(azimuth.value, sph3.phi.value));
    }

    #[test]
    fn test_rcp() {
        assert!(ulp_eq(rcp_f32(1.0), 1.0));
        assert!(ulp_eq(rcp_f32(2.0), 0.5));
        assert!(ulp_eq(rcp_f32(4.0), 0.25));
        assert!(ulp_eq(rcp_f32(8.0), 0.125));
        assert!(ulp_eq(rcp_f32(16.0), 0.0625));
        assert!(ulp_eq(rcp_f32(32.0), 0.03125));
        assert!(ulp_eq(rcp_f32(64.0), 0.015625));
        assert!(ulp_eq(rcp_f32(128.0), 0.0078125));
        assert!(ulp_eq(rcp_f32(256.0), 0.00390625));
        assert!(ulp_eq(rcp_f32(512.0), 0.001953125));
        assert!(ulp_eq(rcp_f32(1024.0), 0.0009765625));
        assert!(ulp_eq(rcp_f32(2048.0), 0.00048828125));
        assert!(ulp_eq(rcp_f32(4096.0), 0.000244140625));
        assert!(ulp_eq(rcp_f32(8192.0), 0.0001220703125));
        assert!(ulp_eq(rcp_f32(16384.0), 6.103515625e-05));
        assert!(ulp_eq(rcp_f32(32768.0), 3.0517578125e-05));
        assert!(ulp_eq(rcp_f32(65536.0), 1.52587890625e-05));
        assert!(ulp_eq(rcp_f32(131072.0), 7.62939453125e-06));
        assert!(ulp_eq(rcp_f32(262144.0), 3.814697265625e-06));
        assert!(ulp_eq(rcp_f32(524288.0), 1.9073486328125e-06));
        assert!(ulp_eq(rcp_f32(1048576.0), 9.5367431640625e-07));
        assert!(ulp_eq(rcp_f32(2097152.0), 4.76837158203125e-07));
        assert!(ulp_eq(rcp_f32(4194304.0), 2.384185791015625e-07));
        assert!(ulp_eq(rcp_f32(8388608.0), 1.1920928955078125e-07));
        assert!(ulp_eq(rcp_f32(3.0), 1.0 / 3.0));
        assert_eq!(1.0 / -0.0, rcp_f32(-0.0));
        assert_eq!(1.0 / 0.0, rcp_f32(0.0));
        assert_eq!(rcp_f32(0.0), f32::INFINITY);
    }

    #[test]
    fn test_rcpf64() {
        use super::rcp_f64;
        assert!(ulp_eq(rcp_f64(1.0) as f32, 1.0));
        assert!(ulp_eq(rcp_f64(2.0) as f32, 0.5));
        assert!(ulp_eq(rcp_f64(4.0) as f32, 0.25));
        assert_eq!(rcp_f64(0.0), f64::INFINITY);
        assert_eq!(1.0 / -0.0, rcp_f64(-0.0));
        assert_eq!(1.0 / 0.0, rcp_f64(0.0));
    }

    #[test]
    fn test_quadratic() {
        assert_eq!(solve_quadratic(1.0, 0.0, 0.0), QuadraticSolution::One(0.0));
        assert_eq!(solve_quadratic(1.0, 0.0, 1.0), QuadraticSolution::None);
        assert_eq!(
            solve_quadratic(2.0, 5.0, 3.0),
            QuadraticSolution::Two(-1.5, -1.0)
        );
        assert_eq!(
            solve_quadratic(5.0, 6.0, 1.0),
            QuadraticSolution::Two(-1.0, -0.2)
        );
        assert_eq!(
            solve_quadratic(-2.0, 2.0, 1.0),
            QuadraticSolution::Two(-0.3660254, 1.3660254)
        );
    }

    #[test]
    fn test_rsqrt() {
        assert!(ulp_eq(rsqrt(1.0), 1.0));
        assert!(ulp_eq(rsqrt(4.0), 0.5));
        assert!(ulp_eq(rsqrt(8.0), 0.35355338));
        assert!(ulp_eq(rsqrt(9.0), 0.33333334));
        assert!(ulp_eq(rsqrt(16.0), 0.25));
        assert!(ulp_eq(rsqrt(64.0), 0.125));
        assert!(ulp_eq(rsqrt(256.0), 0.0625));
        assert!(ulp_eq(rsqrt(1024.0), 0.03125));
        assert!(ulp_eq(rsqrt(4096.0), 0.015625));
        assert!(ulp_eq(rsqrt(16384.0), 0.0078125));
        assert!(ulp_eq(rsqrt(65536.0), 0.00390625));
        assert!(ulp_eq(rsqrt(262144.0), 0.001953125));
        println!("{:.20} - {:.20}", rsqrt(3.0), rcp_f32(3.0f32.sqrt()));
    }

    #[test]
    fn test_msub() {
        assert_eq!(msub(1.0, 2.0, 3.0), -1.0);
        assert_eq!(msub(2.0, 3.0, 4.0), 2.0);
        assert_eq!(msub(3.0, 4.0, 5.0), 7.0);
        assert_eq!(msub(4.0, 5.0, 6.0), 14.0);
    }

    #[test]
    fn test_madd() {
        assert_eq!(madd(1.0, 2.0, 3.0), 5.0);
        assert_eq!(madd(2.0, 4.0, 6.0), 14.0);
        assert_eq!(madd(3.0, 6.0, 9.0), 27.0);
        assert_eq!(madd(4.0, 8.0, 12.0), 44.0);
    }

    #[test]
    fn test_nmsub() {
        assert_eq!(nmsub(1.0, 2.0, 3.0), -5.0);
        assert_eq!(nmsub(2.0, 4.0, 6.0), -14.0);
        assert_eq!(nmsub(3.0, 6.0, 9.0), -27.0);
        assert_eq!(nmsub(4.0, 8.0, 12.0), -44.0);
    }

    #[test]
    fn test_nmadd() {
        assert_eq!(nmadd(1.0, 2.0, 3.0), 1.0);
        assert_eq!(nmadd(2.0, 4.0, 6.0), -2.0);
        assert_eq!(nmadd(3.0, 6.0, 9.0), -9.0);
        assert_eq!(nmadd(4.0, 8.0, 12.0), -20.0);
    }
}
