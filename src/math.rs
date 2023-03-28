use crate::{
    common::{ulp_eq, Handedness, SphericalCoord},
    units::{degrees, radians, Radians},
};
pub use glam::*;

pub const IDENTITY_MAT4: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Conversion from spherical coordinate system to cartesian coordinate system.
///
/// # Arguments
///
/// * `r` - radius
/// * `zenith` - polar angle
/// * `azimuth` - azimuthal angle
/// * `handedness` - handedness of the cartesian coordinate system
pub fn spherical_to_cartesian(
    r: f32,
    zenith: Radians,
    azimuth: Radians,
    handedness: Handedness,
) -> Vec3 {
    let a = r * zenith.sin() * azimuth.cos();
    let b = r * zenith.sin() * azimuth.sin();
    let c = r * zenith.cos();

    match handedness {
        Handedness::RightHandedZUp => Vec3::new(a, b, c),
        Handedness::RightHandedYUp => Vec3::new(a, c, b),
    }
}

/// Conversion from cartesian coordinate system to spherical coordinate system.
///
/// # Arguments
///
/// * `v` - vector in cartesian coordinate system
/// * `handedness` - handedness of the cartesian coordinate system
///
/// # Returns
///
/// * `r` - radius
/// * `zenith` - polar angle
/// * `azimuth` - azimuthal angle
pub fn cartesian_to_spherical(v: Vec3, handedness: Handedness) -> SphericalCoord {
    let radius = v.length();
    let (zenith, azimuth) = match handedness {
        Handedness::RightHandedZUp => {
            let zenith = radians!((v.z * rcp(radius)).acos());
            let azimuth = radians!(v.y.atan2(v.x));
            (zenith, azimuth)
        }
        Handedness::RightHandedYUp => {
            let zenith = radians!((v.y * rcp(radius)).acos());
            let azimuth = radians!(v.z.atan2(v.x));
            (zenith, azimuth)
        }
    };
    SphericalCoord {
        radius,
        zenith,
        azimuth,
    }
}

#[test]
fn spherical_cartesian_conversion() {
    let r = 1.0;
    let zenith = radians!(0.0);
    let azimuth = radians!(0.0);
    let v = spherical_to_cartesian(r, zenith, azimuth, Handedness::RightHandedZUp);
    let sph = cartesian_to_spherical(v, Handedness::RightHandedZUp);
    assert!(ulp_eq(r, sph.radius));
    assert!(ulp_eq(zenith.value, sph.zenith.value));
    assert!(ulp_eq(azimuth.value, sph.azimuth.value));

    let r = 2.0;
    let zenith = degrees!(45.0).into();
    let azimuth = degrees!(120.0).into();
    let v = spherical_to_cartesian(r, zenith, azimuth, Handedness::RightHandedZUp);
    let sph = cartesian_to_spherical(v, Handedness::RightHandedZUp);
    assert!(ulp_eq(r, sph.radius));
    assert!(ulp_eq(zenith.value, sph.zenith.value));
    assert!(ulp_eq(azimuth.value, sph.azimuth.value));

    let r = 1.5;
    let zenith = degrees!(60.0).in_radians();
    let azimuth = degrees!(90.0).in_radians();
    let v = spherical_to_cartesian(r, zenith, azimuth, Handedness::RightHandedYUp);
    let sph = cartesian_to_spherical(v, Handedness::RightHandedYUp);
    assert!(ulp_eq(r, sph.radius));
    assert!(ulp_eq(zenith.value, sph.zenith.value));
    assert!(ulp_eq(azimuth.value, sph.azimuth.value));
}

/// Returns the accurate reciprocal of the given value.
///
/// Newton-Raphson iteration is used to compute the reciprocal.
#[inline(always)]
pub fn rcp(x: f32) -> f32 {
    if cfg!(target_arch = "x86_64") {
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

#[test]
fn test_rcp() {
    assert_eq!(rcp(1.0), 1.0);
    assert_eq!(rcp(2.0), 0.5);
    assert_eq!(rcp(4.0), 0.25);
    assert_eq!(rcp(8.0), 0.125);
    assert_eq!(rcp(16.0), 0.0625);
    assert_eq!(rcp(32.0), 0.03125);
    assert_eq!(rcp(64.0), 0.015625);
    assert_eq!(rcp(128.0), 0.0078125);
    assert_eq!(rcp(256.0), 0.00390625);
    assert_eq!(rcp(512.0), 0.001953125);
    assert_eq!(rcp(1024.0), 0.0009765625);
    assert_eq!(rcp(2048.0), 0.00048828125);
    assert_eq!(rcp(4096.0), 0.000244140625);
    assert_eq!(rcp(8192.0), 0.0001220703125);
    assert_eq!(rcp(16384.0), 6.103515625e-05);
    assert_eq!(rcp(32768.0), 3.0517578125e-05);
    assert_eq!(rcp(65536.0), 1.52587890625e-05);
    assert_eq!(rcp(131072.0), 7.62939453125e-06);
    assert_eq!(rcp(262144.0), 3.814697265625e-06);
    assert_eq!(rcp(524288.0), 1.9073486328125e-06);
    assert_eq!(rcp(1048576.0), 9.5367431640625e-07);
    assert_eq!(rcp(2097152.0), 4.76837158203125e-07);
    assert_eq!(rcp(4194304.0), 2.384185791015625e-07);
    assert_eq!(rcp(8388608.0), 1.1920928955078125e-07);

    assert_eq!(rcp(3.0), 1.0 / 3.0);
}

/// Returns the square of the given value.
#[inline(always)]
pub fn sqr(x: f32) -> f32 { x * x }

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

#[test]
fn test_msub() {
    assert_eq!(msub(1.0, 2.0, 3.0), -1.0);
    assert_eq!(msub(2.0, 3.0, 4.0), 2.0);
    assert_eq!(msub(3.0, 4.0, 5.0), 7.0);
    assert_eq!(msub(4.0, 5.0, 6.0), 14.0);
}

/// Returns the fused multiply-add of the given values.
///
/// This is equivalent to `a * b + c`. However, this function may fall back to
/// a non-fused multiply-add on some platforms.
#[inline(always)]
pub fn madd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(target_arch = "x86_64") && cfg!(target_feature = "fma") {
        use std::arch::x86_64::{_mm_cvtss_f32, _mm_fmadd_ss, _mm_set_ss};
        unsafe { _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c))) }
    } else {
        a * b + c
    }
}

#[test]
fn test_madd() {
    assert_eq!(madd(1.0, 2.0, 3.0), 5.0);
    assert_eq!(madd(2.0, 4.0, 6.0), 14.0);
    assert_eq!(madd(3.0, 6.0, 9.0), 27.0);
    assert_eq!(madd(4.0, 8.0, 12.0), 44.0);
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

#[test]
fn test_nmsub() {
    assert_eq!(nmsub(1.0, 2.0, 3.0), -5.0);
    assert_eq!(nmsub(2.0, 4.0, 6.0), -14.0);
    assert_eq!(nmsub(3.0, 6.0, 9.0), -27.0);
    assert_eq!(nmsub(4.0, 8.0, 12.0), -44.0);
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

#[test]
fn test_nmadd() {
    assert_eq!(nmadd(1.0, 2.0, 3.0), 1.0);
    assert_eq!(nmadd(2.0, 4.0, 6.0), -2.0);
    assert_eq!(nmadd(3.0, 6.0, 9.0), -9.0);
    assert_eq!(nmadd(4.0, 8.0, 12.0), -20.0);
}

/// Quadratic equation can have 0, 1 or 2 real solutions.
#[derive(Debug, Copy, Clone)]
pub enum QuadraticSolution {
    None,
    One(f32),
    Two(f32, f32),
}

impl PartialEq for QuadraticSolution {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (QuadraticSolution::None, QuadraticSolution::None) => true,
            (QuadraticSolution::One(x), QuadraticSolution::One(y)) => x == y,
            (QuadraticSolution::Two(x1, x2), QuadraticSolution::Two(y1, y2)) => {
                (x1 == y1 && x2 == y2) || (x1 == y2 && x2 == y1)
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
    let rcp_2a = 0.5 * rcp(a);
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

#[test]
fn test_quadratic() {
    assert_eq!(solve_quadratic(1.0, 0.0, 0.0), QuadraticSolution::One(0.0));
    assert_eq!(solve_quadratic(1.0, 0.0, 1.0), QuadraticSolution::None);
    assert_eq!(
        solve_quadratic(2.0, 5.0, 3.0),
        QuadraticSolution::Two(-1.0, -1.5)
    );
    assert_eq!(
        solve_quadratic(5.0, 6.0, 1.0),
        QuadraticSolution::Two(-0.2, -1.0)
    );
    assert_eq!(
        solve_quadratic(-2.0, 2.0, 1.0),
        QuadraticSolution::Two(1.3660254, -0.3660254)
    );
}
