use crate::{
    ulp_eq,
    units::{radians, Radians},
    Handedness, SphericalCoord,
};
use cfg_if::cfg_if;
pub use glam::*;

/// Trait for converting from one primitive numeric type to another.
#[const_trait]
pub trait NumericCast<T> {
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

impl_as_primitive!(f32 as f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(f64 as f32, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(i32 as f32, f64, u32, i64, u64, i128, u128);
impl_as_primitive!(u32 as f32, f64, i32, i64, u64, i128, u128);
impl_as_primitive!(i64 as f32, f64, i32, u32, u64, i128, u128);
impl_as_primitive!(u64 as f32, f64, i32, u32, i64, i128, u128);
impl_as_primitive!(i128 as f32, f64, i32, u32, i64, u64, u128);
impl_as_primitive!(u128 as f32, f64, i32, u32, i64, u64, i128);
impl_as_primitive!(u8 as f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(i8 as f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(u16 as f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(i16 as f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(usize as f32, f64, i32, u32, i64, u64, i128, u128);
impl_as_primitive!(isize as f32, f64, i32, u32, i64, u64, i128, u128);

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
pub fn cartesian_to_spherical(v: Vec3, radius: f32, handedness: Handedness) -> SphericalCoord {
    let (zenith, azimuth) = match handedness {
        Handedness::RightHandedZUp => (
            radians!((v.z * rcp(radius)).acos()),
            radians!(v.y.atan2(v.x)),
        ),
        Handedness::RightHandedYUp => (
            radians!((v.y * rcp(radius)).acos()),
            radians!(v.z.atan2(v.x)),
        ),
    };

    SphericalCoord {
        radius,
        zenith: if zenith < radians!(0.0) {
            zenith + Radians::PI
        } else {
            zenith
        },
        azimuth: if azimuth < radians!(0.0) {
            azimuth + Radians::TAU
        } else {
            azimuth
        },
    }
}

// TODO: improve accuracy
#[test]
fn spherical_cartesian_conversion() {
    use crate::{ulp_eq, units::degrees};

    println!(
        "{:?}",
        spherical_to_cartesian(
            1.0,
            radians!(0.0),
            radians!(0.0),
            Handedness::RightHandedYUp
        )
    );
    println!(
        "{:?}",
        cartesian_to_spherical(Vec3::new(0.0, 1.0, 0.0), 1.0, Handedness::RightHandedYUp)
    );

    let r = 1.0;
    let zenith = radians!(0.0);
    let azimuth = radians!(0.0);
    let v = spherical_to_cartesian(r, zenith, azimuth, Handedness::RightHandedZUp);
    let sph = cartesian_to_spherical(v, r, Handedness::RightHandedZUp);
    assert!(ulp_eq(r, sph.radius));
    println!("{:.20} {:.20}", zenith.value, sph.zenith.value);
    assert!(ulp_eq(zenith.value, sph.zenith.value));
    assert!(ulp_eq(azimuth.value, sph.azimuth.value));

    let r = 2.0;
    let zenith = degrees!(45.0).into();
    let azimuth = degrees!(120.0).into();
    let v = spherical_to_cartesian(r, zenith, azimuth, Handedness::RightHandedZUp);
    let sph = cartesian_to_spherical(v, r, Handedness::RightHandedZUp);
    assert!(ulp_eq(r, sph.radius));
    assert!(ulp_eq(zenith.value, sph.zenith.value));
    assert!(ulp_eq(azimuth.value, sph.azimuth.value));

    let r = 1.5;
    let zenith = degrees!(60.0).in_radians();
    let azimuth = degrees!(90.0).in_radians();
    let v = spherical_to_cartesian(r, zenith, azimuth, Handedness::RightHandedYUp);
    let sph = cartesian_to_spherical(v, 1.5, Handedness::RightHandedYUp);
    assert!(ulp_eq(r, sph.radius));
    assert!(ulp_eq(zenith.value, sph.zenith.value));
    assert!(ulp_eq(azimuth.value, sph.azimuth.value));
}

/// Returns the accurate reciprocal of the given value.
///
/// Newton-Raphson iteration is used to compute the reciprocal.
#[inline(always)]
pub fn rcp(x: f32) -> f32 {
    // Intel' intrinsic will give us NaN if x is 0.0 or -0.0
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

#[test]
fn test_rcp() {
    use crate::ulp_eq;
    assert!(ulp_eq(rcp(1.0), 1.0));
    assert!(ulp_eq(rcp(2.0), 0.5));
    assert!(ulp_eq(rcp(4.0), 0.25));
    assert!(ulp_eq(rcp(8.0), 0.125));
    assert!(ulp_eq(rcp(16.0), 0.0625));
    assert!(ulp_eq(rcp(32.0), 0.03125));
    assert!(ulp_eq(rcp(64.0), 0.015625));
    assert!(ulp_eq(rcp(128.0), 0.0078125));
    assert!(ulp_eq(rcp(256.0), 0.00390625));
    assert!(ulp_eq(rcp(512.0), 0.001953125));
    assert!(ulp_eq(rcp(1024.0), 0.0009765625));
    assert!(ulp_eq(rcp(2048.0), 0.00048828125));
    assert!(ulp_eq(rcp(4096.0), 0.000244140625));
    assert!(ulp_eq(rcp(8192.0), 0.0001220703125));
    assert!(ulp_eq(rcp(16384.0), 6.103515625e-05));
    assert!(ulp_eq(rcp(32768.0), 3.0517578125e-05));
    assert!(ulp_eq(rcp(65536.0), 1.52587890625e-05));
    assert!(ulp_eq(rcp(131072.0), 7.62939453125e-06));
    assert!(ulp_eq(rcp(262144.0), 3.814697265625e-06));
    assert!(ulp_eq(rcp(524288.0), 1.9073486328125e-06));
    assert!(ulp_eq(rcp(1048576.0), 9.5367431640625e-07));
    assert!(ulp_eq(rcp(2097152.0), 4.76837158203125e-07));
    assert!(ulp_eq(rcp(4194304.0), 2.384185791015625e-07));
    assert!(ulp_eq(rcp(8388608.0), 1.1920928955078125e-07));
    assert!(ulp_eq(rcp(3.0), 1.0 / 3.0));
    assert_eq!(1.0 / -0.0, rcp(-0.0));
    assert_eq!(1.0 / 0.0, rcp(0.0));
    assert_eq!(rcp(0.0), f32::INFINITY);
}

/// Returns the square of the given value.
#[inline(always)]
pub fn sqr(x: f32) -> f32 { x * x }

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
    println!("{:.20} - {:.20}", rsqrt(3.0), rcp(3.0f32.sqrt()));
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
    cfg_if! {
        if #[cfg(target_arch = "x86_64")] {
            use std::arch::x86_64::{_mm_cvtss_f32, _mm_fmadd_ss, _mm_set_ss};
            unsafe { _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c))) }
        } else {
            a * b + c
        }
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
pub fn generate_parametric_hemisphere_triangles(
    theta_steps: u32,
    phi_steps: u32,
) -> (Vec<Vec3>, Vec<UVec3>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let theta_step_size = Radians::HALF_PI / theta_steps as f32;
    let phi_step_size = Radians::TWO_PI / phi_steps as f32;

    // Generate top vertex
    vertices.push(spherical_to_cartesian(
        1.0,
        Radians::ZERO,
        Radians::ZERO,
        Handedness::RightHandedYUp,
    ));

    for i in 1..=theta_steps {
        let theta = theta_step_size * i as f32;
        for j in 0..phi_steps {
            let phi = phi_step_size * j as f32;
            vertices.push(spherical_to_cartesian(
                1.0,
                theta,
                phi,
                Handedness::RightHandedYUp,
            ));
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
    vertices.push(spherical_to_cartesian(
        1.0,
        Radians::ZERO,
        Radians::ZERO,
        Handedness::RightHandedYUp,
    ));

    for i in 1..=theta_steps {
        let theta = theta_step_size * i as f32;
        for j in 0..phi_steps {
            let phi = phi_step_size * j as f32;
            vertices.push(spherical_to_cartesian(
                1.0,
                theta,
                phi,
                Handedness::RightHandedYUp,
            ));
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
