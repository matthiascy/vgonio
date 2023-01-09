use rand_distr::num_traits::abs;
use std::path::{Path, PathBuf};

/// Machine epsilon for double precision floating point numbers.
pub const MACHINE_EPSILON_F64: f64 = f64::EPSILON * 0.5;

/// Machine epsilon for single precision floating point numbers.
pub const MACHINE_EPSILON_F32: f32 = f32::EPSILON * 0.5;

/// Compute the conservative bounding of $(1 \pm \epsilon_{m})^n$ for a given
/// $n$.
pub const fn gamma(n: u32) -> f32 {
    (n as f32 * MACHINE_EPSILON_F32) / (1.0 - n as f32 * MACHINE_EPSILON_F32)
}

/// Equality test of two floating point numbers.
///
/// # Arguments
///
/// * `a`: The first number.
/// * `b`: The second number.
///
/// returns: bool
pub fn ulp_eq(a: f32, b: f32) -> bool {
    let diff = abs(a - b);
    let a_abs = abs(a);
    let b_abs = abs(b);
    if a == b {
        true
    } else if a == 0.0 || b == 0.0 || a_abs + b_abs < f32::MIN_POSITIVE {
        diff < (f32::MIN_POSITIVE * f32::EPSILON)
    } else {
        (diff / f32::min(a_abs + b_abs, f32::MAX)) < f32::EPSILON
    }
}

#[test]
fn test_approx_eq() {
    assert!(ulp_eq(0.0, 0.0));
    assert!(ulp_eq(1.0, 1.0 + MACHINE_EPSILON_F32));
    assert!(ulp_eq(1.0, 1.0 + 1e-7 * 0.5));
    assert!(ulp_eq(1.0, 1.0 - 1e-7 * 0.5));
    assert!(!ulp_eq(1.0, 1.0 + 1e-6));
    assert!(!ulp_eq(1.0, 1.0 - 1e-6));
}
