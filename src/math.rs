use std::arch::x86_64::{_mm_cvtss_f32, _mm_mul_ss, _mm_sub_ss};
use cfg_if::cfg_if;
pub use glam::*;

pub const IDENTITY_MAT4: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Returns the reciprocal of the given value.
#[inline(always)]
pub fn rcp(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_rcp_ss, _mm_set_ss};
        unsafe {
            let a = _mm_set_ss(x);

            let r = {
                cfg_if! {
                    if #[cfg(target_feature = "avx512vl")] {
                        _mm_rcp14_ss(_mm_set_ss(0.0), a)
                    } else {
                        _mm_rcp_ss(a)
                    }
                }
            };

            cfg_if! {
                if #[cfg(target_feature = "avx512vl")] {
                    _mm_cvtss_f32(_mm_mul_ss(r,_mm_fnmadd_ss(r, a, _mm_set_ss(2.0))))
                } else {
                    _mm_cvtss_f32(_mm_mul_ss(r,_mm_sub_ss(_mm_set_ss(2.0), _mm_mul_ss(r, a))))
                }
            }
        }
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
}
