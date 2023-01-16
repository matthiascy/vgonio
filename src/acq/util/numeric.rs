/// Interprets a floating-point value in terms of its constituent bits.
#[inline]
pub fn f32_to_bits(f: f32) -> u32 { unsafe { f.to_bits() } }

/// Interprets a integer as if its bits constituted a floating-point value.
#[inline]
pub fn bits_to_f32(b: u32) -> f32 { unsafe { f32::from_bits(b) } }

/// Bumps a floating-point value up to the next representable value.
#[inline]
pub fn next_f32_up(f: f32) -> f32 {
    if f.is_infinite() && f > 0.0 {
        f
    } else if f == -0.0 {
        0.0
    } else {
        let bits = f32_to_bits(f);
        if f >= 0.0 {
            bits_to_f32(bits + 1)
        } else {
            bits_to_f32(bits - 1)
        }
    }
}

/// Bumps a floating-point value down to the next representable value.
#[inline]
pub fn next_f32_down(f: f32) -> f32 {
    if f.is_infinite() && f < 0.0 {
        f
    } else if f == -0.0 {
        0.0
    } else {
        let bits = f32_to_bits(f);
        if f > 0.0 {
            bits_to_f32(bits - 1)
        } else {
            bits_to_f32(bits + 1)
        }
    }
}
