/// Bumps a floating-point value up to the next representable value.
#[inline]
pub fn next_f32_up(f: f32) -> f32 {
    if f.is_infinite() && f > 0.0 {
        f
    } else if f == -0.0 {
        0.0
    } else {
        let bits = f.to_bits();
        if f >= 0.0 {
            f32::from_bits(bits + 1)
        } else {
            f32::from_bits(bits - 1)
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
        let bits = f.to_bits();
        if f > 0.0 {
            f32::from_bits(bits - 1)
        } else {
            f32::from_bits(bits + 1)
        }
    }
}
