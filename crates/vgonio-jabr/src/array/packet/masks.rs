mod sealed {
    pub trait Sealed {}
}
use crate::array::packet::{PacketLaneCount, SimdElement, SupportedPacketLaneCount};
use sealed::Sealed;

/// A mask element that can be used to create a mask for SIMD operations.
pub unsafe trait MaskElement: SimdElement<Mask = Self> + Sealed {
    type Unsigned: SimdElement;

    fn eq(self, other: Self) -> bool;
    fn to_usize(self) -> usize;
    fn max_unsigned() -> u64;

    const TRUE: Self;
    const FALSE: Self;
}

macro_rules! impl_mask_element {
    ($($t:ty, $ut:ty);*) => {
        $(
            unsafe impl MaskElement for $t {
                type Unsigned = $ut;

                #[inline]
                fn eq(self, other: Self) -> bool { self == other }

                #[inline]
                fn to_usize(self) -> usize { self as usize }
                #[inline]
                fn max_unsigned() -> u64 { <$ut>::MAX as u64 }

                const TRUE: Self = -1;
                const FALSE: Self = 0;
            }
        )*
    };
}

impl_mask_element!(i8, u8; i16, u16; i32, u32; i64, u64; i128, u128);

pub struct Mask<T, const N: usize>(PacketLaneCount<T, N>::SimdType)
where
    T: MaskElement,
    PacketLaneCount<T, N>: SupportedPacketLaneCount;
