use crate::array::packet::masks::MaskElement;

mod masks;

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

pub unsafe trait SimdElement: Sealed + Copy {
    /// The mask element type corresponding to this element type.
    type Mask: MaskElement;
}

/// SIMD packet of `N` elements of a type `T`.
///
/// This is a wrapper around the SIMD type that provides a more ergonomic API.
pub struct Packet<T, const N: usize>(PacketLaneCount<T, N>::SimdType)
where
    T: SimdElement,
    PacketLaneCount<T, N>: SupportedPacketLaneCount;

// copy + clone

pub trait SupportedPacketLaneCount {
    type SimdType;
}

pub struct PacketLaneCount<T, const N: usize>;

// impl SupportedPacketLaneCount for PacketLaneCount<f32, 4> {
//     type SimdType = __m128
// }

// pub trait StorageSelect<T, const N: usize> {
//     type DataType;
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_packet_creation() {
//         let packet: Packet<f32, 2, f32> = Packet([1.0, 2.0]);
//         assert_eq!(packet.0, [1.0, 2.0]);
//
//         let packet: Packet<f32, 1, f32> = Packet(1.0);
//         assert_eq!(packet.0, 1.0);
//     }
// }
