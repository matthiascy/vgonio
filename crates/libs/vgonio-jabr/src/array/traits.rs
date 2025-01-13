mod decay;

use crate::array::traits::decay::Decay;

/// Base trait for n-dimensional arrays.
pub trait NdArray<T> {
    /// Type of the elements in the array.
    type Elem = T;
    /// Scalar data type all the way to the lowest level.
    type Scalar = <T as Scalar<T>>::Type;
}

pub trait Scalar<T> {
    type Type;
}

impl<T> Scalar<T> for T {
    type Type = <T as Decay>::Output;
}

// TODO: #[const_trait], blocked by effects feature and new const traits
// implementation See: https://github.com/rust-lang/rust/issues/110395
/// Tells whether a type is arithmetic.
pub trait IsArithmetic {
    // NOTE: not using const value here because of `specialization` feature being
    // unsound.
    fn is_arithmetic() -> bool;
}

impl<T> IsArithmetic for T {
    default fn is_arithmetic() -> bool { false }
}

macro_rules! impl_is_arithmetic {
    ($($t:ty),*) => {
        $(
            impl IsArithmetic for $t {
                fn is_arithmetic() -> bool { true }
            }
        )*
    };
}

impl_is_arithmetic!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

// TODO: #[const_trait], blocked by effects feature and new const traits
// implementation See: https://github.com/rust-lang/rust/issues/110395
/// Tells whether a type is integral.
pub trait IsIntegral<T> {
    fn is_integral() -> bool;
}

impl<T> IsIntegral<T> for T {
    default fn is_integral() -> bool { false }
}

macro_rules! impl_is_integral {
    ($($t:ty),*) => {
        $(
            impl IsIntegral<$t> for $t {
                fn is_integral() -> bool { true }
            }
        )*
    };
}

impl_is_integral!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

// TODO: #[const_trait], blocked by effects feature and new const traits
// implementation See: https://github.com/rust-lang/rust/issues/110395
/// Tells whether a type is a floating point.
pub trait IsFloat<T> {
    fn is_float() -> bool;
}

impl<T> IsFloat<T> for T {
    default fn is_float() -> bool { false }
}

macro_rules! impl_is_float {
    ($($t:ty),*) => {
        $(
            impl IsFloat<$t> for $t {
                fn is_float() -> bool { true }
            }
        )*
    };
}

impl_is_float!(f32, f64);
