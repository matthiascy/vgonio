mod decay;

use crate::{array::traits::decay::Decay, utils::IsFalse};

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

/// Tells whether a type is arithmetic.
#[const_trait]
pub trait IsArithmetic {
    // NOTE: not using const value here because of `specialization` feature being
    // unsound.
    fn is_arithmetic() -> bool;
}

impl<T> const IsArithmetic for T {
    default fn is_arithmetic() -> bool { false }
}

macro_rules! impl_is_arithmetic {
    ($($t:ty),*) => {
        $(
            impl const IsArithmetic for $t {
                fn is_arithmetic() -> bool { true }
            }
        )*
    };
}

impl_is_arithmetic!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

/// Tells whether a type is integral.
#[const_trait]
pub trait IsIntegral<T> {
    fn is_integral() -> bool;
}

impl<T> const IsIntegral<T> for T {
    default fn is_integral() -> bool { false }
}

macro_rules! impl_is_integral {
    ($($t:ty),*) => {
        $(
            impl const IsIntegral<$t> for $t {
                fn is_integral() -> bool { true }
            }
        )*
    };
}

impl_is_integral!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

/// Tells whether a type is a floating point.
#[const_trait]
pub trait IsFloat<T> {
    fn is_float() -> bool;
}

impl<T> const IsFloat<T> for T {
    default fn is_float() -> bool { false }
}

macro_rules! impl_is_float {
    ($($t:ty),*) => {
        $(
            impl const IsFloat<$t> for $t {
                fn is_float() -> bool { true }
            }
        )*
    };
}

impl_is_float!(f32, f64);
