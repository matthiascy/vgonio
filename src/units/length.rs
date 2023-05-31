use crate::{error::Error, math::NumericCast, ulp_eq};
use core::fmt::Debug;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    fmt::{Display, Formatter},
    str::FromStr,
};

/// Represents a unit of length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum LengthUnit {
    /// Metres.
    M = 0,
    /// Centimetres.
    CM = 1,
    /// Millimetres.
    MM = 2,
    /// Micrometres.
    UM = 3,
    /// Nanometres.
    NM = 4,
}

impl From<u8> for LengthUnit {
    fn from(value: u8) -> Self {
        debug_assert!(value <= 4, "Invalid length unit: {}", value);
        match value {
            0 => Self::M,
            1 => Self::CM,
            2 => Self::MM,
            3 => Self::UM,
            4 => Self::NM,
            _ => unreachable!(),
        }
    }
}

impl Display for LengthUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LengthUnit::M => {
                write!(f, "m")
            }
            LengthUnit::CM => {
                write!(f, "cm")
            }
            LengthUnit::MM => {
                write!(f, "mm")
            }
            LengthUnit::UM => {
                write!(f, "um")
            }
            LengthUnit::NM => {
                write!(f, "nm")
            }
        }
    }
}

impl LengthUnit {
    /// Returns the conversion factor from the unit to the specified unit.
    pub const fn factor_convert_to<U: LengthMeasurement>(&self) -> f32 {
        match self {
            LengthUnit::M => U::FACTOR_FROM_METRE,
            LengthUnit::CM => U::FACTOR_FROM_CENTIMETRE,
            LengthUnit::MM => U::FACTOR_FROM_MILLIMETRE,
            LengthUnit::UM => U::FACTOR_FROM_MICROMETRE,
            LengthUnit::NM => U::FACTOR_FROM_NANOMETRE,
        }
    }

    /// Returns the conversion factor from the specified unit to the unit.
    pub const fn factor_convert_from<U: LengthMeasurement>(&self) -> f32 {
        match self {
            LengthUnit::M => U::FACTOR_TO_METRE,
            LengthUnit::CM => U::FACTOR_TO_CENTIMETRE,
            LengthUnit::MM => U::FACTOR_TO_MILLIMETRE,
            LengthUnit::UM => U::FACTOR_TO_MICROMETRE,
            LengthUnit::NM => U::FACTOR_TO_NANOMETRE,
        }
    }
}

/// Trait representing a unit of length.
pub trait LengthMeasurement: Debug + Copy + Clone {
    /// The name of the unit.
    const NAME: &'static str;

    /// The SI symbol of the unit.
    const SYMBOL: &'static str;

    /// The factor to convert from meters to the unit.
    const FACTOR_FROM_METRE: f32;

    /// The factor to convert from the unit to meters.
    const FACTOR_TO_METRE: f32 = 1.0 / Self::FACTOR_FROM_METRE;

    /// The factor to convert from centimeters to the unit.
    const FACTOR_FROM_CENTIMETRE: f32;

    /// The factor to convert from the unit to centimeters.
    const FACTOR_TO_CENTIMETRE: f32 = 1.0 / Self::FACTOR_FROM_CENTIMETRE;

    /// The factor to convert from millimeters to the unit.
    const FACTOR_FROM_MILLIMETRE: f32;

    /// The factor to convert from the unit to millimeters.
    const FACTOR_TO_MILLIMETRE: f32 = 1.0 / Self::FACTOR_FROM_MILLIMETRE;

    /// The factor to convert from micrometers to the unit.
    const FACTOR_FROM_MICROMETRE: f32;

    /// The factor to convert from the unit to micrometers.
    const FACTOR_TO_MICROMETRE: f32 = 1.0 / Self::FACTOR_FROM_MICROMETRE;

    /// The factor to convert from nanometers to the unit.
    const FACTOR_FROM_NANOMETRE: f32;

    /// The factor to convert from the unit to nanometers.
    const FACTOR_TO_NANOMETRE: f32 = 1.0 / Self::FACTOR_FROM_NANOMETRE;

    /// The enum value of the unit.
    const UNIT: LengthUnit;
}

/// Meters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UMetre;

/// Centimeters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UCentimetre;

/// Millimeters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UMillimetre;

/// Micrometers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UMicrometre;

/// Nanometers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UNanometre;

impl LengthMeasurement for UMetre {
    const NAME: &'static str = "metre";
    const SYMBOL: &'static str = "m";
    const FACTOR_FROM_METRE: f32 = 1.0;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e-2;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e-3;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e-6;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-9;
    const UNIT: LengthUnit = LengthUnit::M;
}

impl LengthMeasurement for UCentimetre {
    const NAME: &'static str = "centimetre";
    const SYMBOL: &'static str = "cm";
    const FACTOR_FROM_METRE: f32 = 100.0;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e-1;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e-4;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-7;
    const UNIT: LengthUnit = LengthUnit::CM;
}

impl LengthMeasurement for UMillimetre {
    const NAME: &'static str = "millimetre";
    const SYMBOL: &'static str = "mm";
    const FACTOR_FROM_METRE: f32 = 1.0e3;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e1;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e-3;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-6;
    const UNIT: LengthUnit = LengthUnit::MM;
}

impl LengthMeasurement for UMicrometre {
    const NAME: &'static str = "micrometre";
    const SYMBOL: &'static str = "um";
    const FACTOR_FROM_METRE: f32 = 1.0e6;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e4;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e3;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-3;
    const UNIT: LengthUnit = LengthUnit::UM;
}

impl LengthMeasurement for UNanometre {
    const NAME: &'static str = "nanometre";
    const SYMBOL: &'static str = "nm";
    const FACTOR_FROM_METRE: f32 = 1.0e9;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e7;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e6;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e3;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0;
    const UNIT: LengthUnit = LengthUnit::NM;
}

/// Length with a unit.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Length<A: LengthMeasurement> {
    pub(crate) value: f32,
    pub(crate) unit: core::marker::PhantomData<A>,
}

impl<A: LengthMeasurement> Default for Length<A> {
    fn default() -> Self { Self::new(0.0) }
}

impl<A: LengthMeasurement> Debug for Length<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.value, A::SYMBOL)
    }
}

impl<A: LengthMeasurement> Display for Length<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.value, A::SYMBOL)
    }
}

impl<A: LengthMeasurement, B: LengthMeasurement> PartialEq<Length<B>> for Length<A> {
    fn eq(&self, other: &Length<B>) -> bool {
        ulp_eq(
            self.value * A::FACTOR_TO_METRE,
            other.value * B::FACTOR_TO_METRE,
        )
    }
}

impl<A: LengthMeasurement, B: LengthMeasurement> PartialOrd<Length<B>> for Length<A> {
    fn partial_cmp(&self, other: &Length<B>) -> Option<Ordering> {
        let a = self.value * A::FACTOR_TO_METRE;
        let b = other.value * B::FACTOR_TO_METRE;
        a.partial_cmp(&b)
    }
}

impl<A: LengthMeasurement> Length<A> {
    /// Creates a new length.
    pub const fn new(value: f32) -> Self {
        Self {
            value,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the value of the length.
    pub const fn value(&self) -> f32 { self.value }

    /// Returns the value of the length without the unit.
    pub const fn as_f32(&self) -> f32 { self.value }

    super::forward_f32_methods!(
        abs,
        "Returns the absolute value of the length.";
        ceil,
        "Returns the smallest length greater than or equal to `self`.";
        round,
        "Returns the nearest value to `self`. Round half-way cases away from 0.0.";
        trunc,
        "Returns the integer part of `self`. Non-integer numbers are always truncated towards \
         zero.";
        fract,
        "Returns the fractional part of `self`.";
        sqrt,
        "Returns the square root of the length.";
        cbrt,
        "returns the cube root of the length."
    );

    /// Tests if the value of length is not a number.
    pub fn is_nan(&self) -> bool { self.value.is_nan() }

    /// Tests if the two lengths are approximately equal using the absolute
    /// difference.
    pub fn abs_diff_eq<B: LengthMeasurement>(&self, other: &Length<B>, epsilon: f32) -> bool {
        let a = self.value * A::FACTOR_TO_METRE;
        let b = other.value * B::FACTOR_TO_METRE;
        (a - b).abs() <= epsilon
    }

    /// Returns the length in metres.
    #[inline]
    pub const fn in_metres(&self) -> Length<UMetre> { Length::new(self.value * A::FACTOR_TO_METRE) }

    /// Returns the length in centimetres.
    #[inline]
    pub const fn in_centimetres(&self) -> Length<UCentimetre> {
        Length::new(self.value * A::FACTOR_TO_CENTIMETRE)
    }

    /// Returns the length in millimetres.
    #[inline]
    pub const fn in_millimetres(&self) -> Length<UMillimetre> {
        Length::new(self.value * A::FACTOR_TO_MILLIMETRE)
    }

    /// Returns the length in micrometres.
    #[inline]
    pub const fn in_micrometres(&self) -> Length<UMicrometre> {
        Length::new(self.value * A::FACTOR_TO_MICROMETRE)
    }

    /// Returns the length in nanometres.
    #[inline]
    pub const fn in_nanometres(&self) -> Length<UNanometre> {
        Length::new(self.value * A::FACTOR_TO_NANOMETRE)
    }

    // TODO: relative_eq
}

impl<A: LengthMeasurement> From<f32> for Length<A> {
    fn from(value: f32) -> Self { Self::new(value) }
}

impl<'a, A: LengthMeasurement> TryFrom<&'a str> for Length<A> {
    type Error = Error;

    fn try_from(s: &'a str) -> Result<Self, Self::Error> {
        let bytes = s.trim().as_bytes();
        let i = super::findr_first_non_ascii_alphabetic(bytes)
            .ok_or(Error::Any("no unit found in length string".to_string()))?;
        let value = std::str::from_utf8(&bytes[..i])
            .map_err(|_| Error::Any("invalid length string".to_string()))?
            .trim()
            .parse::<f32>()
            .map_err(|_| Error::Any("invalid length value".to_string()))?;
        let unit = std::str::from_utf8(&bytes[i..])
            .map_err(|err| Error::Any(format!("invalid length unit: {err}")))?
            .trim();
        match unit {
            "m" => Ok(Self::new(A::FACTOR_FROM_METRE * value)),
            "cm" => Ok(Self::new(A::FACTOR_FROM_CENTIMETRE * value)),
            "mm" => Ok(Self::new(A::FACTOR_FROM_MILLIMETRE * value)),
            "um" => Ok(Self::new(A::FACTOR_FROM_MICROMETRE * value)),
            "nm" => Ok(Self::new(A::FACTOR_FROM_NANOMETRE * value)),
            _ => Err(Error::Any("invalid length unit".to_string())),
        }
    }
}

impl<A: LengthMeasurement> FromStr for Length<A> {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> { Self::try_from(s) }
}

super::impl_serialization!(Length<A> where A: LengthMeasurement, #[doc = "Customized serialization for the `Length` type."]);

// Customized deserialization for the `Length` type.
impl<'de, A: LengthMeasurement> serde::Deserialize<'de> for Length<A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct LengthVisitor<T>(core::marker::PhantomData<T>);

        impl<'de, T: LengthMeasurement> serde::de::Visitor<'de> for LengthVisitor<T> {
            type Value = Length<T>;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a string containing a number and a unit of length"
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Length::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }

        deserializer.deserialize_str(LengthVisitor::<A>(core::marker::PhantomData))
    }
}

/// Type alias for a length in metres.
pub type Metres = Length<UMetre>;

/// Type alias for a length in centimetres.
pub type Centimetres = Length<UCentimetre>;

/// Type alias for a length in millimetres.
pub type Millimetres = Length<UMillimetre>;

/// Type alias for a length in micrometres.
pub type Micrometres = Length<UMicrometre>;

/// Type alias for a length in nanometres.
pub type Nanometres = Length<UNanometre>;

/// Macro for creating a new length type in metres.
pub macro metres($val:expr) {
    $crate::units::Length::<$crate::units::UMetre>::new($val)
}

/// Macro for creating a new length type in metres.
pub macro m($val:expr) {
    $crate::units::Length::<$crate::units::UMetre>::new($val)
}

/// Macro for creating a new length type in centimetres.
pub macro centimetres($val:expr) {
    $crate::units::Length::<$crate::units::UCentimetre>::new($val)
}

/// Macro for creating a new length type in centimetres.
pub macro cm($val:expr) {
    $crate::units::Length::<$crate::units::UCentimetre>::new($val)
}

/// Macro for creating a new length type in millimetres.
pub macro millimetres($val:expr) {
    $crate::units::Length::<$crate::units::UMillimetre>::new($val)
}

/// Macro for creating a new length type in millimetres.
pub macro mm($val:expr) {
    $crate::units::Length::<$crate::units::UMillimetre>::new($val)
}

/// Macro for creating a new length type in micrometres.
pub macro micrometres($val:expr) {
    $crate::units::Length::<$crate::units::UMicrometre>::new($val)
}

/// Macro for creating a new length type in micrometres.
pub macro um($val:expr) {
    $crate::units::Length::<$crate::units::UMicrometre>::new($val)
}

/// Macro for creating a new length type in nanometres.
pub macro nanometres($val:expr) {
    $crate::units::Length::<$crate::units::UNanometre>::new($val)
}

/// Macro for creating a new length type in nanometres.
pub macro nm($val:expr) {
    $crate::units::Length::<$crate::units::UNanometre>::new($val)
}

macro impl_conversion($from:ident => $($to:ident, $factor:expr);*) {
    $(
        impl From<Length<$from>> for Length<$to> {
            #[inline(always)]
            fn from(other: Length<$from>) -> Self {
                Length {
                    value: other.value * $factor,
                    unit: core::marker::PhantomData,
                }
            }
        }
    )*
}

impl_conversion!(UMetre => UCentimetre, 1e2; UMillimetre, 1e3; UMicrometre, 1e6; UNanometre, 1e9);
impl_conversion!(UCentimetre => UMetre, 1e-2; UMillimetre, 1e1; UMicrometre, 1e4; UNanometre, 1e7);
impl_conversion!(UMillimetre => UMetre, 1e-3; UCentimetre, 1e-1; UMicrometre, 1e3; UNanometre, 1e6);
impl_conversion!(UMicrometre => UMetre, 1e-6; UCentimetre, 1e-4; UMillimetre, 1e-3; UNanometre, 1e3);
impl_conversion!(UNanometre => UMetre, 1e-9; UCentimetre, 1e-7; UMillimetre, 1e-6; UMicrometre, 1e-3);

super::impl_ops!(Add, Sub for Length where A, B: LengthMeasurement);
super::impl_ops_with_f32!(Mul, Div for Length where A: LengthMeasurement);

impl<A: LengthMeasurement> core::ops::Div<Length<A>> for Length<A> {
    type Output = f32;

    #[inline(always)]
    fn div(self, other: Length<A>) -> Self::Output { self.value / other.value }
}

impl<A: LengthMeasurement> core::ops::Mul<Length<A>> for f32 {
    type Output = Length<A>;

    fn mul(self, rhs: Length<A>) -> Self::Output {
        Length {
            value: self * rhs.value,
            unit: core::marker::PhantomData,
        }
    }
}

impl<A: LengthMeasurement> core::ops::Neg for Length<A> {
    type Output = Length<A>;

    fn neg(self) -> Self::Output {
        Length {
            value: -self.value,
            unit: core::marker::PhantomData,
        }
    }
}

super::impl_ops_assign!(AddAssign, SubAssign for Length where A, B: LengthMeasurement);

impl<U: LengthMeasurement> NumericCast<f32> for Length<U> {
    #[inline(always)]
    fn cast(&self) -> f32 { self.value }
}

#[cfg(test)]
mod length_unit_tests {
    use super::*;
    use crate::ulp_eq;
    use paste::paste;

    macro_rules! test_conversion {
        ($from:ident, $init:expr => $($to:ident, $val:expr);*) => {
            paste! {
                $(
                    {
                        let a = Length::<[<U $from>]>::new($init);
                        let b: Length<[<U $to>]> = a.into();
                        let c = a.[<in_$to:lower s>]();
                        assert!(ulp_eq(b.value, $val));
                        assert!(ulp_eq(c.value, $val));
                        assert_eq!(a, b);
                    }
                )*
            }
        };
    }

    #[test]
    fn conversion() {
        test_conversion!(Metre, 1.0 => Centimetre, 100.0; Millimetre, 1000.0; Micrometre, 1e6; Nanometre, 1e9);
        test_conversion!(Centimetre, 1.234 => Metre, 1.234e-2; Millimetre, 12.34; Micrometre, 1.234e4; Nanometre, 1.234e7);
        test_conversion!(Millimetre, 1.234 => Metre, 0.001234; Centimetre, 0.1234; Micrometre, 1.234e3; Nanometre, 1.234e6);
        test_conversion!(Micrometre, 1.234 => Metre, 1.234e-6; Centimetre, 1.234e-4; Millimetre, 0.001234; Nanometre, 1.234e3);
        test_conversion!(Nanometre, 1.234 => Metre, 1.234e-9; Centimetre, 1.234e-7; Millimetre, 1.234e-6; Micrometre, 0.001234);
    }

    macro_rules! test_equivalence {
        ($from:ident, $init:expr => $($to:ident, $val:expr);*) => {
            paste! {
                $(
                    {
                        let a = Length::<$from>::new($init);
                        let b = Length::<$to>::new($val);
                        assert_eq!(a, b);
                    }
                )*
            }
        };
    }

    #[test]
    fn equivalence() {
        test_equivalence!(UMetre, 1.5 => UCentimetre, 150.0; UMillimetre, 1500.0; UMicrometre, 1.5e6; UNanometre, 1.5e9);
        test_equivalence!(UCentimetre, 1.234 => UMetre, 1.234e-2; UMillimetre, 12.34; UMicrometre, 1.234e4; UNanometre, 1.234e7);
        test_equivalence!(UMillimetre, 1.234 => UMetre, 0.001234; UCentimetre, 0.1234; UMicrometre, 1.234e3; UNanometre, 1.234e6);
        test_equivalence!(UMicrometre, 1.234 => UMetre, 1.234e-6; UCentimetre, 1.234e-4; UMillimetre, 0.001234; UNanometre, 1.234e3);
        test_equivalence!(UNanometre, 1.234 => UMetre, 1.234e-9; UCentimetre, 1.234e-7; UMillimetre, 1.234e-6; UMicrometre, 0.001234);
    }

    #[test]
    fn add_and_assign() {
        let a = Length::<UMetre>::new(1.0);
        let b = Length::<UMetre>::new(2.0);
        let c = a + b;
        assert_eq!(c.value, 3.0);

        let a = Length::<UMetre>::new(1.0);
        let b = Length::<UNanometre>::new(2000.0);
        let c = a + b;
        assert_eq!(c, Length::<UMetre>::new(1.000002));

        let mut a = Length::<UMetre>::new(1.0);
        let b = Length::<UMicrometre>::new(2.0);
        a += b;
        assert_eq!(a, Length::<UMetre>::new(1.000002));
    }

    #[test]
    fn sub() {
        let a = Length::<UMetre>::new(1.0);
        let b = Length::<UMetre>::new(2.0);
        let c = a - b;
        assert_eq!(c.value, -1.0);

        let a = Length::<UMetre>::new(1.0);
        let b = Length::<UNanometre>::new(2000.0);
        let c = a - b;
        assert_eq!(c, Length::<UMetre>::new(0.999998));

        let mut a = Length::<UMetre>::new(1.0);
        let b = Length::<UMicrometre>::new(2.0);
        a -= b;
        assert_eq!(a, Length::<UMetre>::new(0.999998));
    }

    #[test]
    fn mul() {
        let a = Length::<UMetre>::new(1.0);
        let b = a * 2.0;
        assert_eq!(b.value, 2.0);

        let a = Length::<UMetre>::new(1.0);
        let b = 2.0 * a;
        assert_eq!(b.value, 2.0);
    }

    #[test]
    fn div() {
        let a = Length::<UMetre>::new(1.0);
        let b = a / 2.0;
        assert_eq!(b.value, 0.5);
    }

    #[test]
    fn de_serialization() {
        let a: Millimetres = metres!(100.2).into();
        let serialized = serde_yaml::to_string(&a).unwrap();
        assert_eq!(serialized, "100200 mm\n");

        let deserialized: Centimetres = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(a, deserialized);
    }
}
