use crate::util::ulp_eq;
use core::fmt::Debug;

/// Trait representing a unit of length.
pub trait LengthUnit: Debug + Copy + Clone {
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

    /// The factor to convert from picometers to the unit.
    const FACTOR_FROM_PICOMETRE: f32;

    /// The factor to convert from the unit to picometers.
    const FACTOR_TO_PICOMETRE: f32 = 1.0 / Self::FACTOR_FROM_PICOMETRE;
}

/// Meters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Metre;

/// Centimeters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Centimetre;

/// Millimeters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Millimetre;

/// Micrometers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Micrometre;

/// Nanometers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Nanometre;

/// Picometers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Picometre;

impl LengthUnit for Metre {
    const NAME: &'static str = "metre";
    const SYMBOL: &'static str = "m";
    const FACTOR_FROM_METRE: f32 = 1.0;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e-2;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e-3;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e-6;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-9;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-12;
}

impl LengthUnit for Centimetre {
    const NAME: &'static str = "centimetre";
    const SYMBOL: &'static str = "cm";
    const FACTOR_FROM_METRE: f32 = 100.0;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e-1;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e-4;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-7;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-10;
}

impl LengthUnit for Millimetre {
    const NAME: &'static str = "millimetre";
    const SYMBOL: &'static str = "mm";
    const FACTOR_FROM_METRE: f32 = 1.0e3;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e1;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e-3;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-6;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-9;
}

impl LengthUnit for Micrometre {
    const NAME: &'static str = "micrometre";
    const SYMBOL: &'static str = "um";
    const FACTOR_FROM_METRE: f32 = 1.0e6;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e4;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e3;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-3;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-6;
}

impl LengthUnit for Nanometre {
    const NAME: &'static str = "nanometre";
    const SYMBOL: &'static str = "nm";
    const FACTOR_FROM_METRE: f32 = 1.0e9;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e7;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e6;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e3;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-3;
}

impl LengthUnit for Picometre {
    const NAME: &'static str = "picometre";
    const SYMBOL: &'static str = "pm";
    const FACTOR_FROM_METRE: f32 = 1.0e12;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e10;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e9;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e6;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e3;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0;
}

/// Length with a unit.
#[derive(Copy, Clone)]
pub struct Length<A: LengthUnit> {
    pub(crate) value: f32,
    pub(crate) unit: core::marker::PhantomData<A>,
}

impl<A: LengthUnit> Debug for Length<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Length {{ value: {}, unit: {} }}", self.value, A::NAME)
    }
}

impl<A: LengthUnit> std::fmt::Display for Length<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.value, A::SYMBOL)
    }
}

impl<A: LengthUnit, B: LengthUnit> PartialEq<Length<B>> for Length<A> {
    fn eq(&self, other: &Length<B>) -> bool {
        ulp_eq(
            self.value * A::FACTOR_TO_METRE,
            other.value * B::FACTOR_TO_METRE,
        )
    }
}

impl<A: LengthUnit> Length<A> {
    /// Creates a new length.
    pub const fn new(value: f32) -> Self {
        Self {
            value,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the value of the length.
    pub const fn value(&self) -> f32 { self.value }
}

impl<A: LengthUnit> From<f32> for Length<A> {
    fn from(value: f32) -> Self { Self::new(value) }
}

impl<'a, A: LengthUnit> TryFrom<&'a str> for Length<A> {
    type Error = &'static str;

    fn try_from(s: &'a str) -> Result<Self, Self::Error> {
        let bytes = s.trim().as_bytes();
        let i =
            super::findr_first_ascii_alphabetic(bytes).ok_or("no unit found in length string")?;
        let value = std::str::from_utf8(&bytes[..i])
            .map_err(|_| "invalid length string")?
            .parse::<f32>()
            .map_err(|_| "invalid length value")?;
        let unit = std::str::from_utf8(&bytes[i..])
            .map_err(|_| "invalid length unit")?
            .trim();
        match unit {
            "m" => Ok(Self::new(A::FACTOR_FROM_METRE * value)),
            "cm" => Ok(Self::new(A::FACTOR_FROM_CENTIMETRE * value)),
            "mm" => Ok(Self::new(A::FACTOR_FROM_MILLIMETRE * value)),
            "um" => Ok(Self::new(A::FACTOR_FROM_MICROMETRE * value)),
            "nm" => Ok(Self::new(A::FACTOR_FROM_NANOMETRE * value)),
            "pm" => Ok(Self::new(A::FACTOR_FROM_PICOMETRE * value)),
            _ => Err("invalid length unit"),
        }
    }
}

super::impl_serialization!(Length<A> where A: LengthUnit, #[doc = "Customized serialization for the `Length` type."]);

// Customized deserialization for the `Length` type.
impl<'de, A: LengthUnit> serde::Deserialize<'de> for Length<A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct LengthVisitor<T>(core::marker::PhantomData<T>);

        impl<'de, T: LengthUnit> serde::de::Visitor<'de> for LengthVisitor<T> {
            type Value = Length<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
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

impl Length<Metre> {
    /// Returns the length in centimetres.
    #[inline(always)]
    pub const fn in_centimetres(self) -> Length<Centimetre> {
        Length {
            value: self.value * 100.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
    pub const fn in_millimetres(self) -> Length<Millimetre> {
        Length {
            value: self.value * 1000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
    pub const fn in_micrometres(self) -> Length<Micrometre> {
        Length {
            value: self.value * 1_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
    pub const fn in_nanometres(self) -> Length<Nanometre> {
        Length {
            value: self.value * 1_000_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
    pub const fn in_picometres(self) -> Length<Picometre> {
        Length {
            value: self.value * 1_000_000_000_000.0,
            unit: core::marker::PhantomData,
        }
    }
}

impl Length<Centimetre> {
    /// Returns the length in metres.
    #[inline(always)]
    pub const fn in_metres(self) -> Length<Metre> {
        Length {
            value: self.value / 100.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
    pub const fn in_millimetres(self) -> Length<Millimetre> {
        Length {
            value: self.value * 10.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
    pub const fn in_micrometres(self) -> Length<Micrometre> {
        Length {
            value: self.value * 10_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
    pub const fn in_nanometres(self) -> Length<Nanometre> {
        Length {
            value: self.value * 10_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
    pub const fn in_picometres(self) -> Length<Picometre> {
        Length {
            value: self.value * 10_000_000_000.0,
            unit: core::marker::PhantomData,
        }
    }
}

impl Length<Millimetre> {
    /// Returns the length in metres.
    #[inline(always)]
    pub const fn in_metres(self) -> Length<Metre> {
        Length {
            value: self.value / 1000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in centimetres.
    #[inline(always)]
    pub const fn in_centimetres(self) -> Length<Centimetre> {
        Length {
            value: self.value / 10.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
    pub const fn in_micrometres(self) -> Length<Micrometre> {
        Length {
            value: self.value * 1000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
    pub const fn in_nanometres(self) -> Length<Nanometre> {
        Length {
            value: self.value * 1_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
    pub const fn in_picometres(self) -> Length<Picometre> {
        Length {
            value: self.value * 1_000_000_000.0,
            unit: core::marker::PhantomData,
        }
    }
}

impl Length<Micrometre> {
    /// Returns the length in metres.
    #[inline(always)]
    pub const fn in_metres(self) -> Length<Metre> {
        Length {
            value: self.value / 1_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in centimetres.
    #[inline(always)]
    pub const fn in_centimetres(self) -> Length<Centimetre> {
        Length {
            value: self.value / 10_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
    pub const fn in_millimetres(self) -> Length<Millimetre> {
        Length {
            value: self.value / 1000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
    pub const fn in_nanometres(self) -> Length<Nanometre> {
        Length {
            value: self.value * 1000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
    pub const fn in_picometres(self) -> Length<Picometre> {
        Length {
            value: self.value * 1_000_000.0,
            unit: core::marker::PhantomData,
        }
    }
}

impl Length<Nanometre> {
    /// Returns the length in metres.
    #[inline(always)]
    pub const fn in_metres(self) -> Length<Metre> {
        Length {
            value: self.value / 1_000_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in centimetres.
    #[inline(always)]
    pub const fn in_centimetres(self) -> Length<Centimetre> {
        Length {
            value: self.value / 10_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
    pub const fn in_millimetres(self) -> Length<Millimetre> {
        Length {
            value: self.value / 1_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
    pub const fn in_micrometres(self) -> Length<Micrometre> {
        Length {
            value: self.value / 1000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
    pub const fn in_picometres(self) -> Length<Picometre> {
        Length {
            value: self.value * 1000.0,
            unit: core::marker::PhantomData,
        }
    }
}

impl Length<Picometre> {
    /// Returns the length in metres.
    #[inline(always)]
    pub const fn in_metres(self) -> Length<Metre> {
        Length {
            value: self.value / 1_000_000_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in centimetres.
    #[inline(always)]
    pub const fn in_centimetres(self) -> Length<Centimetre> {
        Length {
            value: self.value / 10_000_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
    pub const fn in_millimetres(self) -> Length<Millimetre> {
        Length {
            value: self.value / 1_000_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
    pub const fn in_micrometres(self) -> Length<Micrometre> {
        Length {
            value: self.value / 1_000_000.0,
            unit: core::marker::PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
    pub const fn in_nanometres(self) -> Length<Nanometre> {
        Length {
            value: self.value / 1000.0,
            unit: core::marker::PhantomData,
        }
    }
}

/// Type alias for a length in metres.
pub type Metres = Length<Metre>;

/// Type alias for a length in centimetres.
pub type Centimetres = Length<Centimetre>;

/// Type alias for a length in millimetres.
pub type Millimetres = Length<Millimetre>;

/// Type alias for a length in micrometres.
pub type Micrometres = Length<Micrometre>;

/// Type alias for a length in nanometres.
pub type Nanometres = Length<Nanometre>;

/// Type alias for a length in picometres.
pub type Picometres = Length<Picometre>;

/// Macro for creating a new length type in metres.
pub macro metres($val:expr) {
    $crate::acq::Length::<$crate::acq::Metre>::new($val)
}

/// Macro for creating a new length type in centimetres.
pub macro centimetres($val:expr) {
    $crate::acq::Length::<$crate::acq::Centimetre>::new($val)
}

/// Macro for creating a new length type in millimetres.
pub macro millimetres($val:expr) {
    $crate::acq::Length::<$crate::acq::Millimetre>::new($val)
}

/// Macro for creating a new length type in micrometres.
pub macro micrometres($val:expr) {
    $crate::acq::Length::<$crate::acq::Micrometre>::new($val)
}

/// Macro for creating a new length type in nanometres.
pub macro nanometres($val:expr) {
    $crate::acq::Length::<$crate::acq::Nanometre>::new($val)
}

/// Macro for creating a new length type in picometres.
pub macro picometres($val:expr) {
    $crate::acq::Length::<$crate::acq::Picometre>::new($val)
}

macro impl_conversion($from:ident => $($to:ident, $factor:expr);*) {
    $(
        impl const From<Length<$from>> for Length<$to> {
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

impl_conversion!(Metre => Centimetre, 1e2; Millimetre, 1e3; Micrometre, 1e6; Nanometre, 1e9; Picometre, 1e12);
impl_conversion!(Centimetre => Metre, 1e-2; Millimetre, 1e1; Micrometre, 1e4; Nanometre, 1e7; Picometre, 1e10);
impl_conversion!(Millimetre => Metre, 1e-3; Centimetre, 1e-1; Micrometre, 1e3; Nanometre, 1e6; Picometre, 1e9);
impl_conversion!(Micrometre => Metre, 1e-6; Centimetre, 1e-4; Millimetre, 1e-3; Nanometre, 1e3; Picometre, 1e6);
impl_conversion!(Nanometre => Metre, 1e-9; Centimetre, 1e-7; Millimetre, 1e-6; Micrometre, 1e-3; Picometre, 1e3);
impl_conversion!(Picometre => Metre, 1e-12; Centimetre, 1e-10; Millimetre, 1e-9; Micrometre, 1e-6; Nanometre, 1e-3);

super::impl_ops!(Add, Sub for Length where A, B: LengthUnit);
super::impl_ops_with_f32!(Mul, Div for Length where A: LengthUnit);

impl<A: LengthUnit> core::ops::Mul<Length<A>> for f32 {
    type Output = Length<A>;

    fn mul(self, rhs: Length<A>) -> Self::Output {
        Length {
            value: self * rhs.value,
            unit: core::marker::PhantomData,
        }
    }
}

super::impl_ops_assign!(AddAssign, SubAssign for Length where A, B: LengthUnit);

#[cfg(test)]
mod length_unit_tests {
    use super::*;
    use crate::util::ulp_eq;
    use paste::paste;

    macro_rules! test_conversion {
        ($from:ident, $init:expr => $($to:ident, $val:expr);*) => {
            paste! {
                $(
                    {
                        let a = Length::<$from>::new($init);
                        let b: Length<$to> = a.into();
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
        test_conversion!(Metre, 1.0 => Centimetre, 100.0; Millimetre, 1000.0; Micrometre, 1e6; Nanometre, 1e9; Picometre, 1e12);
        test_conversion!(Centimetre, 1.234 => Metre, 1.234e-2; Millimetre, 12.34; Micrometre, 1.234e4; Nanometre, 1.234e7; Picometre, 1.234e10);
        test_conversion!(Millimetre, 1.234 => Metre, 0.001234; Centimetre, 0.1234; Micrometre, 1.234e3; Nanometre, 1.234e6; Picometre, 1.234e9);
        test_conversion!(Micrometre, 1.234 => Metre, 1.234e-6; Centimetre, 1.234e-4; Millimetre, 0.001234; Nanometre, 1.234e3; Picometre, 1.234e6);
        test_conversion!(Nanometre, 1.234 => Metre, 1.234e-9; Centimetre, 1.234e-7; Millimetre, 1.234e-6; Micrometre, 0.001234; Picometre, 1.234e3);
        test_conversion!(Picometre, 1.234 => Metre, 1.234e-12; Centimetre, 1.234e-10; Millimetre, 1.234e-9; Micrometre, 1.234e-6; Nanometre, 0.001234);
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
        test_equivalence!(Metre, 1.5 => Centimetre, 150.0; Millimetre, 1500.0; Micrometre, 1.5e6; Nanometre, 1.5e9; Picometre, 1.5e12);
        test_equivalence!(Centimetre, 1.234 => Metre, 1.234e-2; Millimetre, 12.34; Micrometre, 1.234e4; Nanometre, 1.234e7; Picometre, 1.234e10);
        test_equivalence!(Millimetre, 1.234 => Metre, 0.001234; Centimetre, 0.1234; Micrometre, 1.234e3; Nanometre, 1.234e6; Picometre, 1.234e9);
        test_equivalence!(Micrometre, 1.234 => Metre, 1.234e-6; Centimetre, 1.234e-4; Millimetre, 0.001234; Nanometre, 1.234e3; Picometre, 1.234e6);
        test_equivalence!(Nanometre, 1.234 => Metre, 1.234e-9; Centimetre, 1.234e-7; Millimetre, 1.234e-6; Micrometre, 0.001234; Picometre, 1.234e3);
        test_equivalence!(Picometre, 1.234 => Metre, 1.234e-12; Centimetre, 1.234e-10; Millimetre, 1.234e-9; Micrometre, 1.234e-6; Nanometre, 0.001234);
    }

    #[test]
    fn add_and_assign() {
        let a = Length::<Metre>::new(1.0);
        let b = Length::<Metre>::new(2.0);
        let c = a + b;
        assert_eq!(c.value, 3.0);

        let a = Length::<Metre>::new(1.0);
        let b = Length::<Nanometre>::new(2000.0);
        let c = a + b;
        assert_eq!(c, Length::<Metre>::new(1.000002));

        let mut a = Length::<Metre>::new(1.0);
        let b = Length::<Micrometre>::new(2.0);
        a += b;
        assert_eq!(a, Length::<Metre>::new(1.000002));
    }

    #[test]
    fn sub() {
        let a = Length::<Metre>::new(1.0);
        let b = Length::<Metre>::new(2.0);
        let c = a - b;
        assert_eq!(c.value, -1.0);

        let a = Length::<Metre>::new(1.0);
        let b = Length::<Nanometre>::new(2000.0);
        let c = a - b;
        assert_eq!(c, Length::<Metre>::new(0.999998));

        let mut a = Length::<Metre>::new(1.0);
        let b = Length::<Micrometre>::new(2.0);
        a -= b;
        assert_eq!(a, Length::<Metre>::new(0.999998));
    }

    #[test]
    fn mul() {
        let a = Length::<Metre>::new(1.0);
        let b = a * 2.0;
        assert_eq!(b.value, 2.0);

        let a = Length::<Metre>::new(1.0);
        let b = 2.0 * a;
        assert_eq!(b.value, 2.0);
    }

    #[test]
    fn div() {
        let a = Length::<Metre>::new(1.0);
        let b = a / 2.0;
        assert_eq!(b.value, 0.5);
    }

    #[test]
    fn de_serialization() {
        let a: Metres = metres!(100.2);
        let serialized = serde_yaml::to_string(&a).unwrap();
        assert_eq!(serialized, "100.2 m\n");

        let deserialized: Centimetres = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(a, deserialized);
    }
}
