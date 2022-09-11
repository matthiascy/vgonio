use std::fmt::Debug;
use std::marker::PhantomData;

/// Trait representing a unit of length.
pub trait LengthUnit {
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
pub struct SiMetre;

/// Centimeters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SiCentimetre;

/// Millimeters.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SiMillimetre;

/// Micrometers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SiMicrometre;

/// Nanometers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SiNanometre;

/// Picometers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SiPicometre;

impl LengthUnit for SiMetre {
    const NAME: &'static str = "metre";
    const SYMBOL: &'static str = "m";
    const FACTOR_FROM_METRE: f32 = 1.0;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e-2;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e-3;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e-6;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-9;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-12;
}

impl LengthUnit for SiCentimetre {
    const NAME: &'static str = "centimetre";
    const SYMBOL: &'static str = "cm";
    const FACTOR_FROM_METRE: f32 = 100.0;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e-1;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e-4;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-7;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-10;
}

impl LengthUnit for SiMillimetre {
    const NAME: &'static str = "millimetre";
    const SYMBOL: &'static str = "mm";
    const FACTOR_FROM_METRE: f32 = 1.0e3;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e1;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e-3;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-6;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-9;
}

impl LengthUnit for SiMicrometre {
    const NAME: &'static str = "micrometre";
    const SYMBOL: &'static str = "um";
    const FACTOR_FROM_METRE: f32 = 1.0e6;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e4;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e3;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0e-3;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-6;
}

impl LengthUnit for SiNanometre {
    const NAME: &'static str = "nanometre";
    const SYMBOL: &'static str = "nm";
    const FACTOR_FROM_METRE: f32 = 1.0e9;
    const FACTOR_FROM_CENTIMETRE: f32 = 1.0e7;
    const FACTOR_FROM_MILLIMETRE: f32 = 1.0e6;
    const FACTOR_FROM_MICROMETRE: f32 = 1.0e3;
    const FACTOR_FROM_NANOMETRE: f32 = 1.0;
    const FACTOR_FROM_PICOMETRE: f32 = 1.0e-3;
}

impl LengthUnit for SiPicometre {
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
pub struct Length<U: LengthUnit> {
    value: f32,
    unit: PhantomData<U>,
}

impl<U: LengthUnit> Debug for Length<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Length {{ value: {}, unit: {} }}", self.value, U::NAME)
    }
}

impl<U: LengthUnit, V: LengthUnit> PartialEq<Length<V>> for Length<U> {
    fn eq(&self, other: &Length<V>) -> bool {
        self.value / U::FACTOR_FROM_METRE == other.value / V::FACTOR_FROM_METRE
    }
}

impl<U: LengthUnit> Length<U> {
    /// Creates a new length.
    pub const fn new(value: f32) -> Self {
        Self {
            value,
            unit: PhantomData,
        }
    }

    /// Returns the value of the length.
    pub const fn value(&self) -> f32 {
        self.value
    }
}

fn findr_first_ascii_alphabetic(s: &[u8]) -> Option<usize> {
    let mut i = s.len() - 1;
    while i > 0 {
        if s[i].is_ascii_alphabetic() {
            i -= 1;
        } else {
            return Some(i);
        }
    }
    None
}

impl<'a, U: LengthUnit> TryFrom<&'a str> for Length<U> {
    type Error = &'static str;

    fn try_from(s: &'a str) -> Result<Self, Self::Error> {
        let bytes = s.trim().as_bytes();
        let i = findr_first_ascii_alphabetic(bytes).ok_or("no unit found in length string")?;
        let value = std::str::from_utf8(&bytes[..i])
            .map_err(|_| "invalid length string")?
            .parse::<f32>().map_err(|_| "invalid length value")?;
        let unit = std::str::from_utf8(&bytes[i..]).map_err(|_| "invalid length unit")?.trim();
        match unit {
            "m" => Ok(Self::new(U::FACTOR_FROM_METRE * value)),
            "cm" => Ok(Self::new(U::FACTOR_FROM_CENTIMETRE * value)),
            "mm" => Ok(Self::new(U::FACTOR_FROM_MILLIMETRE * value)),
            "um" => Ok(Self::new(U::FACTOR_FROM_MICROMETRE * value)),
            "nm" => Ok(Self::new(U::FACTOR_FROM_NANOMETRE * value)),
            "pm" => Ok(Self::new(U::FACTOR_FROM_PICOMETRE * value)),
            _ => Err("invalid length unit"),
        }
    }
}

impl<U> From<f32> for Length<U>
where
    U: LengthUnit,
{
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

/// Customized serialization for the `Length` type.
impl<U: LengthUnit> serde::Serialize for Length<U> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer
    {
        let s = format!("{}", self);
        serializer.serialize_str(&s)
    }
}

// Customized deserialization for the `Length` type.
impl<'de, U: LengthUnit> serde::Deserialize<'de> for Length<U> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: serde::Deserializer<'de>
    {
        struct LengthVisitor<T>(PhantomData<T>);

        impl<'de, T: LengthUnit> serde::de::Visitor<'de> for LengthVisitor<T> {
            type Value = Length<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "a string containing a number and a unit of length")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> where E: serde::de::Error {
                Length::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }

        deserializer.deserialize_str(LengthVisitor::<U>(PhantomData))
    }
}

impl<U: LengthUnit> std::fmt::Display for Length<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.value, U::SYMBOL)
    }
}

impl Length<SiMetre> {
    /// Returns the length in centimetres.
    #[inline(always)]
    pub const fn to_centimetre(self) -> Length<SiCentimetre> {
        Length {
            value: self.value * 100.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
    pub const fn to_millimetre(self) -> Length<SiMillimetre> {
        Length {
            value: self.value * 1000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
    pub const fn to_micrometre(self) -> Length<SiMicrometre> {
        Length {
            value: self.value * 1_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
pub const fn to_nanometre(self) -> Length<SiNanometre> {
        Length {
            value: self.value * 1_000_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
pub const fn to_picometre(self) -> Length<SiPicometre> {
        Length {
            value: self.value * 1_000_000_000_000.0,
            unit: PhantomData,
        }
    }
}

impl Length<SiCentimetre> {
    /// Returns the length in metres.
    #[inline(always)]
pub const fn to_metre(self) -> Length<SiMetre> {
        Length {
            value: self.value / 100.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
pub const fn to_millimetre(self) -> Length<SiMillimetre> {
        Length {
            value: self.value * 10.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
    pub const fn to_micrometre(self) -> Length<SiMicrometre> {
        Length {
            value: self.value * 10_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
pub const fn to_nanometre(self) -> Length<SiNanometre> {
        Length {
            value: self.value * 10_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
pub const fn to_picometre(self) -> Length<SiPicometre> {
        Length {
            value: self.value * 10_000_000_000.0,
            unit: PhantomData,
        }
    }
}

impl Length<SiMillimetre> {
    /// Returns the length in metres.
    #[inline(always)]
pub const fn to_metre(self) -> Length<SiMetre> {
        Length {
            value: self.value / 1000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in centimetres.
    #[inline(always)]
pub const fn to_centimetre(self) -> Length<SiCentimetre> {
        Length {
            value: self.value / 10.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
pub const fn to_micrometre(self) -> Length<SiMicrometre> {
        Length {
            value: self.value * 1000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
pub const fn to_nanometre(self) -> Length<SiNanometre> {
        Length {
            value: self.value * 1_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
pub const fn to_picometre(self) -> Length<SiPicometre> {
        Length {
            value: self.value * 1_000_000_000.0,
            unit: PhantomData,
        }
    }
}

impl Length<SiMicrometre> {
    /// Returns the length in metres.
    #[inline(always)]
pub const fn to_metre(self) -> Length<SiMetre> {
        Length {
            value: self.value / 1_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in centimetres.
    #[inline(always)]
pub const fn to_centimetre(self) -> Length<SiCentimetre> {
        Length {
            value: self.value / 10_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
pub const fn to_millimetre(self) -> Length<SiMillimetre> {
        Length {
            value: self.value / 1000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
pub const fn to_nanometre(self) -> Length<SiNanometre> {
        Length {
            value: self.value * 1000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
pub const fn to_picometre(self) -> Length<SiPicometre> {
        Length {
            value: self.value * 1_000_000.0,
            unit: PhantomData,
        }
    }
}

impl Length<SiNanometre> {
    /// Returns the length in metres.
    #[inline(always)]
pub const fn to_metre(self) -> Length<SiMetre> {
        Length {
            value: self.value / 1_000_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in centimetres.
    #[inline(always)]
pub const fn to_centimetre(self) -> Length<SiCentimetre> {
        Length {
            value: self.value / 10_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
pub const fn to_millimetre(self) -> Length<SiMillimetre> {
        Length {
            value: self.value / 1_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
pub const fn to_micrometre(self) -> Length<SiMicrometre> {
        Length {
            value: self.value / 1000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in picometres.
    #[inline(always)]
pub const fn to_picometre(self) -> Length<SiPicometre> {
        Length {
            value: self.value * 1000.0,
            unit: PhantomData,
        }
    }
}

impl Length<SiPicometre> {
    /// Returns the length in metres.
    #[inline(always)]
pub const fn to_metre(self) -> Length<SiMetre> {
        Length {
            value: self.value / 1_000_000_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in centimetres.
    #[inline(always)]
pub const fn to_centimetre(self) -> Length<SiCentimetre> {
        Length {
            value: self.value / 10_000_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in millimetres.
    #[inline(always)]
pub const fn to_millimetre(self) -> Length<SiMillimetre> {
        Length {
            value: self.value / 1_000_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in micrometres.
    #[inline(always)]
pub const fn to_micrometre(self) -> Length<SiMicrometre> {
        Length {
            value: self.value / 1_000_000.0,
            unit: PhantomData,
        }
    }

    /// Returns the length in nanometres.
    #[inline(always)]
pub const fn to_nanometre(self) -> Length<SiNanometre> {
        Length {
            value: self.value / 1000.0,
            unit: PhantomData,
        }
    }
}

/// Type alias for a length in metres.
pub type Metres = Length<SiMetre>;

/// Type alias for a length in centimetres.
pub type Centimetres = Length<SiCentimetre>;

/// Type alias for a length in millimetres.
pub type Millimetres = Length<SiMillimetre>;

/// Type alias for a length in micrometres.
pub type Micrometres = Length<SiMicrometre>;

/// Type alias for a length in nanometres.
pub type Nanometres = Length<SiNanometre>;

/// Type alias for a length in picometres.
pub type Picometres = Length<SiPicometre>;

/// Macro for creating a new length type in metres.
#[macro_export]
macro_rules! metres {
    ($val:expr) => {
        $crate::acq::Length::<$crate::acq::SiMetre>::new($val)
    };
}

/// Macro for creating a new length type in centimetres.
#[macro_export]
macro_rules! centimetres {
    ($val:expr) => {
        $crate::acq::Length::<$crate::acq::SiCentimetre>::new($val)
    };
}

/// Macro for creating a new length type in millimetres.
#[macro_export]
macro_rules! millimetres {
    ($val:expr) => {
        $crate::acq::Length::<$crate::acq::SiMillimetre>::new($val)
    };
}

/// Macro for creating a new length type in micrometres.
#[macro_export]
macro_rules! micrometres {
    ($val:expr) => {
        $crate::acq::Length::<$crate::acq::SiMicrometre>::new($val)
    };
}

/// Macro for creating a new length type in nanometres.
#[macro_export]
macro_rules! nanometres {
    ($val:expr) => {
        $crate::acq::Length::<$crate::acq::SiNanometre>::new($val)
    };
}

/// Macro for creating a new length type in picometres.
#[macro_export]
macro_rules! picometres {
    ($val:expr) => {
        $crate::acq::Length::<$crate::acq::SiPicometre>::new($val)
    };
}

#[cfg(test)]
mod unit_test_si_units {
    use crate::acq::{Centimetres, Metres};

    #[test]
    fn equality() {
        let a = Metres::new(1.0);
        let b = Metres::new(1.0);
        assert_eq!(a, b);

        let c = micrometres!(0.58);
        let d = nanometres!(580.0);
        assert_eq!(c, d);
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