use crate::util::ulp_eq;
use core::{
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};
use paste::paste;

/// Radian unit.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Radian;

/// Degree unit.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Degree;

/// Unit trait for angle units.
pub trait AngleUnit: Debug + Copy + Clone {
    /// The name of the unit.
    const NAME: &'static str;

    /// The symbol of the unit.
    const SYMBOL: &'static str;

    /// The conversion factor from radians.
    const FACTOR_FROM_RAD: f32;

    /// The conversion factor to radians.
    const FACTOR_TO_RAD: f32 = 1.0 / Self::FACTOR_FROM_RAD;

    /// The conversion factor from degrees.
    const FACTOR_FROM_DEG: f32;

    /// The conversion factor to degrees.
    const FACTOR_TO_DEG: f32 = 1.0 / Self::FACTOR_FROM_DEG;
}

impl AngleUnit for Radian {
    const NAME: &'static str = "radian";
    const SYMBOL: &'static str = "rad";
    const FACTOR_FROM_RAD: f32 = 1.0;
    const FACTOR_FROM_DEG: f32 = 1.0 / 180.0 * std::f32::consts::PI;
}

impl AngleUnit for Degree {
    const NAME: &'static str = "degree";
    const SYMBOL: &'static str = "deg";
    const FACTOR_FROM_RAD: f32 = 180.0 / std::f32::consts::PI;
    const FACTOR_FROM_DEG: f32 = 1.0;
}

/// Angle with unit.
#[derive(Copy, Clone)]
pub struct Angle<A: AngleUnit> {
    value: f32,
    unit: PhantomData<A>,
}

impl<A: AngleUnit> Debug for Angle<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Angle {{ value: {}, unit: {} }}", self.value, A::SYMBOL)
    }
}

impl<A: AngleUnit> Display for Angle<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.value, A::SYMBOL)
    }
}

impl<A: AngleUnit, B: AngleUnit> PartialEq<Angle<B>> for Angle<A> {
    fn eq(&self, other: &Angle<B>) -> bool {
        ulp_eq(
            self.value * A::FACTOR_TO_RAD,
            other.value * B::FACTOR_TO_RAD,
        )
    }
}

impl<A: AngleUnit> Angle<A> {
    /// Create a new angle with unit.
    pub fn new(value: f32) -> Self {
        Angle {
            value,
            unit: PhantomData,
        }
    }

    /// Get the value of the angle.
    pub fn value(&self) -> f32 { self.value }
}

impl<A: AngleUnit> From<f32> for Angle<A> {
    fn from(value: f32) -> Self { Angle::new(value) }
}

impl<'a, A: AngleUnit> TryFrom<&'a str> for Angle<A> {
    type Error = &'static str;

    fn try_from(s: &'a str) -> Result<Self, Self::Error> {
        let bytes = s.trim().as_bytes();
        let i =
            super::findr_first_ascii_alphabetic(bytes).ok_or("no unit found in angle string")?;
        let value = std::str::from_utf8(&bytes[..i])
            .map_err(|_| "invalid angle string")?
            .parse::<f32>()
            .map_err(|_| "invalid angle value")?;
        let unit = std::str::from_utf8(&bytes[i..])
            .map_err(|_| "invalid angle unit")?
            .trim();
        match unit {
            "rad" | "rads" | "radians" => Ok(Self::new(A::FACTOR_FROM_RAD * value)),
            "deg" | "degs" | "degrees" => Ok(Self::new(A::FACTOR_FROM_DEG * value)),
            _ => Err("invalid angle unit"),
        }
    }
}

impl_serialization!(Angle<A> where A: AngleUnit, #[doc = "Customized serialization for the `Angle` type."]);

/// Customized deserialization for the `Angle` type.
impl<'de, A: AngleUnit> serde::Deserialize<'de> for Angle<A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct AngleVisitor<T>(PhantomData<T>);

        impl<'de, T: AngleUnit> serde::de::Visitor<'de> for AngleVisitor<T> {
            type Value = Angle<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a string containing a number and a unit of angle"
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Angle::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }

        deserializer.deserialize_str(AngleVisitor::<A>(PhantomData))
    }
}

impl Angle<Radian> {
    /// Convert to degree.
    pub fn in_degrees(&self) -> Angle<Degree> { Angle::new(self.value * Degree::FACTOR_FROM_RAD) }
}

impl Angle<Degree> {
    /// Convert to radian.
    pub fn in_radians(&self) -> Angle<Radian> { Angle::new(self.value * Radian::FACTOR_FROM_DEG) }
}

/// Type alias for `Angle<Radian>`.
pub type Radians = Angle<Radian>;

/// Type alias for `Angle<Degree>`.
pub type Degrees = Angle<Degree>;

/// Helper creating a new `Angle<Radian>`.
#[macro_export]
macro_rules! radians {
    ($value:expr) => {
        $crate::acq::Angle::<$crate::acq::Radian>::new($value)
    };
}

/// Helper creating a new `Angle<Degree>`
#[macro_export]
macro_rules! degrees {
    ($value:expr) => {
        $crate::acq::Angle::<$crate::acq::Degree>::new($value)
    };
}

impl From<Angle<Degree>> for Angle<Radian> {
    fn from(angle: Angle<Degree>) -> Self { angle.in_radians() }
}

impl From<Angle<Radian>> for Angle<Degree> {
    fn from(angle: Angle<Radian>) -> Self { angle.in_degrees() }
}

impl_ops!(Add, Sub for Angle where A, B: AngleUnit);
impl_ops_with_f32!(Mul, Div for Angle where A: AngleUnit);

impl<A: AngleUnit> Mul<Angle<A>> for f32 {
    type Output = Angle<A>;

    fn mul(self, rhs: Angle<A>) -> Self::Output {
        Angle {
            value: self * rhs.value,
            unit: PhantomData,
        }
    }
}

impl_ops_assign!(AddAssign, SubAssign for Angle where A, B: AngleUnit);

#[cfg(test)]
mod angle_unit_tests {
    use super::*;
    use crate::util::ulp_eq;

    #[test]
    fn conversion() {
        let a = Angle::<Radian>::new(1.0);
        let b: Angle<Degree> = a.into();
        let c = a.in_degrees();
        assert!(ulp_eq(b.value, 1.0f32.to_degrees()));
        assert!(ulp_eq(c.value, 1.0f32.to_degrees()));
        assert_eq!(a, b);

        {
            let a = Angle::<Degree>::new(180.0);
            let b: Angle<Radian> = a.into();
            let c = a.in_radians();
            assert!(ulp_eq(b.value, core::f32::consts::PI));
            assert!(ulp_eq(c.value, core::f32::consts::PI));
            assert_eq!(a, b);
        }
    }

    #[test]
    fn equivalence() {
        let a = Angle::<Radian>::new(1.0);
        let b = Angle::<Degree>::new(1.0f32.to_degrees());
        assert_eq!(a, b);
    }

    #[test]
    fn add_and_assign() {
        let a = Angle::<Degree>::new(1.0);
        let b = Angle::<Degree>::new(2.0);
        let c = a + b;
        assert_eq!(c.value, 3.0);

        let a = Angle::<Radian>::new(1.0);
        let b = Angle::<Degree>::new(180.0);
        let c = a + b;
        assert_eq!(c.value, 1.0 + core::f32::consts::PI);

        let mut a = Angle::<Degree>::new(1.0);
        let b = Angle::<Degree>::new(2.0);
        a += b;
        assert_eq!(a, Angle::<Degree>::new(3.0));

        let mut a = Angle::<Radian>::new(1.0);
        let b = Angle::<Degree>::new(180.0);
        a += b;
        assert_eq!(a, Angle::<Radian>::new(1.0 + core::f32::consts::PI));
    }

    #[test]
    fn sub() {
        let a = Angle::<Degree>::new(1.0);
        let b = Angle::<Degree>::new(2.0);
        let c = a - b;
        assert_eq!(c.value, -1.0);

        let a = Angle::<Radian>::new(1.0);
        let b = Angle::<Degree>::new(180.0);
        let c = a - b;
        assert!(ulp_eq(c.value, 1.0 - core::f32::consts::PI));

        let mut a = Angle::<Degree>::new(1.0);
        let b = Angle::<Degree>::new(2.0);
        a -= b;
        assert_eq!(a, Angle::<Degree>::new(-1.0));

        let mut a = Angle::<Degree>::new(1.0);
        let b = Angle::<Radian>::new(1.0);
        a -= b;
        assert_eq!(a, Angle::<Radian>::new(1.0f32.to_radians() - 1.0));
    }

    #[test]
    fn mul() {
        let a = Angle::<Radian>::new(1.0);
        let b = a * 2.0;
        assert_eq!(b.value, 2.0);

        let a = Angle::<Degree>::new(1.0);
        let b = 2.0 * a;
        assert_eq!(b.value, 2.0);
    }

    #[test]
    fn div() {
        let a = Angle::<Degree>::new(1.0);
        let b = a / 2.0;
        assert_eq!(b.value, 0.5);
    }

    #[test]
    fn de_serialization() {
        let a: Degrees = degrees!(180.0);
        let serialized = serde_yaml::to_string(&a).unwrap();
        assert_eq!(serialized, "180 deg\n");

        let deserialized: Radians = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(a, deserialized);

        let deserialized2: Degrees = serde_yaml::from_str("180.0 degs").unwrap();
        assert_eq!(a, deserialized2);

        let deserialized3: Degrees = serde_yaml::from_str("180.0 degrees").unwrap();
        assert_eq!(a, deserialized3);
    }
}
