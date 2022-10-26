use crate::util::ulp_eq;
use core::fmt::{Debug, Display};
use std::ops::{Deref, DerefMut};

/// Radian unit.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct URadian;

/// Degree unit.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UDegree;

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

impl AngleUnit for URadian {
    const NAME: &'static str = "radian";
    const SYMBOL: &'static str = "rad";
    const FACTOR_FROM_RAD: f32 = 1.0;
    const FACTOR_FROM_DEG: f32 = 1.0 / 180.0 * std::f32::consts::PI;
}

impl AngleUnit for UDegree {
    const NAME: &'static str = "degree";
    const SYMBOL: &'static str = "deg";
    const FACTOR_FROM_RAD: f32 = 180.0 / std::f32::consts::PI;
    const FACTOR_FROM_DEG: f32 = 1.0;
}

/// Angle with unit.
#[derive(Copy, Clone)]
pub struct Angle<A: AngleUnit> {
    pub(crate) value: f32,
    pub(crate) unit: core::marker::PhantomData<A>,
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

impl<A: AngleUnit, B: AngleUnit> PartialOrd<Angle<B>> for Angle<A> {
    fn partial_cmp(&self, other: &Angle<B>) -> Option<core::cmp::Ordering> {
        let self_rad = self.value * A::FACTOR_TO_RAD;
        let other_rad = other.value * B::FACTOR_TO_RAD;
        Some(self_rad.partial_cmp(&other_rad)?)
    }
}

impl<A: AngleUnit> Angle<A> {
    /// Create a new angle with unit.
    pub fn new(value: f32) -> Self {
        Angle {
            value,
            unit: core::marker::PhantomData,
        }
    }

    /// Get the value of the angle.
    pub fn value(&self) -> f32 { self.value }

    super::forward_f32_methods!(abs, ceil, round, trunc, fract, sqrt, cbrt);
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

super::impl_serialization!(Angle<A> where A: AngleUnit, #[doc = "Customized serialization for the `Angle` type."]);

/// Customized deserialization for the `Angle` type.
impl<'de, A: AngleUnit> serde::Deserialize<'de> for Angle<A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct AngleVisitor<T>(core::marker::PhantomData<T>);

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

        deserializer.deserialize_str(AngleVisitor::<A>(core::marker::PhantomData))
    }
}

impl Angle<URadian> {
    /// Convert to degree.
    pub fn in_degrees(&self) -> Angle<UDegree> { Angle::new(self.value * UDegree::FACTOR_FROM_RAD) }
    pub fn sin(&self) -> f32 { self.value.sin() }
    pub fn sinh(&self) -> f32 { self.value.sinh() }
    pub fn cos(&self) -> f32 { self.value.cos() }
    pub fn cosh(&self) -> f32 { self.value.cosh() }
    pub fn tan(&self) -> f32 { self.value.tan() }
    pub fn tanh(&self) -> f32 { self.value.tanh() }
}

impl Angle<UDegree> {
    /// Convert to radian.
    pub fn in_radians(&self) -> Angle<URadian> { Angle::new(self.value * URadian::FACTOR_FROM_DEG) }
    pub fn sin(&self) -> f32 { self.value.to_radians().sin() }
    pub fn sinh(&self) -> f32 { self.value.to_radians().sinh() }
    pub fn cos(&self) -> f32 { self.value.to_radians().cos() }
    pub fn cosh(&self) -> f32 { self.value.to_radians().cosh() }
    pub fn tan(&self) -> f32 { self.value.to_radians().tan() }
    pub fn tanh(&self) -> f32 { self.value.to_radians().tanh() }
}

/// Type alias for `Angle<Radian>`.
pub type Radians = Angle<URadian>;

/// Type alias for `Angle<Degree>`.
pub type Degrees = Angle<UDegree>;

/// Helper creating a new `Angle<Radian>`.
pub macro radians($value:expr) {
    $crate::acq::Angle::<$crate::acq::URadian>::new($value)
}

/// Helper creating a new `Angle<Degree>`.
pub macro degrees($value:expr) {
    $crate::acq::Angle::<$crate::acq::UDegree>::new($value)
}

impl From<Angle<UDegree>> for Angle<URadian> {
    fn from(angle: Angle<UDegree>) -> Self { angle.in_radians() }
}

impl From<Angle<URadian>> for Angle<UDegree> {
    fn from(angle: Angle<URadian>) -> Self { angle.in_degrees() }
}

super::impl_ops!(Add, Sub for Angle where A, B: AngleUnit);
super::impl_ops_with_f32!(Mul, Div for Angle where A: AngleUnit);

impl<A: AngleUnit, B: AngleUnit> core::ops::Div<Angle<B>> for Angle<A> {
    type Output = f32;

    fn div(self, rhs: Angle<B>) -> Self::Output { self.value.to_radians() / rhs.value.to_radians() }
}

impl<A: AngleUnit> core::ops::Mul<Angle<A>> for f32 {
    type Output = Angle<A>;

    fn mul(self, rhs: Angle<A>) -> Self::Output {
        Angle {
            value: self * rhs.value,
            unit: core::marker::PhantomData,
        }
    }
}

impl<A: AngleUnit> core::ops::Neg for Angle<A> {
    type Output = Angle<A>;

    fn neg(self) -> Self::Output {
        Angle {
            value: -self.value,
            unit: core::marker::PhantomData,
        }
    }
}

super::impl_ops_assign!(AddAssign, SubAssign for Angle where A, B: AngleUnit);

#[cfg(test)]
mod angle_unit_tests {
    use super::*;
    use crate::util::ulp_eq;

    #[test]
    fn conversion() {
        let a = Angle::<URadian>::new(1.0);
        let b: Angle<UDegree> = a.into();
        let c = a.in_degrees();
        assert!(ulp_eq(b.value, 1.0f32.to_degrees()));
        assert!(ulp_eq(c.value, 1.0f32.to_degrees()));
        assert_eq!(a, b);

        {
            let a = Angle::<UDegree>::new(180.0);
            let b: Angle<URadian> = a.into();
            let c = a.in_radians();
            assert!(ulp_eq(b.value, core::f32::consts::PI));
            assert!(ulp_eq(c.value, core::f32::consts::PI));
            assert_eq!(a, b);
        }
    }

    #[test]
    fn equivalence() {
        let a = Angle::<URadian>::new(1.0);
        let b = Angle::<UDegree>::new(1.0f32.to_degrees());
        assert_eq!(a, b);
    }

    #[test]
    fn add_and_assign() {
        let a = Angle::<UDegree>::new(1.0);
        let b = Angle::<UDegree>::new(2.0);
        let c = a + b;
        assert_eq!(c.value, 3.0);

        let a = Angle::<URadian>::new(1.0);
        let b = Angle::<UDegree>::new(180.0);
        let c = a + b;
        assert_eq!(c.value, 1.0 + core::f32::consts::PI);

        let mut a = Angle::<UDegree>::new(1.0);
        let b = Angle::<UDegree>::new(2.0);
        a += b;
        assert_eq!(a, Angle::<UDegree>::new(3.0));

        let mut a = Angle::<URadian>::new(1.0);
        let b = Angle::<UDegree>::new(180.0);
        a += b;
        assert_eq!(a, Angle::<URadian>::new(1.0 + core::f32::consts::PI));
    }

    #[test]
    fn sub() {
        let a = Angle::<UDegree>::new(1.0);
        let b = Angle::<UDegree>::new(2.0);
        let c = a - b;
        assert_eq!(c.value, -1.0);

        let a = Angle::<URadian>::new(1.0);
        let b = Angle::<UDegree>::new(180.0);
        let c = a - b;
        assert!(ulp_eq(c.value, 1.0 - core::f32::consts::PI));

        let mut a = Angle::<UDegree>::new(1.0);
        let b = Angle::<UDegree>::new(2.0);
        a -= b;
        assert_eq!(a, Angle::<UDegree>::new(-1.0));

        let mut a = Angle::<UDegree>::new(1.0);
        let b = Angle::<URadian>::new(1.0);
        a -= b;
        assert_eq!(a, Angle::<URadian>::new(1.0f32.to_radians() - 1.0));
    }

    #[test]
    fn mul() {
        let a = Angle::<URadian>::new(1.0);
        let b = a * 2.0;
        assert_eq!(b.value, 2.0);

        let a = Angle::<UDegree>::new(1.0);
        let b = 2.0 * a;
        assert_eq!(b.value, 2.0);
    }

    #[test]
    fn div() {
        let a = Angle::<UDegree>::new(1.0);
        let b = a / 2.0;

        let c = Angle::<UDegree>::new(90.0);
        let d = Angle::<URadian>::new(0.78);
        let e = c / d;

        assert_eq!(b.value, 0.5);
        assert_eq!(e.round(), 2.0);
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
