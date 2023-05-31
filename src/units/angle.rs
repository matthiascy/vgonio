use crate::{math::NumericCast, ulp_eq};
use core::fmt::{Debug, Display};
use std::str::FromStr;

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
    const SYMBOLS: &'static [&'static str];

    /// The conversion factor from radians.
    const FACTOR_FROM_RAD: f32;

    /// The conversion factor to radians.
    const FACTOR_TO_RAD: f32 = 1.0 / Self::FACTOR_FROM_RAD;

    /// The conversion factor from degrees.
    const FACTOR_FROM_DEG: f32;

    /// The conversion factor to degrees.
    const FACTOR_TO_DEG: f32 = 1.0 / Self::FACTOR_FROM_DEG;

    /// The value of 2*PI in the unit.
    const TAU: f32;

    /// The value of PI in the unit.
    const PI: f32;

    /// The value of PI/2 in the unit.
    const FRAC_PI_2: f32;
}

impl AngleUnit for URadian {
    const NAME: &'static str = "radian";
    const SYMBOLS: &'static [&'static str] = &["rad"];
    const FACTOR_FROM_RAD: f32 = 1.0;
    const FACTOR_FROM_DEG: f32 = 1.0 / 180.0 * std::f32::consts::PI;
    const TAU: f32 = 2.0 * std::f32::consts::PI;
    const PI: f32 = std::f32::consts::PI;
    const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
}

impl AngleUnit for UDegree {
    const NAME: &'static str = "degree";
    const SYMBOLS: &'static [&'static str] = &["deg", "°"];
    const FACTOR_FROM_RAD: f32 = 180.0 / std::f32::consts::PI;
    const FACTOR_FROM_DEG: f32 = 1.0;
    const TAU: f32 = 360.0;
    const PI: f32 = 180.0;
    const FRAC_PI_2: f32 = 90.0;
}

/// Angle with unit.
#[derive(Copy, Clone)]
pub struct Angle<A: AngleUnit> {
    pub(crate) value: f32,
    pub(crate) unit: core::marker::PhantomData<A>,
}

impl<A: AngleUnit> Debug for Angle<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.value, A::SYMBOLS[0])
    }
}

impl<A: AngleUnit> Display for Angle<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.value, A::SYMBOLS[0])
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
        self_rad.partial_cmp(&other_rad)
    }
}

impl<A: AngleUnit> Angle<A> {
    /// Zero angle.
    pub const ZERO: Self = Self::new(0.0);

    /// PI in radians.
    pub const PI: Self = Self::new(A::PI);

    /// PI/2 in radians.
    pub const HALF_PI: Self = Self::new(A::FRAC_PI_2);

    /// 2 * PI in radians.
    pub const TWO_PI: Self = Self::new(A::TAU);

    /// 2 * PI in radians.
    pub const TAU: Self = Self::new(A::TAU);

    /// Create a new angle with unit.
    pub const fn new(value: f32) -> Self {
        Angle {
            value,
            unit: core::marker::PhantomData,
        }
    }

    /// Get the value of the angle.
    pub const fn value(&self) -> f32 { self.value }

    /// Determines whether the angle is greater than zero.
    #[inline(always)]
    pub const fn is_positive(&self) -> bool { self.value > 0.0 }

    /// Prints the angle in human readable format in degrees.
    #[inline]
    pub fn prettified(&self) -> String {
        format!("{}{}", self.value * A::FACTOR_TO_DEG, UDegree::SYMBOLS[1])
    }

    /// Converts the angle to radians.
    #[inline]
    pub const fn to_radians(&self) -> Angle<URadian> { Angle::new(self.value * A::FACTOR_TO_RAD) }

    /// Converts the angle to degrees.
    #[inline]
    pub const fn to_degrees(&self) -> Angle<UDegree> { Angle::new(self.value * A::FACTOR_TO_DEG) }

    super::forward_f32_methods!(
        abs,
        "Returns the absolute value of the angle.";
        ceil,
        "Returns the smallest angle greater than or equal to `self`.";
        round,
        "Returns the nearest value to `self`. Round half-way cases away from 0.0.";
        trunc,
        "Returns the integer part of `self`. Non-integer numbers are always truncated towards \
         zero.";
        fract,
        "Returns the fractional part of `self`.";
        sqrt,
        "Returns the square root of the angle.";
        cbrt,
        "returns the cube root of the angle."
    );

    /// Clamps the angle between `min` and `max`.
    #[inline]
    pub fn clamp<B, C>(self, min: Angle<B>, max: Angle<C>) -> Self
    where
        B: AngleUnit,
        C: AngleUnit,
        Self: From<Angle<URadian>>,
    {
        let value = self.value * A::FACTOR_TO_RAD;

        Angle::<URadian>::new(
            value.clamp(min.value * B::FACTOR_TO_RAD, max.value * C::FACTOR_TO_RAD),
        )
        .into()
    }

    /// Wraps the angle to the range `[0, 2*PI)`.
    pub fn wrap_to_tau(self) -> Self { Self::new((self.value % A::TAU + A::TAU) % A::TAU) }

    /// Returns the opposite angle; i.e. the angle that is formed by the same
    /// initial and terminal sides but lies in the opposite direction.
    /// The result is always wrapped to the range [0, 2PI).
    pub fn opposite(self) -> Self { Self::new(self.value + A::PI).wrap_to_tau() }

    /// Returns the smallest angle between `self` and `other`.
    pub fn min(self, other: Self) -> Self { Self::new(self.value.min(other.value)) }

    /// Returns the largest angle between `self` and `other`.
    pub fn max(self, other: Self) -> Self { Self::new(self.value.max(other.value)) }
}

impl<A: AngleUnit> From<f32> for Angle<A> {
    fn from(value: f32) -> Self { Angle::new(value) }
}

impl<'a, A: AngleUnit> TryFrom<&'a str> for Angle<A> {
    type Error = &'static str;

    fn try_from(s: &'a str) -> Result<Self, Self::Error> {
        let bytes = s.trim().as_bytes();
        let i = super::findr_first_non_ascii_alphabetic(bytes)
            .ok_or("no unit found in angle string")?;
        let value = std::str::from_utf8(&bytes[..i])
            .map_err(|_| "invalid angle string")?
            .trim()
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

impl<A: AngleUnit> FromStr for Angle<A> {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> { Self::try_from(s) }
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

trait Cond<const B: bool> {}

struct SameUnit<A: AngleUnit, B: AngleUnit>(core::marker::PhantomData<(A, B)>);

impl Cond<true> for SameUnit<URadian, URadian> {}
impl Cond<true> for SameUnit<UDegree, UDegree> {}
impl Cond<false> for SameUnit<URadian, UDegree> {}
impl Cond<false> for SameUnit<UDegree, URadian> {}

impl<A: AngleUnit> From<Angle<URadian>> for Angle<A>
where
    SameUnit<A, URadian>: Cond<false>,
{
    fn from(value: Angle<URadian>) -> Self { Angle::new(value.value * A::FACTOR_FROM_RAD) }
}

impl<A: AngleUnit> From<Angle<UDegree>> for Angle<A>
where
    SameUnit<A, UDegree>: Cond<false>,
{
    fn from(value: Angle<UDegree>) -> Self { Angle::new(value.value * A::FACTOR_FROM_DEG) }
}

impl Angle<URadian> {
    /// Converts to degrees.
    pub fn in_degrees(&self) -> Angle<UDegree> { Angle::new(self.value * UDegree::FACTOR_FROM_RAD) }
    /// Computes the sine of the angle.
    pub fn sin(&self) -> f32 { self.value.sin() }
    /// Computes the hyperbolic sine of the angle.
    pub fn sinh(&self) -> f32 { self.value.sinh() }
    /// Computes the cosine of the angle.
    pub fn cos(&self) -> f32 { self.value.cos() }
    /// Computes the hyperbolic cosine of the angle.
    pub fn cosh(&self) -> f32 { self.value.cosh() }
    /// Computes the tangent of the angle.
    pub fn tan(&self) -> f32 { self.value.tan() }
    /// Computes the hyperbolic tangent of the angle.
    pub fn tanh(&self) -> f32 { self.value.tanh() }
}

impl Angle<UDegree> {
    /// Converts to radians.
    pub const fn in_radians(&self) -> Angle<URadian> {
        Angle::new(self.value * URadian::FACTOR_FROM_DEG)
    }
    /// Computes the sine of the angle.
    pub fn sin(&self) -> f32 { self.value.to_radians().sin() }
    /// Computes the hyperbolic sine of the angle.
    pub fn sinh(&self) -> f32 { self.value.to_radians().sinh() }
    /// Computes the cosine of the angle.
    pub fn cos(&self) -> f32 { self.value.to_radians().cos() }
    /// Computes the hyperbolic cosine of the angle.
    pub fn cosh(&self) -> f32 { self.value.to_radians().cosh() }
    /// Computes the tangent of the angle.
    pub fn tan(&self) -> f32 { self.value.to_radians().tan() }
    /// Computes the hyperbolic tangent of the angle.
    pub fn tanh(&self) -> f32 { self.value.to_radians().tanh() }
}

/// Type alias for `Angle<Radian>`.
pub type Radians = Angle<URadian>;
/// Type alias for `Angle<URadian>`.
pub type Rads = Angle<URadian>;

/// Type alias for `Angle<Degree>`.
pub type Degrees = Angle<UDegree>;
/// Type alias for `Angle<Degree>`.
pub type Degs = Angle<UDegree>;

/// Helper creating a new `Angle<Radian>`.
pub macro radians($value:expr) {
    $crate::units::Angle::<$crate::units::URadian>::new($value)
}

/// Helper creating a new `Angle<Radian>`.
pub macro rad($value:expr) {
    $crate::units::Angle::<$crate::units::URadian>::new($value)
}

/// Helper creating a new `Angle<Degree>`.
pub macro degrees($value:expr) {
    $crate::units::Angle::<$crate::units::UDegree>::new($value)
}

/// Helper creating a new `Angle<Degree>`.
pub macro deg($value:expr) {
    $crate::units::Angle::<$crate::units::UDegree>::new($value)
}

super::impl_ops!(Add, Sub for Angle where A, B: AngleUnit);
super::impl_ops_with_f32!(Mul, Div for Angle where A: AngleUnit);

impl<A: AngleUnit, B: AngleUnit> core::ops::Div<Angle<B>> for Angle<A>
where
    Angle<A>: From<Angle<B>>,
{
    type Output = f32;

    fn div(self, rhs: Angle<B>) -> Self::Output {
        let other = Angle::<A>::from(rhs);
        self.value / other.value
    }
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

impl const NumericCast<Angle<URadian>> for usize {
    fn cast(&self) -> Angle<URadian> { Angle::new(*self as f32) }
}

impl<A: AngleUnit> const NumericCast<f32> for Angle<A> {
    fn cast(&self) -> f32 { self.value }
}

impl<A: AngleUnit> const NumericCast<Angle<A>> for f32 {
    fn cast(&self) -> Angle<A> { Angle::new(*self) }
}

#[cfg(test)]
mod angle_unit_tests {
    use super::*;
    use crate::ulp_eq;

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

        println!("{e}");

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

        let deserialized2: Degrees = serde_yaml::from_str("180.0degs").unwrap();
        assert_eq!(a, deserialized2);

        let deserialized3: Degrees = serde_yaml::from_str("180.0 degrees").unwrap();
        assert_eq!(a, deserialized3);
    }

    #[test]
    fn wrapping() {
        let a = Radians::new(0.0);
        assert_eq!(a.wrap_to_tau(), a);

        let b = Radians::PI;
        assert!(ulp_eq(b.wrap_to_tau().value, Radians::PI.value));

        let c = Radians::TAU;
        assert_eq!(c.wrap_to_tau(), a);

        let d = Radians::PI * 3.0;
        assert!(ulp_eq(d.wrap_to_tau().value, Radians::PI.value));

        let e = Radians::PI * -0.5;
        assert!(ulp_eq(e.wrap_to_tau().value, (Radians::PI * 1.5).value));
    }

    #[test]
    fn test_opposite_angle() {
        let angle = rad!(0.0);
        assert!(ulp_eq(angle.opposite().value, Rads::PI.value));

        let angle = Rads::PI;
        assert_eq!(angle.opposite(), Rads::ZERO);

        let angle = Rads::HALF_PI;
        assert!(ulp_eq(angle.opposite().value, std::f32::consts::PI * 1.5));

        let angle = Rads::PI * 1.5;
        assert!(ulp_eq(angle.opposite().value, std::f32::consts::PI * 0.5));

        let angle = rad!(std::f32::consts::PI * -0.5).wrap_to_tau();
        assert!(ulp_eq(angle.opposite().value, std::f32::consts::PI * 0.5));
    }
}
