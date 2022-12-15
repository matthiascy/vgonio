use crate::units::{Angle, AngleUnit, Length, LengthUnit};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    ops::Sub,
    str::FromStr,
};

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given step.
#[derive(Debug, Copy, Clone)]
pub struct RangeByStepSize<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,

    /// Final value of the range.
    pub stop: T,

    /// Increment between two consecutive values of the range.
    pub step_size: T,
}

impl<T: Copy + Clone> PartialEq for RangeByStepSize<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.stop == other.stop && self.step_size == other.step_size
    }
}

impl<T: Copy + Clone> Eq for RangeByStepSize<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> RangeByStepSize<T> {
    /// Create a new range.
    pub fn new(start: T, stop: T, step_size: T) -> Self {
        Self {
            start,
            stop,
            step_size,
        }
    }

    /// Maps a function over the start and stop of the range.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> RangeByStepSize<U> {
        RangeByStepSize {
            start: f(self.start),
            stop: f(self.stop),
            step_size: f(self.step_size),
        }
    }

    /// Returns the span of the range.
    pub fn span(&self) -> T
    where
        T: Sub<Output = T>,
    {
        self.stop - self.start
    }
}

impl<T: Copy> From<[T; 3]> for RangeByStepSize<T> {
    fn from(vals: [T; 3]) -> Self {
        Self {
            start: vals[0],
            stop: vals[1],
            step_size: vals[2],
        }
    }
}

impl<T: Copy> From<RangeByStepSize<T>> for [T; 3] {
    fn from(range: RangeByStepSize<T>) -> Self { [range.start, range.stop, range.step_size] }
}

impl<T: Copy> From<(T, T, T)> for RangeByStepSize<T> {
    fn from(vals: (T, T, T)) -> Self {
        Self {
            start: vals.0,
            stop: vals.1,
            step_size: vals.2,
        }
    }
}

impl<T: Copy> From<RangeByStepSize<T>> for (T, T, T) {
    fn from(range: RangeByStepSize<T>) -> Self { (range.start, range.stop, range.step_size) }
}

impl<T: Default + Copy> Default for RangeByStepSize<T> {
    fn default() -> Self {
        Self {
            start: T::default(),
            stop: T::default(),
            step_size: T::default(),
        }
    }
}

impl<T> Display for RangeByStepSize<T>
where
    T: Display + Copy,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ~ {} per {}", self.start, self.stop, self.step_size)
    }
}

impl<T> Serialize for RangeByStepSize<T>
where
    T: Serialize + Copy + Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!(
            "{} ~ {} / {}",
            self.start, self.stop, self.step_size
        ))
    }
}

impl<'a, T> TryFrom<&'a str> for RangeByStepSize<T>
where
    T: Copy + FromStr,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let mut parts = value.split("~");
        let start = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {}", value))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range start value: {}", value))?;
        let mut parts = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {}", value))?
            .trim()
            .split("/");
        let stop = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {}", value))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range stop value: {}", value))?;
        let step_size = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {}", value))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range step size value: {}", value))?;
        Ok(Self {
            start,
            stop,
            step_size,
        })
    }
}

impl<'d, T> Deserialize<'d> for RangeByStepSize<T>
where
    T: Deserialize<'d> + Copy + FromStr,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        struct RangeByStepSizeVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T> serde::de::Visitor<'de> for RangeByStepSizeVisitor<T>
        where
            T: Copy + Deserialize<'de> + FromStr,
        {
            type Value = RangeByStepSize<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a range by step size in the from of 'start to stop per
step'"
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                RangeByStepSize::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }
        deserializer.deserialize_str(RangeByStepSizeVisitor::<T>(core::marker::PhantomData))
    }
}

macro impl_step_count_for_range_by_step_size($self:ident) {{
    #[cfg(debug_assertions)]
    assert!(
        $self.step_size.value() > 0.0,
        "step_size must not be greater than 0.0"
    );
    let count = $self.span() / $self.step_size;
    if count == 0.0 {
        1
    } else {
        count.ceil() as usize
    }
}}

impl RangeByStepSize<f32> {
    pub fn step_count(&self) -> usize {
        #[cfg(debug_assertions)]
        assert!(
            self.step_size > 0.0,
            "step_size must not be greater than 0.0"
        );
        let count = self.span() / self.step_size;
        if count == 0.0 {
            1
        } else {
            count.ceil() as usize
        }
    }
}

impl<A: AngleUnit> RangeByStepSize<Angle<A>> {
    pub fn step_count(&self) -> usize { impl_step_count_for_range_by_step_size!(self) }
}

impl<A: LengthUnit> RangeByStepSize<Length<A>> {
    pub fn step_count(&self) -> usize { impl_step_count_for_range_by_step_size!(self) }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given number of steps.
#[derive(Debug, Copy, Clone)]
pub struct RangeByStepCount<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,

    /// Final value of the range.
    pub stop: T,

    /// Number of samples.
    pub step_count: usize,
}

impl<T: Copy + Clone> PartialEq for RangeByStepCount<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.stop == other.stop && self.step_count == other.step_count
    }
}

impl<T: Copy + Clone> Eq for RangeByStepCount<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> RangeByStepCount<T> {
    /// Create a new range.
    pub fn new(start: T, stop: T, step_count: usize) -> Self {
        Self {
            start,
            stop,
            step_count,
        }
    }

    /// Maps a function over the start and stop of the range.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> RangeByStepCount<U> {
        RangeByStepCount {
            start: f(self.start),
            stop: f(self.stop),
            step_count: self.step_count,
        }
    }

    /// Returns the span of the range.
    pub const fn span(&self) -> T
    where
        T: ~const Sub<Output = T>,
    {
        self.stop - self.start
    }
}

impl<T: Copy> From<(T, T, usize)> for RangeByStepCount<T> {
    fn from(vals: (T, T, usize)) -> Self {
        Self {
            start: vals.0,
            stop: vals.1,
            step_count: vals.2,
        }
    }
}

impl<T: Copy> From<RangeByStepCount<T>> for (T, T, usize) {
    fn from(range: RangeByStepCount<T>) -> Self { (range.start, range.stop, range.step_count) }
}

impl<T: Default + Copy> Default for RangeByStepCount<T> {
    fn default() -> Self {
        Self {
            start: T::default(),
            stop: T::default(),
            step_count: 1,
        }
    }
}

impl<'a, T> TryFrom<&'a str> for RangeByStepCount<T>
where
    T: Copy + FromStr,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let mut parts = value.split("~");
        let start = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {}", value))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range start value: {}", value))?;
        let mut parts = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {}", value))?
            .split(",");
        let stop = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {}", value))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range stop value: {}", value))?;
        let step_count = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {}", value))?
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("Invalid range step count: {}", value))?;
        Ok(Self {
            start,
            stop,
            step_count,
        })
    }
}

impl<T> Serialize for RangeByStepCount<T>
where
    T: Copy + Serialize + Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!(
            "{} ~ {}, {}",
            self.start, self.stop, self.step_count
        ))
    }
}

impl<'d, T> Deserialize<'d> for RangeByStepCount<T>
where
    T: Deserialize<'d> + Copy + FromStr,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        struct RangeByStepCountVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T> serde::de::Visitor<'de> for RangeByStepCountVisitor<T>
        where
            T: Copy + Deserialize<'de> + FromStr,
        {
            type Value = RangeByStepCount<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a range by step size in the from of 'start to stop per
step'"
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                RangeByStepCount::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }
        deserializer.deserialize_str(RangeByStepCountVisitor::<T>(core::marker::PhantomData))
    }
}

impl RangeByStepCount<f32> {
    /// Returns the step between two consecutive values in the range.
    #[inline]
    pub const fn step_size(&self) -> f32 { self.span() / self.step_count as f32 }
}

impl<A: LengthUnit> RangeByStepCount<Length<A>> {
    /// Returns the size of the step bewteen two consecutive values in the
    /// range.
    #[inline]
    pub fn step_size(&self) -> Length<A> { self.span() / self.step_count as f32 }
}

#[test]
fn range_by_step_size_try_from_str() {
    let range = RangeByStepSize::<f32>::try_from("0.0 ~ 1.0 / 0.1").unwrap();
    assert_eq!(range.start, 0.0);
    assert_eq!(range.stop, 1.0);
    assert_eq!(range.step_size, 0.1);
}

#[test]
fn range_by_step_size_try_from_str_with_angle() {
    use crate::units::{radians, Radians, URadian};
    let range = RangeByStepSize::<Radians>::try_from("0.0rad ~ 1.0rad / 0.1rad").unwrap();
    assert_eq!(range.start, radians!(0.0));
    assert_eq!(range.stop, radians!(1.0));
    assert_eq!(range.step_size, Angle::<URadian>::new(0.1));
}

#[test]
fn range_by_step_count_try_from_str() {
    let range = RangeByStepCount::<f32>::try_from("0.0 ~ 1.0, 19").unwrap();
    assert_eq!(range.start, 0.0);
    assert_eq!(range.stop, 1.0);
    assert_eq!(range.step_count, 19);
}

#[test]
fn range_by_step_count_try_from_str_with_angle() {
    use crate::units::{radians, Radians};
    let range = RangeByStepCount::<Radians>::try_from("0.0rad ~ 1.0rad, 4").unwrap();
    assert_eq!(range.start, radians!(0.0));
    assert_eq!(range.stop, radians!(1.0));
    assert_eq!(range.step_count, 4);
}
