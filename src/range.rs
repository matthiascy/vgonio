use crate::math::NumericCast;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    ops::{Deref, Div, Sub},
    str::FromStr,
};

/// Defines a range of values with a given step size or count.
struct RangeInner<T, U> {
    /// Initial value of the range.
    pub start: T,
    /// Final value of the range.
    pub stop: T,
    /// Step size or count.
    pub step_size_or_count: U,
}

impl<T: Clone, U: Clone> Clone for RangeInner<T, U> {
    fn clone(&self) -> Self {
        Self {
            start: self.start.clone(),
            stop: self.stop.clone(),
            step_size_or_count: self.step_size_or_count.clone(),
        }
    }
}

impl<T: Copy, U: Copy> Copy for RangeInner<T, U> {}

impl<T: Debug, U: Debug> Debug for RangeInner<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RangeInner")
            .field("start", &self.start)
            .field("stop", &self.stop)
            .field("step_size_or_count", &self.step_size_or_count)
            .finish()
    }
}

impl<T: Default, U: Default> Default for RangeInner<T, U> {
    fn default() -> Self {
        Self {
            start: Default::default(),
            stop: Default::default(),
            step_size_or_count: Default::default(),
        }
    }
}

impl<T: PartialEq, U: PartialEq> PartialEq for RangeInner<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start
            && self.stop == other.stop
            && self.step_size_or_count == other.step_size_or_count
    }
}

impl<T: PartialEq + Eq, U: PartialEq + Eq> Eq for RangeInner<T, U> {}

impl<T, U> RangeInner<T, U> {
    /// Returns the initial value of the range.
    pub const fn start(&self) -> &T { &self.start }

    /// Returns the final value of the range.
    pub const fn stop(&self) -> &T { &self.stop }

    /// Returns the span of the range.
    pub const fn span(&self) -> T
    where
        T: Sub<Output = T> + Copy,
    {
        self.stop - self.start
    }

    /// Returns all possible values in the range, including the start and stop
    /// values.
    pub fn values_inclusive(&self) -> impl Iterator<Item = T> { todo!() }

    /// Returns all possible values in the range, excluding the stop value.
    pub fn values_exclusive(&self) -> impl Iterator<Item = T> { todo!() }

    /// Returns the range as a `std::ops::Range`.
    pub fn range_bound(&self) -> std::ops::Range<T> { self.start..self.stop }

    /// Returns the range as a `std::ops::RangeInclusive`.
    pub fn range_bound_inclusive(&self) -> std::ops::RangeInclusive<T> { self.start..=self.stop }
}

impl<T> From<[T; 3]> for RangeInner<T, T> {
    fn from(value: [T; 3]) -> Self {
        let [start, stop, step_size_or_count] = value;
        Self {
            start,
            stop,
            step_size_or_count,
        }
    }
}

impl<T> From<RangeInner<T, T>> for [T; 3] {
    fn from(range: RangeInner<T, T>) -> Self { [range.start, range.stop, range.step_size_or_count] }
}

impl<T, U> From<(T, T, U)> for RangeInner<T, U> {
    fn from(value: (T, T, U)) -> Self {
        let (start, stop, step_size_or_count) = value;
        Self {
            start,
            stop,
            step_size_or_count,
        }
    }
}

impl<T, U> From<RangeInner<T, U>> for (T, T, U) {
    fn from(range: RangeInner<T, U>) -> Self { (range.start, range.stop, range.step_size_or_count) }
}

/// Defines a right inclusive range [a, b] of values with a given step.
///
/// The range is always inclusive, even if the step size does not divide the
/// range evenly.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct RangeByStepSizeInclusive<T: Copy + Clone> {
    inner: RangeInner<T, T>,
}

impl<T: Copy + Clone> Eq for RangeByStepSizeInclusive<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> Deref for RangeByStepSizeInclusive<T> {
    type Target = RangeInner<T, T>;

    fn deref(&self) -> &Self::Target { &self.inner }
}

macro impl_range_by_step_size_common_methods($self:ident, $T:ident) {
    /// Create a new range.
    pub const fn new(start: $T, stop: $T, step_size: $T) -> Self {
        Self {
            inner: RangeInner {
                start,
                stop,
                step_size_or_count: step_size,
            },
        }
    }

    /// Returns the step size of the range.
    pub const fn step_size(&self) -> &$T { &self.inner.step_size_or_count }
}

impl<T: Copy + Clone> RangeByStepSizeInclusive<T> {
    impl_range_by_step_size_common_methods!(self, T);

    /// Returns the step count of the range.
    pub fn step_count(&self) -> usize
    where
        T: Div<Output = T> + Sub<Output = T> + NumericCast<f32>,
    {
        let step_size = self.inner.step_size_or_count.cast();
        let start = self.inner.start.cast();
        let stop = self.inner.stop.cast();
        (((stop - start) / step_size).ceil() as usize).min(2)
    }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given step.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct RangeByStepSizeExclusive<T: Copy + Clone> {
    inner: RangeInner<T, T>,
}

impl<T: Copy + Clone> Eq for RangeByStepSizeExclusive<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> RangeByStepSizeExclusive<T> {
    impl_range_by_step_size_common_methods!(self, T);

    /// Returns the step count of the exclusive range.
    pub fn step_count(&self) -> usize
    where
        T: Div<Output = T> + Sub<Output = T> + NumericCast<f32>,
    {
        let step_size = self.inner.step_size_or_count.cast();
        let start = self.inner.start.cast();
        let stop = self.inner.stop.cast();
        ((stop - start) / step_size).ceil() as usize
    }
}

macro impl_display_serialisation($($t:ident<$tp:ident>, $string:literal, $symbol:literal);*) {
    $(
        impl<$tp> Display for $t<$tp>
        where
            $tp: Display + Copy,
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    stringify!($string),
                    self.inner.start, self.inner.stop, self.inner.step_size_or_count
                )
            }
        }

        impl<$tp> Serialize for $t<$tp>
        where
            $tp: Serialize + Copy + Display,
        {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&format!(stringify!($string), self.start, self.stop, self.step_size))
            }
        }

        impl<'a, $tp> TryFrom<&'a str> for RangeByStepSizeInclusive<$tp>
        where
            $tp: Copy + FromStr,
        {
            type Error = String;

            fn try_from(value: &'a str) -> Result<Self, Self::Error> {
                let mut parts = value.split(',');
                let start = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim_matches('[')
                    .parse::<$tp>()
                    .map_err(|_| format!("Invalid range start value: {value}"))?;
                let mut parts = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .split('/');
                let stop = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim_end_matches($symbol)
                    .parse::<$tp>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                let step_size = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .parse::<$tp>()
                    .map_err(|_| format!("Invalid range step size value: {value}"))?;
                Ok(Self::new(start, stop, step_size))
            }
        }

        impl<'d, $tp> Deserialize<'d> for $t<$tp>
        where
            $tp: Deserialize<'d> + Copy + FromStr,
        {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'d>,
            {
                struct RangeByStepSizeVisitor<$tp>(std::marker::PhantomData<$tp>);

                impl<'de, $tp> serde::de::Visitor<'de> for RangeByStepSizeVisitor<$tp>
                where
                    $tp: Copy + Deserialize<'de> + FromStr,
                {
                    type Value = RangeByStepSize<$tp>;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        write!(
                            formatter,
                            "a range by step size in the form of '[start, stop{} / step",
                            stringify!($symbol)
                        )
                    }

                    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        $t::<$tp>::try_from(v).map_err(|e| E::custom(e))
                    }
                }
                deserializer.deserialize_str(RangeByStepSizeVisitor::<$tp>(core::marker::PhantomData))
            }
        }
    )*
}

impl_display_serialisation!(RangeByStepSizeExclusive<T>, "[{}, {}) / {}", ')'; RangeByStepSizeInclusive<T>, "[{}, {}] / {}", ']');

/// Defines a range from a start value to a stop value with a given step size.
///
/// Left of the range is always inclusive, right of the range can be inclusive
/// or exclusive.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum RangeByStepSize<T> {
    Inclusive(RangeByStepSizeInclusive<T>),
    Exclusive(RangeByStepSizeExclusive<T>),
}

impl<T> Deref for RangeByStepSize<T> {
    type Target = RangeInner<T, T>;

    fn deref(&self) -> &Self::Target {
        match self {
            RangeByStepSize::Inclusive(range) => &range.inner,
            RangeByStepSize::Exclusive(range) => &range.inner,
        }
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
        match self {
            RangeByStepSize::Inclusive(range) => range.serialize(serializer),
            RangeByStepSize::Exclusive(range) => range.serialize(serializer),
        }
    }
}

impl<'a, T> TryFrom<&'a str> for RangeByStepSize<T>
where
    T: Copy + FromStr,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let mut parts = value.split(',');
        let start = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim_matches('[')
            .parse::<T>()
            .map_err(|_| format!("Invalid range start value: {value}"))?;
        let mut parts = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .split('/');

        let stop_str = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim();

        let step_size = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range step size value: {value}"))?;

        match stop_str.chars().last() {
            None => {
                return Err(format!("Invalid range stop value: {value}"));
            }
            Some(s) => {
                match s {
                    ']' => {
                        // Inclusive
                        let stop = stop_str[..stop_str.len() - 1]
                            .parse::<T>()
                            .map_err(|_| format!("Invalid range stop value: {value}"))?;
                        Ok(Self::Inclusive(RangeByStepSizeInclusive::new(
                            start, stop, step_size,
                        )))
                    }
                    ')' => {
                        // Exclusive
                        let stop = stop_str[..stop_str.len() - 1]
                            .parse::<T>()
                            .map_err(|_| format!("Invalid range stop value: {value}"))?;
                        Ok(Self::Exclusive(RangeByStepSizeExclusive::new(
                            start, stop, step_size,
                        )))
                    }
                    _ => {
                        return Err(format!("Invalid range stop value: {value}"));
                    }
                }
            }
        }
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
                    "a range by step size in the form of '[start, stop] / step or '[start, stop) \
                     / step'"
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

// ------

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

impl<A: AngleUnit> RangeByStepSizeExclusive<Angle<A>> {
    /// Returns the number of steps in this range of angles.
    pub fn step_count(&self) -> usize { impl_step_count_for_range_by_step_size!(self) }
}

impl<A: LengthMeasurement> RangeByStepSizeExclusive<Length<A>> {
    /// Returns the number of steps in this range of lengths.
    pub fn step_count(&self) -> usize { impl_step_count_for_range_by_step_size!(self) }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given number of steps.
#[derive(Debug, Copy, Clone)]
pub struct RangeByStepCountExclusive<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,

    /// Final value of the range.
    pub stop: T,

    /// Number of samples.
    pub step_count: usize,
}

impl<T: Copy + Clone> PartialEq for RangeByStepCountExclusive<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.stop == other.stop && self.step_count == other.step_count
    }
}

impl<T: Copy + Clone> Eq for RangeByStepCountExclusive<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> RangeByStepCountExclusive<T> {
    /// Create a new range.
    pub fn new(start: T, stop: T, step_count: usize) -> Self {
        Self {
            start,
            stop,
            step_count,
        }
    }
}

impl<T: Copy> From<(T, T, usize)> for RangeByStepCountExclusive<T> {
    fn from(vals: (T, T, usize)) -> Self {
        Self {
            start: vals.0,
            stop: vals.1,
            step_count: vals.2,
        }
    }
}

impl<T: Copy> From<RangeByStepCountExclusive<T>> for (T, T, usize) {
    fn from(range: RangeByStepCountExclusive<T>) -> Self {
        (range.start, range.stop, range.step_count)
    }
}

impl<T: Default + Copy> Default for RangeByStepCountExclusive<T> {
    fn default() -> Self {
        Self {
            start: T::default(),
            stop: T::default(),
            step_count: 1,
        }
    }
}

impl<'a, T> TryFrom<&'a str> for RangeByStepCountExclusive<T>
where
    T: Copy + FromStr,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let mut parts = value.split('~');
        let start = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range start value: {value}"))?;
        let mut parts = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .split(',');
        let stop = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range stop value: {value}"))?;
        let step_count = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("Invalid range step count: {value}"))?;
        Ok(Self {
            start,
            stop,
            step_count,
        })
    }
}

impl<T> Serialize for RangeByStepCountExclusive<T>
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

impl<'d, T> Deserialize<'d> for RangeByStepCountExclusive<T>
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
            type Value = RangeByStepCountExclusive<T>;

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
                RangeByStepCountExclusive::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }
        deserializer.deserialize_str(RangeByStepCountVisitor::<T>(core::marker::PhantomData))
    }
}

impl RangeByStepCountExclusive<f32> {
    /// Returns the step between two consecutive values in the range.
    #[inline]
    pub fn step_size(&self) -> f32 { self.span() / self.step_count as f32 }
}

impl<A: LengthMeasurement> RangeByStepCountExclusive<Length<A>> {
    /// Returns the size of the step bewteen two consecutive values in the
    /// range.
    #[inline]
    pub fn step_size(&self) -> Length<A> { self.span() / self.step_count as f32 }
}

#[test]
fn range_by_step_size_try_from_str() {
    let range = RangeByStepSizeExclusive::<f32>::try_from("0.0 ~ 1.0 / 0.1").unwrap();
    assert_eq!(range.start, 0.0);
    assert_eq!(range.stop, 1.0);
    assert_eq!(range.step_size, 0.1);
}

#[test]
fn range_by_step_size_try_from_str_with_angle() {
    use crate::units::{radians, Radians, URadian};
    let range = RangeByStepSizeExclusive::<Radians>::try_from("0.0rad ~ 1.0rad / 0.1rad").unwrap();
    assert_eq!(range.start, radians!(0.0));
    assert_eq!(range.stop, radians!(1.0));
    assert_eq!(range.step_size, Angle::<URadian>::new(0.1));
}

#[test]
fn range_by_step_count_try_from_str() {
    let range = RangeByStepCountExclusive::<f32>::try_from("0.0 ~ 1.0, 19").unwrap();
    assert_eq!(range.start, 0.0);
    assert_eq!(range.stop, 1.0);
    assert_eq!(range.step_count, 19);
}

#[test]
fn range_by_step_count_try_from_str_with_angle() {
    use crate::units::{radians, Radians};
    let range = RangeByStepCountExclusive::<Radians>::try_from("0.0rad ~ 1.0rad, 4").unwrap();
    assert_eq!(range.start, radians!(0.0));
    assert_eq!(range.stop, radians!(1.0));
    assert_eq!(range.step_count, 4);
}

pub enum RangeByStepCount<T> {
    Inclusive(RangeByStepCountInclusive<T>),
    Exclusive(RangeByStepCountExclusive<T>),
}

pub enum SteppedRange<T> {
    ByStepSize(RangeByStepSize<T>),
    ByStepCount(RangeByStepCount<T>),
}

/// Defines a inclusive range [a, b] of values with a given step count.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct RangeByStepCountInclusive<T: Copy + Clone> {
    inner: RangeInner<T, usize>,
}

impl<T: Copy + Clone> Eq for RangeByStepCountInclusive<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> Deref for RangeByStepCountInclusive<T> {
    type Target = RangeInner<T, usize>;

    fn deref(&self) -> &Self::Target { &self.inner }
}

impl<T: Copy + Clone> RangeByStepCountInclusive<T> {
    /// Create a new range.
    pub const fn new(start: T, stop: T, step_count: usize) -> Self {
        Self {
            inner: RangeInner {
                start,
                stop,
                step_size_or_count: step_count,
            },
        }
    }

    /// Returns the step count of the range.
    pub const fn step_count(&self) -> usize { self.inner.step_size_or_count }

    /// Returns the step size of the range.
    pub fn step_size(&self) -> T
    where
        T: Div<Output = T> + Sub<Output = T>,
        usize: NumericCast<T>,
    {
        let step_size = (self.inner.stop - self.inner.start) / self.inner.step_size_or_count.cast();
        step_size
    }
}

impl<T> Display for RangeByStepCountInclusive<T>
where
    T: Display + Copy,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}, {}], {}",
            self.inner.start, self.inner.stop, self.inner.step_size_or_count
        )
    }
}

#[cfg(test)]
mod range_by_step_count_inclusive_tests {
    use super::*;

    #[test]
    fn test_step_size() {
        let range = RangeByStepCountInclusive::new(0, 10, 3);
        assert_eq!(range.step_size(), 3);

        let range = RangeByStepCountInclusive::new(0, 10, 4);
        assert_eq!(range.step_size(), 2);

        let range = RangeByStepCountInclusive::new(0, 10, 5);
        assert_eq!(range.step_size(), 2);

        let range = RangeByStepCountInclusive::new(1.0, 13.0, 6);
        assert_eq!(range.step_size(), 2.0);
    }
}
