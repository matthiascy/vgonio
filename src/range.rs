use crate::{
    math::NumericCast,
    units::{Angle, AngleUnit, Radians},
};
use approx::AbsDiffEq;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    ops::{Add, Div, Mul, RangeInclusive, Sub},
    str::FromStr,
};

/// Defines a right inclusive range [a, b] of values with a given step.
///
/// The range is always inclusive, even if the step size does not divide the
/// range evenly.
#[derive(Clone, Copy)]
pub struct RangeByStepSizeInclusive<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,
    /// Final value of the range.
    pub stop: T,
    /// Step size.
    pub step_size: T,
}

macro_rules! impl_range_by_step_size_sub_types {
    ($($range:ident<$T:ident>);*) => {
        $(
            impl<$T: Copy + Clone + Debug> Debug for $range<$T> {
                fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                    f.debug_struct(stringify!($range))
                        .field("start", &self.start)
                        .field("stop", &self.stop)
                        .field("step_size", &self.step_size)
                        .finish()
                }
            }

            impl<$T: Copy + Clone + Default> Default for $range<$T> {
                fn default() -> Self {
                    Self {
                        start: Default::default(),
                        stop: Default::default(),
                        step_size: Default::default(),
                    }
                }
            }

            impl<$T: Copy + Clone + PartialEq> PartialEq for $range<$T> {
                fn eq(&self, other: &Self) -> bool {
                    self.start == other.start
                    && self.stop == other.stop
                    && self.step_size == other.step_size
                }
            }

            impl<$T: Copy + Clone + PartialEq + Eq> Eq for $range<$T> {}

            impl<$T: Copy + Clone> $range<$T> {
                /// Creates a new range with the given start, stop and step size.
                pub const fn new(start: $T, stop: $T, step_size: $T) -> Self {
                    Self {
                        start,
                        stop,
                        step_size,
                    }
                }

                /// Returns the span of the range.
                pub fn span(&self) -> $T
                where
                    $T: Sub<Output = $T>,
                {
                    self.stop - self.start
                }

                /// Maps the range to a new range with a different type.
                pub fn map<F, U>(self, mut f: F) -> $range<U>
                where
                    F: FnMut($T) -> U,
                    U: Copy + Clone,
                {
                    $range::new(f(self.start), f(self.stop), f(self.step_size))
                }

                /// Returns the range as a rust range.
                pub fn range_bound(&self) -> std::ops::Range<$T> {
                    self.start..self.stop
                }

                /// Returns the range as a rust inclusive range.
                pub fn range_bound_inclusive(&self) -> std::ops::RangeInclusive<$T> {
                    self.start..=self.stop
                }
            }
        )*
    };
}

impl_range_by_step_size_sub_types!(RangeByStepSizeInclusive<T>; RangeByStepSizeExclusive<T>);

impl<T> RangeByStepSizeInclusive<T>
where
    T: ~const NumericCast<f32> + Copy + Clone + From<f32>,
{
    /// Returns all possible values of the range.
    pub fn values(&self) -> impl Iterator<Item = T> {
        let step_size = self.step_size.cast();
        let start = self.start.cast();
        let stop = self.stop.cast();
        let step_count = self.step_count();
        (0..step_count).map(move |i| (start + step_size * i as f32).min(stop).into())
    }

    /// Returns all possible values of the range in reverse order.
    pub fn values_rev(&self) -> impl Iterator<Item = T> {
        let step_size = self.step_size.cast();
        let start = self.start.cast();
        let stop = self.stop.cast();
        let step_count = self.step_count();
        (0..step_count)
            .rev()
            .map(move |i| (start + step_size * i as f32).min(stop).into())
    }
}

impl<T> RangeByStepSizeInclusive<T>
where
    T: ~const NumericCast<f32> + Copy + Clone,
{
    /// Returns the step count of the range.
    pub fn step_count(&self) -> usize {
        let step_size = self.step_size.cast();
        let start = self.start.cast();
        let stop = self.stop.cast();
        (((stop - start) / step_size).round() as usize).max(2) + 1
    }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given step.
#[derive(Copy, Clone)]
pub struct RangeByStepSizeExclusive<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,
    /// Final value of the range.
    pub stop: T,
    /// Step size.
    pub step_size: T,
}

impl<T> RangeByStepSizeExclusive<T>
where
    T: Div<Output = T> + Sub<Output = T> + ~const NumericCast<f32> + Copy + Clone,
{
    /// Returns the step count of the exclusive range.
    pub fn step_count(&self) -> usize {
        let step_size = self.step_size.cast();
        let start = self.start.cast();
        let stop = self.stop.cast();
        ((stop - start) / step_size).ceil() as usize
    }
}
impl<T> RangeByStepSizeExclusive<T>
where
    T: Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + NumericCast<f32>
        + Copy
        + Clone,
    usize: NumericCast<T>,
{
    /// Returns all possible values of the range.
    pub fn values(&self) -> impl Iterator<Item = T>
where {
        let step_size = self.step_size;
        let start = self.start;
        let step_count = self.step_count();
        (0..step_count).map(move |i| start + step_size * i.cast())
    }
}

macro_rules! impl_serialisation {
    (@inclusive $range:ident<$T:ident>, $split_char:literal, $step_type:ident, $step:ident) => {
        impl<$T> Display for $range<$T>
        where
            $T: Display + Copy + Clone,
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "{} .. ={} {} {}",
                    self.start, self.stop, $split_char, self.$step
                )
            }
        }

        impl<'a, $T> TryFrom<&'a str> for $range<$T>
        where
            $T: Copy + Clone + FromStr,
        {
            type Error = String;

            fn try_from(value: &'a str) -> Result<Self, Self::Error> {
                let mut parts = value.split(&"..");
                let start = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .parse::<$T>()
                    .map_err(|_| format!("Invalid range start value: {value}"))?;
                let mut parts = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .split($split_char);
                let stop = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .trim_matches('=')
                    .parse::<$T>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                let step = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .parse::<$step_type>()
                    .map_err(|_| format!("Invalid range step size/count value: {value}"))?;
                Ok(Self::new(start, stop, step))
            }
        }

        impl<$T> Serialize for $range<$T>
        where
            $T: Serialize + Copy + Display + Clone,
        {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&format!("{}", self))
            }
        }

        impl<'d, $T> Deserialize<'d> for $range<$T>
        where
            $T: Deserialize<'d> + Copy + FromStr + Clone,
        {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'d>,
            {
                struct RangeVisitor<$T>(std::marker::PhantomData<$T>);

                impl<'de, $T> serde::de::Visitor<'de> for RangeVisitor<$T>
                where
                    $T: Copy + Deserialize<'de> + FromStr,
                {
                    type Value = $range<$T>;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        write!(
                            formatter,
                            "an inclusive range by step size in the form of \"start .. =stop {} \
                             step\"",
                            $split_char
                        )
                    }

                    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        $range::<$T>::try_from(v).map_err(|e| E::custom(e))
                    }
                }
                deserializer.deserialize_str(RangeVisitor::<$T>(core::marker::PhantomData))
            }
        }
    };
    (@exclusive $range:ident<$T:ident>, $split_char:literal, $step_type:ident, $step:ident) => {
        impl<$T> Display for $range<$T>
        where
            $T: Display + Copy + Clone,
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "{} .. {} {} {}",
                    self.start, self.stop, $split_char, self.$step
                )
            }
        }

        impl<'a, $T> TryFrom<&'a str> for $range<$T>
        where
            $T: Copy + Clone + FromStr,
        {
            type Error = String;

            fn try_from(value: &'a str) -> Result<Self, Self::Error> {
                let mut parts = value.split(&"..");
                let start = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .parse::<$T>()
                    .map_err(|_| format!("Invalid range start value: {value}"))?;
                let mut parts = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .split($split_char);
                let stop = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .parse::<$T>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                let step = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .parse::<$step_type>()
                    .map_err(|_| format!("Invalid range step size/count value: {value}"))?;
                Ok(Self::new(start, stop, step))
            }
        }

        impl<$T> Serialize for $range<$T>
        where
            $T: Serialize + Copy + Display + Clone,
        {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&format!("{}", self))
            }
        }

        impl<'d, $T> Deserialize<'d> for $range<$T>
        where
            $T: Deserialize<'d> + Copy + FromStr + Clone,
        {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'d>,
            {
                struct RangeVisitor<$T>(std::marker::PhantomData<$T>);

                impl<'de, $T> serde::de::Visitor<'de> for RangeVisitor<$T>
                where
                    $T: Copy + Deserialize<'de> + FromStr,
                {
                    type Value = $range<$T>;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        write!(
                            formatter,
                            "an inclusive range by step size in the form of \"start .. stop {} \
                             step\"",
                            $split_char
                        )
                    }

                    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        $range::<$T>::try_from(v).map_err(|e| E::custom(e))
                    }
                }
                deserializer.deserialize_str(RangeVisitor::<$T>(core::marker::PhantomData))
            }
        }
    };
}

impl_serialisation!(@exclusive RangeByStepSizeExclusive<T>, '/', T, step_size);
impl_serialisation!(@inclusive RangeByStepSizeInclusive<T>, '/', T, step_size);

impl<A: AngleUnit> RangeByStepSizeInclusive<Angle<A>> {
    /// Returns the number of steps in this range of angles.
    /// If the range's stop value is the same as the start value after wrapping
    /// it to the range [0, 2π), the range is considered to be a full circle.
    pub fn step_count_wrapped(&self) -> usize {
        if self
            .start
            .wrap_to_tau()
            .value
            .abs_diff_eq(&self.stop.wrap_to_tau().value, 1e-6)
        {
            (self.span() / self.step_size).round() as usize
        } else {
            ((self.span() / self.step_size).round() as usize + 1).max(2)
        }
    }

    /// Returns the index of the angle in the range of the measurement.
    ///
    /// The index of the bin is determined by testing if the angle falls
    /// inside half of the bin width from the bin boundary.
    pub fn index_of(&self, angle: Radians) -> usize {
        let angle = angle.to_degrees();
        let start = self.start.to_degrees();
        let step_size = self.step_size.to_degrees();
        ((angle - start).value / step_size.value).round() as usize % self.step_count_wrapped()
    }

    /// Returns the range of angles as a range of floating point values.
    pub fn range_bound_inclusive_f32(&self) -> RangeInclusive<f32> {
        self.start.value..=self.stop.value
    }
}

/// Defines a left inclusive, right inclusive range [a, b] of values with a
/// given number of steps.
#[derive(Copy, Clone)]
pub struct RangeByStepCountInclusive<T: Copy + Clone> {
    /// The start value of the range.
    pub start: T,
    /// The stop value of the range.
    pub stop: T,
    /// The number of steps in the range.
    pub step_count: usize,
}

macro_rules! impl_range_by_step_count_sub_types {
    ($($range:ident<$T:ident>);*) => {
        $(
            impl<$T: Copy + Clone + PartialEq> PartialEq for $range<$T> {
                fn eq(&self, other: &Self) -> bool {
                    self.start == other.start && self.stop == other.stop && self.step_count == other.step_count
                }
            }

            impl<$T: Copy + Clone> Eq for $range<$T> where $T: PartialEq + Eq {}

            impl<$T: Copy + Clone + Debug> Debug for $range<$T> {
                fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                    f.debug_struct(stringify!($range))
                        .field("start", &self.start)
                        .field("stop", &self.stop)
                        .field("step_count", &self.step_count)
                        .finish()
                }
            }

            impl<$T: Copy + Clone> $range<$T> {
                /// Creates a new range with the given start and stop values and the given number of steps.
                pub const fn new(start: $T, stop: $T, step_count: usize) -> Self {
                    Self { start, stop, step_count }
                }

                /// Maps the start and stop values of this range using the given function.
                pub fn map(&self, f: impl Fn($T) -> $T) -> Self {
                    Self::new(f(self.start), f(self.stop), self.step_count)
                }

                /// Returns the span of values in this range.
                pub fn span(&self) -> $T
                where
                    $T: Sub<Output = $T>
                {
                    self.stop - self.start
                }

                /// Returns the range of values as a range of exclusive values.
                pub fn range_bound(&self) -> std::ops::Range<$T> {
                    self.start..self.stop
                }

                /// Returns the range of values as a range of inclusive values.
                pub fn range_bound_inclusive(&self) -> std::ops::RangeInclusive<$T> {
                    self.start..=self.stop
                }
            }
        )*
    };
}

impl_range_by_step_count_sub_types!(
    RangeByStepCountInclusive<T>;
    RangeByStepCountExclusive<T>
);

impl<T> RangeByStepCountInclusive<T>
where
    T: NumericCast<f32> + Copy + Clone + Sub<Output = T>,
    f32: NumericCast<T>,
{
    /// Returns the step size of this range.
    pub fn step_size(&self) -> T { (self.span().cast() / (self.step_count as f32 - 1.0f32)).cast() }
}

impl<T: Copy + Clone> RangeByStepCountInclusive<T>
where
    T: Add<Output = T> + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + NumericCast<f32>,
    f32: NumericCast<T>,
    usize: NumericCast<T>,
{
    /// Returns all possible values in this range.
    pub fn values(&self) -> impl Iterator<Item = T> {
        let step_size = self.step_size();
        let start = self.start;
        (0..self.step_count).map(move |i| start + step_size * i.cast())
    }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given number of steps.
#[derive(Copy, Clone)]
pub struct RangeByStepCountExclusive<T: Copy + Clone> {
    /// The start value of the range.
    pub start: T,
    /// The stop value of the range.
    pub stop: T,
    /// The number of steps in the range.
    pub step_count: usize,
}

impl<T> RangeByStepCountExclusive<T>
where
    T: NumericCast<f32> + Copy + Clone + Sub<Output = T>,
    f32: NumericCast<T>,
{
    /// Returns the step size of this range.
    pub fn step_size(&self) -> T { (self.span().cast() / self.step_count as f32).cast() }
}

impl<T: Copy + Clone> RangeByStepCountExclusive<T>
where
    T: Add<Output = T> + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + NumericCast<f32>,
    f32: NumericCast<T>,
    usize: NumericCast<T>,
{
    /// Returns all possible values in this range.
    pub fn values(&self) -> impl Iterator<Item = T> + '_ {
        (0..self.step_count).map(move |i| self.start + self.step_size() * i.cast())
    }
}

impl_serialisation!(@exclusive RangeByStepCountExclusive<T>, '|', usize, step_count);
impl_serialisation!(@inclusive RangeByStepCountInclusive<T>, '|', usize, step_count);

/// Defines a range from a start value to a stop value with a given step size.
///
/// Left of the range is always inclusive, right of the range can be inclusive
/// or exclusive.
#[derive(Copy, Clone)]
pub enum RangeByStepSize<T: Copy + Clone> {
    /// Defines a right inclusive range.
    Inclusive(RangeByStepSizeInclusive<T>),
    /// Defines a right exclusive range.
    Exclusive(RangeByStepSizeExclusive<T>),
}

impl<T> Serialize for RangeByStepSize<T>
where
    T: Serialize + Copy + Clone + Display,
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
        let mut parts = value.split(&"..");
        let start = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
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

        match stop_str.chars().next() {
            Some('=') => {
                // Inclusive
                let stop = stop_str[1..]
                    .parse::<T>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                Ok(Self::Inclusive(RangeByStepSizeInclusive::new(
                    start, stop, step_size,
                )))
            }
            Some(_) => {
                // Exclusive
                let stop = stop_str[..]
                    .parse::<T>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                Ok(Self::Exclusive(RangeByStepSizeExclusive::new(
                    start, stop, step_size,
                )))
            }
            None => Err(format!("Invalid range stop value: {value}")),
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
                    "a range by step size in the form of 'start .. stop / step' or 'start .. stop \
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

/// Defines a range of values with a given number of steps.
#[derive(Copy, Clone)]
pub enum RangeByStepCount<T: Copy + Clone> {
    /// Defines a right inclusive range.
    Inclusive(RangeByStepCountInclusive<T>),
    /// Defines a right exclusive range.
    Exclusive(RangeByStepCountExclusive<T>),
}

impl<T: Copy + Clone> RangeByStepCount<T> {
    /// Returns the step size of the range.
    pub fn step_size(&self) -> T
    where
        T: NumericCast<f32> + Copy + Clone + Sub<Output = T>,
        f32: NumericCast<T>,
    {
        match self {
            RangeByStepCount::Inclusive(range) => range.step_size(),
            RangeByStepCount::Exclusive(range) => range.step_size(),
        }
    }
}

impl<T> Serialize for RangeByStepCount<T>
where
    T: Serialize + Copy + Clone + Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            RangeByStepCount::Inclusive(range) => range.serialize(serializer),
            RangeByStepCount::Exclusive(range) => range.serialize(serializer),
        }
    }
}

impl<'a, T> TryFrom<&'a str> for RangeByStepCount<T>
where
    T: Copy + FromStr,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let mut parts = value.split(&"..");
        let start = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<T>()
            .map_err(|_| format!("Invalid range start value: {value}"))?;
        let mut parts = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .split('|');
        let stop_str = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim();
        let step_count = parts
            .next()
            .ok_or_else(|| format!("Invalid range: {value}"))?
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("Invalid range step size value: {value}"))?;

        match stop_str.chars().next() {
            Some('=') => {
                // Inclusive
                let stop = stop_str[1..]
                    .parse::<T>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                Ok(Self::Inclusive(RangeByStepCountInclusive::new(
                    start, stop, step_count,
                )))
            }
            Some(_) => {
                // Exclusive
                let stop = stop_str[..]
                    .parse::<T>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                Ok(Self::Exclusive(RangeByStepCountExclusive::new(
                    start, stop, step_count,
                )))
            }
            None => Err(format!("Invalid range stop value: {value}")),
        }
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
                    "a range by step size in the form of '[start, stop] | step or '[start, stop) \
                     | step'"
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

/// Defines a range of values either by a step size or by a step count.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SteppedRange<T: Copy + Clone> {
    /// Defines a range by a step size.
    ByStepSize(RangeByStepSize<T>),
    /// Defines a range by a step count.
    ByStepCount(RangeByStepCount<T>),
}

impl<T> Serialize for SteppedRange<T>
where
    T: Serialize + Copy + Clone + Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            SteppedRange::ByStepSize(range) => range.serialize(serializer),
            SteppedRange::ByStepCount(range) => range.serialize(serializer),
        }
    }
}

impl<'a, T> TryFrom<&'a str> for SteppedRange<T>
where
    T: Copy + FromStr,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        if value.contains('|') {
            Ok(Self::ByStepCount(RangeByStepCount::try_from(value)?))
        } else {
            Ok(Self::ByStepSize(RangeByStepSize::try_from(value)?))
        }
    }
}

impl<'d, T> Deserialize<'d> for SteppedRange<T>
where
    T: Deserialize<'d> + Copy + FromStr,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'d>,
    {
        struct SteppedRangeVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T> serde::de::Visitor<'de> for SteppedRangeVisitor<T>
        where
            T: Copy + Deserialize<'de> + FromStr,
        {
            type Value = SteppedRange<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a range by step size in the form of '[start, stop] | step_count', '[start, \
                     stop) | step_count', '[start, stop) / step_size' or '[start, stop] / \
                     step_size'"
                )
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                SteppedRange::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }
        deserializer.deserialize_str(SteppedRangeVisitor::<T>(core::marker::PhantomData))
    }
}

macro_rules! impl_range_by_step {
    ($($range:ident<$T:ident>, $step_size_or_count:ident, $step_size_or_count_type:ty, #[$comments:meta]);*) => {
        $(
            paste::paste! {
                impl<$T: Copy + Clone + PartialEq> PartialEq for $range<$T> {
                    fn eq(&self, other: &Self) -> bool {
                        match (self, other) {
                            (Self::Inclusive(range), Self::Inclusive(other)) => range == other,
                            (Self::Exclusive(range), Self::Exclusive(other)) => range == other,
                            _ => false,
                        }
                    }
                }

                impl<$T: Copy + Clone + PartialEq + Eq> Eq for $range<$T> {}

                impl<$T: Copy + Clone> $range<$T> {
                    /// Creates a new right inclusive range.
                    pub const fn new_inclusive(start: $T, stop: $T, [<step_ $step_size_or_count>]: $step_size_or_count_type) -> Self {
                        Self::Inclusive([<$range Inclusive>]::new(start, stop, [<step_ $step_size_or_count>]))
                    }

                    /// Creates a new right exclusive range.
                    pub const fn new_exclusive(start: $T, stop: $T, [<step_ $step_size_or_count>]: $step_size_or_count_type) -> Self {
                        Self::Exclusive([<$range Exclusive>]::new(start, stop, [<step_ $step_size_or_count>]))
                    }

                    #[$comments]
                    pub const fn [<step_ $step_size_or_count>](&self) -> $step_size_or_count_type {
                        match self {
                            Self::Inclusive(range) => range.[<step_ $step_size_or_count>],
                            Self::Exclusive(range) => range.[<step_ $step_size_or_count>],
                        }
                    }

                    /// Returns true if the range is right inclusive.
                    pub const fn is_inclusive(&self) -> bool {
                        match self {
                            Self::Inclusive(_) => true,
                            Self::Exclusive(_) => false,
                        }
                    }

                    /// Returns true if the range is right exclusive.
                    pub const fn is_exclusive(&self) -> bool {
                        match self {
                            Self::Inclusive(_) => false,
                            Self::Exclusive(_) => true,
                        }
                    }

                    /// Returns the start of the range.
                    pub const fn start(&self) -> $T {
                        match self {
                            Self::Inclusive(range) => range.start,
                            Self::Exclusive(range) => range.start,
                        }
                    }

                    /// Returns the stop of the range.
                    pub const fn stop(&self) -> $T {
                        match self {
                            Self::Inclusive(range) => range.stop,
                            Self::Exclusive(range) => range.stop,
                        }
                    }
                }

                impl<$T: Copy + Clone + Debug> Debug for $range<$T> {
                    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                        f.debug_struct(stringify!($range))
                            .field("start", &self.start())
                            .field("stop", &self.stop())
                            .field(stringify!($step_size_or_count), &self.[<step_ $step_size_or_count>]())
                            .finish()
                    }
                }

                impl<$T: Copy + Clone + Display> Display for $range<$T> {
                    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                        match self {
                            Self::Inclusive(range) => write!(f, "{}", range),
                            Self::Exclusive(range) => write!(f, "{}", range),
                        }
                    }
                }
            }
        )*
    };
}

impl_range_by_step!(
    RangeByStepCount<T>, count, usize, #[doc = "Returns the step count of the range."];
    RangeByStepSize<T>, size, T, #[doc = "Returns the step size of the range."]
);

#[cfg(test)]
mod range_by_step_count_tests {
    use super::*;

    #[test]
    fn serialization_inclusive() {
        let range = RangeByStepCountInclusive::new(0, 10, 3);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(serialized, "0 .. =10 | 3\n");
        let deserialized: RangeByStepCountInclusive<i32> =
            serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, range);

        let range = RangeByStepCount::new_inclusive(0, 10, 3);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(serialized, "0 .. =10 | 3\n");
        let deserialized: RangeByStepCount<i32> = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, range);
        assert!(deserialized.is_inclusive());
    }

    #[test]
    fn serialization_exclusive() {
        let range = RangeByStepCountExclusive::new(0, 10, 3);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(serialized, "0 .. 10 | 3\n");
        let deserialized: RangeByStepCountExclusive<i32> =
            serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, range);

        let range = RangeByStepCount::new_exclusive(0, 10, 3);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(serialized, "0 .. 10 | 3\n");
        let deserialized: RangeByStepCount<i32> = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, range);
        assert!(deserialized.is_exclusive());
    }

    #[test]
    fn test_step_size() {
        let range = RangeByStepCountInclusive::new(0, 10, 3);
        assert_eq!(range.step_size(), 5);

        let range = RangeByStepCountInclusive::new(0, 10, 4);
        assert_eq!(range.step_size(), 3);

        let range = RangeByStepCountInclusive::new(0, 10, 5);
        assert_eq!(range.step_size(), 2);

        let range = RangeByStepCountInclusive::new(1.0, 13.0, 7);
        assert_eq!(range.step_size(), 2.0);

        let range = RangeByStepCountExclusive::new(1.0, 13.0, 8);
        assert_eq!(range.step_size(), 1.5);
    }
}

#[cfg(test)]
mod range_by_step_size_tests {
    use super::*;
    use crate::units::{deg, degrees, rad, radians, Degrees, Rads};

    #[test]
    fn try_from_str_inclusive() {
        let range = RangeByStepSize::<f32>::try_from("0.0 .. =12.0 / 0.3").unwrap();
        assert_eq!(range.start(), 0.0);
        assert_eq!(range.stop(), 12.0);
        assert_eq!(range.step_size(), 0.3);
        assert!(!range.is_exclusive());

        let range = RangeByStepSizeInclusive::<f32>::try_from("0.0 .. =16.0 / 0.2").unwrap();
        assert_eq!(range.start, 0.0);
        assert_eq!(range.stop, 16.0);
        assert_eq!(range.step_size, 0.2);
    }

    #[test]
    fn try_from_str_exclusive() {
        let range = RangeByStepSize::<f32>::try_from("0.0 .. 1.0 / 0.1").unwrap();
        assert_eq!(range.start(), 0.0);
        assert_eq!(range.stop(), 1.0);
        assert_eq!(range.step_size(), 0.1);
        assert!(range.is_exclusive());

        let range = RangeByStepSizeExclusive::<f32>::try_from("0.0 .. 1.0 / 0.1").unwrap();
        assert_eq!(range.start, 0.0);
        assert_eq!(range.stop, 1.0);
        assert_eq!(range.step_size, 0.1);
    }

    #[test]
    fn try_from_str_inclusive_angle() {
        let range = RangeByStepSize::<Radians>::try_from("0.0 rad .. =10.0rad / 0.1deg").unwrap();
        assert_eq!(range.start(), radians!(0.0));
        assert_eq!(range.stop(), radians!(10.0));
        assert_eq!(range.step_size(), degrees!(0.1).to_radians());
        assert!(!range.is_exclusive());
    }

    #[test]
    fn try_from_str_exclusive_angle() {
        let range = RangeByStepSize::<Degrees>::try_from("0.0rad .. 360 deg / 10.0deg").unwrap();
        assert_eq!(range.start(), degrees!(0.0));
        assert_eq!(range.stop(), degrees!(360.0));
        assert_eq!(range.step_size(), degrees!(10.0));
        assert!(range.is_exclusive());
    }

    #[test]
    fn serialisation_exclusive() {
        let range = RangeByStepSize::new_exclusive(0.0, 20.0, 0.5);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(
            serialized, "0 .. 20 / 0.5\n",
            "serilisation failed: {}",
            serialized
        );
        let deserialized: RangeByStepSize<f32> = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(
            deserialized, range,
            "deserilisation failed: {}",
            deserialized
        );
    }

    #[test]
    fn deserialisation_exclusive() {
        let range_str = "0 .. 20 / 0.5";
        let deserialized: RangeByStepSize<f32> = serde_yaml::from_str(&range_str).unwrap();
        let range = RangeByStepSize::new_exclusive(0.0, 20.0, 0.5);
        assert_eq!(
            deserialized, range,
            "deserilisation failed: {}",
            deserialized
        );
    }

    #[test]
    fn serialisation_inclusive() {
        let range = RangeByStepSizeInclusive::new(0.0, 20.0, 0.5f32);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(
            serialized, "0 .. =20 / 0.5\n",
            "serilisation failed: {}",
            serialized
        );
        let deserialized: RangeByStepSizeInclusive<f32> =
            serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(
            deserialized, range,
            "deserilisation failed: {}",
            deserialized
        );
    }

    #[test]
    fn display() {
        let range = RangeByStepSize::new_exclusive(0.5, 20.5, 0.5);
        assert_eq!(format!("{}", range), "0.5 .. 20.5 / 0.5");
        let range = RangeByStepSize::new_inclusive(0.5, 20.5, 0.5);
        assert_eq!(format!("{}", range), "0.5 .. =20.5 / 0.5");
    }

    #[test]
    fn step_count() {
        let range = RangeByStepSizeInclusive::new(0.0, 10.0, 1.0);
        assert_eq!(range.step_count(), 11);
        let range = RangeByStepSizeInclusive::new(0.0, 10.0, 2.0);
        assert_eq!(range.step_count(), 6);

        let range = RangeByStepSizeInclusive::new(380.0, 780.0, 300.0);
        assert_eq!(range.step_count(), 3);

        let range =
            RangeByStepSizeInclusive::new(rad!(0.0), Rads::HALF_PI, deg!(30.0).to_radians());
        assert_eq!(range.step_count(), 4);
        assert_eq!(range.step_count_wrapped(), 4);

        let range = RangeByStepSizeInclusive::new(rad!(0.0), Rads::TAU, deg!(30.0).to_radians());
        assert_eq!(range.step_count(), 13);
        assert_eq!(range.step_count_wrapped(), 12);

        let range = RangeByStepSizeInclusive::new(deg!(0.0), deg!(360.0), deg!(90.0));
        assert_eq!(range.step_count(), 5);
        assert_eq!(range.step_count_wrapped(), 4);

        let range = RangeByStepSizeInclusive::new(0.0, 50.0, 12.5);
        assert_eq!(range.step_count(), 5);

        let range = RangeByStepSizeInclusive::new(0.0, 10.0, 4.0);
        assert_eq!(range.step_count(), 4);

        let range = RangeByStepSizeInclusive::new(0.0, 10.0, 2.5);
        assert_eq!(range.step_count(), 5);
    }
}
