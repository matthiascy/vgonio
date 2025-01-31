//! Defines types for ranges of values with a given step size or step count.

#[cfg(feature = "io")]
use crate::units::Nanometres;
use crate::{
    math,
    math::NumericCast,
    units::{Angle, AngleUnit, Radians},
};
use approx::AbsDiffEq;
#[cfg(feature = "serde")]
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
pub struct StepRangeIncl<T: Copy + Clone> {
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

impl_range_by_step_size_sub_types!(StepRangeIncl<T>; StepRangeExcl<T>);

impl<T> StepRangeIncl<T>
where
    T: NumericCast<f32> + Copy + Clone + From<f32>,
{
    /// Returns all possible values of the range.
    pub fn values(&self) -> impl ExactSizeIterator<Item = T> {
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

impl<T> StepRangeIncl<T>
where
    T: NumericCast<f32> + Copy + Clone,
{
    /// Returns the step count of the range.
    pub fn step_count(&self) -> usize {
        let step_size = self.step_size.cast();
        let start = self.start.cast();
        let stop = self.stop.cast();
        if math::ulp_eq(stop, start) {
            1
        } else if math::ulp_eq(step_size, stop - start) {
            2
        } else {
            (((stop - start) / step_size).round() as usize).max(2) + 1
        }
    }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given step.
#[derive(Copy, Clone)]
pub struct StepRangeExcl<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,
    /// Final value of the range.
    pub stop: T,
    /// Step size.
    pub step_size: T,
}

impl<T> StepRangeExcl<T>
where
    T: Div<Output = T> + Sub<Output = T> + NumericCast<f32> + Copy + Clone,
{
    /// Returns the step count of the exclusive range.
    pub fn step_count(&self) -> usize {
        let step_size = self.step_size.cast();
        let start = self.start.cast();
        let stop = self.stop.cast();
        ((stop - start) / step_size).ceil() as usize
    }
}
impl<T> StepRangeExcl<T>
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

        #[cfg(feature = "serde")]
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

        #[cfg(feature = "serde")]
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

        #[cfg(feature = "serde")]
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

        #[cfg(feature = "serde")]
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

impl_serialisation!(@exclusive StepRangeExcl<T>, '/', T, step_size);
impl_serialisation!(@inclusive StepRangeIncl<T>, '/', T, step_size);

macro_rules! impl_angle_related_common_methods {
    (@by_step_size $($range:ident, $inclusive:literal);*) => {
        $(
            impl<A: AngleUnit> $range<Angle<A>> {
                /// Pretty prints the range of angles.
                pub fn pretty_print(&self) -> String {
                    format!(
                        "{}° .. {}{}° per {}°",
                        self.start.to_degrees().value(),
                        $inclusive,
                        self.stop.to_degrees().value(),
                        self.step_size.to_degrees().value()
                    )
                }
            }
        )*
    };
    (@by_step_count $($range:ident, $inclusive:literal);*) => {
        $(
            impl<A: AngleUnit> $range<Angle<A>> {
                /// Pretty prints the range of angles.
                pub fn pretty_print(&self) -> String {
                    format!(
                        "{}° ..{}{}° in {} steps",
                        self.start.to_degrees().value(),
                        $inclusive,
                        self.stop.to_degrees().value(),
                        self.step_count
                    )
                }
            }
        )*
    };
}

impl<A: AngleUnit> StepRangeIncl<Angle<A>> {
    /// Returns the number of steps in this range of angles.
    /// If the range's stop value is the same as the start value after wrapping
    /// it to the range `[0, 2π)`, the range is considered to be a full circle.
    pub fn step_count_wrapped(&self) -> usize {
        if self.start.as_f32().abs_diff_eq(&self.stop.as_f32(), 1e-6) {
            return 1;
        }
        if self
            .start
            .wrap_to_tau()
            .value()
            .abs_diff_eq(&self.stop.wrap_to_tau().value(), 1e-6)
        {
            (self.span() / self.step_size).round() as usize
        } else {
            ((self.span() / self.step_size).round() as usize + 1).max(2)
        }
    }

    /// Returns an iterator over the values of the range of angles.
    /// The range is considered to be a full circle if the stop value is the
    /// same as the start value after wrapping it to the range [0, 2π).
    pub fn values_wrapped(&self) -> impl Iterator<Item = Angle<A>> {
        let step_size = self.step_size;
        let step_count = self.step_count_wrapped();
        let start = self.start;
        (0..step_count).map(move |i| start + step_size * i as f32)
    }

    /// Returns the index of the angle in the range of the measurement.
    ///
    /// The index of the bin is determined by testing if the angle falls
    /// inside half of the bin width from the bin boundary.
    pub fn index_of(&self, angle: Radians) -> usize {
        let angle = angle.to_degrees();
        let start = self.start.to_degrees();
        let step_size = self.step_size.to_degrees();
        ((angle - start).value() / step_size.value()).round() as usize % self.step_count_wrapped()
    }

    /// Returns the angle at the given index.
    pub fn step(&self, idx: usize) -> Angle<A> {
        (self.start + self.step_size * idx as f32).min(self.stop)
    }

    /// Returns the range of angles as a range of floating point values.
    pub fn range_bound_inclusive_f32(&self) -> RangeInclusive<f32> {
        self.start.value()..=self.stop.value()
    }

    /// Constructs a new range of angles from 0 to 2π with the given step size.
    pub const fn zero_to_tau(step_size: Angle<A>) -> StepRangeIncl<Angle<A>> {
        StepRangeIncl::new(Angle::ZERO, Angle::TAU, step_size)
    }

    /// Constructs a new range of angles from 0 to π/2 with the given step size.
    pub const fn zero_to_half_pi(step_size: Angle<A>) -> StepRangeIncl<Angle<A>> {
        StepRangeIncl::new(Angle::ZERO, Angle::HALF_PI, step_size)
    }
}

impl_angle_related_common_methods!(@by_step_size StepRangeIncl, " ="; StepRangeExcl, " ");

/// Defines a left inclusive, right inclusive range [a, b] of values with a
/// given number of steps.
#[derive(Copy, Clone)]
pub struct CountRangeIncl<T: Copy + Clone> {
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
    CountRangeIncl<T>;
    CountRangeExcl<T>
);

impl<T> CountRangeIncl<T>
where
    T: NumericCast<f32> + Copy + Clone + Sub<Output = T>,
    f32: NumericCast<T>,
{
    /// Returns the step size of this range.
    pub fn step_size(&self) -> T { (self.span().cast() / (self.step_count as f32 - 1.0f32)).cast() }
}

impl<T: Copy + Clone> CountRangeIncl<T>
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
pub struct CountRangeExcl<T: Copy + Clone> {
    /// The start value of the range.
    pub start: T,
    /// The stop value of the range.
    pub stop: T,
    /// The number of steps in the range.
    pub step_count: usize,
}

impl<T> CountRangeExcl<T>
where
    T: NumericCast<f32> + Copy + Clone + Sub<Output = T>,
    f32: NumericCast<T>,
{
    /// Returns the step size of this range.
    pub fn step_size(&self) -> T { (self.span().cast() / self.step_count as f32).cast() }
}

impl<T: Copy + Clone> CountRangeExcl<T>
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

impl_serialisation!(@exclusive CountRangeExcl<T>, '|', usize, step_count);
impl_serialisation!(@inclusive CountRangeIncl<T>, '|', usize, step_count);

impl_angle_related_common_methods!(@by_step_count CountRangeIncl, " ="; CountRangeExcl, " ");

/// Defines a range from a start value to a stop value with a given step size.
///
/// Left of the range is always inclusive, right of the range can be inclusive
/// or exclusive.
#[derive(Copy, Clone)]
pub enum StepRange<T: Copy + Clone> {
    /// Defines a right inclusive range.
    Inclusive(StepRangeIncl<T>),
    /// Defines a right exclusive range.
    Exclusive(StepRangeExcl<T>),
}

#[cfg(feature = "serde")]
impl<T> Serialize for StepRange<T>
where
    T: Serialize + Copy + Clone + Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            StepRange::Inclusive(range) => range.serialize(serializer),
            StepRange::Exclusive(range) => range.serialize(serializer),
        }
    }
}

impl<'a, T> TryFrom<&'a str> for StepRange<T>
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
                Ok(Self::Inclusive(StepRangeIncl::new(start, stop, step_size)))
            },
            Some(_) => {
                // Exclusive
                let stop = stop_str[..]
                    .parse::<T>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                Ok(Self::Exclusive(StepRangeExcl::new(start, stop, step_size)))
            },
            None => Err(format!("Invalid range stop value: {value}")),
        }
    }
}

#[cfg(feature = "serde")]
impl<'d, T> Deserialize<'d> for StepRange<T>
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
            type Value = StepRange<T>;

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
                StepRange::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }
        deserializer.deserialize_str(RangeByStepSizeVisitor::<T>(core::marker::PhantomData))
    }
}

/// Defines a range of values with a given number of steps.
#[derive(Copy, Clone)]
pub enum CountRange<T: Copy + Clone> {
    /// Defines a right inclusive range.
    Inclusive(CountRangeIncl<T>),
    /// Defines a right exclusive range.
    Exclusive(CountRangeExcl<T>),
}

impl<T: Copy + Clone> CountRange<T> {
    /// Returns the step size of the range.
    pub fn step_size(&self) -> T
    where
        T: NumericCast<f32> + Copy + Clone + Sub<Output = T>,
        f32: NumericCast<T>,
    {
        match self {
            CountRange::Inclusive(range) => range.step_size(),
            CountRange::Exclusive(range) => range.step_size(),
        }
    }
}

#[cfg(feature = "serde")]
impl<T> Serialize for CountRange<T>
where
    T: Serialize + Copy + Clone + Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            CountRange::Inclusive(range) => range.serialize(serializer),
            CountRange::Exclusive(range) => range.serialize(serializer),
        }
    }
}

impl<'a, T> TryFrom<&'a str> for CountRange<T>
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
                Ok(Self::Inclusive(CountRangeIncl::new(
                    start, stop, step_count,
                )))
            },
            Some(_) => {
                // Exclusive
                let stop = stop_str[..]
                    .parse::<T>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                Ok(Self::Exclusive(CountRangeExcl::new(
                    start, stop, step_count,
                )))
            },
            None => Err(format!("Invalid range stop value: {value}")),
        }
    }
}

#[cfg(feature = "serde")]
impl<'d, T> Deserialize<'d> for CountRange<T>
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
            type Value = CountRange<T>;

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
                CountRange::<T>::try_from(v).map_err(|e| E::custom(e))
            }
        }
        deserializer.deserialize_str(RangeByStepCountVisitor::<T>(core::marker::PhantomData))
    }
}

/// Defines a range of values either by a step size or by a step count.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Range<T: Copy + Clone> {
    /// Defines a range by a step size.
    ByStepSize(StepRange<T>),
    /// Defines a range by a step count.
    ByStepCount(CountRange<T>),
}

#[cfg(feature = "serde")]
impl<T> Serialize for Range<T>
where
    T: Serialize + Copy + Clone + Display,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Range::ByStepSize(range) => range.serialize(serializer),
            Range::ByStepCount(range) => range.serialize(serializer),
        }
    }
}

impl<'a, T> TryFrom<&'a str> for Range<T>
where
    T: Copy + FromStr,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        if value.contains('|') {
            Ok(Self::ByStepCount(CountRange::try_from(value)?))
        } else {
            Ok(Self::ByStepSize(StepRange::try_from(value)?))
        }
    }
}

#[cfg(feature = "serde")]
impl<'d, T> Deserialize<'d> for Range<T>
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
            type Value = Range<T>;

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
                Range::<T>::try_from(v).map_err(|e| E::custom(e))
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
                        Self::Inclusive([<$range Incl>]::new(start, stop, [<step_ $step_size_or_count>]))
                    }

                    /// Creates a new right exclusive range.
                    pub const fn new_exclusive(start: $T, stop: $T, [<step_ $step_size_or_count>]: $step_size_or_count_type) -> Self {
                        Self::Exclusive([<$range Excl>]::new(start, stop, [<step_ $step_size_or_count>]))
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
    CountRange<T>, count, usize, #[doc = "Returns the step count of the range."];
    StepRange<T>, size, T, #[doc = "Returns the step size of the range."]
);

impl<T> From<CountRangeIncl<T>> for StepRangeIncl<T>
where
    T: NumericCast<f32> + Copy + Clone + Sub<Output = T>,
    f32: NumericCast<T>,
{
    fn from(range: CountRangeIncl<T>) -> Self {
        Self::new(range.start, range.stop, range.step_size())
    }
}

impl<T> From<StepRangeIncl<T>> for CountRangeIncl<T>
where
    T: NumericCast<f32> + Copy + Clone,
{
    fn from(value: StepRangeIncl<T>) -> Self {
        Self::new(value.start, value.stop, value.step_count())
    }
}

#[cfg(feature = "io")]
macro_rules! impl_range_by_step_size_inclusive_read_write {
    ($($T:ty, $step_count:ident);*) => {
        $(paste::paste! {
            impl StepRangeIncl<$T> {
                #[doc = "Writes the RangeByStepSizeInclusive<`" $T "`> into the given buffer, following the order: start, stop, step_size, step_count."]
                #[track_caller]
                pub fn write_to_buf(&self, buf: &mut [u8]) {
                    debug_assert!(buf.len() >= 16, "RangeByStepSizeInclusive needs at least 16 bytes of space");
                    buf[0..4].copy_from_slice(&self.start.value().to_le_bytes());
                    buf[4..8].copy_from_slice(&self.stop.value().to_le_bytes());
                    buf[8..12].copy_from_slice(&self.step_size.value().to_le_bytes());
                    buf[12..16].copy_from_slice(&(self.$step_count() as u32).to_le_bytes());
                }

                #[doc = "Reads the RangeByStepSizeInclusive<`" $T "`> from the given buffer, checking that the step count matches the expected value."]
                #[track_caller]
                pub fn read_from_buf(buf: &[u8]) -> Self {
                    debug_assert!(
                        buf.len() >= 16,
                        "RangeByStepSizeInclusive needs at least 16 bytes of space"
                    );
                    let start = <$T>::new(f32::from_le_bytes(buf[0..4].try_into().unwrap()));
                    let end = <$T>::new(f32::from_le_bytes(buf[4..8].try_into().unwrap()));
                    let step_size = <$T>::new(f32::from_le_bytes(buf[8..12].try_into().unwrap()));
                    let step_count = u32::from_le_bytes(buf[12..16].try_into().unwrap());
                    let range = Self::new(start, end, step_size);
                    assert_eq!(
                        step_count,
                        range.$step_count() as u32,
                        "RangeByStepSizeInclusive: step count mismatch"
                    );
                    range
                }
            }
        })*
    };
}

#[cfg(feature = "io")]
impl_range_by_step_size_inclusive_read_write!(
    Radians, step_count_wrapped;
    Nanometres, step_count
);

#[cfg(feature = "io")]
impl CountRangeIncl<Radians> {
    /// Writes the `RangeByStepCountInclusive<Radians>` into the given
    /// buffer, following the order: start, stop, step_size,
    /// step_count.
    pub fn write_to_buf(&self, buf: &mut [u8]) {
        debug_assert!(
            buf.len() >= 16,
            "RangeByStepCountInclusive<Radians> needs at least 16 bytes of space"
        );
        buf[0..4].copy_from_slice(&self.start.value().to_le_bytes());
        buf[4..8].copy_from_slice(&self.stop.value().to_le_bytes());

        buf[8..12].copy_from_slice(&self.step_size().value().to_le_bytes());
        buf[12..16].copy_from_slice(&(self.step_count as u32).to_le_bytes());
    }

    /// Reads the `RangeByStepCountInclusive<Radians>` from the given
    /// buffer, checking that the step size matches the expected
    /// value.
    pub fn read_from_buf(buf: &[u8]) -> Self {
        debug_assert!(
            buf.len() >= 16,
            "RangeByStepCountInclusive needs at least 16 bytes of space"
        );
        let start = Radians::new(f32::from_le_bytes(buf[0..4].try_into().unwrap()));
        let end = Radians::new(f32::from_le_bytes(buf[4..8].try_into().unwrap()));
        let step_size = Radians::new(f32::from_le_bytes(buf[8..12].try_into().unwrap()));
        let step_count = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let range = Self::new(start, end, step_count as usize);
        assert!(
            math::ulp_eq(range.step_size().value(), step_size.value()),
            "RangeByStepCountInclusive<Radians> step size mismatch: expected {}, got {}",
            range.step_size().value(),
            step_size.value()
        );
        range
    }
}

#[cfg(test)]
mod range_by_step_count_tests {
    use super::*;

    #[test]
    #[cfg(feature = "serde")]
    fn serialization_inclusive() {
        let range = CountRangeIncl::new(0, 10, 3);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(serialized, "0 .. =10 | 3\n");
        let deserialized: CountRangeIncl<i32> = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, range);

        let range = CountRange::new_inclusive(0, 10, 3);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(serialized, "0 .. =10 | 3\n");
        let deserialized: CountRange<i32> = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, range);
        assert!(deserialized.is_inclusive());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serialization_exclusive() {
        let range = CountRangeExcl::new(0, 10, 3);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(serialized, "0 .. 10 | 3\n");
        let deserialized: CountRangeExcl<i32> = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, range);

        let range = CountRange::new_exclusive(0, 10, 3);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(serialized, "0 .. 10 | 3\n");
        let deserialized: CountRange<i32> = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, range);
        assert!(deserialized.is_exclusive());
    }

    #[test]
    fn test_step_size() {
        let range = CountRangeIncl::new(0, 10, 3);
        assert_eq!(range.step_size(), 5);

        let range = CountRangeIncl::new(0, 10, 4);
        assert_eq!(range.step_size(), 3);

        let range = CountRangeIncl::new(0, 10, 5);
        assert_eq!(range.step_size(), 2);

        let range = CountRangeIncl::new(1.0, 13.0, 7);
        assert_eq!(range.step_size(), 2.0);

        let range = CountRangeExcl::new(1.0, 13.0, 8);
        assert_eq!(range.step_size(), 1.5);
    }

    #[test]
    fn test_conversion() {
        let range = CountRangeIncl::new(0, 10, 3);
        let range: StepRangeIncl<i32> = range.into();
        assert_eq!(range, StepRangeIncl::new(0, 10, 5));
    }
}

#[cfg(test)]
mod range_by_step_size_tests {
    use super::*;
    use crate::units::{deg, rad, Degrees, Rads};

    #[test]
    fn try_from_str_inclusive() {
        let range = StepRange::<f32>::try_from("0.0 .. =12.0 / 0.3").unwrap();
        assert_eq!(range.start(), 0.0);
        assert_eq!(range.stop(), 12.0);
        assert_eq!(range.step_size(), 0.3);
        assert!(!range.is_exclusive());

        let range = StepRangeIncl::<f32>::try_from("0.0 .. =16.0 / 0.2").unwrap();
        assert_eq!(range.start, 0.0);
        assert_eq!(range.stop, 16.0);
        assert_eq!(range.step_size, 0.2);
    }

    #[test]
    fn try_from_str_exclusive() {
        let range = StepRange::<f32>::try_from("0.0 .. 1.0 / 0.1").unwrap();
        assert_eq!(range.start(), 0.0);
        assert_eq!(range.stop(), 1.0);
        assert_eq!(range.step_size(), 0.1);
        assert!(range.is_exclusive());

        let range = StepRangeExcl::<f32>::try_from("0.0 .. 1.0 / 0.1").unwrap();
        assert_eq!(range.start, 0.0);
        assert_eq!(range.stop, 1.0);
        assert_eq!(range.step_size, 0.1);
    }

    #[test]
    fn try_from_str_inclusive_angle() {
        let range = StepRange::<Radians>::try_from("0.0 rad .. =10.0rad / 0.1deg").unwrap();
        assert_eq!(range.start(), rad!(0.0));
        assert_eq!(range.stop(), rad!(10.0));
        assert_eq!(range.step_size(), deg!(0.1).to_radians());
        assert!(!range.is_exclusive());
    }

    #[test]
    fn try_from_str_exclusive_angle() {
        let range = StepRange::<Degrees>::try_from("0.0rad .. 360 deg / 10.0deg").unwrap();
        assert_eq!(range.start(), deg!(0.0));
        assert_eq!(range.stop(), deg!(360.0));
        assert_eq!(range.step_size(), deg!(10.0));
        assert!(range.is_exclusive());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serialisation_exclusive() {
        let range = StepRange::new_exclusive(0.0, 20.0, 0.5);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(
            serialized, "0 .. 20 / 0.5\n",
            "serilisation failed: {}",
            serialized
        );
        let deserialized: StepRange<f32> = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(
            deserialized, range,
            "deserilisation failed: {}",
            deserialized
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn deserialisation_exclusive() {
        let range_str = "0 .. 20 / 0.5";
        let deserialized: StepRange<f32> = serde_yaml::from_str(&range_str).unwrap();
        let range = StepRange::new_exclusive(0.0, 20.0, 0.5);
        assert_eq!(
            deserialized, range,
            "deserilisation failed: {}",
            deserialized
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serialisation_inclusive() {
        let range = StepRangeIncl::new(0.0, 20.0, 0.5f32);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(
            serialized, "0 .. =20 / 0.5\n",
            "serilisation failed: {}",
            serialized
        );
        let deserialized: StepRangeIncl<f32> = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(
            deserialized, range,
            "deserilisation failed: {}",
            deserialized
        );
    }

    #[test]
    fn display() {
        let range = StepRange::new_exclusive(0.5, 20.5, 0.5);
        assert_eq!(format!("{}", range), "0.5 .. 20.5 / 0.5");
        let range = StepRange::new_inclusive(0.5, 20.5, 0.5);
        assert_eq!(format!("{}", range), "0.5 .. =20.5 / 0.5");
    }

    #[test]
    fn step_count() {
        let range = StepRangeIncl::new(0.0, 10.0, 1.0);
        assert_eq!(range.step_count(), 11);
        let range = StepRangeIncl::new(0.0, 10.0, 2.0);
        assert_eq!(range.step_count(), 6);

        let range = StepRangeIncl::new(380.0, 780.0, 300.0);
        assert_eq!(range.step_count(), 3);

        let range = StepRangeIncl::new(rad!(0.0), Rads::HALF_PI, deg!(30.0).to_radians());
        assert_eq!(range.step_count(), 4);
        assert_eq!(range.step_count_wrapped(), 4);

        let range = StepRangeIncl::new(rad!(0.0), Rads::TAU, deg!(30.0).to_radians());
        assert_eq!(range.step_count(), 13);
        assert_eq!(range.step_count_wrapped(), 12);

        let range = StepRangeIncl::new(deg!(0.0), deg!(360.0), deg!(90.0));
        assert_eq!(range.step_count(), 5);
        assert_eq!(range.step_count_wrapped(), 4);

        let range = StepRangeIncl::new(0.0, 50.0, 12.5);
        assert_eq!(range.step_count(), 5);

        let range = StepRangeIncl::new(0.0, 10.0, 4.0);
        assert_eq!(range.step_count(), 4);

        let range = StepRangeIncl::new(0.0, 10.0, 2.5);
        assert_eq!(range.step_count(), 5);

        let range = StepRangeIncl::new(0.0, 10.0, 10.0);
        assert_eq!(range.step_count(), 2);

        let range = StepRangeIncl::new(10.0, 10.0, 0.0);
        assert_eq!(range.step_count(), 1);

        let range = StepRangeIncl::new(10.0, 10.0, 1.0);
        assert_eq!(range.step_count(), 1);
    }

    #[test]
    fn values() {
        let range = StepRangeIncl::new(0.0, 10.0, 1.0f32);
        let values: Vec<f32> = range.values().collect();
        assert_eq!(
            values,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );

        let range = StepRangeIncl::new(10.0, 10.0, 2.0f32);
        let values: Vec<f32> = range.values().collect();
        assert_eq!(values, vec![10.0]);

        let range = StepRangeIncl::new(Radians::TAU, Radians::TAU, Radians::ZERO);
        let values = range.values_wrapped().collect::<Vec<Radians>>();
        assert_eq!(values, vec![Radians::TAU]);
    }
}
