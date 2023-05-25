use crate::{
    math,
    math::NumericCast,
    units::{Angle, AngleUnit, Radians},
};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    ops::{Add, Deref, DerefMut, Div, Mul, RangeInclusive, Sub},
    str::FromStr,
};

/// Defines a range of values with a given step size or count.
pub struct RangeInner<T, U> {
    /// Initial value of the range.
    pub(crate) start: T,
    /// Final value of the range.
    pub(crate) stop: T,
    /// Step size or count.
    pub(crate) step_size_or_count: U,
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

impl<T, U> RangeInner<T, U>
where
    T: Copy,
{
    /// Returns the range as a `std::ops::Range`.
    pub fn range_bound(&self) -> std::ops::Range<T> { self.start..self.stop }

    /// Returns the range as a `std::ops::RangeInclusive`.
    pub fn range_bound_inclusive(&self) -> std::ops::RangeInclusive<T> { self.start..=self.stop }
}

impl<T: Sub<Output = T> + Copy + Clone, U> RangeInner<T, U> {
    /// Returns the span of the range.
    pub fn span(&self) -> T { self.stop - self.start }
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
pub struct RangeByStepSizeInclusive<T: Copy + Clone>(RangeInner<T, T>);

impl<T: Copy + Clone> Eq for RangeByStepSizeInclusive<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> Deref for RangeByStepSizeInclusive<T> {
    type Target = RangeInner<T, T>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T: Copy + Clone> DerefMut for RangeByStepSizeInclusive<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<T: Copy + Clone + Debug> Debug for RangeByStepSizeInclusive<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RangeByStepSizeInclusive")
            .field("start", &self.start)
            .field("stop", &self.stop)
            .field("step_size", &self.0.step_size_or_count)
            .finish()
    }
}

impl<T: Copy + Clone> RangeByStepSizeInclusive<T> {
    /// Creates a new range with the given start, stop and step size.
    pub const fn new(start: T, stop: T, step_size: T) -> Self {
        Self(RangeInner {
            start,
            stop,
            step_size_or_count: step_size,
        })
    }

    /// Returns the step size of the range.
    pub const fn step_size(&self) -> &T { &self.0.step_size_or_count }

    /// Returns the step size of the range.
    pub fn step_size_mut(&mut self) -> &mut T { &mut self.0.step_size_or_count }

    /// Maps the range to a new range with a different type.
    pub fn map<F, U>(self, mut f: F) -> RangeByStepSizeInclusive<U>
    where
        F: FnMut(T) -> U,
        U: Copy + Clone,
    {
        RangeByStepSizeInclusive(RangeInner {
            start: f(self.start),
            stop: f(self.stop),
            step_size_or_count: f(self.0.step_size_or_count),
        })
    }
}

impl<T> RangeByStepSizeInclusive<T>
where
    T: ~const NumericCast<f32> + Copy + Clone + From<f32>,
{
    /// Returns all possible values of the range.
    pub fn values(&self) -> impl Iterator<Item = T> {
        let step_size = self.0.step_size_or_count.cast();
        let start = self.0.start.cast();
        let stop = self.0.stop.cast();
        let step_count = self.step_count();
        (0..step_count).map(move |i| (start + step_size * i as f32).min(stop).into())
    }

    /// Returns all possible values of the range in reverse order.
    pub fn values_rev(&self) -> impl Iterator<Item = T> {
        let step_size = self.0.step_size_or_count.cast();
        let start = self.0.start.cast();
        let stop = self.0.stop.cast();
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
        let step_size = self.0.step_size_or_count.cast();
        let start = self.0.start.cast();
        let stop = self.0.stop.cast();
        (((stop - start) / step_size).ceil() as usize).min(2)
    }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given step.
#[derive(Copy, Clone, Default, PartialEq)]
pub struct RangeByStepSizeExclusive<T: Copy + Clone>(RangeInner<T, T>);

impl<T: Copy + Clone> Eq for RangeByStepSizeExclusive<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> Deref for RangeByStepSizeExclusive<T> {
    type Target = RangeInner<T, T>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T: Copy + Clone> DerefMut for RangeByStepSizeExclusive<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<T: Copy + Clone + Debug> Debug for RangeByStepSizeExclusive<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RangeByStepSizeExclusive")
            .field("start", &self.start)
            .field("stop", &self.stop)
            .field("step_size", &self.0.step_size_or_count)
            .finish()
    }
}

impl<T: Copy + Clone> RangeByStepSizeExclusive<T> {
    /// Creates a new range with the given start, stop and step size.
    pub const fn new(start: T, stop: T, step_size: T) -> Self {
        Self(RangeInner {
            start,
            stop,
            step_size_or_count: step_size,
        })
    }

    /// Returns the step size of the range.
    pub const fn step_size(&self) -> &T { &self.0.step_size_or_count }

    /// Returns the step size of the range.
    pub fn step_size_mut(&mut self) -> &mut T { &mut self.0.step_size_or_count }

    /// Maps the range to a new range with a different type.
    pub fn map<F, U>(self, mut f: F) -> RangeByStepSizeInclusive<U>
    where
        F: FnMut(T) -> U,
        U: Copy + Clone,
    {
        RangeByStepSizeInclusive(RangeInner {
            start: f(self.start),
            stop: f(self.stop),
            step_size_or_count: f(self.0.step_size_or_count),
        })
    }
}

impl<T> RangeByStepSizeExclusive<T>
where
    T: Div<Output = T> + Sub<Output = T> + ~const NumericCast<f32> + Copy + Clone,
{
    /// Returns the step count of the exclusive range.
    pub fn step_count(&self) -> usize {
        let step_size = self.0.step_size_or_count.cast();
        let start = self.0.start.cast();
        let stop = self.0.stop.cast();
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
        let step_size = self.0.step_size_or_count;
        let start = self.0.start;
        let step_count = self.step_count();
        (0..step_count).map(move |i| start + step_size * i.cast())
    }
}

macro impl_display_serialisation($($t:ident<$T:ident>, $string:literal, $open_symbol:literal, $split_char:literal, $step_type:ident);*) {
    $(
        impl<$T> Display for $t<$T>
        where
            $T: Display + Copy + Clone,
        {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    stringify!($string),
                    self.0.start, self.0.stop, self.0.step_size_or_count
                )
            }
        }

        impl<$T> Serialize for $t<$T>
        where
            $T: Serialize + Copy + Display + Clone,
        {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&format!(stringify!($string), self.0.start, self.0.stop, self.0.step_size_or_count))
            }
        }

        impl<'a, $T> TryFrom<&'a str> for $t<$T>
        where
            $T: Copy + Clone + FromStr,
        {
            type Error = String;

            fn try_from(value: &'a str) -> Result<Self, Self::Error> {
                let mut parts = value.split(',');
                println!("parts: {:?}", parts);
                let start = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?;
                println!("start: {:?}", start);
                let start = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim_matches('[')
                    .parse::<$T>()
                    .map_err(|_| format!("Invalid range start value: {value}"))?;
                let mut parts = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .split($split_char);
                let stop = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?;
                println!("stop: {:?}", stop);
                let stop = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim_end_matches($open_symbol)
                    .parse::<$T>()
                    .map_err(|_| format!("Invalid range stop value: {value}"))?;
                let step_size = parts
                    .next()
                    .ok_or_else(|| format!("Invalid range: {value}"))?
                    .trim()
                    .parse::<$step_type>()
                    .map_err(|_| format!("Invalid range step size/count value: {value}"))?;
                Ok(Self::new(start, stop, step_size))
            }
        }

        impl<'d, $T> Deserialize<'d> for $t<$T>
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
                    type Value = $t<$T>;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        write!(
                            formatter,
                            "a range by step size in the form of '[start, stop{} {} step",
                            stringify!($open_symbol), stringify!($split_char)
                        )
                    }

                    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        $t::<$T>::try_from(v).map_err(|e| E::custom(e))
                    }
                }
                deserializer.deserialize_str(RangeVisitor::<$T>(core::marker::PhantomData))
            }
        }
    )*
}

impl_display_serialisation!(RangeByStepSizeExclusive<T>, "[{}, {}) / {}", ')', '/', T; RangeByStepSizeInclusive<T>, "[{}, {}] / {}", ']', '/', T);

/// Defines a range from a start value to a stop value with a given step size.
///
/// Left of the range is always inclusive, right of the range can be inclusive
/// or exclusive.
#[derive(Copy, Clone, PartialEq)]
pub enum RangeByStepSize<T: Copy + Clone> {
    Inclusive(RangeByStepSizeInclusive<T>),
    Exclusive(RangeByStepSizeExclusive<T>),
}

impl<T: Copy + Clone + Debug> Debug for RangeByStepSize<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            RangeByStepSize::Inclusive(range) => write!(f, "{:?}", range),
            RangeByStepSize::Exclusive(range) => write!(f, "{:?}", range),
        }
    }
}

impl<T: Copy + Clone> RangeByStepSize<T> {
    pub fn start(&self) -> T {
        match self {
            RangeByStepSize::Inclusive(range) => range.0.start,
            RangeByStepSize::Exclusive(range) => range.0.start,
        }
    }

    pub fn stop(&self) -> T {
        match self {
            RangeByStepSize::Inclusive(range) => range.0.stop,
            RangeByStepSize::Exclusive(range) => range.0.stop,
        }
    }

    pub fn step_size(&self) -> T {
        match self {
            RangeByStepSize::Inclusive(range) => range.0.step_size_or_count,
            RangeByStepSize::Exclusive(range) => range.0.step_size_or_count,
        }
    }

    pub fn is_inclusive(&self) -> bool {
        match self {
            RangeByStepSize::Inclusive(_) => true,
            RangeByStepSize::Exclusive(_) => false,
        }
    }

    pub fn is_exclusive(&self) -> bool {
        match self {
            RangeByStepSize::Inclusive(_) => false,
            RangeByStepSize::Exclusive(_) => true,
        }
    }
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

impl<A: AngleUnit> RangeByStepSizeInclusive<Angle<A>> {
    /// Returns the number of steps in this range of angles.
    /// If the range's stop value is the same as the start value after wrapping
    /// it to the range [0, 2π), the range is considered to be a full circle.
    pub fn step_count_wrapped(&self) -> usize {
        let start = self.start.to_radians().value;
        let stop = self.stop.to_radians().value;
        let step_size = self.0.step_size_or_count.to_radians().value;
        let span = stop - start;
        if math::wrap_angle_to_tau_exclusive(stop) == math::wrap_angle_to_tau_exclusive(start) {
            (span / step_size).ceil() as usize
        } else {
            ((span / step_size).ceil() as usize).min(2)
        }
    }

    /// Returns the index of the angle in the range of the measurement.
    ///
    /// The index of the bin is determined by testing if the angle falls
    /// inside half of the bin width from the bin boundary.
    pub fn index_of(&self, angle: Radians) -> usize
    where
        Radians: From<Angle<A>>,
    {
        ((angle - self.start).value / self.step_size().value).round() as usize
            % self.step_count() as usize
    }

    pub fn range_bound_inclusive_f32(&self) -> RangeInclusive<f32> {
        self.start.value..=self.stop.value
    }
}

/// Defines a left inclusive, right inclusive range [a, b] of values with a
/// given number of steps.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RangeByStepCountInclusive<T: Copy + Clone>(RangeInner<T, usize>);

impl<T: Copy + Clone> Eq for RangeByStepCountInclusive<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> Deref for RangeByStepCountInclusive<T> {
    type Target = RangeInner<T, usize>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T: Copy + Clone> DerefMut for RangeByStepCountInclusive<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<T: Copy + Clone> RangeByStepCountInclusive<T> {
    pub const fn new(start: T, stop: T, step_count: usize) -> Self {
        Self(RangeInner {
            start,
            stop,
            step_size_or_count: step_count,
        })
    }

    /// Returns the number of steps in this range.
    pub const fn step_count(&self) -> &usize { &self.0.step_size_or_count }

    /// Returns the number of steps in this range.
    pub fn step_count_mut(&mut self) -> &mut usize { &mut self.0.step_size_or_count }
}

impl<T> RangeByStepCountInclusive<T>
where
    T: Div<Output = T> + Sub<Output = T> + NumericCast<f32> + Copy + Clone,
    f32: NumericCast<T>,
{
    /// Returns the step size of this range.
    pub fn step_size(&self) -> T {
        (self.span().cast() / (self.0.step_size_or_count as f32 - 1.0f32)).cast()
    }
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
        let start = self.0.start;
        (0..self.0.step_size_or_count).map(move |i| start + step_size * i.cast())
    }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a
/// given number of steps.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct RangeByStepCountExclusive<T: Copy + Clone>(RangeInner<T, usize>);

impl<T: Copy + Clone> Eq for RangeByStepCountExclusive<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> Deref for RangeByStepCountExclusive<T> {
    type Target = RangeInner<T, usize>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<T: Copy + Clone> DerefMut for RangeByStepCountExclusive<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl<T: Copy + Clone> RangeByStepCountExclusive<T> {
    pub const fn new(start: T, stop: T, step_count: usize) -> Self {
        Self(RangeInner {
            start,
            stop,
            step_size_or_count: step_count,
        })
    }

    /// Returns the number of steps in this range.
    pub const fn step_count(&self) -> usize { self.0.step_size_or_count }
}

impl<T> RangeByStepCountExclusive<T>
where
    T: Div<Output = T> + Sub<Output = T> + Copy + Clone,
    usize: NumericCast<T>,
{
    /// Returns the step size of this range.
    pub fn step_size(&self) -> T { self.0.span() / self.0.step_size_or_count.cast() }
}

impl<T: Copy + Clone> RangeByStepCountExclusive<T>
where
    T: Add<Output = T> + Sub<Output = T> + Div<Output = T> + Mul<Output = T> + NumericCast<f32>,
    f32: NumericCast<T>,
    usize: NumericCast<T>,
{
    /// Returns all possible values in this range.
    pub fn values(&self) -> impl Iterator<Item = T> {
        let step_size = self.step_size();
        let start = self.0.start;
        (0..self.0.step_size_or_count).map(move |i| start + step_size * i.cast())
    }
}

impl_display_serialisation!(RangeByStepCountExclusive<T>, "[{}, {}) | {}", ')', '|', usize; RangeByStepCountInclusive<T>, "[{}, {}] | {}", ']', '|', usize);

/// Defines a range of values with a given number of steps.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum RangeByStepCount<T: Copy + Clone> {
    Inclusive(RangeByStepCountInclusive<T>),
    Exclusive(RangeByStepCountExclusive<T>),
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
                        Ok(Self::Inclusive(RangeByStepCountInclusive::new(
                            start, stop, step_count,
                        )))
                    }
                    ')' => {
                        // Exclusive
                        let stop = stop_str[..stop_str.len() - 1]
                            .parse::<T>()
                            .map_err(|_| format!("Invalid range stop value: {value}"))?;
                        Ok(Self::Exclusive(RangeByStepCountExclusive::new(
                            start, stop, step_count,
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
    ByStepSize(RangeByStepSize<T>),
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

#[cfg(test)]
mod range_by_step_count_inclusive_tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let range = RangeByStepCountInclusive::new(0, 10, 3);
        let serialized = serde_yaml::to_string(&range).unwrap();
        assert_eq!(serialized, "[0 ~ 10] | 3");
        let deserialized: RangeByStepCountInclusive<i32> =
            serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, range);
    }

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

#[test]
fn range_by_step_size_exclusive_try_from_str() {
    let range = RangeByStepSizeExclusive::<f32>::try_from("[0.0, 1.0) / 0.1").unwrap();
    assert_eq!(range.start, 0.0);
    assert_eq!(range.stop, 1.0);
    assert_eq!(*range.step_size(), 0.1);
}

#[test]
fn range_by_step_size_try_from_str_with_angle() {
    use crate::units::{radians, Radians, URadian};
    let range = RangeByStepSizeExclusive::<Radians>::try_from("[0.0rad, 1.0rad) / 0.1rad").unwrap();
    assert_eq!(range.start, radians!(0.0));
    assert_eq!(range.stop, radians!(1.0));
    assert_eq!(*range.step_size(), Angle::<URadian>::new(0.1));
}

#[test]
fn range_by_step_count_try_from_str() {
    let range = RangeByStepCountExclusive::<f32>::try_from("0.0 ~ 1.0, 19").unwrap();
    assert_eq!(range.start, 0.0);
    assert_eq!(range.stop, 1.0);
    assert_eq!(range.step_count(), 19);
}

#[test]
fn range_by_step_count_try_from_str_with_angle() {
    use crate::units::{radians, Radians};
    let range = RangeByStepCountExclusive::<Radians>::try_from("0.0rad ~ 1.0rad, 4").unwrap();
    assert_eq!(range.start, radians!(0.0));
    assert_eq!(range.stop, radians!(1.0));
    assert_eq!(range.step_count(), 4);
}

#[cfg(test)]
mod range_by_step_size_tests {
    use super::*;
    use crate::units::{degrees, radians, Degrees};

    #[test]
    fn try_from_str_inclusive() {
        let range = RangeByStepSize::<f32>::try_from("[0.0, 10.0] / 0.1").unwrap();
        assert_eq!(range.start(), 0.0);
        assert_eq!(range.stop(), 10.0);
        assert_eq!(range.step_size(), 0.1);
        assert!(!range.is_exclusive());

        let range = RangeByStepSizeInclusive::<f32>::try_from("[0.0, 10.0] / 0.1").unwrap();
        assert_eq!(range.start, 0.0);
        assert_eq!(range.stop, 10.0);
        assert_eq!(*range.step_size(), 0.1);
    }

    #[test]
    fn try_from_str_exclusive() {
        let range = RangeByStepSize::<f32>::try_from("[0.0, 1.0) / 0.1").unwrap();
        assert_eq!(range.start(), 0.0);
        assert_eq!(range.stop(), 1.0);
        assert_eq!(range.step_size(), 0.1);
        assert!(range.is_exclusive());
    }

    #[test]
    fn try_from_str_inclusive_angle() {
        let range = RangeByStepSize::<Radians>::try_from("[0.0 rad, 10.0rad] / 0.1deg").unwrap();
        assert_eq!(range.start(), radians!(0.0));
        assert_eq!(range.stop(), radians!(10.0));
        assert_eq!(range.step_size(), degrees!(0.1).to_radians());
        assert!(!range.is_exclusive());
    }

    #[test]
    fn try_from_str_exclusive_angle() {
        let range = RangeByStepSize::<Degrees>::try_from("[0.0rad, 360 deg) / 10.0deg").unwrap();
        assert_eq!(range.start(), degrees!(0.0));
        assert_eq!(range.stop(), degrees!(360.0));
        assert_eq!(range.step_size(), degrees!(10.0));
        assert!(range.is_exclusive());
    }
}
