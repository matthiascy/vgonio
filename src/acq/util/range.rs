use std::fmt::Display;
use crate::acq::{Angle, AngleUnit, Length, LengthUnit};
use std::ops::{Div, Sub};
use std::process::Output;

/// Defines a left inclusive, right exclusive range [a, b) of values with a given step.
#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize)]
#[serde(into = "[T; 3]", from = "[T; 3]")]
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

macro impl_step_count_for_range_by_step_size($self:ident) {
{
        #[cfg(debug_assertions)]
        assert!($self.step_size.value() > 0.0, "step_size must not be greater than 0.0");
        let count = $self.span() / $self.step_size;
        if count == 0.0 {
            1
        } else {
            count.ceil() as usize
        }
    }
}

impl RangeByStepSize<f32> {
    pub fn step_count(&self) -> usize {
        #[cfg(debug_assertions)]
        assert!(self.step_size > 0.0, "step_size must not be greater than 0.0");
        let count = self.span() / self.step_size;
        if count == 0.0 {
            1
        } else {
            count.ceil() as usize
        }
    }
}

impl<A: AngleUnit> RangeByStepSize<Angle<A>> {
    pub fn step_count(&self) -> usize {
        impl_step_count_for_range_by_step_size!(self)
    }
}


impl<A: LengthUnit> RangeByStepSize<Length<A>> {
    pub fn step_count(&self) -> usize {
        impl_step_count_for_range_by_step_size!(self)
    }
}

/// Defines a left inclusive, right exclusive range [a, b) of values with a given number of steps.
#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize)]
#[serde(into = "(T, T, usize)", from = "(T, T, usize)")]
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

impl RangeByStepCount<f32> {
    /// Returns the step between two consecutive values in the range.
    #[inline]
    pub const fn step_size(&self) -> f32 {
        self.span() / self.step_count as f32
    }
}

impl<A: LengthUnit> RangeByStepCount<Length<A>> {
    #[inline]
    pub fn step_size(&self) -> Length<A> {
        self.span() / self.step_count as f32
    }
}
