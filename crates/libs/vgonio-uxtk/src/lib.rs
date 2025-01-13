//! This crate is a collection of widgets and utilities for building user
//! interfaces for VGonio.

macro_rules! profile_function {
    ($($arg:tt)*) => {
        #[cfg(feature = "profile")]
        puffin::profile_function!($($arg)*);
    };
}

macro_rules! profile_scope {
    ($($arg:tt)*) => {
        #[cfg(feature = "profile")]
        puffin::profile_scope!($($arg)*);
    };
}

mod renderer;
pub mod theme;

pub use renderer::*;

pub mod widgets;
