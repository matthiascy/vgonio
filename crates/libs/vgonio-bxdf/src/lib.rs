#![feature(associated_type_defaults)]
#![feature(downcast_unchecked)]
#![feature(allocator_api)]
#![feature(adt_const_params)]
//! Bxdf models and utilities.
pub mod brdf;
pub mod distro;

#[cfg(feature = "fitting")]
pub mod fitting;
