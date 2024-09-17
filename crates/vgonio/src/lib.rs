//! vgonio is a library for micro-level light transport simulation.

#![feature(async_closure)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(vec_push_within_capacity)]
#![feature(assert_matches)]
#![feature(stmt_expr_attributes)]
#![feature(adt_const_params)]
#![feature(seek_stream_len)]
#![feature(trait_upcasting)]
#![feature(portable_simd)]
#![feature(slice_pattern)]
#![feature(os_str_display)]
#![feature(let_chains)]
#![feature(generic_const_exprs)]

extern crate core;

mod app;
mod error;
#[cfg(feature = "fitting")]
pub mod fitting;
mod io;
pub mod measure;
pub(crate) mod pyplot;

pub use app::run;

/// Machine epsilon for `f32`.
pub const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;

/// Returns the gamma factor for a floating point number.
pub const fn gamma_f32(n: f32) -> f32 { (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON) }
