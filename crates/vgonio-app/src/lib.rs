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
#![feature(iter_map_windows)]
extern crate core;

mod app;
mod error;
#[cfg(not(any(test, feature = "test-support")))]
mod io;

// The VGMO codec (`BsdfMeasurement::{read,write}_to_vgmo`) lives here; its
// inherent methods are only reachable from integration tests if the enclosing
// module is public. Expose it solely under `test-support`.
#[cfg(any(test, feature = "test-support"))]
pub mod io;
pub mod measure;
mod orchestration;
pub(crate) mod pyplot;

#[cfg(feature = "fitting")]
pub use app::cli::FitOptions;
pub use app::{cli::MeasureOptions, run};

/// Machine epsilon for `f32`.
pub const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;

/// Returns the gamma factor for a floating point number.
#[must_use]
pub const fn gamma_f32(n: f32) -> f32 { (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON) }

pub fn run_vgonio_compute() {
    std::process::exit(match run() {
        Ok(_) => 0,
        Err(ref e) => {
            eprintln!("{e}");
            1
        },
    })
}
