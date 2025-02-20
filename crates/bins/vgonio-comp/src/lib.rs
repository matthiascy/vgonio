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
#[cfg(feature = "fitting")]
pub mod fitting;
pub mod measure;
pub(crate) mod pyplot;

pub use app::run;

pub fn run_vgonio_compute() {
    std::process::exit(match run() {
        Ok(_) => 0,
        Err(ref e) => {
            eprintln!("{e}");
            1
        },
    })
}
