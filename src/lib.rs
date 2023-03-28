//! vgonio is a library for micro-level light transport simulation.

#![feature(async_closure)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(is_some_and)]
#![feature(vec_push_within_capacity)]
#![warn(missing_docs)]
#![feature(stdsimd)] // to enable _mm_rcp14_ss

extern crate core;

mod app;
mod common;
mod error;
mod io;
mod math;
pub mod measure;
pub mod msurf;
pub mod optics;
pub mod units;

use crate::error::Error;

/// Main entry point for the application.
pub fn run() -> Result<(), Error> {
    use app::args::CliArgs;
    use clap::Parser;

    let args = CliArgs::parse();
    let config = app::init(&args, std::time::SystemTime::now())?;

    match args.command {
        None => app::gui::run(config),
        Some(cmd) => app::cli::run(cmd, config),
    }
}
