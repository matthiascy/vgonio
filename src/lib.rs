//! vgonio is a library for micro-level light transport simulation.

#![feature(async_closure)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(is_some_and)]
#![warn(missing_docs)]

extern crate core;

mod app;
mod common;
mod error;
mod io;
pub mod isect;
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
        None => app::gui::launch(config),
        Some(cmd) => app::cli::execute(cmd, config),
    }
}
