//! vgonio is a library for micro-level light transport simulation.

#![feature(async_closure)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(is_some_and)]
#![warn(missing_docs)]

extern crate core;

pub mod acq;
mod app;
mod error;
pub mod htfld;
mod io;
pub mod isect;
mod math;
mod mesh;
pub mod units;
mod util;

use crate::error::Error;

/// Main entry point for the application.
pub fn run() -> Result<(), Error> {
    use app::args::VgonioArgs;
    use clap::Parser;

    let launch_time = std::time::SystemTime::now();

    println!(
        "Vgonio launched at {} on {}.",
        chrono::DateTime::<chrono::Utc>::from(launch_time),
        std::env::consts::OS
    );

    let args = VgonioArgs::parse();
    let config = app::init(&args, launch_time)?;

    match args.command {
        None => app::gui::launch(config),
        Some(cmd) => app::cli::execute(cmd, config),
    }
}
