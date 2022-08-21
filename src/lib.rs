//! vgonio is a library for micro-level light transport simulation.

#![feature(is_some_with)]
#![feature(const_fn_floating_point_arithmetic)]
#![warn(missing_docs)]

extern crate core;

pub mod acq;
mod app;
mod error;
mod gfx;
pub mod htfld;
mod io;
pub mod isect;
mod math;
mod mesh;
mod util;

use crate::error::Error;

/// Main entry point.
#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> Result<(), Error> {
    use app::VgonioArgs;
    use clap::Parser;

    let launch_time = std::time::SystemTime::now();

    println!(
        "Vgonio launched at {} on {}",
        chrono::DateTime::<chrono::Utc>::from(launch_time),
        std::env::consts::OS
    );

    // Parse the command line arguments
    let args: VgonioArgs = VgonioArgs::parse();

    // Initialize vgonio application
    let config = app::init(&args, launch_time)?;

    // Dispatch subcommands
    match args.command {
        None => app::launch_gui(config),
        Some(cmd) => app::execute_command(cmd, config),
    }
}
