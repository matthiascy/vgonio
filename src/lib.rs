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
#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> Result<(), Error> {
    use app::VgonioArgs;
    use clap::Parser;

    let launch_time = std::time::SystemTime::now();

    println!(
        "Vgonio launched at {} on {}.",
        chrono::DateTime::<chrono::Utc>::from(launch_time),
        std::env::consts::OS
    );

    // Parse the command line arguments
    let args: VgonioArgs = VgonioArgs::parse();

    // Initialize vgonio application
    let config = app::init(&args, launch_time)?;

    match args.command {
        None => app::gui::launch(config),
        Some(cmd) => app::cli::execute(cmd, config),
    }
}

/// Main entry point for the application when compiled to WebAssembly.
///
/// Only runs the Vgonio GUI.
///
/// wasm_bindgen(start) requires that this function must take no arguments and
/// must either return `()` or `Result<(), JsValue>`.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn run() -> Result<(), wasm_bindgen::JsValue> {
    app::init_wasm().map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    app::gui::launch_wasm().map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}
