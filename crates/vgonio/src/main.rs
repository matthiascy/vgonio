#![warn(clippy::all, rust_2021_compatibility)]
// Hide the console window on Windows when in release mode.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::Parser;

#[cfg(all(not(target_arch = "wasm32"), feature = "viewer_native"))]
fn main() { view::run_vgonio_viewer_native() }

#[cfg(all(target_arch = "wasm32", feature = "viewer_web"))]
fn main() { view::run_vgonio_viewer_web() }

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(any(
        all(not(target_arch = "wasm32"), feature = "viewer_native"),
        all(not(target_arch = "wasm32"), feature = "compute"),
        all(target_arch = "wasm32", feature = "viewer_web"),
    )))]
    {
        eprintln!(
            "No feature enabled. Please enable one of the following features: viewer_native, \
             compute, viewer_web"
        );
        std::process::exit(1);
    }

    let args = vgonio::Args::parse();

    println!("VGonio v{}", env!("CARGO_PKG_VERSION"));
    println!("  by {}", env!("CARGO_PKG_AUTHORS"));
    println!();
    println!("  Running with the following arguments:");
    println!("    {:?}", args);

    #[cfg(all(not(target_arch = "wasm32"), feature = "compute"))]
    comp::run_vgonio_compute();

    Ok(())
}
