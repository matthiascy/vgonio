#![warn(clippy::all, rust_2021_compatibility)]
// Hide the console window on Windows when in release mode.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// TODO: rewrite in client/server model to separate the UI from the computation
// to allow for a web UI using wasm.
//

#[cfg(all(not(target_arch = "wasm32"), feature = "viewer_native"))]
fn main() { view::run_vgonio_viewer_native() }

#[cfg(all(not(target_arch = "wasm32"), feature = "compute"))]
fn main() { comp::run_vgonio_compute() }

#[cfg(all(target_arch = "wasm32", feature = "viewer_web"))]
fn main() { view::run_vgonio_viewer_web() }
