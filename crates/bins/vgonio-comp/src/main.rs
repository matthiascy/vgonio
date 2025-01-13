#![warn(clippy::all, rust_2021_compatibility)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    std::process::exit(match vgonio_comp::run() {
        Ok(_) => 0,
        Err(ref e) => {
            eprintln!("{e}");
            1
        },
    })
}
