#![forbid(unsafe_code)]
#![warn(clippy::all, rust_2018_idioms)]

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    ::std::process::exit(match vgonio::run() {
        Ok(_) => 0,
        Err(ref e) => {
            eprintln!("{}", e);
            1
        }
    })
}
