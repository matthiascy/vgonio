#![forbid(unsafe_code)]
#![warn(clippy::all, rust_2021_compatibility)]

// TODO: rewrite in client/server model to separate the UI from the computation
// to allow for a web UI using wasm.

fn main() {
    std::process::exit(match vgonio::run() {
        Ok(_) => 0,
        Err(ref e) => {
            eprintln!("{}", e);
            1
        }
    })
}
