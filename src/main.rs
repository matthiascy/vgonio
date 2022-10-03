#![forbid(unsafe_code)]
#![warn(clippy::all, rust_2018_idioms)]

fn main() {
    std::process::exit(match vgonio::run() {
        Ok(_) => 0,
        Err(ref e) => {
            eprintln!("{}", e);
            1
        }
    })
}
