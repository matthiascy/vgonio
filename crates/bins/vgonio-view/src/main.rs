fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    match vgonio_view::run_native() {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        },
    }

    #[cfg(target_arch = "wasm32")]
    vgonio_view::run_wasm32();
}
