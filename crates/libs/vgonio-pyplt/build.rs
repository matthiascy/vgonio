use std::{fs, path::Path};

// This file is used to bundle the python code with the rust code.
fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let target_dir = Path::new(&out_dir)
        .parent()
        .unwrap() // target/debug/build
        .parent()
        .unwrap() // target/debug
        .parent()
        .unwrap(); // target
    let py_dst_dir = target_dir.join("vgplt");

    // Create the directory if it doesn't exist
    if !py_dst_dir.exists() {
        fs::create_dir_all(&py_dst_dir).unwrap();
    }

    // Copy the python files to the build directory
    let py_src_path = Path::new("vgplt");
    for entry in fs::read_dir(py_src_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() && path.extension().unwrap() == "py" {
            let file_name = path.file_name().unwrap();
            let dest_path = py_dst_dir.join(file_name);
            fs::copy(&path, &dest_path).unwrap();
        }
    }

    // Re-run the build script if any of the python files change
    println!("cargo:rerun-if-changed={}", py_src_path.display());
}
