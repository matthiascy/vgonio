const SPIRV_DST_PATH: &str = "src/app/assets/shaders/spirv";
const GLSL_SRC_PATH: &str = "src/app/assets/shaders/glsl";

fn main() {
    // Rerun the build if GLSL shaders have changed.
    println!("cargo:rerun-if-changed={}", GLSL_SRC_PATH);

    // Create destination directory if it doesn't exist.
    std::fs::create_dir_all(SPIRV_DST_PATH).expect("Couldn't create destination directory.");

    // Compile GLSL shaders to SPIRV.
    let mut compiler = shaderc::Compiler::new().unwrap();
    // let mut options = shaderc::CompileOptions::new().unwrap();

    for entry in std::fs::read_dir(GLSL_SRC_PATH).expect("Couldn't read glsl source directory.") {
        let entry = entry.unwrap();
        if let Ok(file_type) = entry.file_type() {
            if file_type.is_file() {
                let path = entry.path();
                let filename = path.file_name().expect("empty file name").to_string_lossy();
                let kind = path
                    .extension()
                    .and_then(|ext| match ext.to_string_lossy().as_ref() {
                        "vert" => Some(shaderc::ShaderKind::Vertex),
                        "frag" => Some(shaderc::ShaderKind::Fragment),
                        _ => None,
                    });

                if let Some(kind) = kind {
                    let spirv = {
                        let source = std::fs::read_to_string(path.as_path())
                            .unwrap_or_else(|_|
                                panic!("Couldn't read source file {:?}", path.as_os_str()));
                        compiler.compile_into_spirv(&source, kind, &filename, "main", None)
                    };

                    match spirv {
                        Ok(compiled) => {
                            let output_filepath =
                                format!("src/app/assets/shaders/spirv/{}.spv", filename);
                            match std::fs::write(&output_filepath, compiled.as_binary_u8()) {
                                Ok(_) => {}
                                Err(err) => {
                                    eprintln!("Failed to write to {:?}\n{}", output_filepath, err);
                                }
                            }
                        }
                        Err(err) => {
                            eprintln!("Failed to compile shader: {:?}\n{}", path.as_os_str(), err);
                        }
                    }
                }
            }
        }
    }
}
