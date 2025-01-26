use std::{ffi::CString, fs::File, io::Read, path::Path};

/// Load the Python source code dynamically from the given path into a
/// `CString`. This function is an alternative to `include_str!` for loading
/// Python source code dynamically without recompiling the crate.
///
/// # Arguments
///
/// * `path` - The path to the Python source code.
pub fn load_python_source_code(path: &Path) -> Result<CString, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(CString::new(contents)?)
}
