//! Xtask to run cargo commands using the repository Python virtual environment.
//!
//! This sets `PYO3_PYTHON` to the repo `.venv` Python executable and ensures
//! the virtualenv's library directory is exposed to the dynamic loader:
//! - `LD_LIBRARY_PATH` on Linux
//! - `DYLD_FALLBACK_LIBRARY_PATH` on macOS
//! - `PATH` on Windows
//!
//! Usage:
//! ```sh
//! cargo run --package xtask -- [--gen-env[=PATH]] <cargo-args>
//! ```
//!
//! Optional Cargo alias (add to `~/.cargo/config.toml` or repo
//! `.cargo/config.toml`):
//!
//! ```toml
//! [alias]
//! x = "run --package xtask --"
//! ```
//!
//! Then invoke the task with:
//! ```sh
//! cargo x <cargo-args>
//! ```
use clap::Parser;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

/// A type alias for a boxed dynamic error.
type DynError = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(long = "bt", help = "Enable Rust backtrace")]
    pub enable_backtrace: bool,

    #[arg(
        long = "gen-env",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "./.cargo/config.toml",
        value_name = "PATH",
        help = "Generate a dotenv file with required variables. If PATH is not provided, the variables will be appended to `./.cargo/config.toml`."
    )]
    pub generate_env: Option<PathBuf>,

    #[arg(
        required_unless_present = "generate_env",
        num_args = 1..,
        value_name = "ARGS",
        trailing_var_arg = true,
        help = "Subcommand and arguments to pass to cargo"
    )]
    pub cargo_args: Vec<String>,

    #[arg(
        long = "--interactive",
        help = "Run vgonio related cargo commands in interactive mode"
    )]
    pub interactive: bool,
}

/// Runs the xtask with the provided arguments.
fn run() -> Result<(), DynError> {
    let args = Args::parse();
    let repo_path = repo_root_path()?;
    let py_path = py_path(&repo_path)?;
    let lib_py_path = lib_python_path(&py_path)?;

    if let Some(env_path) = args.generate_env.as_deref() {
        generate_env_file(
            &repo_path,
            env_path,
            &py_path,
            lib_py_path.as_deref(),
            args.enable_backtrace,
        )?;
    }

    if args.cargo_args.is_empty() {
        return Ok(());
    }

    run_cargo_commands(&repo_path, &py_path, lib_py_path.as_deref(), &args)?;

    Ok(())
}

/// Gets the root path of the repository.
///
/// This is determined by taking the parent directory of the Cargo manifest
/// directory.
///
/// # Errors
///
/// Returns an error if the `CARGO_MANIFEST_DIR` environment variable is not set
/// or if the parent directory cannot be determined.
fn repo_root_path() -> Result<PathBuf, DynError> {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR")?)
        .parent()
        .map(Path::to_path_buf)
        .ok_or("Failed to get repo root path".into())
}

/// Attempts to find the path to the Python executable in the repository virtual
/// environment.
///
/// The expected location is `.venv/bin/python` on Unix and
/// `.venv\Scripts\python.exe` on Windows. If the executable is not found at the
/// expected location, an error is returned. This function does not search for
/// Python outside of the expected virtual environment location, as the purpose
/// of this xtask is to ensure that the repository virtual environment is used
/// for cargo commands.
fn py_path(repo_path: &Path) -> Result<PathBuf, DynError> {
    let py_path = if cfg!(windows) {
        repo_path.join(".venv").join("Scripts").join("python.exe")
    } else {
        repo_path.join(".venv").join("bin").join("python")
    };

    if !py_path.is_file() {
        return Err(format!(
            "Virtual environment Python executable not found at {}",
            py_path.display()
        )
        .into());
    }

    Ok(py_path)
}

/// Attempts to find the path to the Python library directory using the provided
/// Python executable.
fn lib_python_path(py_path: &Path) -> Result<Option<PathBuf>, DynError> {
    // As the venv can be created with different tools, we need to find the correct
    // libpython.so or .dll path.
    let out = Command::new(py_path)
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')",
        ])
        .output()?;
    if !out.status.success() {
        return Err(format!(
            "Failed to get libpython path {}",
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let s = s.trim();
    Ok(if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    })
}

/// Runs the specified cargo command with the appropriate environment variables.
fn run_cargo_commands(
    repo_path: &Path,
    py_path: &Path,
    lib_py_path: Option<&Path>,
    args: &Args,
) -> Result<(), DynError> {
    let mut command = Command::new("cargo");
    command.args(&args.cargo_args);
    command.current_dir(repo_path);
    command.env("PYO3_PYTHON", py_path);

    if args.enable_backtrace {
        command.env("RUST_BACKTRACE", "1");
    }

    if cfg!(windows) {
        let path_prefix = windows_runtime_path(py_path, lib_py_path)?;
        if !path_prefix.is_empty() {
            prepend_env(&mut command, "PATH", &path_prefix);
        }
    } else if let Some(ld) = lib_py_path {
        prepend_env(
            &mut command,
            loader_library_path_key(),
            ld.to_string_lossy().as_ref(),
        );
    }

    println!("Running {command:?}");

    let status = command.status()?;
    if !status.success() {
        return Err(format!("Cargo command failed with status {status}").into());
    }

    Ok(())
}

/// Generates a dotenv file with the required environment variables.
///
/// The generated file will include:
/// - `PYO3_PYTHON` pointing to the Python executable in the repo virtual environment.
/// - `RUST_BACKTRACE=1` if backtrace is enabled.
/// - On Windows, the virtual environment's `Scripts` directory and the libpython directory (if
///   found) will be prepended to `PATH`.
/// - On macOS, the libpython directory (if found) will be prepended to
///   `DYLD_FALLBACK_LIBRARY_PATH`.
/// - On Linux, the libpython directory (if found) will be prepended to `LD_LIBRARY_PATH`.
///
/// # Errors
///
/// Returns an error if the output path cannot be created or written to, or if
/// the Python executable path is invalid.
fn generate_env_file(
    repo_path: &Path,
    output_path: &Path,
    py_path: &Path,
    lib_py_path: Option<&Path>,
    enable_backtrace: bool,
) -> Result<(), DynError> {
    let output_path = if output_path.is_absolute() {
        output_path.to_path_buf()
    } else {
        repo_path.join(output_path)
    };

    let assignments = build_env_assignments(py_path, lib_py_path, enable_backtrace)?;

    if is_cargo_config_path(&output_path) {
        upsert_cargo_env_assignments(&output_path, &assignments)?;
    } else {
        upsert_dotenv_assignments(&output_path, &assignments)?;
    }

    Ok(())
}

/// Builds the list of environment variable assignments to be added to the dotenv or Cargo config
/// file.
fn build_env_assignments(
    py_path: &Path,
    lib_py_path: Option<&Path>,
    enable_backtrace: bool,
) -> Result<Vec<(String, String)>, DynError> {
    let mut assignments = vec![(
        "PYO3_PYTHON".to_string(),
        py_path.to_string_lossy().to_string(),
    )];

    if enable_backtrace {
        assignments.push(("RUST_BACKTRACE".to_string(), "1".to_string()));
    }

    if cfg!(windows) {
        let path_value = windows_runtime_path(py_path, lib_py_path)?;
        if !path_value.is_empty() {
            assignments.push(("PATH".to_string(), path_value));
        }
    } else if let Some(ld) = lib_py_path {
        assignments.push((
            loader_library_path_key().to_string(),
            ld.to_string_lossy().to_string(),
        ));
    }

    Ok(assignments)
}

/// Constructs the value to prepend to `PATH` on Windows, which includes the virtual environment's
/// `Scripts` directory and the libpython directory (if found).
fn windows_runtime_path(py_path: &Path, lib_py_path: Option<&Path>) -> Result<String, DynError> {
    let py_dir = py_path
        .parent()
        .ok_or("Failed to get python executable directory")?;
    let mut entries = vec![py_dir.to_string_lossy().to_string()];

    if let Some(lib_path) = lib_py_path {
        entries.push(lib_path.to_string_lossy().to_string());
    }

    Ok(entries.join(";"))
}

/// Gets the appropriate environment variable key for specifying library search paths.
fn loader_library_path_key() -> &'static str {
    if cfg!(target_os = "macos") {
        "DYLD_FALLBACK_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// Checks if the given path is a Cargo config file (i.e. ends with `.cargo/config.toml`).
fn is_cargo_config_path(path: &Path) -> bool {
    path.parent()
        .and_then(Path::file_name)
        .is_some_and(|name| name == ".cargo")
        && path.file_name().is_some_and(|name| name == "config.toml")
}

/// Upserts environment variable assignments into a dotenv file.
fn upsert_dotenv_assignments(
    path: &Path,
    assignments: &[(String, String)],
) -> Result<(), DynError> {
    ensure_parent_dir(path)?;
    let mut lines = read_lines(path)?;

    for (key, value) in assignments {
        let rendered = format!("{key}={}", quote_dotenv_value(value));
        if let Some(idx) = find_key_line(&lines, key, 0, lines.len()) {
            if lines[idx] != rendered {
                lines[idx] = rendered;
                println!("Updated variable {key} in {}", path.display());
            }
        } else {
            lines.push(rendered);
            println!("Added variable {key} to {}", path.display());
        }
    }

    write_lines(path, &lines)
}

/// Upserts environment variable assignments into a Cargo config file.
fn upsert_cargo_env_assignments(
    path: &Path,
    assignments: &[(String, String)],
) -> Result<(), DynError> {
    ensure_parent_dir(path)?;
    let mut lines = read_lines(path)?;

    let env_start = if let Some(idx) = lines.iter().position(|line| line.trim() == "[env]") {
        idx
    } else {
        if !lines.is_empty() && !lines.last().is_some_and(|line| line.trim().is_empty()) {
            lines.push(String::new());
        }
        lines.push("[env]".to_string());
        lines.len() - 1
    };

    let mut env_end = lines
        .iter()
        .enumerate()
        .skip(env_start + 1)
        .find_map(|(idx, line)| is_toml_table_header(line).then_some(idx))
        .unwrap_or(lines.len());

    for (key, value) in assignments {
        let rendered = format!("{key} = {}", quote_dotenv_value(value));
        if let Some(idx) = find_key_line(&lines, key, env_start + 1, env_end) {
            if lines[idx] != rendered {
                lines[idx] = rendered;
                println!("Updated variable {key} in {}", path.display());
            }
        } else {
            lines.insert(env_end, rendered);
            env_end += 1;
            println!("Added variable {key} to {}", path.display());
        }
    }

    write_lines(path, &lines)
}

/// Ensures that the parent directory of the given path exists, creating it if necessary.
fn ensure_parent_dir(path: &Path) -> Result<(), DynError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Reads lines from the specified path, returning an empty vector if the file does not exist.
fn read_lines(path: &Path) -> Result<Vec<String>, DynError> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    Ok(fs::read_to_string(path)?
        .lines()
        .map(ToString::to_string)
        .collect())
}

/// Writes the provided lines to the specified path, joining them with newlines.
fn write_lines(path: &Path, lines: &[String]) -> Result<(), DynError> {
    let mut content = lines.join("\n");
    if !content.is_empty() {
        content.push('\n');
    }
    fs::write(path, content)?;
    Ok(())
}

/// Finds the line index of the given key in the provided lines.
fn find_key_line(lines: &[String], key: &str, start: usize, end: usize) -> Option<usize> {
    lines
        .iter()
        .enumerate()
        .skip(start)
        .take(end.saturating_sub(start))
        .find_map(|(idx, line)| {
            if parse_assignment_key(line) == Some(key) {
                Some(idx)
            } else {
                None
            }
        })
}

/// Parses a line to find an assignment key (e.g. `KEY=value` or `KEY = value`).
fn parse_assignment_key(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('[') {
        return None;
    }
    let (key, _) = trimmed.split_once('=')?;
    let key = key.trim();
    if key.is_empty() {
        None
    } else {
        Some(key)
    }
}

/// Checks if a line is a TOML table header (e.g. `[env]`).
fn is_toml_table_header(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('[') && trimmed.ends_with(']')
}

/// Prepends a value to an environment variable in the given command.
fn prepend_env(cmd: &mut Command, key: &str, value: &str) {
    let sep = if cfg!(windows) { ";" } else { ":" };
    let old = env::var(key).unwrap_or_default();
    let new_val = if old.is_empty() {
        value.to_string()
    } else {
        format!("{value}{sep}{old}")
    };
    cmd.env(key, new_val);
}

/// Quotes a value for inclusion in a dotenv file.
///
/// This is necessary to ensure that special characters in the value (like
/// backslashes or quotes) are properly escaped and that the value is treated as
/// a single string when the dotenv file is loaded.
fn quote_dotenv_value(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}
