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
//! cargo run --package xtask -- <cargo-args>
//! ```
//!
//! Optional Cargo alias (add to `~/.cargo/config.toml` or repo
//! `.cargo/config.toml`): ```toml
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
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(long = "bt", help = "Enable Rust backtrace")]
    pub enable_backtrace: bool,

    #[arg(required = true, num_args = 1.., value_name = "ARGS", trailing_var_arg = true, help = "Subcommand and arguments to pass to cargo")]
    pub cargo_args: Vec<String>,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let repo_path = repo_root_path()?;
    let py_path = py_path(&repo_path)?;
    let lib_py_path = lib_python_path(&py_path)?;

    run_cargo_commands(&repo_path, &py_path, &lib_py_path, &args)?;

    Ok(())
}

fn repo_root_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR")?)
        .parent()
        .map(|p| p.to_path_buf())
        .ok_or("Failed to get repo root path".into())
}

fn py_path(repo_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let py_path = if cfg!(windows) {
        repo_path.join(".venv").join("Scripts").join("python.exe")
    } else {
        repo_path.join(".venv").join("bin").join("python")
    };

    if !py_path.exists() {
        return Err(format!("Virtual environment not found at {:?}", py_path).into());
    }

    Ok(py_path)
}

fn lib_python_path(py_path: &Path) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
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

fn run_cargo_commands(
    repo_path: &Path,
    py_path: &Path,
    lib_py_path: &Option<PathBuf>,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut command = Command::new("cargo");
    command.args(&args.cargo_args);
    command.current_dir(repo_path);
    command.env("PYO3_PYTHON", py_path);

    if args.enable_backtrace {
        command.env("RUST_BACKTRACE", "1");
    }

    if cfg!(windows) {
        let py_dir = py_path.parent().unwrap().to_string_lossy().to_string();
        prepend_env(&mut command, "PATH", &py_dir);
        if let Some(ld) = lib_py_path {
            let ld = ld
                .parent()
                .map(|p| p.to_string_lossy())
                .ok_or("Failed to get libpython parent dir")?;
            prepend_env(&mut command, "PATH", ld.as_ref());
        }
    } else if cfg!(target_os = "macos") {
        if let Some(ld) = lib_py_path {
            prepend_env(
                &mut command,
                "DYLD_FALLBACK_LIBRARY_PATH",
                ld.to_string_lossy().as_ref(),
            );
        }
    } else if let Some(ld) = lib_py_path {
        prepend_env(
            &mut command,
            "LD_LIBRARY_PATH",
            ld.to_string_lossy().as_ref(),
        );
    }

    println!("Running {:?}", command);

    let status = command.status()?;
    if !status.success() {
        return Err(format!("Cargo command failed with status {}", status).into());
    }

    Ok(())
}

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
