use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(clap::Parser, Debug)]
#[clap(
    author,
    version,
    about = "Micro-geometry level light transport simulation tool.",
    arg_required_else_help(true),
    allow_external_subcommands(true)
)]
pub struct Args {
    #[clap(subcommand)]
    pub subcmd: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// List all external commands in the PATH.
    List,
}

/// Check if a file is executable.
pub fn is_executable<P: AsRef<Path>>(path: P) -> bool {
    let p = path.as_ref();

    #[cfg(not(target_os = "windows"))]
    {
        use std::os::unix::fs::PermissionsExt;
        p.is_file()
            && p.metadata()
                .map_or(false, |m| m.permissions().mode() & 0o111 != 0)
    }

    #[cfg(target_os = "windows")]
    {
        p.is_file()
            && p.extension()
                .map_or(false, |ext| ext == "exe" || ext == "bat" || ext == "cmd")
    }
}

/// Search path for external commands.
pub fn search_paths() -> Vec<PathBuf> {
    let exe_path = std::env::current_exe().expect("Failed to get current executable path");
    let exe_dir = exe_path
        .parent()
        .expect("Failed to get parent directory of executable");
    let path = format!(
        "{}:{}",
        exe_dir.to_str().unwrap(),
        std::env::var("PATH").unwrap_or_default()
    );

    std::env::split_paths(&path).collect()
}

/// List all external commands starting with vgonio- in the PATH.
///
/// The search path includes the directories listed in the PATH environment
/// variable and the current working directory where the executable is located.
///
/// The listed commands are the full path to the executable.
pub fn list_all_external_commands(paths: &[PathBuf]) -> BTreeMap<String, PathBuf> {
    let mut commands = BTreeMap::new();

    for dir in paths {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if is_executable(&path) {
                        if let Some(name) = path.file_name() {
                            if let Some(name) = name.to_str() {
                                if name.starts_with("vgonio-") {
                                    commands.insert(name.to_string(), path);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    commands
}
