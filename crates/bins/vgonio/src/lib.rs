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
    let exe_dir =
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| std::env::current_dir().unwrap());
    let mut paths: Vec<PathBuf> = std::env::var_os("PATH")
        .as_deref()
        .map(std::env::split_paths)
        .map(|it| it.collect())
        .unwrap_or_default();

    // Search first in the directory where the executable is located.
    paths.insert(0, exe_dir);
    paths
}

/// List all external commands starting with vgn- in the PATH.
///
/// The search path includes the directories listed in the PATH environment
/// variable and the current working directory where the executable is located.
///
/// The listed commands are the full path to the executable.
pub fn list_all_external_commands(paths: &[PathBuf]) -> BTreeMap<String, PathBuf> {
    const PREFIXES: &[&str] = &["vgn-", "vgonio-"];
    let mut commands = BTreeMap::new();

    for dir in paths {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if !is_executable(&path) {
                        continue;
                    }
                    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    let stem = path.file_stem().and_then(|n| n.to_str()).unwrap_or(file_name);

                    if PREFIXES.iter().any(|prefix| stem.starts_with(prefix)) {
                        // key by canonical subcommand name without prefix & extension
                        if let Some((_, cmd)) = PREFIXES.iter().find_map(|prefix| stem.strip_prefix(prefix).map(|c| (prefix, c))) {
                            commands.entry(cmd.to_string()).or_insert(path);
                        }
                    }
                }
            }
        }
    }

    commands
}
