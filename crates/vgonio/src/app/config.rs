use crate::error::RuntimeError;
use base::error::VgonioError;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    io::Write,
    path::{Path, PathBuf},
};
use surf::TriangulationPattern;

/// Vgonio configuration.
#[derive(Debug)]
pub struct Config {
    /// Path to the configuration directory.
    pub(crate) sys_config_dir: PathBuf,

    /// Path to the cache directory.
    pub(crate) sys_cache_dir: PathBuf,

    /// Path to the data files.
    pub(crate) sys_data_dir: PathBuf,

    /// Current working directory (where the user started the program).
    /// CWD will be used in case [`UserConfig::output_dir`] is not defined.
    pub(crate) cwd: PathBuf,

    /// User-defined configuration.
    pub(crate) user: UserConfig,
}

/// Options configured by user.
#[derive(Debug, Serialize, Deserialize)]
pub struct UserConfig {
    /// Path to user-defined cache directory.
    /// If not set, the default cache directory is used.
    pub cache_dir: Option<PathBuf>,

    /// Path to the user-defined output directory.
    /// If not set, the default output directory is used.
    pub output_dir: Option<PathBuf>,

    /// Path to the user-defined data files directory.
    pub data_dir: Option<PathBuf>,

    /// Triangulation pattern for heightfield.
    #[serde(default = "TriangulationPattern::default")]
    pub triangulation: TriangulationPattern,
}

impl UserConfig {
    /// Load [`UserConfig`] from a .toml file.
    pub fn load(path: &Path) -> Result<Self, VgonioError> {
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let string = std::fs::read_to_string(path).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!("Failed to read user configuration file: {}", path.display()),
            )
        })?;
        let mut config: UserConfig = toml::from_str(&string).map_err(|err| {
            VgonioError::new(
                format!(
                    "Failed to parse user configuration file: {}",
                    path.display()
                ),
                Some(Box::new(RuntimeError::from(err))),
            )
        })?;
        if let Some(cache_dir) = config.cache_dir {
            config.cache_dir = Some(canonicalize_path(base, Some(&cache_dir)));
        }
        if let Some(output_dir) = config.output_dir {
            config.output_dir = Some(canonicalize_path(base, Some(&output_dir)));
        }
        if let Some(data_dir) = config.data_dir {
            config.data_dir = Some(canonicalize_path(base, Some(&data_dir)));
        }
        log::info!("    - User cache directory: {:?}", config.cache_dir);
        log::info!("    - User output directory: {:?}", config.output_dir);
        log::info!("    - User data directory: {:?}", config.data_dir);
        log::info!("    - Triangulation pattern: {:?}", config.triangulation);
        Ok(config)
    }
}

impl Config {
    /// Load the configuration from the config directory.
    ///
    ///
    /// This function accepts a file path to the user-defined configuration
    /// file. If it's not set, the function tries to load the user-defined
    /// configuration from the current working directory (named as
    /// vgonio.toml). If the file still doesn't exist, the function will
    /// try to load the config file located in the configuration folder.
    /// If this config file doesn't exist neither, `vgonio.toml` will
    /// be created inside the default configuration direction.
    ///
    /// # Arguments
    ///
    /// * `user_config` - Path to the user configuration file.
    ///
    /// # Default configuration directory
    ///
    /// + On *nix system: "$XDG_CONFIG_HOME" or "$HOME/.config"
    ///
    /// + On windows system: `%APPDATA%` which is usually
    ///   "C:\Users\username\AppData\Roaming"
    ///
    /// + On macos system: "$HOME/Library/Application Support"
    ///
    /// # Default cache directory
    ///
    /// + On *nix system: "$XDG_CACHE_HOME" or "$HOME/.cache"
    ///
    /// + On windows system: `%LOCALAPPDATA%` which is usually
    ///   "C:\Users\username\AppData\Local"
    ///
    /// + On macos system: "$HOME/Library/Caches"
    ///
    /// # Default data files directory
    ///
    /// + On *nix system: "$XDG_DATA_HOME" or "$HOME/.local/share"
    ///
    /// + On windows system: `%LOCALAPPDATA%` which is usually
    ///   "C:\Users\username\AppData\Local" (same as cache directory)
    ///
    /// + On macos system: "$HOME/Library/Application Support" (same as
    ///   configuration directory)
    pub fn load_config(filepath: Option<&Path>) -> Result<Self, VgonioError> {
        log::info!("Loading configurations...");
        let sys_config_dir = {
            let mut config_dir = dirs::config_dir().ok_or(VgonioError::new(
                "System configuration directory not found.",
                Some(Box::new(RuntimeError::SysConfigDirNotFound)),
            ))?;
            config_dir.push("vgonio");
            config_dir
        };
        let sys_cache_dir = {
            let mut cache_dir = dirs::cache_dir().ok_or(VgonioError::new(
                "System cache directory not found.",
                Some(Box::new(RuntimeError::SysCacheDirNotFound)),
            ))?;
            cache_dir.push("vgonio");
            cache_dir
        };
        let sys_data_dir = {
            let mut data_dir = dirs::data_dir().ok_or(VgonioError::new(
                "System data files directory not found.",
                Some(Box::new(RuntimeError::SysDataDirNotFound)),
            ))?;
            data_dir.push("vgonio");
            data_dir
        };

        log::info!(
            "  - Sys configuration directory: {}",
            sys_config_dir.display()
        );
        log::info!("  - Sys cache directory: {}", sys_cache_dir.display());
        log::info!("  - Sys data files directory: {}", sys_data_dir.display());

        // Create the default directories if they don't exist.
        if !sys_config_dir.exists() {
            std::fs::create_dir_all(&sys_config_dir).map_err(|err| {
                VgonioError::from_io_error(
                    err,
                    format!(
                        "Failed to create system configuration directory {}",
                        sys_config_dir.display()
                    ),
                )
            })?;
        }

        if !sys_cache_dir.exists() {
            std::fs::create_dir_all(&sys_cache_dir).map_err(|err| {
                VgonioError::from_io_error(
                    err,
                    format!(
                        "Failed to create system cache directory {}",
                        sys_cache_dir.display()
                    ),
                )
            })?;
        }

        if !sys_data_dir.exists() {
            std::fs::create_dir_all(&sys_data_dir).map_err(|err| {
                VgonioError::from_io_error(
                    err,
                    format!(
                        "Failed to create system data files directory {}",
                        sys_data_dir.display()
                    ),
                )
            })?;
        }

        let cwd = std::env::current_dir().map_err(|err| {
            VgonioError::from_io_error(err, "Failed to get current working directory.")
        })?;

        let user_config = if filepath.is_some_and(|p| p.exists()) {
            let user_config_path = filepath.unwrap().canonicalize().unwrap();
            log::info!(
                "  Load user specified configuration from {}",
                user_config_path.display()
            );
            UserConfig::load(&user_config_path)?
        } else {
            // No configuration file specified, try to load from CWD
            let config_in_cwd = cwd.join("vgonio.toml");
            if config_in_cwd.exists() {
                log::info!("  Load configuration in CWD: {}", config_in_cwd.display());
                UserConfig::load(&config_in_cwd)?
            } else {
                // No configuration file exists in CWD, try load from system
                // configuration folder.
                let config_in_sys = sys_config_dir.join("vgonio.toml");
                if config_in_sys.exists() {
                    log::info!("  Loading configuration from {}", config_in_sys.display());
                    UserConfig::load(&config_in_sys)?
                } else {
                    // No configuration file in system level, create one and
                    // fill with default values.
                    let user_config = UserConfig {
                        cache_dir: Some(sys_cache_dir.clone()),
                        output_dir: None,
                        data_dir: Some(sys_data_dir.clone()),
                        triangulation: TriangulationPattern::default(),
                    };
                    let serialized = toml::to_string(&user_config).map_err(|err| {
                        VgonioError::new(
                            "Failed to serialize default configuration.",
                            Some(Box::new(RuntimeError::from(err))),
                        )
                    })?;
                    let mut config_file = std::fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .open(&config_in_sys)
                        .map_err(|err| {
                            VgonioError::from_io_error(
                                err,
                                format!(
                                    "Failed to create configuration file {}",
                                    config_in_sys.display()
                                ),
                            )
                        })?;
                    config_file.write_all(serialized.as_bytes()).unwrap();
                    user_config
                }
            }
        };
        Ok(Self {
            sys_config_dir,
            sys_cache_dir,
            sys_data_dir,
            cwd,
            user: user_config,
        })
    }

    /// Returns the current working directory.
    pub fn cwd(&self) -> &Path { &self.cwd }

    /// Returns the cache directory.
    pub fn cache_dir(&self) -> &Path {
        self.user
            .cache_dir
            .as_ref()
            .map_or(self.sys_cache_dir.as_path(), |cache_dir| {
                cache_dir.as_path()
            })
    }

    /// Returns the output directory.
    /// If the output directory is not set, the current working directory will
    /// be returned.
    pub fn output_dir(&self) -> &Path {
        self.user
            .output_dir
            .as_ref()
            .map_or(self.cwd.as_path(), |output_dir| output_dir.as_path())
    }

    /// Returns the user data files directory.
    pub fn user_data_dir(&self) -> Option<&Path> { self.user.data_dir.as_deref() }

    /// Returns the default data files directory.
    pub fn sys_data_dir(&self) -> &Path { &self.sys_data_dir }

    /// Returns the configuration directory.
    pub fn sys_config_dir(&self) -> &Path { &self.sys_config_dir }

    /// Returns the default cache directory.
    pub fn sys_cache_dir(&self) -> &Path { &self.sys_cache_dir }

    /// Resolves the given path to an absolute path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to resolve. If the path is relative, it will be
    ///   resolved against the current working directory. If the path starts
    ///   with `usr://`, it will be resolved against the user data files
    ///   directory. If the path starts with `sys://`, it will be resolved
    ///   against the system data files directory. The user data files directory
    ///   and the system data files directory can be set in the configuration
    ///   file.
    pub fn resolve_path(&self, path: &Path) -> Option<PathBuf> {
        if let Ok(stripped) = path.strip_prefix("usr://") {
            if self.user_data_dir().is_some() {
                self.user_data_dir().map(|p| p.join(stripped))
            } else {
                log::error!(
                    "The file path begins with `usr://`: {}, but the user data directory is not \
                     configured.",
                    path.display()
                );
                None
            }
        } else if let Ok(stripped) = path.strip_prefix("sys://") {
            Some(self.sys_data_dir().join(stripped))
        } else {
            Some(canonicalize_path(&self.cwd, Some(path)))
        }
    }

    /// Returns the output directory in canonical form.
    /// If the output directory is not specified, returns config's output
    /// directory.
    /// # Arguments
    ///
    /// * `output` - The output directory specified by the user.
    pub fn resolve_output_dir(&self, output_dir: Option<&Path>) -> Result<PathBuf, VgonioError> {
        match output_dir {
            Some(dir) => {
                let resolved = self.resolve_path(dir);
                if resolved.is_none() {
                    Err(VgonioError::new(
                        format!("failed to resolve the output directory: {}", dir.display()),
                        Some(Box::new(RuntimeError::InvalidOutputDir)),
                    ))
                } else if !resolved.as_ref().unwrap().is_dir() {
                    Err(VgonioError::new(
                        format!("{} is not a directory", dir.display()),
                        Some(Box::new(RuntimeError::InvalidOutputDir)),
                    ))
                } else {
                    Ok(resolved.unwrap())
                }
            },
            None => Ok(self.output_dir().to_path_buf()),
        }
    }
}

/// Resolves the path to canonical form even if the path doesn't exist.
///
/// # Arguments
///
/// * `base` - The base path to resolve against.
/// * `path` - The path to resolve.
///
/// # Returns
///
/// A `PathBuf` indicating the resolved path. It differs according to the
/// base path and patterns inside `path`.
///
///   1. `path` is `None`
///
///      Returns the `base` path.
///
///   2. `path` is relative
///
///      Returns the path which is relative to `base` path, with
///      the remaining of the `path` appended.
///
///   3. `path` is absolute
///
///      Returns the `path` as is.
pub(crate) fn canonicalize_path(base: &Path, path: Option<&Path>) -> PathBuf {
    log::trace!(
        "Canonicalizing path: base={}, path={:?}",
        base.display(),
        path
    );
    const HOME: [&str; 6] = [
        "~/",
        "~\\",
        "$HOME",
        "%USERPROFILE%/",
        "%HOMEPATH%/",
        "%HOME%/",
    ];
    path.map_or(base.to_path_buf(), |path| {
        let resolved = if base.is_relative() {
            base.join(path)
        } else {
            let mut i = 0;
            loop {
                if i == HOME.len() {
                    break path.to_path_buf();
                }

                if path.starts_with(HOME[i]) {
                    break dirs::home_dir()
                        .unwrap()
                        .join(path.strip_prefix(HOME[i]).unwrap());
                }
                i += 1;
            }
        };

        if !resolved.exists() {
            normalise_path(&resolved)
        } else {
            resolved.canonicalize().unwrap()
        }
    })
}

/// Resolves the path to canonical form even if the path does not exist.
pub(crate) fn normalise_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut components = path.components().peekable();
    let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().cloned() {
        components.next();
        PathBuf::from(c.as_os_str())
    } else {
        PathBuf::new()
    };

    for component in components {
        match component {
            Component::Prefix(..) => unreachable!(),
            Component::RootDir => {
                ret.push(component.as_os_str());
            },
            Component::CurDir => {},
            Component::ParentDir => {
                ret.pop();
            },
            Component::Normal(c) => {
                ret.push(c);
            },
        }
    }
    ret
}

#[test]
fn test_normalise_path() {
    let path = Path::new("/a/b/c/../../d");
    let normalised = normalise_path(path);
    assert_eq!(normalised, Path::new("/a/d"));
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Configuration directory: {}\nSystrem cache directory: {}\nSystem datafiles \
             directory: {}\nCurrent working directory: {}\nOutput directory: {}\nUser datafiles \
             directory: {}\nUser cache directory: {}",
            self.sys_config_dir.display(),
            self.sys_cache_dir.display(),
            self.sys_data_dir.display(),
            self.cwd.display(),
            self.output_dir().display(),
            if self.user.data_dir.is_some() {
                self.user_data_dir().as_ref().unwrap().display().to_string()
            } else {
                "None".to_string()
            },
            if self.user.cache_dir.is_some() {
                self.cache_dir().display().to_string()
            } else {
                "None".to_string()
            }
        )
    }
}
