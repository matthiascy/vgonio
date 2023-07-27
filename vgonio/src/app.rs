use crate::error::RuntimeError;
use serde::{Deserialize, Serialize};
use std::{fmt, io::Write, path::Path};

pub(crate) mod args;

pub mod cache;
pub mod cli;
pub mod gfx;
pub(crate) mod gui;

use crate::app::cache::resolve_path;
use args::CliArgs;
use vgcore::error::VgonioError;
use vgsurf::TriangulationPattern;

/// Vgonio configuration.
#[derive(Debug)]
pub struct Config {
    /// Path to the configuration directory.
    sys_config_dir: std::path::PathBuf,

    /// Path to the cache directory.
    sys_cache_dir: std::path::PathBuf,

    /// Path to the data files.
    sys_data_dir: std::path::PathBuf,

    /// Current working directory (where the user started the program).
    /// CWD will be used in case [`UserConfig::output_dir`] is not defined.
    cwd: std::path::PathBuf,

    /// User-defined configuration.
    user: UserConfig,
}

/// Options can be configured by user.
#[derive(Debug, Serialize, Deserialize)]
pub struct UserConfig {
    /// Path to user-defined cache directory.
    /// If not set, the default cache directory is used.
    pub cache_dir: Option<std::path::PathBuf>,

    /// Path to the user-defined output directory.
    /// If not set, the default output directory is used.
    pub output_dir: Option<std::path::PathBuf>,

    /// Path to the user-defined data files directory.
    pub data_dir: Option<std::path::PathBuf>,

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
            config.cache_dir = Some(resolve_path(base, Some(&cache_dir)));
        }
        if let Some(output_dir) = config.output_dir {
            config.output_dir = Some(resolve_path(base, Some(&output_dir)));
        }
        if let Some(data_dir) = config.data_dir {
            config.data_dir = Some(resolve_path(base, Some(&data_dir)));
        }
        log::trace!("  - User cache directory: {:?}", config.cache_dir);
        log::trace!("  - User output directory: {:?}", config.output_dir);
        log::trace!("  - User data directory: {:?}", config.data_dir);
        log::trace!("  - Triangulation pattern: {:?}", config.triangulation);
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
    /// "C:\Users\username\AppData\Roaming"
    ///
    /// + On macos system: "$HOME/Library/Application Support"
    ///
    /// # Default cache directory
    ///
    /// + On *nix system: "$XDG_CACHE_HOME" or "$HOME/.cache"
    ///
    /// + On windows system: `%LOCALAPPDATA%` which is usually
    /// "C:\Users\username\AppData\Local"
    ///
    /// + On macos system: "$HOME/Library/Caches"
    ///
    /// # Default data files directory
    ///
    /// + On *nix system: "$XDG_DATA_HOME" or "$HOME/.local/share"
    ///
    /// + On windows system: `%LOCALAPPDATA%` which is usually
    /// "C:\Users\username\AppData\Local" (same as cache directory)
    ///
    /// + On macos system: "$HOME/Library/Application Support"
    /// (same as configuration directory)
    pub fn load_config(filepath: Option<&Path>) -> Result<Self, VgonioError> {
        log::info!("Loading configurations...");
        let sys_config_dir = {
            let mut config_dir = dirs::config_dir().ok_or(VgonioError::new(
                format!("System configuration directory not found."),
                Some(Box::new(RuntimeError::SysConfigDirNotFound)),
            ))?;
            config_dir.push("vgonio");
            config_dir
        };
        let sys_cache_dir = {
            let mut cache_dir = dirs::cache_dir().ok_or(VgonioError::new(
                format!("System cache directory not found."),
                Some(Box::new(RuntimeError::SysCacheDirNotFound)),
            ))?;
            cache_dir.push("vgonio");
            cache_dir
        };
        let sys_data_dir = {
            let mut data_dir = dirs::data_dir().ok_or(VgonioError::new(
                format!("System data files directory not found."),
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
            VgonioError::from_io_error(err, format!("Failed to get current working directory."))
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
                            format!("Failed to serialize default configuration."),
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

pub fn log_filter_from_level(level: u8) -> log::LevelFilter {
    match level {
        0 => log::LevelFilter::Error,
        1 => log::LevelFilter::Warn,
        2 => log::LevelFilter::Info,
        3 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    }
}

/// Initialises settings for the vgonio program.
///
/// This function will set up the logger and the thread pool.
/// It will also set up the cache directory if the user has enabled it.
/// This function will return the configuration of the program.
///
/// # Arguments
///
/// * `args` - The CLI arguments passed to the program.
/// * `launch_time` - The time when the program is launched.
pub fn init(args: &CliArgs, launch_time: std::time::SystemTime) -> Result<Config, VgonioError> {
    let log_level = if args.verbose { 4 } else { args.log_level };
    let log_level_wgpu = if args.debug_wgpu { 3 } else { 0 };
    let log_level_winit = if args.debug_winit { 3 } else { 0 };

    // Initialize logger settings.
    let timestamp = args.log_timestamp;
    env_logger::builder()
        .format(move |buf, record| {
            let top_level_module = record.module_path().unwrap().split("::").next().unwrap();
            if timestamp {
                let duration = launch_time.elapsed().unwrap();
                let millis = duration.as_millis() % 1000;
                let seconds = duration.as_secs() % 60;
                let minutes = (duration.as_secs() / 60) % 60;
                let hours = (duration.as_secs() / 60) / 60;
                writeln!(
                    buf,
                    "{}:{}:{}.{:03} {:5} [{}]: {}",
                    hours,
                    minutes,
                    seconds,
                    millis,
                    record.level(),
                    top_level_module,
                    record.args()
                )
            } else {
                writeln!(
                    buf,
                    "{:5} [{}]: {}",
                    record.level(),
                    top_level_module,
                    record.args()
                )
            }
        })
        .filter(Some("wgpu"), log_filter_from_level(log_level_wgpu))
        .filter(Some("naga"), log_filter_from_level(log_level_wgpu))
        .filter(Some("winit"), log_filter_from_level(log_level_winit))
        .filter_level(log_filter_from_level(log_level))
        .init();

    log::info!(
        "Vgonio launched at {} on {}.",
        chrono::DateTime::<chrono::Utc>::from(launch_time),
        std::env::consts::OS
    );

    Config::load_config(args.config.as_deref())
}
