use crate::error::Error;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::{io::Write, path::PathBuf};

pub mod cache;
pub mod cli;
pub mod gfx;
pub(crate) mod gui;

/// Vgonio configuration.
#[derive(Debug)]
pub struct Config {
    /// Path to the configuration directory.
    pub config_dir: std::path::PathBuf,

    /// Path to the cache directory.
    pub cache_dir: std::path::PathBuf,

    /// Path to the data files.
    pub data_dir: std::path::PathBuf,

    /// Current working directory (where the user started the program).
    pub cwd: std::path::PathBuf,

    /// Path to the user-defined configuration.
    pub user_config: UserConfig,
}

/// User-defined configuration.
#[derive(Debug, Serialize, Deserialize)]
pub struct UserConfig {
    /// Path to user-defined cache directory.
    /// If not set, the default cache directory is used.
    pub cache_dir: std::path::PathBuf,

    /// Path to the user-defined output directory.
    /// If not set, the default output directory is used.
    pub output_dir: std::path::PathBuf,

    /// Path to the user-defined data files directory.
    pub data_dir: std::path::PathBuf,
}

impl UserConfig {
    /// Convert the possibly relative paths to absolute.
    fn to_absolute(self) -> Self {
        Self {
            cache_dir: self.cache_dir.canonicalize().expect(&format!(
                "failed to convert relative path '{}' to absolute",
                self.cache_dir.display(),
            )),
            output_dir: self.output_dir.canonicalize().expect(&format!(
                "failed to convert relative path '{}' to absolute",
                self.output_dir.display(),
            )),
            data_dir: self.data_dir.canonicalize().expect(&format!(
                "failed to convert relative path '{}' to absolute",
                self.data_dir.display(),
            )),
        }
    }
}

impl Config {
    /// Load the configuration from the config directory.
    ///
    /// There is a difference between the default onfiguration and the
    /// user-defined configuration. The default configuration is reserved for
    /// the vgonio application itself and can not be modified while the
    /// user-defined configuration is freely editable by the user.
    ///
    /// The default configuration is firstly loaded from the default
    /// configuration directory if it exists. Otherwise, the default
    /// configuration is created and saved to the default configuration
    /// directory.
    ///
    /// This function accepts a file path to the user-defined configuration
    /// file. If it's not set, the function tries to load the user-defined
    /// configuration from the current working directory (named as
    /// vgonio-user.toml). If the file still doesn't exist, the function will
    /// create a new user-specific configuration file and save it to the current
    /// working directory as vgonio-user.toml with current working directory
    /// as the default output directory, the default cache directory as the
    /// user-defined cache directory and the default data directory as the
    /// user-defined data directory.
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
    pub fn load_config(user_config_file: Option<&PathBuf>) -> Result<Self, Error> {
        log::info!("Loading configurations...");
        // Load or create the default configuration.
        let config_dir = {
            let mut config_dir = dirs::config_dir().ok_or(Error::SysConfigDirNotFound)?;
            config_dir.push("vgonio");
            config_dir
        };
        let cache_dir = {
            let mut cache_dir = dirs::cache_dir().ok_or(Error::SysCacheDirNotFound)?;
            cache_dir.push("vgonio");
            cache_dir
        };
        let data_dir = {
            let mut data_dir = dirs::data_dir().ok_or(Error::SysDataDirNotFound)?;
            data_dir.push("vgonio");
            data_dir
        };

        log::info!("Configuration directory: {}", config_dir.display());
        log::info!("Cache directory: {}", cache_dir.display());
        log::info!("Data files directory: {}", data_dir.display());

        // Create the default directories if they don't exist.
        if !config_dir.exists() {
            std::fs::create_dir_all(&config_dir)?;
        }

        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)?;
        }

        if !data_dir.exists() {
            std::fs::create_dir_all(&data_dir)?;
        }

        let cwd = std::env::current_dir()?;

        if let Some(path) = user_config_file {
            if !path.exists() {
                Err(Error::UserConfigNotFound)
            } else {
                log::info!("Loading user configuration from {}", path.display());
                let user_config: UserConfig = {
                    let user_config_string = std::fs::read_to_string(path)?;
                    toml::from_str(&user_config_string)?
                };
                Ok(Self {
                    config_dir,
                    cache_dir,
                    data_dir,
                    cwd,
                    user_config: user_config.to_absolute(),
                })
            }
        } else {
            // Try to load the user-defined configuration in the current working
            // directory. If it doesn't exist, create a new one.
            let cwd_config_path = cwd.join("vgonio-user.toml");
            let user_config = if cwd_config_path.exists() {
                log::info!(
                    "Loading user configuration from cwd: {}",
                    cwd_config_path.display()
                );
                let user_config_string = std::fs::read_to_string(&cwd_config_path)?;
                let user_config: UserConfig = toml::from_str(&user_config_string)?;
                user_config.to_absolute()
            } else {
                log::info!("Creating user configuration in cwd: {}", cwd.display());
                let mut user_config = UserConfig {
                    cache_dir: cache_dir.clone(),
                    output_dir: cwd.clone(),
                    data_dir: data_dir.clone(),
                };
                let mut file = std::fs::File::create(&cwd_config_path)?;
                println!("Serialised user config: {}", toml::to_string(&user_config)?);
                file.write_all(toml::to_string(&user_config)?.as_bytes())?;
                user_config
            };
            Ok(Self {
                config_dir,
                cache_dir,
                data_dir,
                user_config,
                cwd,
            })
        }
    }
}

/// Top-level CLI arguments.
#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Micro-geometry level light transportation simulation."
)]
pub struct VgonioArgs {
    /// Whether to print any information to stdout.
    #[clap(short, long, help = "Silent output printed to stdout")]
    pub quite: bool,

    /// Whether to print verbose information to stdout.
    #[clap(short, long, help = "Use verbose output (log level = 4)")]
    pub verbose: bool,

    /// Path to the file where to output the log to.
    #[clap(long, help = "Set a file to output the log to")]
    pub log_file: Option<PathBuf>,

    /// File descriptor where to output the log to.
    #[clap(long, help = "Set a file descriptor as log output [2 = stderr]")]
    pub log_fd: Option<u32>,

    /// Whether to show the timestamp in the log.
    #[clap(
        long,
        help = "Show timestamp for each log message in seconds since\nprogram starts"
    )]
    pub log_timestamp: bool,

    /// Verbosity level for the log.
    #[clap(
        long,
        help = "Setting logging verbosity level (higher for more\ndetails)\n  0 - error\n  1 - \
                warn + error\n  2 - info + warn + error\n  3 - debug + info + warn + error\n  4 - \
                trace + debug + info + warn + error\n\x08",
        default_value_t = 1
    )]
    pub log_level: u8,

    #[clap(long, help = "Enable debug messages from `wgpu-rs` and `naga`")]
    pub debug_wgpu: bool,

    #[clap(long, help = "Enable debug messages from `winit`")]
    pub debug_winit: bool,

    /// Command to execute.
    #[clap(subcommand)]
    pub command: Option<VgonioCommand>,

    /// Path to the user config file. If not specified, vgonio will
    /// load the default config file.
    #[clap(short, long, help = "Path to the user config file")]
    pub config: Option<PathBuf>,
}

/// Micro-surface information that can be retrieved.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, ValueEnum)]
pub enum MicroSurfaceInfo {
    /// Normal vectors per vertex.
    VertexNormal,

    /// Normal vectors per triangle.
    SurfaceNormal,
}

/// Vgonio command.
#[derive(Subcommand, Debug)]
pub enum VgonioCommand {
    /// Measures micro-geometry level light transport related metrics.
    Measure(MeasureOptions),

    /// Extracts micro-surface information from a mesh.
    Extract(ExtractOptions),
}

/// Options for the `measure` command.
#[derive(Args, Debug)]
#[clap(about = "Measure different aspects of the micro-surface.")]
pub struct MeasureOptions {
    #[clap(short, long, help = "The input measurement description file.")]
    input_path: PathBuf,

    #[clap(
        short,
        long,
        help = "The path where stores the simulation data. Use //\nat the start of the path to \
                set the output path\nrelative to the input file location. If not \nspecified, \
                current working directory will be used"
    )]
    output_path: Option<PathBuf>,

    #[clap(
        short,
        long = "num-threads",
        help = "The number of threads in the thread pool"
    )]
    nthreads: Option<u32>,

    #[clap(long, help = "Use caches to minimize the processing time")]
    enable_cache: bool,

    #[clap(
        long,
        help = "Show detailed statistics about memory and time\nusage during the measurement"
    )]
    print_stats: bool,
}

/// Options for the `extract` command.
#[derive(Args, Debug)]
#[clap(about = "Extract information from micro-surface.")]
pub struct ExtractOptions {
    #[clap(value_enum, short, long, help = "Type of information to be extracted.")]
    kind: MicroSurfaceInfo,

    #[clap(
        short,
        long,
        help = "The input micro-surface profile, it can be either\nmicro-surface height field or \
                micro-surface mesh\ncache"
    )]
    input_path: PathBuf,

    #[clap(
        short,
        long,
        help = "The path where stores the simulation data. Use //\nat the start of the path to \
                set the output path\nrelative to the input file location. If not \nspecified, \
                current working directory will be used"
    )]
    output_path: Option<PathBuf>,

    #[clap(long, help = "Use caches to minimize the processing time")]
    enable_cache: bool,
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
pub fn init(args: &VgonioArgs, launch_time: std::time::SystemTime) -> Result<Config, Error> {
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

    log::info!("Initialising...");

    // Load the configuration file.
    Config::load_config(args.config.as_ref())
}
