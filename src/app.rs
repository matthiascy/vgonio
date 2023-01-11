use crate::error::Error;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    io::Write,
    path::{Path, PathBuf},
};

pub mod cache;
pub mod cli;
pub mod gfx;
pub(crate) mod gui;

/// Vgonio configuration.
#[derive(Debug)]
pub struct VgonioConfig {
    /// Path to the configuration directory.
    pub sys_config_dir: std::path::PathBuf,

    /// Path to the cache directory.
    pub sys_cache_dir: std::path::PathBuf,

    /// Path to the data files.
    pub sys_data_dir: std::path::PathBuf,

    /// Current working directory (where the user started the program).
    /// CWD will be used in case [`UserConfig::output_dir`] is not defined.
    pub cwd: std::path::PathBuf,

    /// User-defined configuration.
    pub user_config: UserConfig,
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
}

impl UserConfig {
    /// Load [`UserConfig`] from a .toml file.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let string = std::fs::read_to_string(&path)?;
        let mut config = toml::from_str(&string)?;
        if let Some(cache_dir) = config.cache_dir {
            config.cache_dir = cache_dir.canonicalize()?;
        }

        if let Some(output_dir) = config.output_dir {
            config.output_dir.canonicalize()?;
        }

        if let Some(data_dir) = config.data_dir {
            config.data_dir.canonicalize()?;
        }
        config
    }
}

impl VgonioConfig {
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
    pub fn load_config(config_path: Option<&PathBuf>) -> Result<Self, Error> {
        log::info!("Loading configurations...");
        let sys_config_dir = {
            let mut config_dir = dirs::config_dir().ok_or(Error::SysConfigDirNotFound)?;
            config_dir.push("vgonio");
            config_dir
        };
        let sys_cache_dir = {
            let mut cache_dir = dirs::cache_dir().ok_or(Error::SysCacheDirNotFound)?;
            cache_dir.push("vgonio");
            cache_dir
        };
        let sys_data_dir = {
            let mut data_dir = dirs::data_dir().ok_or(Error::SysDataDirNotFound)?;
            data_dir.push("vgonio");
            data_dir
        };

        log::info!(
            "Default configuration directory: {}",
            sys_config_dir.display()
        );
        log::info!("Default cache directory: {}", sys_cache_dir.display());
        log::info!("Default data files directory: {}", sys_data_dir.display());

        // Create the default directories if they don't exist.
        if !sys_config_dir.exists() {
            std::fs::create_dir_all(&sys_config_dir)?;
        }

        if !sys_cache_dir.exists() {
            std::fs::create_dir_all(&sys_cache_dir)?;
        }

        if !sys_data_dir.exists() {
            std::fs::create_dir_all(&sys_data_dir)?;
        }

        let cwd = std::env::current_dir()?;

        let Some(user_config_path) = config_path else {
            log::warn!("Configuration file '{}' specified but doesn't exist!", config_path.display());
        };

        let user_config = if user_config_path.exists() {
            log::info!("Loading configuration from {}", user_config_path.display());
            Some(UserConfig::load(&user_config_path)?)
        } else {
            // No configuration file specified, try to load from CWD
            let mut config_in_cwd = cwd.clone();
            config_in_cwd.push("vgonio.toml");
            if config_in_cwd.exists() {
                log::info!("Loading configuration from {}", config_in_cwd.display());
                Some(UserConfig::load(&config_in_cwd)?)
            }

            // No configuration file exists in CWD, try load from system
            // configuration folder.
            let mut config_in_sys = sys_config_dir.clone();
            if config_in_sys.exists() {
                log::info!("Loading configuration from {}", config_in_sys.display());
                Some(UserConfig::load(&config_in_sys)?)
            }

            // No configuration file in system level, create one and
            // fill with default values.
            let user_config = UserConfig {
                cache_dir: Some(sys_cache_dir.clone()),
                output_dir: Some(cwd.clone()),
                data_dir: Some(sys_data_dir.clone()),
            };
            let serialized = serde_yaml::to_string(&user_config).unwrap();
            let mut config_file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .open(&config_in_sys)?;
            config_file.write_all(serialized.as_bytes()).unwrap();
            None
        };
        Ok(Self {
            sys_config_dir,
            sys_cache_dir,
            sys_data_dir,
            cwd,
            user_config,
        })
    }

    /// Returns the cache directory.
    pub fn cache_dir(&self) -> &Path {
        self.user_config
            .cache_dir
            .map_or(self.sys_cache_dir.as_path(), |cache_dir| {
                cache_dir.as_path()
            })
    }

    /// Returns the output directory.
    pub fn output_dir(&self) -> &Path {
        self.user_config
            .output_dir
            .map_or(self.cwd.as_path(), |output_dir| output_dir.as_path())
    }

    /// Returns the user data files directory.
    pub fn user_data_dir(&self) -> Option<&Path> {
        self.user_config.data_dir.map(|data_dir| data_dir.as_path())
    }

    /// Returns the default data files directory.
    pub fn default_data_dir(&self) -> &Path { &self.sys_data_dir }
}

impl fmt::Display for VgonioConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Configuration directory: {}\nCache directory: {}\nDatafiles directory: {}\nCurrent \
             working directory: {}\nCache directory: {}\nOutput directory: {}\nUser datafiles \
             directory: {}\nVgonio datafiles directory: {}",
            self.sys_config_dir.display(),
            self.cache_dir().display(),
            self.default_data_dir.display(),
            self.cwd.display(),
            self.cache_dir.display(),
            self.output_dir.display(),
            self.user_data_dir.display(),
            self.default_data_dir(),
        )
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

/// Micro-surface's intrinsic property.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, ValueEnum)]
pub enum MicroSurfaceProperty {
    /// Micro-facet normal vectors.
    #[clap(name = "facet-normal")]
    MicroFacetNormal,

    /// Micro-facet distribution function (i.e. NDF/D term).
    #[clap(name = "facet-distrb")]
    MicroFacetDistribution,

    /// Micro-facet shadowing-masking function (i.e. Geometric term).
    #[clap(name = "facet-shadow")]
    MicroFacetMaskingShadowing,
}

/// Vgonio command.
#[derive(Subcommand, Debug)]
pub enum VgonioCommand {
    /// Measures micro-geometry level light transport related metrics.
    Measure(MeasureOptions),

    /// Extracts micro-surface's intrinsic properties.
    Extract(ExtractOptions),

    /// Prints the current configurations.
    Info,
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
#[clap(about = "Extract micro-surface's intrinsic properties.")]
pub struct ExtractOptions {
    #[clap(value_enum)]
    property: MicroSurfaceProperty,

    #[clap(
        help = "The input micro-surface profile description files. Can be a list of files or a \
                directory. Use `user://` or `local://` to indicate the user-defined data file \
                path or system-defined data file path respectively."
    )]
    inputs: Vec<PathBuf>,

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
pub fn init(args: &VgonioArgs, launch_time: std::time::SystemTime) -> Result<VgonioConfig, Error> {
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

    VgonioConfig::load_config(args.config.as_ref())
}
