use crate::{
    acq::{
        desc::{MeasurementDesc, MeasurementKind},
        RayTracingMethod,
    },
    app::cache::{VgonioCache, VgonioDatafiles},
    error::Error,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::{io::Write, path::PathBuf};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder},
    window::WindowBuilder,
};

pub mod cache;
pub mod cli;
pub(crate) mod gui;
pub mod state;

const WIN_INITIAL_WIDTH: u32 = 1600;
const WIN_INITIAL_HEIGHT: u32 = 900;

/// Vgonio configuration.
#[derive(Debug)]
pub struct Config {
    /// Path to the configuration directory.
    pub config_dir: std::path::PathBuf,

    /// Path to the cache directory.
    pub cache_dir: std::path::PathBuf,

    /// Path to the data files.
    pub data_files_dir: std::path::PathBuf,

    /// Path to the user-defined configuration.
    pub user_config: VgonioUserConfig,
}

/// User-defined configuration.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VgonioUserConfig {
    /// Path to the output directory.
    pub output_dir: std::path::PathBuf,

    /// Path to the user-defined data files.
    pub data_files_dir: std::path::PathBuf,
}

impl Config {
    /// Load the configuration from the config directory.
    pub fn load_config() -> Result<Self, Error> {
        // MacOS: ~/Library/Application Support/vgonio
        // Windows: %APPDATA%\vgonio
        // Unix-like: ~/.config/vgonio
        let config_dir = {
            let mut config_dir = dirs::config_dir().ok_or(Error::ConfigDirNotFound)?;
            config_dir.push("vgonio");
            config_dir
        };

        let cache_dir = config_dir.join("cache");
        let data_files_dir = config_dir.join("datafiles");

        // Create the config directory if it doesn't exist.
        if !config_dir.exists() {
            std::fs::create_dir_all(&config_dir)?;
        }

        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)?;
        }

        if !data_files_dir.exists() {
            std::fs::create_dir_all(&data_files_dir)?;
        }

        let current_dir = std::env::current_dir()?;

        // Try to load user config file.
        let (user_config_file, relative_to) = {
            let current_dir_user_config = current_dir.join("vgonio.toml");
            let default_user_config = config_dir.join("vgonio.toml");
            if current_dir_user_config.exists() {
                // Try to load the user config file under the current directory.
                (
                    std::fs::read_to_string(&current_dir_user_config).ok(),
                    &current_dir,
                )
            } else if default_user_config.exists() {
                // Otherwise, try to load the user config file under the config directory.
                (
                    std::fs::read_to_string(&default_user_config).ok(),
                    &config_dir,
                )
            } else {
                (None, &current_dir)
            }
        };

        // If the user config file exists, parse it.
        let user_config = if let Some(user_config_file) = user_config_file {
            let mut config: VgonioUserConfig = toml::from_str(&user_config_file)?;
            // Convert relative paths to absolute paths.
            if config.data_files_dir.is_relative() {
                let data_files_dir = relative_to.join(&config.data_files_dir);
                config.data_files_dir = data_files_dir.canonicalize().map_err(|err| {
                    std::io::Error::new(
                        err.kind(),
                        format!(
                            "failed to convert relative path '{}' to absolute path: {}",
                            data_files_dir.display(),
                            err
                        ),
                    )
                })?;
            }
            if config.output_dir.is_relative() {
                let output_dir = relative_to.join(&config.output_dir);
                config.output_dir =
                    output_dir
                        .join(config.output_dir)
                        .canonicalize()
                        .map_err(|err| {
                            std::io::Error::new(
                                err.kind(),
                                format!(
                                    "failed to convert relative path '{}' to absolute path: {}",
                                    output_dir.display(),
                                    err
                                ),
                            )
                        })?;
            }
            config
        } else {
            // If the user config file doesn't exist, create it.
            let user_config = VgonioUserConfig {
                output_dir: config_dir.join("output"),
                data_files_dir: config_dir.join("datafiles"),
            };
            let user_config_file = config_dir.join("vgonio.toml");
            std::fs::write(&user_config_file, toml::to_string(&user_config)?)?;

            user_config
        };

        Ok(Config {
            config_dir,
            cache_dir,
            data_files_dir,
            user_config,
        })
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
        default_value_t = 2
    )]
    pub log_level: u8,

    /// Command to execute.
    #[clap(subcommand)]
    pub command: Option<VgonioCommand>,
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

    // Only enable info logging for wgpu only if the log level is greater than 2.
    let wgpu_log_level = if log_level > 2 {
        log::LevelFilter::Info
    } else {
        log::LevelFilter::Error
    };

    // Initialize logger settings.
    let timestamp = args.log_timestamp;
    env_logger::builder()
        .format(move |buf, record| {
            if timestamp {
                let duration = launch_time.elapsed().unwrap();
                let millis = duration.as_millis() % 1000;
                let seconds = duration.as_secs() % 60;
                let minutes = (duration.as_secs() / 60) % 60;
                let hours = (duration.as_secs() / 60) / 60;
                // Show log level only in Warn and Error level
                if record.level() <= log::Level::Warn {
                    writeln!(
                        buf,
                        "{}:{}:{}.{:03} {}: {}",
                        hours,
                        minutes,
                        seconds,
                        millis,
                        record.level(),
                        record.args()
                    )
                } else {
                    writeln!(
                        buf,
                        "{}:{}:{}.{:03}: {}",
                        hours,
                        minutes,
                        seconds,
                        millis,
                        record.args()
                    )
                }
            } else if record.level() <= log::Level::Warn {
                writeln!(buf, "{}: {}", record.level(), record.args())
            } else {
                writeln!(buf, "{}", record.args())
            }
        })
        .filter(Some("wgpu"), wgpu_log_level)
        .filter_level(match log_level {
            0 => log::LevelFilter::Error,
            1 => log::LevelFilter::Warn,
            2 => log::LevelFilter::Info,
            3 => log::LevelFilter::Debug,
            4 => log::LevelFilter::Trace,
            _ => log::LevelFilter::Info,
        })
        .init();

    // Load the configuration file.
    Config::load_config()
}

/// Runs the GUI application.
pub fn launch_gui(config: Config) -> Result<(), Error> {
    use crate::app::gui::VgonioEvent;
    use state::VgonioState;

    let event_loop = EventLoopBuilder::<VgonioEvent>::with_user_event().build();

    let window = WindowBuilder::new()
        .with_decorations(true)
        .with_resizable(true)
        .with_transparent(false)
        .with_inner_size(PhysicalSize {
            width: WIN_INITIAL_WIDTH,
            height: WIN_INITIAL_HEIGHT,
        })
        .with_title("vgonio")
        .build(&event_loop)
        .unwrap();

    let mut vgonio = pollster::block_on(VgonioState::new(config, &window, &event_loop))?;

    let mut last_frame_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let now = std::time::Instant::now();
        let dt = now - last_frame_time;
        last_frame_time = now;

        match event {
            Event::UserEvent(VgonioEvent::Quit) => {
                *control_flow = ControlFlow::Exit;
            }
            Event::UserEvent(event) => vgonio.handle_user_event(event),
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == window.id() => {
                if !vgonio.handle_input(event) {
                    match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

                        WindowEvent::Resized(new_size) => vgonio.resize(*new_size),

                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            vgonio.resize(**new_inner_size);
                        }

                        _ => {}
                    }
                }
            }

            Event::RedrawRequested(window_id) if window_id == window.id() => {
                vgonio.update(dt);
                match vgonio.render(&window) {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => vgonio.resize(PhysicalSize {
                        width: vgonio.surface_width(),
                        height: vgonio.surface_height(),
                    }),
                    // The system is out of memory, we should quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }

            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually request it.
                window.request_redraw()
            }

            _ => {}
        }
    })
}

/// Execute a vgonio subcommand.
pub fn execute_command(cmd: VgonioCommand, config: Config) -> Result<(), Error> {
    match cmd {
        VgonioCommand::Measure(opts) => cli::measure(opts, config),
        VgonioCommand::Extract(opts) => cli::extract(opts, config),
    }
}
