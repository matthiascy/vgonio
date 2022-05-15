use crate::acq::bxdf::BxdfKind;
use crate::acq::desc::{MeasurementDesc, MeasurementKind};
use crate::acq::ior::RefractiveIndexDatabase;
use crate::acq::resolve_file_path;
use crate::app::gui::RepaintSignal;
use crate::error::Error;
use crate::htfld::{AxisAlignment, Heightfield};
use clap::{AppSettings, ArgEnum, Args, Parser, Subcommand};
use std::io::Write;
use std::path::PathBuf;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

pub(crate) mod gui;
pub mod state;

const WIN_INITIAL_WIDTH: u32 = 1280;
const WIN_INITIAL_HEIGHT: u32 = 720;

const BRIGHT_CYAN: &'static str = "\u{001b}[36m";
const BRIGHT_YELLOW: &'static str = "\u{001b}[33m";
const RESET: &'static str = "\u{001b}[0m";

/// Vgonio configuration.
#[derive(Debug)]
pub struct VgonioConfig {
    pub config_dir: std::path::PathBuf,
    pub cache_dir: std::path::PathBuf,
    pub data_files_dir: std::path::PathBuf,
    pub user_config: VgonioUserConfig,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct VgonioUserConfig {
    pub output_dir: std::path::PathBuf,
    pub data_files_dir: std::path::PathBuf,
}

impl VgonioConfig {
    /// Load the configuration from the config directory.
    fn load_config() -> Result<Self, Error> {
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

        Ok(VgonioConfig {
            config_dir,
            cache_dir,
            data_files_dir,
            user_config,
        })
    }
}

#[derive(Parser, Debug)]
#[clap(
    author = "Yang Chen <matthiasychen@gmail.com/y.chen-14@tudelft.nl>",
    version = "0.1.0",
    about = "Micro-geometry level light transportation simulation.",
    setting = AppSettings::DeriveDisplayOrder,
)]
pub struct VgonioArgs {
    #[clap(short, long, help = "Silent output printed to stdout")]
    pub quite: bool,

    #[clap(short, long, help = "Use verbose output (log level = 4)")]
    pub verbose: bool,

    #[clap(long, help = "Set a file to output the log to")]
    pub log_file: Option<std::path::PathBuf>,

    #[clap(long, help = "Set a file descriptor as log output [2 = stderr]")]
    pub log_fd: Option<u32>,

    #[clap(
        long,
        help = "Show timestamp for each log message in seconds since\
                \nprogram starts"
    )]
    pub log_timestamp: bool,

    #[clap(
        long,
        help = "Setting logging verbosity level (higher for more\n\
                details)\
                \n  0 - error\
                \n  1 - warn + error\
                \n  2 - info + warn + error\
                \n  3 - debug + info + warn + error\
                \n  4 - trace + debug + info + warn + error\n\x08",
        default_value_t = 2
    )]
    pub log_level: u8,

    #[clap(subcommand)]
    pub command: Option<VgonioCommand>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, ArgEnum)]
pub enum MicroSurfaceInfo {
    VertexNormal,
    SurfaceNormal,
}

#[derive(Subcommand, Debug)]
pub enum VgonioCommand {
    Measure(MeasureOptions),
    Extract(ExtractOptions),
}

#[derive(Args, Debug)]
#[clap(
about = "Measure different aspects of the micro-surface.",
setting = AppSettings::DeriveDisplayOrder
)]
pub struct MeasureOptions {
    #[clap(short, long, help = "The input measurement description file.")]
    input_path: std::path::PathBuf,

    #[clap(
        short,
        long,
        help = "The path where stores the simulation data. Use //\n\
                    at the start of the path to set the output path\n\
                    relative to the input file location. If not \n\
                    specified, current working directory will be used"
    )]
    output_path: Option<std::path::PathBuf>,

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
        help = "Show detailed statistics about memory and time\n\
                             usage during the measurement"
    )]
    print_stats: bool,
}

#[derive(Args, Debug)]
#[clap(
about = "Extract information from micro-surface.",
setting = AppSettings::DeriveDisplayOrder
)]
pub struct ExtractOptions {
    #[clap(arg_enum, short, long, help = "Type of information to be extracted.")]
    kind: MicroSurfaceInfo,

    #[clap(
        short,
        long,
        help = "The input micro-surface profile, it can be either\n\
                    micro-surface height field or micro-surface mesh\n\
                    cache"
    )]
    input_path: std::path::PathBuf,

    #[clap(
        short,
        long,
        help = "The path where stores the simulation data. Use //\n\
                    at the start of the path to set the output path\n\
                    relative to the input file location. If not \n\
                    specified, current working directory will be used"
    )]
    output_path: Option<std::path::PathBuf>,

    #[clap(long, help = "Use caches to minimize the processing time")]
    enable_cache: bool,
}

pub fn init(args: &VgonioArgs, launch_time: std::time::SystemTime) -> Result<VgonioConfig, Error> {
    let log_level = if args.verbose { 4 } else { args.log_level };

    // Only enable info logging for wgpu only if the log level is greater than 2.
    let wgpu_log_level = if log_level > 2 {
        log::LevelFilter::Info
    } else {
        log::LevelFilter::Error
    };

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Initialize logger settings.
        let timestamp = args.log_timestamp;
        env_logger::builder()
            .format(move |buf, record| {
                if timestamp {
                    let duration = launch_time.elapsed().unwrap();
                    let milis = duration.as_millis() % 1000;
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
                            milis,
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
                            milis,
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
        VgonioConfig::load_config()
    }

    #[cfg(target_arch = "wasm32")]
    {
        todo!()
    }
}

pub fn launch_gui_client(config: VgonioConfig) -> Result<(), Error> {
    use crate::app::gui::UserEvent;
    use state::VgonioApp;

    let event_loop = EventLoop::<UserEvent>::with_user_event();

    let window = WindowBuilder::new()
        .with_decorations(true)
        .with_resizable(true)
        .with_transparent(false)
        .with_inner_size(winit::dpi::PhysicalSize {
            width: WIN_INITIAL_WIDTH,
            height: WIN_INITIAL_HEIGHT,
        })
        .with_title("vgonio")
        .build(&event_loop)
        .unwrap();

    let mut vgonio = pollster::block_on(VgonioApp::new(&window, event_loop.create_proxy()))?;
    let repaint_signal = std::sync::Arc::new(RepaintSignal(std::sync::Mutex::new(
        event_loop.create_proxy(),
    )));

    let mut last_frame_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let now = std::time::Instant::now();
        let dt = now - last_frame_time;
        last_frame_time = now;

        match event {
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
                match vgonio.render(&window, repaint_signal.clone()) {
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
    });
}

pub fn execute_command(cmd: VgonioCommand, config: VgonioConfig) -> Result<(), Error> {
    match cmd {
        VgonioCommand::Measure(opts) => measure(opts, config),
        VgonioCommand::Extract(opts) => extract(opts, config),
    }
}

fn measure(opts: MeasureOptions, config: VgonioConfig) -> Result<(), Error> {
    // Configure thread pool for parallelism.
    if let Some(nthreads) = opts.nthreads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads as usize)
            .build_global()
            .unwrap();
    }
    println!(
        "{BRIGHT_YELLOW}>{RESET} Executing 'vgonio measure' with a thread pool of size: {}",
        rayon::current_num_threads()
    );

    // Load data files: refractive indices, spd etc.
    println!("  {BRIGHT_YELLOW}>{RESET} Loading data files (refractive indices, spd etc.)...");

    let ior_db = RefractiveIndexDatabase::load_from_config_dirs(&config);

    println!("  {BRIGHT_YELLOW}>{RESET} Reading measurement description file...");
    let desc = MeasurementDesc::load_from_file(&opts.input_path)?;
    println!("    - Resolving surfaces' file path...");
    let surface_paths: Vec<PathBuf> = desc
        .surfaces
        .iter()
        .map(|s| {
            if let Ok(stripped) = s.strip_prefix("user://") {
                config.user_config.data_files_dir.join(stripped)
            } else if let Ok(stripped) = s.strip_prefix("local://") {
                config.data_files_dir.join(stripped)
            } else {
                resolve_file_path(&opts.input_path, Some(s))
            }
        })
        .collect();
    println!("    - Loading surfaces...");
    let surfaces = load_surfaces_from_files(&surface_paths)?;
    println!("    {BRIGHT_CYAN}✓{RESET} Successfully read scene description file");

    let start = std::time::SystemTime::now();
    let measurement = match desc.measurement_kind {
        MeasurementKind::Bxdf { kind } => match kind {
            BxdfKind::InPlane => {
                println!(
                    "  {BRIGHT_YELLOW}>{RESET} Launch in-plane BRDF measurement at {}
    • parameters:
      + incident medium: {:?}
      + transmitted medium: {:?}
      + surfaces: {:?}
      + emitter:
        - radius: {}
        - num rays: {}
        - max bounces: {}
        - spectrum: {} - {}, step size {}
        - partition:
          - polar angle: {}
          - azimuthal angle: {}
      + collector:
        - radius: {}
        - shape: {:?}
        - partition:
          - type: {}
          - polar angle: {}
          - azimuthal angle: {}",
                    chrono::DateTime::<chrono::Utc>::from(start),
                    desc.incident_medium,
                    desc.transmitted_medium,
                    surface_paths,
                    desc.emitter.radius,
                    desc.emitter.num_rays,
                    desc.emitter.max_bounces,
                    desc.emitter.spectrum.start,
                    desc.emitter.spectrum.stop,
                    desc.emitter.spectrum.step,
                    desc.emitter.partition.theta_range_str(),
                    desc.emitter.partition.phi_range_str(),
                    desc.collector.radius,
                    desc.collector.shape,
                    desc.collector.partition.kind_str(),
                    desc.collector.partition.theta_range_str(),
                    desc.collector.partition.phi_range_str()
                );
                println!("    {BRIGHT_YELLOW}>{RESET} Measuring in-plane BRDF...");
                crate::acq::bxdf::measure_in_plane_brdf(&desc, &ior_db, &surfaces)
            }
        },
        MeasurementKind::Ndf => {
            // todo: measure ndf
            vec![]
        }
    };

    println!(
        "    {BRIGHT_CYAN}✓{RESET} Finished in {:.2} s",
        start.elapsed().unwrap().as_secs_f32()
    );

    println!("  {BRIGHT_YELLOW}>{RESET} Saving results...");
    // todo: save to file
    println!(
        "    {BRIGHT_CYAN}✓{RESET} Successfully saved to \"{}\"",
        config.user_config.output_dir.display()
    );

    Ok(())
}

fn extract(_opts: ExtractOptions, _config: VgonioConfig) -> Result<(), Error> {
    Ok(())
}

fn load_surfaces_from_files(surfaces_paths: &[PathBuf]) -> Result<Vec<Heightfield>, Error> {
    let mut surfaces = Vec::new();
    for path in surfaces_paths {
        let surface = Heightfield::read_from_file(path, None, Some(AxisAlignment::XZ))?;
        surfaces.push(surface);
    }
    Ok(surfaces)
}
