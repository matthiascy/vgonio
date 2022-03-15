use crate::app::state::{RepaintSignal, UserEvent};
use crate::error::Error;
use clap::{AppSettings, ArgEnum, Parser, Subcommand};
use std::io::Write;
use winit::dpi::PhysicalSize;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub(crate) mod gfx;
pub(crate) mod grid;
pub mod state;
pub mod texture;
pub(crate) mod ui;

const WIN_INITIAL_WIDTH: u32 = 1280;
const WIN_INITIAL_HEIGHT: u32 = 720;

#[derive(Parser, Debug)]
#[clap(
    author = "Yang Chen <matthiasychen@gmail.com/y.chen-14@tudelft.nl>",
    version = "0.1.0",
    about = "Micro-geometry level light transportation simulation.",
    setting = AppSettings::DeriveDisplayOrder,
)]
pub struct VgonioArgs {
    /// Silent the output.
    #[clap(short, long, help = "Silent output printed to stdout")]
    pub quite: bool,

    #[clap(short, long, help = "Use verbose output")]
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
pub enum MeasurementType {
    Brdf,
    Ndf,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, ArgEnum)]
pub enum MicroSurfaceInfo {
    VertexNormal,
    SurfaceNormal,
}

#[derive(Subcommand, Debug)]
pub enum VgonioCommand {
    #[clap(
        about = "Measure different aspects of the micro-surface.",
        setting = AppSettings::DeriveDisplayOrder
    )]
    Measure {
        #[clap(arg_enum, short, long, help = "Type of measurement.")]
        kind: MeasurementType,

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
    },

    #[clap(
        about = "Extract information from micro-surface.",
        setting = AppSettings::DeriveDisplayOrder
    )]
    Extract {
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
    },
}

pub fn init(args: &VgonioArgs, launch_time: std::time::SystemTime) {
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
            .filter_level(match args.log_level {
                0 => log::LevelFilter::Error,
                1 => log::LevelFilter::Warn,
                2 => log::LevelFilter::Info,
                3 => log::LevelFilter::Debug,
                4 => log::LevelFilter::Trace,
                _ => log::LevelFilter::Info,
            })
            .init();
    }

    #[cfg(target_arch = "wasm32")]
    {
        todo!()
    }
}

pub fn launch_gui_client() -> Result<(), Error> {
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

    let mut state = pollster::block_on(VgonioApp::new(&window))?;
    let repaint_signal = std::sync::Arc::new(RepaintSignal(std::sync::Mutex::new(
        event_loop.create_proxy(),
    )));

    let mut last_frame_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let now = std::time::Instant::now();
        let dt = now - last_frame_time;
        last_frame_time = now;

        match event {
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == window.id() => {
                if !state.process_input(event) {
                    match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

                        WindowEvent::Resized(new_size) => state.resize(*new_size),

                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        }

                        _ => {}
                    }
                }
            }

            Event::RedrawRequested(window_id) if window_id == window.id() => {
                state.update(dt);
                match state.render(&window, repaint_signal.clone()) {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(PhysicalSize {
                        width: state.gpu.surface_config.width,
                        height: state.gpu.surface_config.height,
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

pub fn execute_command(cmd: VgonioCommand) -> Result<(), Error> {
    match cmd {
        VgonioCommand::Measure { .. } => capture(),
        VgonioCommand::Extract { .. } => extract(),
    }
}

fn capture() -> Result<(), Error> {
    Ok(())
}

fn extract() -> Result<(), Error> {
    Ok(())
}
