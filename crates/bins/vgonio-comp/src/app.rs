use args::CliArgs;
use vgonio_core::{cli as core_cli, error::VgonioError};

pub(crate) mod args;

pub mod cache;
pub mod cli;
pub(crate) mod gui;

/// Main entry point for the VGonio app.
pub fn run() -> Result<(), VgonioError> {
    let (args, launch_time) = core_cli::parse_args::<CliArgs>("vgonio-comp");

    let timestamp = if args.log_timestamp {
        Some(launch_time)
    } else {
        None
    };

    let log_level_wgpu = if args.debug_wgpu {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Error
    };
    let log_level_winit = if args.debug_winit {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Error
    };

    let filters = [
        ("wgpu", log_level_wgpu),
        ("naga", log_level_wgpu),
        ("winit", log_level_winit),
        ("calloop", log::LevelFilter::Debug),
    ];

    core_cli::setup_logging(timestamp, args.log_level, &filters);

    let config = vgonio_core::config::Config::load_config(args.config.as_deref())?;

    match args.command {
        None => gui::run(config),
        Some(cmd) => cli::run(cmd, config),
    }
}
