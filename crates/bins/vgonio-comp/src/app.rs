use args::CliArgs;
use std::io::Write;
use vgonio_core::error::VgonioError;

pub(crate) mod args;

pub mod cache;
pub mod cli;
mod config;
pub(crate) mod gui;

pub use config::*;

/// Main entry point for the VGonio app.
pub fn run() -> Result<(), VgonioError> {
    use clap::Parser;

    let args = CliArgs::parse();
    let launch_time = std::time::SystemTime::now();
    log::info!(
        "Vgonio launched at {} on {}.",
        chrono::DateTime::<chrono::Utc>::from(launch_time),
        std::env::consts::OS
    );

    setup_logging(&args, launch_time);

    let config = Config::load_config(args.config.as_deref())?;

    match args.command {
        None => gui::run(config),
        Some(cmd) => cli::run(cmd, config),
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
pub fn setup_logging(args: &CliArgs, launch_time: std::time::SystemTime) {
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
        .filter(Some("calloop"), log_filter_from_level(3))
        .filter_level(log_filter_from_level(log_level))
        .init();
}
