use args::CliArgs;
use std::io::IsTerminal;
use vgn_core::{
    cli::{self as core_cli, SilentSink},
    error::VgonioError,
    utils::medium,
};

pub(crate) mod args;

pub mod cache;
pub mod cli;
pub(crate) mod gui;

/// Main entry point for the VGonio app.
pub fn run() -> Result<(), VgonioError> {
    let (args, launch_time) = core_cli::parse_args::<CliArgs>("vgonio-comp");

    let base_status_verbosity = if args.verbose || args.log_level >= 3 {
        1
    } else {
        0
    };
    let color_mode = match args.status_color {
        args::StatusColorMode::Auto => core_cli::ColorMode::Auto,
        args::StatusColorMode::Always => core_cli::ColorMode::Always,
        args::StatusColorMode::Never => core_cli::ColorMode::Never,
    };
    core_cli::setup_printer(core_cli::PrinterConfig {
        quiet: args.quiet,
        verbosity: args.status_verbosity.max(base_status_verbosity),
        color_mode,
    });
    let serialize_env = matches!(
        std::env::var("VGN_CLI_SERIALIZE")
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    );
    let serialize_auto = !std::io::stdout().is_terminal();
    match args.status_sink {
        args::StatusSinkMode::Silent => core_cli::set_status_sink(SilentSink),
        args::StatusSinkMode::Cli | args::StatusSinkMode::Auto => {
            if args.serialize_output || serialize_env || serialize_auto {
                core_cli::use_cli_sink(true);
            }
        },
    }

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

    let config = vgn_core::config::Config::load_config(args.config.as_deref())?;

    // Bootstrap the medium registry before any IOR / measurement code runs.
    // Baseline media ship embedded (`include_str!`-ed builtin.toml); only the
    // system and user `media.toml` overrides are read from disk. No repo-root
    // discovery.
    if medium::registry().is_none() {
        log::info!("Bootstrapping medium registry...");
        let sys_media = Some(config.sys_data_dir().join("media.toml"));
        let user_media = config.user_data_dir().map(|dir| dir.join("media.toml"));
        medium::bootstrap(sys_media.as_deref(), user_media.as_deref()).map_err(|e| {
            VgonioError::new("Failed to bootstrap medium registry", Some(Box::new(e)))
        })?;
    }

    match args.command {
        None => gui::run(config),
        Some(cmd) => cli::run(cmd, config),
    }
}
