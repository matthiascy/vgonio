//! CLI shape for the `measure` subcommand: clap parsing + thread-pool wrapper.
//! Orchestration lives in [`crate::measure_orchestration`].

use crate::app::args::OutputFormat;
use std::path::PathBuf;
use vgn_core::{
    cli::{cli_step, Indent},
    config::Config,
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
};

/// Measure different metrics of the micro-surface.
pub fn measure(opts: MeasureOptions, config: Config) -> Result<(), VgonioError> {
    log::info!("{:#?}", config);

    // [0.5] TODO: later when extracting capabilites (ADAPTER print, moved here
    // from measure_orchestration::run), this will stay in the command file, because
    // this is the top-level `vgonio measure` banner; this is a CLI-shell
    // concern, not orchestration.
    let dispatch = |opts: MeasureOptions, config: Config| {
        cli_step!(
            Indent::ROOT,
            "Executing 'vgonio measure' with a thread pool of size: {}",
            rayon::current_num_threads()
        );
        crate::orchestration::measure::run(opts, config)
    };

    if let Some(nthreads) = opts.nthreads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads as usize)
            .build()
            .map_err(|err| {
                VgonioError::new(
                    &format!(
                        "Failed to create measurement thread pool with {} threads: {}",
                        nthreads, err
                    ),
                    None,
                )
            })?;
        pool.install(|| dispatch(opts, config))
    } else {
        dispatch(opts, config)
    }
}

/// Options for the `measure` command.
#[derive(clap::Args, Debug)]
#[clap(about = "Measure different aspects of the micro-surface.")]
pub struct MeasureOptions {
    #[arg(
        short,
        long,
        required = true,
        num_args(1..),
        help = "The measurement description files or directories."
    )]
    pub inputs: Vec<PathBuf>,

    #[arg(
        short,
        long,
        help = "The path where stores the simulation data. Use // at the start of the\npath to \
                set the output path relative to the input file location.\nOutput path can also be \
                specified in configuration file."
    )]
    pub output: Option<PathBuf>,

    #[arg(
        short = 'f',
        long,
        default_value_t = OutputFormat::Vgmo,
        help = "The format of the measurement output. If not specified, the format\nwill be the \
                vgonio internal file format.",
    )]
    pub output_format: OutputFormat,

    #[arg(
        short,
        long,
        default_value_t = 512,
        help = "The resolution of the measurement output in case the output is image.\nIf not \
                specified, the resolution will be 512."
    )]
    pub resolution: u32,

    #[arg(
        short,
        long,
        required_if_eq("output_format", "vgms"),
        default_value_t = FileEncoding::Binary,
        help = "Data format for the measurement output.\nOnly used when output format is vgms."
    )]
    pub encoding: FileEncoding,

    #[arg(
    short,
    long,
    required_if_eq("output_format", "vgms"),
    default_value_t = CompressionScheme::None,
    help = "Data compression for the measurement output."
    )]
    pub compression: CompressionScheme,

    #[arg(
        short,
        long = "num-threads",
        help = "The number of threads in the thread pool"
    )]
    pub nthreads: Option<u32>,

    #[clap(
        long,
        help = "Show detailed statistics about memory and time\nusage during the measurement"
    )]
    pub print_stats: bool,
}
