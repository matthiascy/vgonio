use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

/// Vgonio command.
#[derive(Subcommand, Debug)]
pub enum VgonioCommand {
    /// Measures micro-geometry level light transport related metrics.
    Measure(MeasureOptions),

    /// Prints the current configurations.
    Info,
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize, ValueEnum)]
pub enum MeasurementKind {
    #[clap(name = "bsdf")]
    /// Bidirectional reflectance distribution function.
    Bsdf,
    #[clap(name = "mfd")]
    /// Micro-facet distribution function.
    MicrofacetDistribution,
    #[clap(name = "mfs")]
    /// Micro-facet shadowing-masking function.
    MicrofacetShadowingMasking,
}

/// Options for the `measure` command.
#[derive(Args, Debug)]
#[clap(about = "Measure different aspects of the micro-surface.")]
pub struct MeasureOptions {
    #[clap(
        short,
        long,
        help = "The measurement description file. If option '--f, --fast'\nis specified, inputs \
                will be interpreted as micro-surface\nprofiles instead of measurement description \
                file."
    )]
    pub inputs: Vec<PathBuf>,

    #[clap(
        short,
        long,
        help = "The path where stores the simulation data. Use //\nat the start of the path to \
                set the output path\nrelative to the input file location. Output path can \
                also\nbe specified in configuration file."
    )]
    pub output: Option<PathBuf>,

    #[clap(
        short,
        long,
        help = "Quickly measure the micro-surface with default parameters.\nCheck default \
                parameters using 'info' command."
    )]
    pub fast: Option<MeasurementKind>,

    #[clap(
        short,
        long = "num-threads",
        help = "The number of threads in the thread pool"
    )]
    pub nthreads: Option<u32>,

    #[clap(long, help = "Use caches to minimize the processing time")]
    pub enable_cache: bool,

    #[clap(
        long,
        help = "Show detailed statistics about memory and time\nusage during the measurement"
    )]
    pub print_stats: bool,
}
