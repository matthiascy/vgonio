use crate::common::DataEncoding;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Vgonio command line interface arguments.
#[derive(clap::Parser, Debug)]
#[clap(
    author,
    version,
    about = "Micro-geometry level light transportation simulation."
)]
pub struct CliArgs {
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

    #[clap(long, help = "Enable debug outputs files.")]
    pub debug_output: bool,

    /// Command to execute.
    #[clap(subcommand)]
    pub command: Option<SubCommand>,

    /// Path to the user config file. If not specified, vgonio will
    /// load the default config file.
    #[clap(short, long, help = "Path to the user config file")]
    pub config: Option<PathBuf>,
}

/// Vgonio command.
#[derive(clap::Subcommand, Debug)]
pub enum SubCommand {
    /// Generates a new micro-geometry level surface.
    Generate(GenerateOptions),

    /// Measures micro-geometry level light transport related metrics.
    Measure(MeasureOptions),

    /// Prints related information about the current vgonio instance.
    #[clap(name = "info")]
    PrintInfo(PrintInfoOptions),
}

#[derive(clap::Args, Debug)]
#[clap(about = "Print information about vgonio.")]
pub struct PrintInfoOptions {
    #[clap(
        help = "Type of information to print. If not specified, all information\nwill be printed."
    )]
    pub kind: Option<PrintInfoKind>,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintInfoKind {
    /// Print the current configuration.
    Config,
    /// Print the default parameters for the `measure` command.
    Defaults,
    /// Print an example measurement description file.
    #[clap(name = "meas-desc")]
    MeasurementDescription,
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
pub enum FastMeasurementKind {
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

#[derive(clap::Args, Debug)]
#[clap(about = "Generate a micro-geometry level surface using Gaussian distribution.")]
pub struct GenerateOptions {
    #[clap(
        help = "Horizontal resolution of the generated micro-geometry profile (height map).",
        default_value_t = 512
    )]
    pub res_x: u32,

    #[clap(
        help = "Vertical resolution of the generated micro-geometry profile (height map).",
        default_value_t = 512
    )]
    pub res_y: u32,

    #[clap(
        long = "amp",
        help = "Amplitude of the 2D gaussian.",
        default_value_t = 1.0
    )]
    pub amplitude: f32,

    #[clap(
        long = "sx",
        help = "Standard deviation of the 2D gaussian distribution in horizontal direction.",
        default_value_t = 1.0
    )]
    pub sigma_x: f32,

    #[clap(
        long = "sy",
        help = "Standard deviation of the 2D gaussian distribution in vertical direction.",
        default_value_t = 1.0
    )]
    pub sigma_y: f32,

    #[clap(
        long = "mx",
        help = "Mean of 2D gaussian distribution on horizontal axis.",
        default_value_t = 0.0
    )]
    pub mean_x: f32,

    #[clap(
        long = "my",
        help = "Mean of 2D gaussian distribution on vertical axis.",
        default_value_t = 0.0
    )]
    pub mean_y: f32,

    #[clap(
        short,
        long,
        help = "Path to the file where to save the generated micro-geometry profile."
    )]
    pub output: Option<PathBuf>,
}

/// Options for the `measure` command.
#[derive(clap::Args, Debug)]
#[clap(about = "Measure different aspects of the micro-surface.")]
pub struct MeasureOptions {
    #[clap(
        short,
        long,
        num_args(1..),
        help = "The measurement description files. If option '-f, --fast-measurement' is\n\
                specified, inputs will be interpreted as micro-surface profiles instead of\n\
                measurement description file."
    )]
    pub inputs: Vec<PathBuf>,

    #[clap(
        short,
        long,
        help = "The path where stores the simulation data. Use // at the start of the path\nto \
                set the output path relative to the input file location. Output path can\nalso be \
                specified in configuration file."
    )]
    pub output: Option<PathBuf>,

    #[clap(
        short,
        long,
        default_value_t = DataEncoding::Binary,
        help = "Data format for the measurement output."
    )]
    pub encoding: DataEncoding,

    #[clap(
        short,
        long,
        num_args(1..),
        help = "Quickly measure the micro-surface with default parameters.\n\
                Check with 'info' command."
    )]
    pub fast_measurement: Option<Vec<FastMeasurementKind>>,

    #[clap(
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
