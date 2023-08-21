use serde::{Deserialize, Serialize};
use std::{fmt::Display, path::PathBuf, str::FromStr};
use vgcore::io::{CompressionScheme, FileEncoding};
use vgsurf::{RandomGenMethod, SurfGenKind};

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

    /// Converts non-vgonio files to vgonio files.
    Convert(ConvertOptions),
}

/// Arguments for the `convert --resize` command.
#[derive(Copy, Clone, Debug)]
pub struct NewSize(pub u32, pub u32);

impl FromStr for NewSize {
    type Err = std::io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        println!("s: {}", s);
        let mut parts = s.trim().split_ascii_whitespace();
        let width: u32 = parts
            .next()
            .ok_or(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No input provided for new width",
            ))?
            .parse()
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Failed to parse new width")
            })?;
        let height: u32 = parts
            .next()
            .ok_or(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No input provided for new height",
            ))?
            .parse()
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Failed to parse new height",
                )
            })?;
        Ok(Self(width, height))
    }
}

#[derive(clap::Args, Debug)]
#[clap(
    about = "Converts non-vgonio surface files to vgonio files or resizes vgonio surface files."
)]
pub struct ConvertOptions {
    /// Path to input files.
    #[clap(
        short,
        long,
        num_args(1..),
        required = true,
        help = "Files to be converted."
    )]
    pub inputs: Vec<PathBuf>,

    /// Path to output file.
    #[clap(short, long, help = "Path to store converted files.")]
    pub output: Option<PathBuf>,

    #[clap(
        short,
        long,
        default_value_t = FileEncoding::Binary,
        help = "Data encoding for the output."
    )]
    pub encoding: FileEncoding,

    #[clap(
        short,
        long,
        default_value_t = CompressionScheme::None,
        help = "Data compression for the output."
    )]
    pub compression: CompressionScheme,

    #[clap(
        short,
        long,
        required = true,
        help = "Type of conversion to perform. If not specified, the\nconversion will be inferred \
                from the file extension."
    )]
    pub kind: ConvertKind,

    #[clap(
        long,
        value_name = "WIDTH HEIGHT",
        num_args(2),
        help = "Resize the micro-surface profile to the given resolution. The \n resolution \
                should besamller than the original."
    )]
    pub resize: Option<Vec<u32>>,

    #[clap(
        long,
        help = "Resize the micro-surface profile to make it square. The\nresolution will be the \
                minimum of the width and height."
    )]
    pub squaring: bool,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertKind {
    #[clap(
        name = "ms",
        help = "Convert a micro-surface profile to .vgms file. Accepts files coming \
                from\n\"Predicting Appearance from Measured Microgeometry of Metal \
                Surfaces\",\nand plain text data coming from µsurf confocal microscope system."
    )]
    MicroSurfaceProfile,
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
    #[clap(name = "madf")]
    /// Micro-facet normal distribution function.
    MicrofacetAreaDistribution,
    #[clap(name = "mmsf")]
    /// Micro-facet masking-shadowing function.
    MicrofacetMaskingShadowing,
}

#[derive(clap::Args, Debug)]
#[clap(about = "Generate a micro-geometry level surface using Gaussian distribution.")]
pub struct GenerateOptions {
    /// Horizontal and vertical resolution of the generated micro-geometry
    /// profile.
    #[arg(long = "res", num_args(2), default_values_t = [1024; 2])]
    pub res: Vec<u32>,

    /// Horizontal and vertical spacing between micro-geometry height samples,
    /// in micrometers.
    #[arg(long = "spacing", num_args(2), default_values_t = [0.1; 2])]
    pub spacing: Vec<f32>,

    /// Highest height of the generated micro-geometry profile, in micrometers.
    /// The lowest height is always 0.
    #[arg(long = "height", default_value_t = 1.0)]
    pub max_height: f32,

    /// Type of surface to generate.
    #[arg(ignore_case = true, long = "kind")]
    pub kind: SurfGenKind,

    /// Amplitude of the 2D gaussian. Only used when `kind` is `gaussian2d`.
    #[arg(long = "amp", required_if_eq("kind", "gaussian2d"))]
    pub amplitude: Option<f32>,

    /// Standard deviation of the 2D gaussian distribution in horizontal
    /// direction. Only used when `kind` is `gaussian2d`.
    #[arg(long = "sx", required_if_eq("kind", "gaussian2d"))]
    pub sigma_x: Option<f32>,

    /// Standard deviation of the 2D gaussian distribution in vertical
    /// direction. Only used when `kind` is `gaussian2d`.
    #[arg(long = "sy", required_if_eq("kind", "gaussian2d"))]
    pub sigma_y: Option<f32>,

    /// Mean of 2D gaussian distribution on horizontal axis. Only used when
    /// `kind` is `gaussian2d`.
    #[arg(long = "mx", required_if_eq("kind", "gaussian2d"))]
    pub mean_x: Option<f32>,

    /// Mean of 2D gaussian distribution on vertical axis. Only used when
    /// `kind` is `gaussian2d`.
    #[arg(long = "my", required_if_eq("kind", "gaussian2d"))]
    pub mean_y: Option<f32>,

    /// Random method to use. Only used when `kind` is `random`.
    #[arg(ignore_case = true, long = "method", required_if_eq("kind", "random"))]
    pub method: Option<RandomGenMethod>,

    /// Number of initial seed points for Worley noise. Should be a power of 2.
    #[arg(long = "num-seeds", required_if_eq_all([("kind", "random"), ("method", "worley-noise")]))]
    pub num_seeds: Option<u32>,

    /// Data format for the surface generation output.
    #[arg(short, long, default_value_t = FileEncoding::Binary)]
    pub encoding: FileEncoding,

    /// Data compression for the surface generation output.
    #[arg(short, long, default_value_t = CompressionScheme::None)]
    pub compression: CompressionScheme,

    /// Path to the file where to save the generated micro-geometry profile.
    #[arg(short, long)]
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
        default_value_t = FileEncoding::Binary,
        help = "Data format for the measurement output."
    )]
    pub encoding: FileEncoding,

    #[clap(
        short,
        long,
        default_value_t = CompressionScheme::None,
        help = "Data compression for the measurement output."
    )]
    pub compression: CompressionScheme,

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
