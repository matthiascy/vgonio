#[cfg(feature = "fitting")]
use crate::app::cli::FitOptions;
#[cfg(feature = "surf-gen")]
use crate::app::cli::GenerateOptions;
use crate::app::cli::{ConvertOptions, DiffOptions, MeasureOptions, PlotOptions};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    path::PathBuf,
};

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
    #[cfg(feature = "surf-gen")]
    /// Generates a new micro-geometry level surface.
    Generate(GenerateOptions),

    /// Measures micro-geometry level light transport related metrics.
    Measure(MeasureOptions),

    /// Prints related information about the current vgonio instance.
    #[clap(name = "info")]
    PrintInfo(PrintInfoOptions),

    /// Converts non-vgonio files to vgonio files.
    Convert(ConvertOptions),

    #[cfg(feature = "fitting")]
    /// Fits a measured data to a model.
    Fit(FitOptions),

    /// Computes the difference between measured data.
    Diff(DiffOptions),

    Plot(PlotOptions),
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
    #[clap(name = "adf")]
    /// Microfacet distribution function.
    AreaDistributionFunction,
    #[clap(name = "msf")]
    /// Micro-facet masking-shadowing function.
    MaskingShadowingFunction,
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// Vgonio internal file format.
    #[default]
    Vgmo,
    // /// Portable float map.
    // Pfm,
    // /// Portable network graphics.
    // Png,
    /// OpenEXR image format with resolution.
    Exr,
    /// Vgonio interal file format together with a EXR file.
    VgmoExr,
}

impl Display for OutputFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vgmo => write!(f, "vgmo"),
            Self::Exr => write!(f, "exr"),
            Self::VgmoExr => write!(f, "vgmo+exr"),
        }
    }
}

impl OutputFormat {
    pub fn is_vgmo(&self) -> bool { matches!(self, Self::Vgmo | Self::VgmoExr) }

    pub fn is_exr(&self) -> bool { matches!(self, Self::Exr | Self::VgmoExr) }
}
