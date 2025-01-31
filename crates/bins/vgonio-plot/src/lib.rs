use std::path::PathBuf;

use vgonio_core::{BrdfLevel, ErrorMetric, Symmetry, Weighting};

#[cfg(feature = "surface")]
use vgonio_surf::subdivision::SubdivisionKind;

mod utils;

/// Kind of plots that can be generated.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotKind {
    /// Compare between VgonioBrdf and ClausenBrdf.
    #[clap(alias = "cmp-vc")]
    ComparisonVgonioClausen,
    /// Compare between VgonioBrdf and VgonioBrdf.
    #[clap(alias = "cmp-vv")]
    ComparisonVgonio,
    /// Plot slices of the input BRDF.
    #[clap(name = "brdf-slice")]
    BrdfSlice,
    /// Plot slices of the input BRDF at the same plane.
    #[clap(name = "brdf-inplane")]
    BrdfSliceInPlane,
    /// Plot the BRDF from saved *.exr file into a embedded map.
    #[clap(name = "brdf-map")]
    BrdfMap,
    /// Plot the BRDF from saved *.vgmo file in 3D
    #[clap(name = "brdf-3d")]
    Brdf3D,
    /// Plot everything related to the fitting process.
    #[clap(name = "brdf-fitting")]
    BrdfFitting,
    /// Plot the NDF from saved *.vgmo file
    #[clap(name = "ndf")]
    Ndf,
    /// Plot the GAF from saved *.vgmo file
    #[clap(name = "gaf")]
    Gaf,
    #[cfg(feature = "surface")]
    /// Plot the micro-surfaces from saved *.vgmo file
    #[clap(name = "surf")]
    Surface,
}

/// Arguments for the `vgonio-plot` command.
#[derive(clap::Parser, Debug, Clone)]
pub struct PlotArgs {
    #[clap(num_args = 1.., value_delimiter = ' ', help = "Input files to plot.")]
    pub inputs: Vec<PathBuf>,

    #[clap(short, long, help = "The kind of plot to generate.")]
    pub kind: PlotKind,

    #[clap(short, long, help = "The log level.")]
    pub log_level: u8,

    #[clap(
        short,
        long,
        value_delimiter = ' ',
        num_args = 1..,
        help = "The roughness parameter along the x-axis and y-axis. Can be multiple values.",
        required_if_eq("kind", "brdf-fitting"),
    )]
    pub alpha: Vec<f64>,

    #[clap(
        long = "err",
        help = "The error metric to use for plotting the error map.",
        required_if_eq("kind", "brdf-fitting")
    )]
    pub err_metric: Option<ErrorMetric>,

    #[clap(
        long,
        help = "The weighting function to use for plotting the error map.",
        required_if_eq("kind", "brdf-fitting")
    )]
    pub weighting: Option<Weighting>,

    #[clap(
        long,
        help = "The symmetry of the model which decides how the roughness parameters been \
                interpreted",
        default_value = "iso"
    )]
    pub symmetry: Symmetry,

    #[clap(
        long = "ti",
        help = "The polar angle to plot the BRDF at.",
        default_value = "0.0",
        required_if_eq("kind", "brdf-slice"),
        required_if_eq("kind", "brdf-map"),
        required_if_eq("kind", "brdf-3d")
    )]
    pub theta_i: f32,

    #[clap(
        long = "pi",
        help = "The azimuthal angle to plot the BRDF at. (in degrees)",
        default_value = "0.0",
        required_if_eq("kind", "brdf-inplane"),
        required_if_eq("kind", "brdf-slice"),
        required_if_eq("kind", "brdf-map"),
        required_if_eq("kind", "brdf-3d"),
        required_if_eq("kind", "ndf")
    )]
    pub phi_i: f32,

    #[clap(
        long = "po",
        help = "The azimuthal angle to plot the BRDF at.",
        default_value = "0.0",
        required_if_eq("kind", "brdf-slice")
    )]
    pub phi_o: f32,

    #[clap(long, help = "Whether to show the legend.", default_value = "false")]
    pub legend: bool,

    #[clap(
        long,
        help = "The labels for potential legends.",
        num_args = 0..,
        value_delimiter = ' ',
    )]
    pub labels: Vec<String>,

    #[clap(
        long = "pv",
        help = "The azimuthal angle of the view direction to plot the GAF at. (in degrees)",
        default_value = "0.0",
        required_if_eq("kind", "gaf")
    )]
    pub phi_v: f32,

    #[clap(
        long = "tm",
        help = "The polar angle of the view direction to plot the GAF at. (in degrees)",
        default_value = "0.0",
        required_if_eq("kind", "gaf")
    )]
    pub theta_m: f32,

    #[clap(
        long = "pm",
        help = "The azimuthal angle of microfacet normal direction to plot the GAF at. (in \
                degrees)",
        default_value = "0.0",
        required_if_eq("kind", "gaf")
    )]
    pub phi_m: f32,

    #[clap(
        long = "pstep",
        help = "The step size for the phi angle (in degrees) in the polar coordinate system.",
        default_value = "45"
    )]
    pub pstep: f32,

    #[clap(
        long = "tstep",
        help = "The step size for the theta angle (in degrees) in the polar coordinate system.",
        default_value = "30"
    )]
    pub tstep: f32,

    #[clap(
        long = "lambda",
        help = "The wavelength to plot the BRDF at.",
        default_value = "550.0",
        required_if_eq("kind", "brdf-map"),
        required_if_eq("kind", "brdf-3d")
    )]
    pub lambda: f32,

    #[clap(
        short,
        long,
        help = "Whether to sample 4x more points than the original data.",
        default_value = "false"
    )]
    pub dense: bool,

    #[clap(short, long, help = "The level of BRDF to plot.", default_value = "l0")]
    pub level: BrdfLevel,

    #[clap(long, help = "The colormap to use.")]
    pub cmap: Option<String>,

    #[clap(long, help = "The scale of the BRDF value.")]
    pub scale: Option<f32>,

    #[clap(long, default_value = "false", help = "Whether to show the colorbar.")]
    pub cbar: bool,

    #[clap(
        long,
        default_value = "false",
        help = "Whether to show the ploar coordinate."
    )]
    pub coord: bool,

    #[clap(
        long,
        default_value = "false",
        help = "Whether to show the difference."
    )]
    pub diff: bool,

    #[clap(long, help = "The font color.", default_value = "w")]
    pub fc: Option<String>,

    #[clap(long, help = "The y-axis limits.")]
    pub ylim: Option<f32>,

    #[clap(long, help = "Use log scale for the plot.", default_value = "false")]
    pub log: bool,

    #[clap(long, help = "The file to save the plot.")]
    pub save: Option<String>,

    #[clap(
        long,
        help = "Whether to run the plot in parallel. Works with `kind = brdf-fitting`."
    )]
    pub parallel: bool,

    #[clap(
        short,
        long,
        help = "Whether to run the plot in interactive mode. Only works with `kind = \
                brdf-fitting`.",
        default_value = "false"
    )]
    pub interactive: bool,

    #[cfg(feature = "surface")]
    #[clap(
        long,
        help = "The downsample factor for the surface plot.",
        default_value_t = 1,
        required_if_eq("kind", "surf")
    )]
    pub downsample: u32,

    #[cfg(feature = "surface")]
    #[clap(
        long = "subdiv_kind",
        help = "The kind of subdivision.",
        required_if_eq("kind", "surf")
    )]
    pub subdiv_kind: Option<SubdivisionKind>,

    #[cfg(feature = "surface")]
    #[clap(
        long = "subdiv_level",
        help = "The level of subdivision.",
        required_if_eq("kind", "surf"),
        default_value = "0"
    )]
    pub subdiv_level: Option<u32>,

    #[cfg(feature = "surface")]
    #[clap(
        long = "offset",
        help = "The offset to add randomly to the z coordinate of the new points.",
        required_if_eq("subdiv_kind", "wiggly"),
        default_value = "100"
    )]
    pub subdiv_offset: Option<u32>,
}
