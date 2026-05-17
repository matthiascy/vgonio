//! CLI shape for the `fit` subcommand. Orchestration lives in
//! [`crate::fitting_orchestration`]; this file is clap parsing + dispatch only.

use crate::fitting_orchestration::{self, FittingMethod};
use clap::builder::ValueParser;
use std::path::PathBuf;
use vgn_bxdf::{brdf::measured::MeasuredBrdfKind, distro::MicrofacetDistroKind, BrdfFamily};
use vgn_core::{config::Config, error::VgonioError, BrdfLevel, ErrorMetric, Symmetry, Weighting};

pub fn fit(opts: FitOptions, config: Config) -> Result<(), VgonioError> {
    crate::fitting_orchestration::run(opts, config)
}

// TODO: separate the NDF fitting & the BRDF fitting

/// Options for the `fit` subcommand.
#[derive(clap::Args, Debug, Clone)]
#[clap(about = "Fits a micro-surface related measurement to a given model.")]
pub struct FitOptions {
    #[clap(num_args = 1.., value_delimiter = ' ', help = "Input files to fit the measurement to.")]
    pub inputs: Vec<PathBuf>,

    #[clap(long, short, help = "Kind of the measured BRDF data.")]
    pub kind: MeasuredBrdfKind,

    #[clap(
        long = "ax",
        value_name = "START:END:STEP",
        value_parser = ValueParser::new(fitting_orchestration::parse_roughness_values),
        requires("ay"),
        help = "Anisotropic roughness in x direction (inclusive)."
    )]
    pub ax: Option<[f64; 3]>,

    #[clap(
        long = "ay",
        value_name = "START:END:STEP",
        value_parser = ValueParser::new(fitting_orchestration::parse_roughness_values),
        requires("ax"),
        help = "Anisotropic roughness in y direction (inclusive)."
    )]
    pub ay: Option<[f64; 3]>,

    #[clap(
        long = "per-wl-ax",
        requires("per_wavelength_ay"),
        requires("per_wavelength"),
        conflicts_with_all(&["ax", "a", "per_wavelength_a"]),
        help = "Anisotropic roughness in y direction for each wavelength. The values are \
                specified as a list of triplets, each triplet is a range of roughness values for \
                a wavelength. The number of triplets should be equal to the number of wavelengths."
    )]
    pub per_wavelength_ax: Option<PathBuf>,

    #[clap(
        long = "per-wl-ay",
        requires("per_wavelength_ax"),
        requires("per_wavelength"),
        conflicts_with_all(&["ay", "a", "per_wavelength_a"]),
        help = "Anisotropic roughness in y direction for each wavelength. The values are \
                specified as a list of triplets, each triplet is a range of roughness values for \
                a wavelength. The number of triplets should be equal to the number of wavelengths."
    )]
    pub per_wavelength_ay: Option<PathBuf>,

    #[clap(
        long = "a",
        value_name = "START:END:STEP",
        value_parser = ValueParser::new(fitting_orchestration::parse_roughness_values),
        conflicts_with_all(&["ax", "ay", "per_wavelength_ax", "per_wavelength_ay", "per_wavelength_a"]),
        help = "Isotropic roughness (inclusive)."
    )]
    pub a: Option<[f64; 3]>,

    // TODO: Add support for per-wavelength isotropic roughness
    #[clap(
        long = "per-wl-a",
        conflicts_with_all(&["per_wavelength_ax", "per_wavelength_ay", "ax", "ay", "a"]),
        help = "Isotropic roughness for each wavelength. The values are specified as a list of \
                triplets, each triplet is a range of roughness values for a wavelength. The \
                number of triplets should be equal to the number of wavelengths."
    )]
    pub per_wavelength_a: Option<PathBuf>,

    #[clap(
        long,
        help = "Whether to match the measured BRDF data to physically measured in-plane BRDF data \
                by O. Clausen. If true, the inputs should be in pairs of measured data and O. \
                Clausen's data.",
        default_value = "false"
    )]
    pub clausen: bool,

    #[clap(
        long,
        help = "Whether to use 4 times mores samples while resampling the measured data to match \
                the resolution of the Clausen's data.",
        default_value = "false"
    )]
    pub dense: bool,

    #[clap(
        long,
        short,
        help = "Specifies the output file for saving the fitted data. If omitted, the data is \
                written to standard output. Use \"auto\" to automatically determine the output \
                filename."
    )]
    pub output: Option<String>,

    #[clap(
        long,
        short,
        help = "Model to fit the measurement to. If not specified, the default model will be used."
    )]
    pub family: BrdfFamily,

    #[clap(long, help = "Symmetry of the microfacet model.")]
    pub symmetry: Symmetry,

    #[clap(
        long,
        short,
        help = "Distribution to use for the microfacet model. If not specified, the default \
                distribution will be used.",
        required_if_eq("family", "microfacet")
    )]
    pub distro: Option<MicrofacetDistroKind>,

    #[clap(
        long,
        short,
        help = "Level of the measured BRDF data to fit. Only used for the Vgonio kind.",
        required_if_eq("kind", "vgonio"),
        default_value = "l0"
    )]
    pub level: BrdfLevel,

    #[clap(long, help = "Theta limit for the fitting in degrees. Default to 90°.")]
    pub theta_limit: Option<f32>,

    #[clap(long, short, help = "Method to use for the fitting.")]
    pub method: FittingMethod,

    #[clap(
        long = "bprec",
        help = "Precision of the brute force fitting: number of digits after the decimal point.",
        default_value = "6"
    )]
    pub brute_precision: u32,

    #[clap(
        long = "err",
        help = "Error metric used during calculation of objective function (residuals).",
        default_value = "mse",
        required_if_eq("method", "brute")
    )]
    pub error_metric: Option<ErrorMetric>,

    #[clap(
        short,
        long,
        help = "The weighting to use to weight the measured data.",
        default_value = "none"
    )]
    pub weighting: Weighting,

    #[clap(
        long = "per-wl",
        help = "Whether to fit the measured data per wavelength.",
        default_value = "false"
    )]
    pub per_wavelength: bool,

    #[clap(
        long,
        help = "Whether to plot the fitted data.",
        default_value = "false"
    )]
    pub plot: bool,

    #[cfg(feature = "cuda")]
    #[clap(long, help = "Enable CUDA acceleration for the fitting.")]
    pub cuda: bool,

    // Temporary fix for adding the NDF fitting
    #[clap(long, help = "NDF fitting.")]
    pub ndf: bool,
}

impl FitOptions {
    #[inline]
    pub fn on_gpu(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    #[inline]
    pub fn on_cpu(&self) -> bool { !self.on_gpu() }
}
