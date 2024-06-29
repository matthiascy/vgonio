use crate::{
    app::{cache::Cache, Config},
    measure::{
        bsdf::{MeasuredBrdfLevel, MeasuredBsdfData},
        mfd::{MeasuredMsfData, MeasuredNdfData},
    },
    pyplot::{
        plot_brdf_map, plot_brdf_slice, plot_brdf_slice_in_plane, plot_brdf_vgonio_clausen,
        plot_gaf, plot_ndf, plot_surfaces,
    },
};
use base::{
    error::VgonioError,
    math::Sph2,
    units::{Degs, Nanometres, Radians, Rads},
};
use bxdf::brdf::measured::ClausenBrdf;
use std::path::PathBuf;
use surf::TriangulationPattern;

#[rustfmt::skip]
// Same BRDF but different incident angles
// vgonio plot -i file -k slice-in-plane --ti 0.0 --pi 0.0 --po 0.0

// Mutiple BRDF
// vgonio plot -i f0 f1 f2 -k slice --ti 0.0 --pi 0.0 --po 0.0 --labels f0 f1 f2

// Single BRDF
// vgonio plot -i f0 -k slice --ti 0.0 --pi 0.0 --po 0.0

// Plot BRDF map
// vgonio plot -i f.exr -k brdf-map --ti 0 --pi 0 --lambda 400 --cmap BuPu

// python tone_mapping.py --input a.exr  b.exr --channel '400 nm' --cbar --cmap plasma --diff --fc w --coord

/// Kind of plot to generate.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotKind {
    /// Compare between VgonioBrdf and ClausenBrdf.
    #[clap(alias = "cmp-vc")]
    ComparisonVgonioClausen,
    /// Compare between VgonioBrdf and VgonioBrdf.
    #[clap(alias = "cmp-vv")]
    ComparisonVgonio,
    /// Plot slices of the input BRDF.
    #[clap(alias = "slice")]
    Slice,

    #[clap(alias = "slice-in-plane")]
    SliceInPlane,

    /// Plot the NDF from saved *.vgmo file
    #[clap(alias = "ndf")]
    Ndf,
    /// Plot the GAF from saved *.vgmo file
    #[clap(alias = "gaf")]
    Gaf,
    /// Plot the BRDF from saved *.exr file
    #[clap(alias = "brdf-map")]
    BrdfMap,
    /// Plot the micro-surfaces from saved *.vgmo file
    #[clap(alias = "surf")]
    Surface,
}

/// Options for the `plot` command.
#[derive(clap::Args, Debug, Clone)]
pub struct PlotOptions {
    #[clap(short, long, num_args = 1.., value_delimiter = ' ', help = "Input files to plot.")]
    pub inputs: Vec<PathBuf>,

    #[clap(long, help = "Whether to show the legend.", default_value = "false")]
    pub legend: bool,

    #[clap(
        long,
        help = "The labels for potential legends.",
        num_args = 0..,
        value_delimiter = ' ',
    )]
    pub labels: Vec<String>,

    #[clap(short, long, help = "The kind of plot to generate.")]
    pub kind: PlotKind,

    #[clap(
        long = "po",
        help = "The azimuthal angle to plot the BRDF at.",
        default_value = "0.0",
        required_if_eq("kind", "slice")
    )]
    pub phi_o: f32,

    #[clap(
        long = "ti",
        help = "The polar angle to plot the BRDF at.",
        default_value = "0.0",
        required_if_eq("kind", "slice"),
        required_if_eq("kind", "brdf-map")
    )]
    pub theta_i: f32,

    #[clap(
        long = "pi",
        help = "The azimuthal angle to plot the BRDF at. (in degrees)",
        default_value = "0.0",
        required_if_eq("kind", "slice"),
        required_if_eq("kind", "slice-in-plane"),
        required_if_eq("kind", "ndf"),
        required_if_eq("kind", "brdf-map")
    )]
    pub phi_i: f32,

    #[clap(
        long = "pv",
        help = "The azimuthal angle of the view direction to plot the GAF at. (in degrees)",
        default_value = "0.0",
        required_if_eq("kind", "gaf")
    )]
    pub phi_v: f32,

    #[clap(
        long = "pm",
        help = "The azimuthal angle of microfacet normal direction to plot the GAF at. (in \
                degrees)",
        default_value = "0.0",
        required_if_eq("kind", "gaf")
    )]
    pub phi_m: f32,

    #[clap(
        long = "tm",
        help = "The polar angle of the view direction to plot the GAF at. (in degrees)",
        default_value = "0.0",
        required_if_eq("kind", "gaf")
    )]
    pub theta_m: f32,

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
        required_if_eq("kind", "brdf-map")
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
    pub level: MeasuredBrdfLevel,

    #[clap(long, help = "The colormap to use.")]
    pub cmap: Option<String>,

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

    #[clap(
        long,
        help = "The downsample factor for the surface plot.",
        default_value_t = 1,
        required_if_eq("kind", "surf")
    )]
    pub downsample: u32,

    #[clap(long, help = "The y-axis limits.")]
    pub ylim: Option<f32>,
}

pub fn plot(opts: PlotOptions, config: Config) -> Result<(), VgonioError> {
    let cache = Cache::new(config.cache_dir());
    cache.write(|cache| {
        cache.load_ior_database(&config);
        match opts.kind {
            PlotKind::ComparisonVgonioClausen => {
                if opts.inputs.len() % 2 != 0 {
                    return Err(VgonioError::new(
                        "The number of input files must be even.",
                        None,
                    ));
                }
                for input in opts.inputs.chunks(2) {
                    let simulated_hdl = cache.load_micro_surface_measurement(&config, &input[0])?;
                    // Measured by Clausen
                    let measured_hdl = cache.load_micro_surface_measurement(&config, &input[1])?;
                    let simulated = cache
                        .get_measurement(simulated_hdl)
                        .unwrap()
                        .measured
                        .downcast_ref::<MeasuredBsdfData>()
                        .unwrap();
                    let meas = cache
                        .get_measurement(measured_hdl)
                        .unwrap()
                        .measured
                        .downcast_ref::<ClausenBrdf>()
                        .expect("Expected BSDF measured by Clausen");
                    let phi_offset = std::env::var("PHI_OFFSET")
                        .ok()
                        .map(|s| s.parse::<f32>().unwrap())
                        .unwrap_or(0.0)
                        .to_radians();
                    let itrp = simulated.resample(
                        &meas.params,
                        opts.level,
                        opts.dense,
                        Rads::new(phi_offset),
                    );
                    plot_brdf_vgonio_clausen(&itrp, meas, opts.dense).unwrap();
                }
                Ok(())
            }
            PlotKind::ComparisonVgonio => {
                todo!("Implement comparison between VgonioBrdf and VgonioBrdf.")
            }
            slc @ (PlotKind::Slice | PlotKind::SliceInPlane) => {
                let brdf_hdls = opts
                    .inputs
                    .iter()
                    .map(|input| cache.load_micro_surface_measurement(&config, input))
                    .collect::<Result<Vec<_>, _>>()?;
                let phi_i = Rads::from_degrees(opts.phi_i);
                let theta_i = Rads::from_degrees(opts.theta_i);
                let wi = Sph2::new(theta_i, phi_i);
                let phi_o = Rads::from_degrees(opts.phi_o);
                if slc == PlotKind::Slice {
                    let labels = if opts.labels.is_empty() {
                        vec!["".to_string(); brdf_hdls.len()]
                    } else {
                        opts.labels.clone()
                    };
                    let brdfs = brdf_hdls
                        .into_iter()
                        .zip(labels.into_iter())
                        .filter_map(|(hdl, label)| {
                            cache
                                .get_measurement(hdl)
                                .unwrap()
                                .measured
                                .downcast_ref::<MeasuredBsdfData>()
                                .and_then(|meas| meas.brdf_at(opts.level).map(|brdf| (brdf, label)))
                        })
                        .collect::<Vec<_>>();
                    let spectrum = brdfs[0].0.spectrum.clone().into_vec();
                    plot_brdf_slice(&brdfs, wi, phi_o, spectrum, opts.legend).unwrap();
                } else {
                    let brdfs = brdf_hdls
                        .into_iter()
                        .filter_map(|hdl| {
                            cache
                                .get_measurement(hdl)
                                .unwrap()
                                .measured
                                .downcast_ref::<MeasuredBsdfData>()
                                .and_then(|meas| meas.brdf_at(opts.level).map(|brdf| brdf))
                        })
                        .collect::<Vec<_>>();
                    let spectrum = brdfs[0].spectrum.clone().into_vec();
                    plot_brdf_slice_in_plane(&brdfs, phi_i, spectrum).unwrap();
                }
                Ok(())
            }
            PlotKind::Ndf => {
                // TODO: unload if the input is not ndf
                let hdls = opts
                    .inputs
                    .iter()
                    .map(|input| cache.load_micro_surface_measurement(&config, input))
                    .collect::<Result<Vec<_>, _>>()?;
                let ndfs = hdls
                    .into_iter()
                    .filter_map(|hdl| {
                        cache
                            .get_measurement(hdl)
                            .unwrap()
                            .measured
                            .downcast_ref::<MeasuredNdfData>()
                    })
                    .collect::<Vec<_>>();
                plot_ndf(
                    &ndfs,
                    Radians::from_degrees(opts.phi_i),
                    opts.labels,
                    opts.ylim,
                )
                .unwrap();
                Ok(())
            }
            PlotKind::Gaf => {
                let hdls = opts
                    .inputs
                    .iter()
                    .map(|input| cache.load_micro_surface_measurement(&config, input))
                    .collect::<Result<Vec<_>, _>>()?;
                let gafs = hdls
                    .into_iter()
                    .filter_map(|hdl| {
                        cache
                            .get_measurement(hdl)
                            .unwrap()
                            .measured
                            .downcast_ref::<MeasuredMsfData>()
                    })
                    .collect::<Vec<_>>();
                let wm = Sph2::new(
                    Radians::from_degrees(opts.theta_m),
                    Radians::from_degrees(opts.phi_m),
                );
                plot_gaf(&gafs, wm, Radians::from_degrees(opts.phi_v), opts.labels).unwrap();
                Ok(())
            }
            PlotKind::BrdfMap => {
                use exr::prelude::*;
                let imgs = opts
                    .inputs
                    .iter()
                    .map(|input| {
                        (
                            read_all_flat_layers_from_file(input).unwrap(),
                            input.file_name().unwrap().to_str().unwrap().to_string(),
                        )
                    })
                    .collect::<Vec<_>>();

                plot_brdf_map(
                    &imgs,
                    Degs::new(opts.theta_i),
                    Degs::new(opts.phi_i),
                    Nanometres::new(opts.lambda),
                    opts.cmap.unwrap_or("viridis".to_string()),
                    opts.cbar,
                    opts.coord,
                    opts.diff,
                    opts.fc.unwrap_or("w".to_string()),
                    Degs::new(opts.pstep),
                    Degs::new(opts.tstep),
                )
                .unwrap();

                Ok(())
            }
            PlotKind::Surface => {
                let hdls = cache.load_micro_surfaces(
                    &config,
                    &opts.inputs,
                    TriangulationPattern::BottomLeftToTopRight,
                )?;
                let surfaces = hdls
                    .into_iter()
                    .map(|hdl| cache.get_micro_surface(hdl).unwrap())
                    .collect::<Vec<_>>();

                plot_surfaces(
                    &surfaces,
                    opts.downsample,
                    opts.cmap.unwrap_or("viridis".to_string()),
                )
                .unwrap();

                Ok(())
            }
        }
    })
}
