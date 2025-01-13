use crate::{
    app::{cache::Cache, Config},
    measure::{
        bsdf::{MeasuredBrdfLevel, MeasuredBsdfData},
        mfd::{MeasuredGafData, MeasuredNdfData},
        params::SurfacePath,
    },
    pyplot::{
        plot_brdf_3d, plot_brdf_fitting, plot_brdf_map, plot_brdf_slice, plot_brdf_slice_in_plane,
        plot_brdf_vgonio_clausen, plot_gaf, plot_ndf, plot_surfaces,
    },
};
use base::{
    error::VgonioError,
    math::Sph2,
    units::{Degs, Nanometres, Radians, Rads},
    Symmetry, MeasurementKind,
};
use bxdf::brdf::measured::ClausenBrdf;
use std::path::PathBuf;
use surf::{
    subdivision::{Subdivision, SubdivisionKind},
    TriangulationPattern,
};

#[rustfmt::skip]
// Same BRDF but different incident angles
// vgonio plot -i file -k slice-in-plane --ti 0.0 --pi 0.0 --po 0.0

// Multiple BRDF
// vgonio plot -i f0 f1 f2 -k slice --ti 0.0 --pi 0.0 --po 0.0 --labels f0 f1 f2

// vgonio plot -i a b c d --ti 30.0 --po 120.0 --labels '$0.5^{\circ}$' '$1^{\circ}$' '$2^{\circ}$' '$5^{\circ}$' --kind brdf-slice --legend --cmap Paired --pi 120
// vgonio plot -i a b c d --ti 30.0 --po 30.0 --labels '$10^5$' '$10^6$' '$10^7$' '$10^8$' --kind brdf-slice --legend --cmap tab10

// Single BRDF
// vgonio plot -i f0 -k slice --ti 0.0 --pi 0.0 --po 0.0

// Plot BRDF map
// vgonio plot -i f.exr -k brdf-map --ti 0 --pi 0 --lambda 400 --cmap BuPu

// python tone_mapping.py --input a.exr b.exr --channel '400 nm' --cbar --cmap plasma --diff --fc w --coord

/// Kind of plot to generate.
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotKind {
    /// Compare between VgonioBrdf and ClausenBrdf.
    #[clap(alias = "cmp-vc")]
    ComparisonVgonioClausen,
    /// Compare between VgonioBrdf and VgonioBrdf.
    #[clap(alias = "cmp-vv")]
    ComparisonVgonio,
    /// Plot the residual map between the input BRDF and the fitted BRDF.
    #[clap(name = "residual")]
    BrdfResidual,
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
    #[clap(name = "brdf-fitting")]
    BrdfFitting,
    /// Plot the NDF from saved *.vgmo file
    #[clap(name = "ndf")]
    Ndf,
    /// Plot the GAF from saved *.vgmo file
    #[clap(name = "gaf")]
    Gaf,
    /// Plot the micro-surfaces from saved *.vgmo file
    #[clap(name = "surf")]
    Surface,
}

/// Options for the `plot` command.
#[derive(clap::Args, Debug, Clone)]
pub struct PlotOptions {
    #[clap(short, long, num_args = 1.., value_delimiter = ' ', help = "Input files to plot.")]
    pub inputs: Vec<PathBuf>,

    #[clap(short, long, help = "The kind of plot to generate.")]
    pub kind: PlotKind,

    #[arg(
        short,
        long,
        value_delimiter = ' ',
        num_args = 1..,
        help = "The roughness parameter along the x-axis and y-axis. Can be multiple values.",
        required_if_eq("kind", "brdf-fitting"),
    )]
    pub alpha: Vec<f64>,

    #[clap(
        long,
        help = "The symmetry of the model which decides how the roughness parameters been \
                interpreted",
        default_value = "iso"
    )]
    pub symmetry: Symmetry,

    #[clap(
        long = "subdiv_kind",
        help = "The kind of subdivision.",
        required_if_eq("kind", "surf")
    )]
    pub subdiv_kind: Option<SubdivisionKind>,

    #[clap(
        long = "subdiv_level",
        help = "The level of subdivision.",
        required_if_eq("kind", "surf"),
        default_value = "0"
    )]
    pub subdiv_level: Option<u32>,

    #[clap(
        long = "offset",
        help = "The offset to add randomly to the z coordinate of the new points.",
        required_if_eq("subdiv_kind", "wiggly"),
        default_value = "100"
    )]
    pub subdiv_offset: Option<u32>,

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
    pub level: MeasuredBrdfLevel,

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

    #[clap(
        long,
        help = "The downsample factor for the surface plot.",
        default_value_t = 1,
        required_if_eq("kind", "surf")
    )]
    pub downsample: u32,

    #[clap(long, help = "The y-axis limits.")]
    pub ylim: Option<f32>,

    #[clap(long, help = "Use log scale for the plot.", default_value = "false")]
    pub log: bool,

    #[clap(long, help = "The file to save the plot.")]
    pub save: Option<String>,
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
            },
            PlotKind::ComparisonVgonio => {
                todo!("Implement comparison between VgonioBrdf and VgonioBrdf.")
            },
            slc @ (PlotKind::BrdfSlice | PlotKind::BrdfSliceInPlane) => {
                let brdf_hdls = opts
                    .inputs
                    .iter()
                    .map(|input| cache.load_micro_surface_measurement(&config, input))
                    .collect::<Result<Vec<_>, _>>()?;
                let phi_i = Rads::from_degrees(opts.phi_i);
                let theta_i = Rads::from_degrees(opts.theta_i);
                let wi = Sph2::new(theta_i, phi_i);
                let phi_o = Rads::from_degrees(opts.phi_o);
                if slc == PlotKind::BrdfSlice {
                    let labels = if opts.labels.is_empty() {
                        vec!["".to_string(); brdf_hdls.len()]
                    } else {
                        opts.labels.clone()
                    };
                    let brdfs = brdf_hdls
                        .into_iter()
                        .zip(labels)
                        .filter_map(|(hdl, label)| {
                            cache
                                .get_measurement(hdl)
                                .unwrap()
                                .measured
                                .downcast_ref::<MeasuredBsdfData>()
                                .and_then(|meas| meas.brdf_at(opts.level).map(|brdf| (brdf, label)))
                        })
                        .collect::<Vec<_>>();
                    plot_brdf_slice(
                        &brdfs,
                        wi,
                        phi_o,
                        opts.legend,
                        opts.cmap.unwrap_or("Set3".to_string()),
                        opts.scale.unwrap_or(1.0),
                        opts.log,
                    )
                    .unwrap();
                } else {
                    let brdfs = brdf_hdls
                        .into_iter()
                        .filter_map(|hdl| {
                            cache
                                .get_measurement(hdl)
                                .unwrap()
                                .measured
                                .downcast_ref::<MeasuredBsdfData>()
                                .and_then(|meas| meas.brdf_at(opts.level))
                        })
                        .collect::<Vec<_>>();
                    plot_brdf_slice_in_plane(&brdfs, phi_i).unwrap();
                }
                Ok(())
            },
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
            },
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
                            .downcast_ref::<MeasuredGafData>()
                    })
                    .collect::<Vec<_>>();
                let wm = Sph2::new(
                    Radians::from_degrees(opts.theta_m),
                    Radians::from_degrees(opts.phi_m),
                );
                plot_gaf(
                    &gafs,
                    wm,
                    Radians::from_degrees(opts.phi_v),
                    opts.labels,
                    opts.save,
                )
                .unwrap();
                Ok(())
            },
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
                    opts.save,
                )
                .unwrap();

                Ok(())
            },
            PlotKind::Brdf3D => {
                let hdls = opts
                    .inputs
                    .iter()
                    .map(|input| cache.load_micro_surface_measurement(&config, input))
                    .collect::<Result<Vec<_>, _>>()?;
                let brdfs = hdls
                    .into_iter()
                    .filter_map(|hdl| {
                        cache
                            .get_measurement(hdl)
                            .unwrap()
                            .measured
                            .downcast_ref::<MeasuredBsdfData>()
                            .and_then(|meas| meas.brdf_at(opts.level))
                    })
                    .collect::<Vec<_>>();
                let lambda = Nanometres::new(opts.lambda);
                for brdf in &brdfs {
                    if !brdf.spectrum.iter().any(|&l| l == lambda) {
                        return Err(VgonioError::new(
                            "The wavelength is not in the spectrum of the BRDF.",
                            None,
                        ));
                    }
                    plot_brdf_3d(
                        brdf,
                        Sph2::new(
                            Radians::from_degrees(opts.theta_i),
                            Radians::from_degrees(opts.phi_i),
                        ),
                        lambda,
                        opts.cmap.clone().unwrap_or("viridis".to_string()),
                        opts.scale.unwrap_or(1.0),
                    )
                    .unwrap();
                }
                Ok(())
            },
            PlotKind::Surface => {
                let surfaces = opts
                    .inputs
                    .iter()
                    .map(|path| SurfacePath {
                        path: path.clone(),
                        subdivision: match (opts.subdiv_kind, opts.subdiv_level) {
                            (Some(SubdivisionKind::Curved), Some(level)) => {
                                Some(Subdivision::Curved(level))
                            },
                            (Some(SubdivisionKind::Wiggly), Some(level)) => {
                                Some(Subdivision::Wiggly {
                                    level,
                                    offset: opts.subdiv_offset.unwrap(),
                                })
                            },
                            _ => None,
                        },
                    })
                    .collect::<Box<_>>();
                let hdls = cache.load_micro_surfaces(
                    &config,
                    &surfaces,
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
            },
            PlotKind::BrdfFitting => {
                if opts.inputs.len() != 1 {
                    return Err(VgonioError::new(
                        "Only one input file is allowed for plotting the BRDF fitting.",
                        None,
                    ));
                }
                let hdl = cache.load_micro_surface_measurement(&config, &opts.inputs[0])?;
                let measured = &cache.get_measurement(hdl).unwrap().measured;
                if measured.kind() != MeasurementKind::Bsdf {
                    return Err(VgonioError::new(
                        "The input file is not a BSDF measurement.",
                        None,
                    ));
                }
                let alphas = extract_alphas(&opts.alpha, opts.symmetry)?;
                plot_brdf_fitting(&measured, &alphas, &cache.iors).unwrap();
                Ok(())
            },
            PlotKind::BrdfResidual => {
                let alphas = extract_alphas(&opts.alpha, opts.symmetry)?;
                todo!("Implement BRDF residual plot.")
            }
        }
    })
}

/// Extracts alpha values from the command-line arguments as a vector of `f64` pairs,
/// duplicating values for isotropic surfaces or pairing them as provided for anisotropic surfaces.
///
/// # Arguments
///
/// * `alphas` - The roughness parameters provided by the user.
/// * `symmetry` - The symmetry of the surface.
///
/// # Returns
///
/// An array of roughness parameters as pairs of `f64` values sorted by the roughness parameter.
fn extract_alphas(alphas: &[f64], symmetry: Symmetry) -> Result<Box<[(f64, f64)]>, VgonioError> {
    let mut processed = if symmetry.is_isotropic() {
        alphas.iter().map(|&a| (a, a)).collect::<Box<_>>()
    } else {
        if alphas.len() % 2 != 0 {
            return Err(VgonioError::new(
                "Two roughness parameters are needed for providing anisotropic roughness. \
                             If you wish to provide multiple roughness parameters, please supply \
                             them in pairs.",
                None,
            ));
        }
       alphas
            .iter()
            .map_windows(|alpha: &[_; 2]| (*alpha[0], *alpha[1]))
            .collect::<Box<_>>()
    };
    processed.sort_by(|(a1, a2), (b1, b2)|
        // Sort first by the first element, then by the second element
        a1.partial_cmp(b1).unwrap().then_with(|| a2.partial_cmp(b2).unwrap()));
    Ok(processed)
}
