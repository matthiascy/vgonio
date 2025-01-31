#![feature(iter_map_windows)]

use vgonio_bxdf::brdf::measured::ClausenBrdf;
use vgonio_core::{
    cli,
    config::Config,
    error::VgonioError,
    math::Sph2,
    res::DataStore,
    units::{Degs, Nanometres, Radians, Rads},
    ErrorMetric, MeasurementKind, Symmetry,
};
use vgonio_plot::{PlotArgs, PlotKind};

#[cfg(feature = "surface")]
use vgonio_surf::subdivision::{Subdivision, SubdivisionKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (args, launch_time) = cli::parse_args::<PlotArgs>("vgonio-plot");

    cli::setup_logging(Some(launch_time), args.log_level, &[]);

    let config = Config::load_config(args.config.as_deref())?;

    let data_store = DataStore::new_from_config(true, &config);

    run(args, config, data_store).map_err(|err| err.into())
}

fn run(opts: PlotArgs, config: Config, data_store: DataStore) -> Result<(), VgonioError> {
    data_store.write(|ds| {
        match opts.kind {
            PlotKind::ComparisonVgonioClausen => {
                if opts.inputs.len() % 2 != 0 {
                    return Err(VgonioError::new(
                        "The number of input files must be even.",
                        None,
                    ));
                }
                for input in opts.inputs.chunks(2) {
                    let simulated_hdl = ds.load_micro_surface_measurement(&config, &input[0])?;
                    // Measured by Clausen
                    let measured_hdl = ds.load_micro_surface_measurement(&config, &input[1])?;
                    let simulated = ds
                        .get_measurement(simulated_hdl)
                        .unwrap()
                        .measured
                        .downcast_ref::<BsdfMeasurement>()
                        .unwrap();
                    let meas = ds
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
                todo!(
                    "Implement comparison between VgonioBrdf and
VgonioBrdf."
                )
            },
            slc @ (PlotKind::BrdfSlice | PlotKind::BrdfSliceInPlane) => {
                let brdf_hdls = opts
                    .inputs
                    .iter()
                    .map(|input| ds.load_micro_surface_measurement(&config, input))
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
                            ds.get_measurement(hdl)
                                .unwrap()
                                .measured
                                .downcast_ref::<BsdfMeasurement>()
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
                            ds.get_measurement(hdl)
                                .unwrap()
                                .measured
                                .downcast_ref::<BsdfMeasurement>()
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
                    .map(|input| ds.load_micro_surface_measurement(&config, input))
                    .collect::<Result<Vec<_>, _>>()?;
                let ndfs = hdls
                    .into_iter()
                    .filter_map(|hdl| {
                        ds.get_measurement(hdl)
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
                    .map(|input| ds.load_micro_surface_measurement(&config, input))
                    .collect::<Result<Vec<_>, _>>()?;
                let gafs = hdls
                    .into_iter()
                    .filter_map(|hdl| {
                        ds.get_measurement(hdl)
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
                    .map(|input| ds.load_micro_surface_measurement(&config, input))
                    .collect::<Result<Vec<_>, _>>()?;
                let brdfs = hdls
                    .into_iter()
                    .filter_map(|hdl| {
                        ds.get_measurement(hdl)
                            .unwrap()
                            .measured
                            .downcast_ref::<BsdfMeasurement>()
                            .and_then(|meas| meas.brdf_at(opts.level))
                    })
                    .collect::<Vec<_>>();
                let lambda = Nanometres::new(opts.lambda);
                for brdf in &brdfs {
                    if !brdf.spectrum.iter().any(|&l| l == lambda) {
                        return Err(VgonioError::new(
                            "The wavelength is not in the spectrum of the
BRDF.",
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
                let hdls = ds.load_micro_surfaces(
                    &config,
                    &surfaces,
                    TriangulationPattern::BottomLeftToTopRight,
                )?;
                let surfaces = hdls
                    .into_iter()
                    .map(|hdl| ds.get_micro_surface(hdl).unwrap())
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
                if opts.interactive {
                    if opts.inputs.len() != 1 {
                        return Err(VgonioError::new(
                            "Only one input file is allowed for plotting the
BRDF fitting.",
                            None,
                        ));
                    }
                    let hdl = ds.load_micro_surface_measurement(&config, &opts.inputs[0])?;
                    let measured = &ds.get_measurement(hdl).unwrap().measured;
                    if measured.kind() != MeasurementKind::Bsdf {
                        return Err(VgonioError::new(
                            "The input file is not a
BSDF measurement.",
                            None,
                        ));
                    }
                    let alphas = extract_alphas(&opts.alpha, opts.symmetry)?;
                    let brdf = measured.as_any_brdf(opts.level).unwrap();
                    BrdfFittingPlotter::plot_interactive(brdf, &alphas, &ds.iors).unwrap();
                } else {
                    if opts.inputs.len() != 1 {
                        return Err(VgonioError::new(
                            "Only one input file is allowed for plotting the
BRDF error map.",
                            None,
                        ));
                    }
                    let hdl = ds.load_micro_surface_measurement(&config, &opts.inputs[0])?;
                    let measured = &ds.get_measurement(hdl).unwrap().measured;
                    if measured.kind() != MeasurementKind::Bsdf {
                        return Err(VgonioError::new(
                            "The input file is not a
BSDF measurement.",
                            None,
                        ));
                    }
                    let name = opts.inputs[0].file_name().unwrap().to_str().unwrap();
                    let alphas = extract_alphas(&opts.alpha, opts.symmetry)?;
                    let error_metric = opts.error_metric.unwrap_or(ErrorMetric::Mse);
                    BrdfFittingPlotter::plot_non_interactive(
                        name,
                        &measured,
                        &alphas,
                        error_metric,
                        opts.weighting.unwrap(),
                        &ds.iors,
                        opts.parallel,
                    )
                    .map_err(|pyerr| {
                        VgonioError::new(
                            format!("Failed to plot the BRDF error map: {}", pyerr),
                            Some(Box::new(pyerr)),
                        )
                    })?;
                }
                Ok(())
            },
        }
    })
}

/// Extracts alpha values from the command-line arguments as a vector of `f64`
/// pairs, duplicating values for isotropic surfaces or pairing them as provided
/// for anisotropic surfaces.
///
/// # Arguments
///
/// * `alphas` - The roughness parameters provided by the user.
/// * `symmetry` - The symmetry of the surface.
///
/// # Returns
///
/// An array of roughness parameters as pairs of `f64` values sorted by the
/// roughness parameter.
fn extract_alphas(alphas: &[f64], symmetry: Symmetry) -> Result<Box<[(f64, f64)]>, VgonioError> {
    let mut processed = if symmetry.is_isotropic() {
        alphas.iter().map(|&a| (a, a)).collect::<Box<_>>()
    } else {
        if alphas.len() % 2 != 0 {
            return Err(VgonioError::new(
                "Two roughness parameters are needed for providing anisotropic roughness. If you \
                 wish to provide multiple roughness parameters, please supply them in pairs.",
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
