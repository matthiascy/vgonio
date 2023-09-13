use crate::{
    app::{
        args::{FastMeasurementKind, MeasureOptions},
        cache::{resolve_path, Cache},
        cli::{write_measured_data_to_file, BRIGHT_CYAN, BRIGHT_RED, BRIGHT_YELLOW, RESET},
        Config,
    },
    measure,
    measure::params::{
        AdfMeasurementParams, BsdfMeasurementParams, Measurement, MeasurementParams,
        MsfMeasurementParams,
    },
};
use std::time::Instant;
use vgcore::error::VgonioError;

/// Measure different metrics of the micro-surface.
pub fn measure(opts: MeasureOptions, config: Config) -> Result<(), VgonioError> {
    log::info!("{:#?}", config);
    // Configure thread pool for parallelism.
    if let Some(nthreads) = opts.nthreads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads as usize)
            .build_global()
            .unwrap();
    }
    println!(
        "{BRIGHT_YELLOW}>{RESET} Executing 'vgonio measure' with a thread pool of size: {}",
        rayon::current_num_threads()
    );

    let mut cache = Cache::new(config.cache_dir());
    println!("  {BRIGHT_YELLOW}>{RESET} Reading measurement description files...");
    let measurements = {
        match opts.fast_measurement {
            None => opts
                .inputs
                .iter()
                .filter_map(|meas_path| {
                    let resolved = resolve_path(&config.cwd, Some(meas_path));
                    match Measurement::load(&resolved) {
                        Ok(meas) => Some(meas),
                        Err(err) => {
                            log::warn!("Failed to load measurement description file: {}", err);
                            None
                        }
                    }
                })
                .flatten()
                .collect::<Vec<_>>(),
            Some(kinds) => kinds
                .iter()
                .map(|kind| match kind {
                    FastMeasurementKind::Bsdf => Measurement {
                        params: MeasurementParams::Bsdf(BsdfMeasurementParams::default()),
                        surfaces: opts.inputs.clone(),
                    },
                    FastMeasurementKind::AreaDistributionFunction => Measurement {
                        params: MeasurementParams::Adf(AdfMeasurementParams::default()),
                        surfaces: opts.inputs.clone(),
                    },
                    FastMeasurementKind::MaskingShadowingFunction => Measurement {
                        params: MeasurementParams::Msf(MsfMeasurementParams::default()),
                        surfaces: opts.inputs.clone(),
                    },
                })
                .collect::<Vec<_>>(),
        }
    };
    println!(
        "    {BRIGHT_CYAN}✓{RESET} {} measurement(s)",
        measurements.len()
    );

    // Load data files: refractive indices, spd etc. if needed.
    if measurements.iter().any(|meas| meas.params.is_bsdf()) {
        println!("  {BRIGHT_YELLOW}>{RESET} Loading data files (refractive indices, spd etc.)...");
        cache.load_ior_database(&config);
        println!("    {BRIGHT_CYAN}✓{RESET} Successfully load data files");
    }

    println!("  {BRIGHT_YELLOW}>{RESET} Resolving and loading micro-surfaces...");
    let tasks = measurements
        .into_iter()
        .filter_map(|meas| {
            cache
                .load_micro_surfaces(&config, &meas.surfaces, config.user.triangulation)
                .ok()
                .map(|surfaces| (meas, surfaces))
        })
        .collect::<Vec<_>>();
    println!(
        "    {BRIGHT_CYAN}✓{RESET} {} micro-surface(s) loaded",
        cache.num_micro_surfaces()
    );

    if cache.num_micro_surfaces() == 0 {
        println!("  {BRIGHT_RED}✗{RESET} No micro-surface to measure. Exiting...");
        return Ok(());
    }

    cache
        .loaded_micro_surface_paths()
        .unwrap()
        .iter()
        .for_each(|s| println!("      {BRIGHT_CYAN}-{RESET} {}", s.display()));

    let start_time = Instant::now();
    for (measurement, surfaces) in tasks {
        let measurement_start_time = std::time::SystemTime::now();
        let measured_data = match measurement.params {
            MeasurementParams::Bsdf(measurement) => {
                println!(
                    "  {BRIGHT_YELLOW}>{RESET} Launch BSDF measurement at {}
    • parameters:
      + incident medium: {:?}
      + transmitted medium: {:?}
      + emitter:
        - num rays: {}
        - max bounces: {}
        - spectrum: {}
        - polar angle: {}
        - azimuthal angle: {}
      + collector:
        - domain: {}
        - precision: {}",
                    chrono::DateTime::<chrono::Utc>::from(measurement_start_time),
                    measurement.incident_medium,
                    measurement.transmitted_medium,
                    measurement.emitter.num_rays,
                    measurement.emitter.max_bounces,
                    measurement.emitter.spectrum,
                    measurement.emitter.zenith.pretty_print(),
                    measurement.emitter.azimuth.pretty_print(),
                    measurement.detector.domain,
                    measurement.detector.precision
                );
                measure::bsdf::measure_bsdf_rt(measurement, &surfaces, measurement.sim_kind, &cache)
            }
            MeasurementParams::Adf(measurement) => {
                println!(
                    "  {BRIGHT_YELLOW}>{RESET} Measuring microfacet area distribution:
    • parameters:
      + azimuth: {}
      + zenith: {}",
                    measurement.azimuth.pretty_print(),
                    measurement.zenith.pretty_print(),
                );
                measure::microfacet::measure_area_distribution(measurement, &surfaces, &cache)
            }
            MeasurementParams::Msf(measurement) => {
                println!(
                    "  {BRIGHT_YELLOW}>{RESET} Measuring microfacet masking-shadowing function:
    • parameters:
      + azimuth: {}
      + zenith: {}
      + resolution: {} x {}",
                    measurement.azimuth.pretty_print(),
                    measurement.zenith.pretty_print(),
                    measurement.resolution,
                    measurement.resolution
                );

                #[cfg(debug_assertions)]
                log::warn!(
                    "Debug mode is enabled. Measuring MMSF in debug mode is not recommended."
                );

                measure::microfacet::measure_masking_shadowing(measurement, &surfaces, &cache)
            }
        };

        println!(
            "    {BRIGHT_CYAN}✓{RESET} Measurement finished in {} secs.",
            measurement_start_time.elapsed().unwrap().as_secs_f32()
        );

        write_measured_data_to_file(
            &measured_data,
            &surfaces,
            &cache,
            &config,
            opts.encoding,
            opts.compression,
            &opts.output,
        )?;

        println!("    {BRIGHT_CYAN}✓{RESET} Done!");
    }

    println!(
        "    {BRIGHT_CYAN}✓{RESET} Finished in {:.2} s",
        start_time.elapsed().as_secs_f32()
    );

    Ok(())
}
