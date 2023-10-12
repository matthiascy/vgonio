use crate::{
    app::{args::MeasureOptions, cache::Cache, cli::ansi, Config},
    measure,
    measure::params::{Measurement, MeasurementParams},
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
        "{}>{} Executing 'vgonio measure' with a thread pool of size: {}",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        rayon::current_num_threads()
    );

    println!(
        "  {}>{} Reading measurement description files...",
        ansi::BRIGHT_YELLOW,
        ansi::RESET
    );
    let measurements = opts
        .inputs
        .iter()
        .flat_map(|meas_path| {
            config
                .resolve_path(meas_path)
                .map(|resolved| match Measurement::load(&resolved) {
                    Ok(meas) => Some(meas),
                    Err(err) => {
                        log::warn!("Failed to load measurement description file: {}", err);
                        None
                    }
                })
        })
        .filter_map(|meas| meas)
        .flatten()
        .collect::<Vec<_>>();
    println!(
        "    {}✓{} {} measurement(s)",
        ansi::BRIGHT_CYAN,
        ansi::RESET,
        measurements.len()
    );

    let cache = Cache::new(config.cache_dir());

    let (tasks, num_surfs) = cache.write(|cache| {
        // Load data files: refractive indices, spd etc. if needed.
        if measurements.iter().any(|meas| meas.params.is_bsdf()) {
            println!(
                "  {}>{} Loading data files (refractive indices, spd etc.)...",
                ansi::BRIGHT_YELLOW,
                ansi::RESET
            );
            cache.load_ior_database(&config);
            println!(
                "    {}✓{} Successfully load data files",
                ansi::BRIGHT_CYAN,
                ansi::RESET
            );
        }

        println!(
            "  {}>{} Resolving and loading micro-surfaces...",
            ansi::BRIGHT_YELLOW,
            ansi::RESET
        );
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
            "    {}✓{} {} micro-surface(s) loaded",
            ansi::BRIGHT_CYAN,
            ansi::RESET,
            cache.num_micro_surfaces()
        );

        #[cfg(debug_assertions)]
        cache
            .loaded_micro_surface_paths()
            .unwrap()
            .iter()
            .for_each(|s| {
                println!(
                    "      {}-{} {}",
                    ansi::BRIGHT_CYAN,
                    ansi::RESET,
                    s.display()
                )
            });

        (tasks, cache.num_micro_surfaces())
    });

    if num_surfs == 0 {
        println!(
            "  {}✗{} No micro-surface to measure. Exiting...",
            ansi::BRIGHT_RED,
            ansi::RESET
        );
        return Ok(());
    }

    let start_time = Instant::now();
    for (measurement, surfaces) in tasks {
        let measurement_start_time = std::time::SystemTime::now();
        let measured_data = match measurement.params {
            MeasurementParams::Bsdf(params) => {
                println!(
                    "  {}>{} Launch BSDF measurement at {}
    • parameters:
      + incident medium: {:?}
      + transmitted medium: {:?}
      + emitter:
        - num rays: {}
        - max bounces: {}
        - spectrum: {}
        - polar angle: {}
        - azimuthal angle: {}
      + receiver:
        - domain: {}
        - scheme: {:?}
        - precision: {}",
                    ansi::BRIGHT_YELLOW,
                    ansi::RESET,
                    chrono::DateTime::<chrono::Utc>::from(measurement_start_time),
                    params.incident_medium,
                    params.transmitted_medium,
                    params.emitter.num_rays,
                    params.emitter.max_bounces,
                    params.emitter.spectrum,
                    params.emitter.zenith.pretty_print(),
                    params.emitter.azimuth.pretty_print(),
                    params.receiver.domain,
                    params.receiver.scheme,
                    params.receiver.precision
                );
                cache.read(|cache| {
                    measure::bsdf::measure_bsdf_rt(params, &surfaces, params.sim_kind, cache)
                })
            }
            MeasurementParams::Adf(measurement) => {
                println!(
                    "  {}>{} Measuring microfacet area distribution:
    • parameters:
      + azimuth: {}
      + zenith: {}",
                    ansi::BRIGHT_YELLOW,
                    ansi::RESET,
                    measurement.azimuth.pretty_print(),
                    measurement.zenith.pretty_print(),
                );
                cache.read(|cache| {
                    measure::microfacet::measure_area_distribution(measurement, &surfaces, cache)
                })
            }
            MeasurementParams::Msf(measurement) => {
                println!(
                    "  {}>{} Measuring microfacet masking-shadowing function:
    • parameters:
      + azimuth: {}
      + zenith: {}
      + resolution: {} x {}",
                    ansi::BRIGHT_YELLOW,
                    ansi::RESET,
                    measurement.azimuth.pretty_print(),
                    measurement.zenith.pretty_print(),
                    measurement.resolution,
                    measurement.resolution
                );

                #[cfg(debug_assertions)]
                log::warn!(
                    "Debug mode is enabled. Measuring MMSF in debug mode is not recommended."
                );
                cache.read(|cache| {
                    measure::microfacet::measure_masking_shadowing(measurement, &surfaces, cache)
                })
            }
        };

        println!(
            "    {}✓{} Measurement finished in {} secs.",
            ansi::BRIGHT_CYAN,
            ansi::RESET,
            measurement_start_time.elapsed().unwrap().as_secs_f32()
        );

        crate::io::write_measured_data_to_file(
            &measured_data,
            &surfaces,
            &cache,
            &config,
            opts.output_format,
            opts.encoding,
            opts.compression,
            opts.output.as_deref(),
        )?;

        println!("    {}✓{} Done!", ansi::BRIGHT_CYAN, ansi::RESET);
    }

    println!(
        "    {}✓{} Finished in {:.2} s",
        ansi::BRIGHT_CYAN,
        ansi::RESET,
        start_time.elapsed().as_secs_f32()
    );

    Ok(())
}
