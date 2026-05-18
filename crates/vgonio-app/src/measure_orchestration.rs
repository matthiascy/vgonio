//! Measurement orchestration: resolves descriptions, loads surfaces, runs the
//! measurement, and writes output. CLI shape (clap + thread-pool) stays in
//! [`crate::app::cli::cmd_measure`]; Phase 2 moves this module into a
//! capability crate.
//!
//! # CLI-print inventory (DIST Plan Task 0.5)
//!
//! Every `cli_*` call here is ORCHESTRATION (the one ADAPTER print — the
//! `vgonio measure` ROOT banner — was moved to `cmd_measure::measure`). When
//! Phase 2 moves this module into a capability crate, `vgn_core::cli` is no
//! longer reachable; Phase 1 replaces each call below with a `ProgressEvent`
//! emission. Mapping (line numbers approximate, see inline `[0.5]` tags):
//!
//! | Site | Text | → Phase 1 |
//! |---|---|---|
//! | "Reading measurement description files..." | step | `ProgressEvent::Step` |
//! | "{N} measurement(s)" | success | `ProgressEvent::Success` |
//! | "Loading data files..." | step | `ProgressEvent::Step` |
//! | "Successfully loaded data files" | success | `ProgressEvent::Success` |
//! | "Resolving and loading micro-surfaces..." | step | `ProgressEvent::Step` |
//! | "{N} micro-surface(s) loaded" | success | `ProgressEvent::Success` |
//! | per-surface path note (debug) | note | `ProgressEvent::Note` |
//! | "No micro-surface to measure" | error | `ProgressEvent::Error` |
//! | per-kind "Launch/Measuring …" banners (BSDF/NDF/GAF/SDF) | step | `ProgressEvent::Step` |
//! | "Measurement finished in … secs" | success | `ProgressEvent::Success` |
//! | "Done!" / "Finished in … s" | success | `ProgressEvent::Success` |
//!
//! Do NOT migrate to `ProgressEvent` in Phase 0 — `vgonio-job-api` does not
//! exist yet. This block is the migration checklist for Phase 1.

use crate::{
    app::{args::OutputFormat, cache::Cache, cli::MeasureOptions},
    io::{OutputFileFormatOption, OutputOptions},
    measure,
    measure::params::{MeasurementDescription, MeasurementParams, NdfMeasurementMode},
};
use std::time::Instant;
use vgn_core::{
    cli::{cli_error, cli_note, cli_step, cli_success, Indent},
    config::Config,
    error::VgonioError,
};

pub fn run(opts: MeasureOptions, config: Config) -> Result<(), VgonioError> {
    // [0.5] ADAPTER print moved to `cmd_measure::measure` (the ROOT
    // "Executing 'vgonio measure' …" banner). Nothing prints here anymore.

    // [0.5] orchestration → Phase 1 ProgressEvent::Step
    cli_step!(Indent::SECTION, "Reading measurement description files...");
    let measurements = opts
        .inputs
        .iter()
        .flat_map(|meas_path| {
            config.resolve_path(meas_path).map(|resolved| {
                match MeasurementDescription::load(&resolved) {
                    Ok(meas) => Some(meas),
                    Err(err) => {
                        log::warn!("Failed to load measurement description file: {}", err);
                        None
                    },
                }
            })
        })
        .filter_map(|meas| meas)
        .flatten()
        .collect::<Vec<_>>();
    // [0.5] orchestration → Phase 1 ProgressEvent::Success
    cli_success!(Indent::SUBSECTION, "{} measurement(s)", measurements.len());

    let cache = Cache::new(config.cache_dir());

    let (tasks, num_surfs) = cache.write(|cache| {
        // Load data files: refractive indices, spd etc. if needed.
        if measurements.iter().any(|meas| meas.params.is_bsdf()) {
            // [0.5] orchestration → Phase 1 ProgressEvent::Step
            cli_step!(
                Indent::SECTION,
                "Loading data files (refractive indices, spd etc.)..."
            );
            cache.load_ior_database(&config);
            // [0.5] orchestration → Phase 1 ProgressEvent::Success
            cli_success!(Indent::SUBSECTION, "Successfully loaded data files");
        }

        // [0.5] orchestration → Phase 1 ProgressEvent::Step
        cli_step!(Indent::SECTION, "Resolving and loading micro-surfaces...");
        let tasks = measurements
            .into_iter()
            .filter_map(|meas| {
                cache
                    .load_micro_surfaces(&config, &meas.surfaces, config.user.triangulation)
                    // TODO [cli-report worthy]: `.ok()` silently drops surfaces
                    // that fail to load — the user gets no diagnostic for a
                    // missing/corrupt surface, just a smaller count. Phase 1
                    // should emit ProgressEvent::Error (or Warning) per failed
                    // `meas.surfaces` entry before discarding it.
                    .ok()
                    .map(|surfaces| (meas, surfaces))
            })
            .collect::<Vec<_>>();
        // [0.5] orchestration → Phase 1 ProgressEvent::Success
        cli_success!(
            Indent::SUBSECTION,
            "{} micro-surface(s) loaded",
            cache.num_micro_surfaces()
        );

        // [0.5] orchestration → Phase 1 ProgressEvent::Note
        #[cfg(debug_assertions)]
        cache
            .loaded_micro_surface_paths()
            .unwrap()
            .iter()
            .for_each(|s| cli_note!(Indent::DETAIL, "{}", s.display()));

        (tasks, cache.num_micro_surfaces())
    });

    if num_surfs == 0 {
        // [0.5] orchestration → Phase 1 ProgressEvent::Error
        cli_error!(Indent::SECTION, "No micro-surface to measure. Exiting...");
        return Ok(());
    }

    let start_time = Instant::now();
    for (desc, surfaces) in tasks {
        let measurement_start_time = std::time::SystemTime::now();
        // [0.5] orchestration: every per-kind launch banner in this `match`
        // (BSDF "Launch BSDF measurement…", NDF by-points/by-partition, GAF
        // "Measuring microfacet masking-shadowing…", SDF "Measuring slope
        // distribution…") → Phase 1 ProgressEvent::Step. The trailing
        // "Measurement finished in … secs" → ProgressEvent::Success.
        // TODO [cli-report worthy]: the actual measurement calls below
        // (measure_bsdf_rt / measure_area_distribution / …) run silently for
        // potentially minutes with no per-progress feedback. Phase 1 should
        // thread a ProgressEvent sink into them for per-incident-direction /
        // per-surface progress, not just a start banner + final success.
        let measured = match desc.params {
            MeasurementParams::Bsdf(params) => {
                if let Err(reason) = params.check_supported_for_measurement() {
                    return Err(VgonioError::new(
                        &format!(
                            "Unsupported simulation method for this build: {}. Rebuild with the \
                             required backend feature or choose another simulation kind.",
                            reason
                        ),
                        None,
                    ));
                }
                cli_step!(
                    Indent::SECTION,
                    "Launch BSDF measurement at {}
    • parameters:
      + incident medium: {:?}
      + transmitted medium: {:?}
      + emitter:
        - num rays: {}
        - num sectors: {}
        - max bounces: {}
        - spectrum: {}
        - polar angle: {}
        - azimuthal angle: {}",
                    chrono::DateTime::<chrono::Utc>::from(measurement_start_time),
                    params.incident_medium,
                    params.transmitted_medium,
                    params.emitter.num_rays,
                    params.emitter.num_sectors,
                    params.emitter.max_bounces,
                    params.emitter.spectrum,
                    params.emitter.zenith.pretty_print(),
                    params.emitter.azimuth.pretty_print()
                );
                for receiver in &params.receivers {
                    println!(
                        "      + receiver:
        - domain: {}
        - scheme: {:?}
        - precision: {}",
                        receiver.domain, receiver.scheme, receiver.precision
                    );
                }
                cache.read(|cache| measure::bsdf::measure_bsdf_rt(params, &surfaces, cache))
            },
            MeasurementParams::Ndf(measurement) => {
                match &measurement.mode {
                    NdfMeasurementMode::ByPoints { azimuth, zenith } => {
                        cli_step!(
                            Indent::SECTION,
                            "Measuring microfacet area distribution:
    • parameters:
      + mode: by points
        + azimuth: {}
        + zenith: {}",
                            azimuth.pretty_print(),
                            zenith.pretty_print()
                        );
                    },
                    NdfMeasurementMode::ByPartition { precision } => {
                        cli_step!(
                            Indent::SECTION,
                            "Measuring microfacet area distribution:
    • parameters:
       + mode: by partition
           + scheme: Beckers
           + precision: {}",
                            precision.prettified()
                        );
                    },
                }
                cache.read(|cache| {
                    measure::mfd::measure_area_distribution(measurement, &surfaces, cache)
                })
            },
            MeasurementParams::Gaf(measurement) => {
                cli_step!(
                    Indent::SECTION,
                    "Measuring microfacet masking-shadowing function:
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
                cache.read(|cache| {
                    measure::mfd::measure_masking_shadowing_function(measurement, &surfaces, cache)
                })
            },
            MeasurementParams::Sdf(params) => {
                cli_step!(Indent::SECTION, "Measuring slope distribution function...");
                cache.read(|cache| {
                    measure::mfd::measure_slope_distribution(&surfaces, params, cache)
                })
            },
        };

        // [0.5] orchestration → Phase 1 ProgressEvent::Success
        cli_success!(
            Indent::SUBSECTION,
            "Measurement finished in {} secs.",
            measurement_start_time.elapsed().unwrap().as_secs_f32()
        );

        let formats = match opts.output_format {
            OutputFormat::Vgmo => vec![OutputFileFormatOption::Vgmo {
                encoding: opts.encoding,
                compression: opts.compression,
            }]
            .into_boxed_slice(),
            OutputFormat::Exr => vec![OutputFileFormatOption::Exr {
                resolution: opts.resolution,
            }]
            .into_boxed_slice(),
            OutputFormat::VgmoExr => vec![
                OutputFileFormatOption::Vgmo {
                    encoding: opts.encoding,
                    compression: opts.compression,
                },
                OutputFileFormatOption::Exr {
                    resolution: opts.resolution,
                },
            ]
            .into_boxed_slice(),
        };

        crate::io::write_measured_data_to_file(
            &measured,
            &cache,
            &config,
            OutputOptions {
                dir: opts.output.clone(),
                formats,
            },
        )?;

        // [0.5] orchestration → Phase 1 ProgressEvent::Success (per-task)
        cli_success!(Indent::SUBSECTION, "Done!");
    }

    // [0.5] orchestration → Phase 1 ProgressEvent::Success (run total)
    cli_success!(
        Indent::SUBSECTION,
        "Finished in {:.2} s",
        start_time.elapsed().as_secs_f32()
    );

    Ok(())
}
