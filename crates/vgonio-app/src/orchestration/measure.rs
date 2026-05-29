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
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc, time::Instant};
use vgn_core::{
    cli::{cli_error, cli_note, cli_step, cli_success, Indent},
    config::Config,
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
};

/// Top-level capability request for `vgonio measure`. Mirrors the runtime
/// fields of [`MeasureOptions`] in a serde-friendly form.
///
/// Phase 1 treats the whole CLI invocation as one batched capability call:
/// the orchestration loads every description from `inputs`, resolves their
/// surfaces, dispatches per-kind internally, and writes outputs. Phase 2
/// will split per-measurement-kind (BSDF / NDF / MSF / SDF), at which point
/// `cmd_measure` will emit one envelope per description instead.
///
/// CLI-shell concerns (`nthreads`, `print_stats`) do not belong on the
/// request: rayon pool sizing is set up at the adapter before submission
/// and `print_stats` is presently unused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureRequest {
    /// Measurement description files (or directories containing them).
    pub inputs: Vec<PathBuf>,
    /// Output directory. `None` defers to the configuration's default.
    pub output: Option<PathBuf>,
    /// Selects which file format(s) the writer emits.
    pub output_format: OutputFormat,
    /// Resolution for image-shaped outputs (EXR; the Vgbsdf disc grid).
    pub resolution: u32,
    /// On-disk encoding for the Vgmo container.
    pub encoding: FileEncoding,
    /// On-disk compression for the Vgmo container.
    pub compression: CompressionScheme,
}

impl From<&MeasureOptions> for MeasureRequest {
    fn from(opts: &MeasureOptions) -> Self {
        Self {
            inputs: opts.inputs.clone(),
            output: opts.output.clone(),
            output_format: opts.output_format,
            resolution: opts.resolution,
            encoding: opts.encoding,
            compression: opts.compression,
        }
    }
}

pub fn run(req: MeasureRequest, config: Arc<Config>) -> Result<(), VgonioError> {
    // [0.5] ADAPTER print moved to `cmd_measure::measure` (the ROOT
    // "Executing 'vgonio measure' …" banner). Nothing prints here anymore.

    // [0.5] orchestration → Phase 1 ProgressEvent::Step
    cli_step!(Indent::SECTION, "Reading measurement description files...");
    let measurements = req
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

        let formats = match req.output_format {
            OutputFormat::Vgmo => vec![OutputFileFormatOption::Vgmo {
                encoding: req.encoding,
                compression: req.compression,
            }]
            .into_boxed_slice(),
            OutputFormat::Exr => vec![OutputFileFormatOption::Exr {
                resolution: req.resolution,
            }]
            .into_boxed_slice(),
            OutputFormat::VgmoExr => vec![
                OutputFileFormatOption::Vgmo {
                    encoding: req.encoding,
                    compression: req.compression,
                },
                OutputFileFormatOption::Exr {
                    resolution: req.resolution,
                },
            ]
            .into_boxed_slice(),
            OutputFormat::Vgbsdf => vec![OutputFileFormatOption::Vgbsdf {
                disc_res: req.resolution,
            }]
            .into_boxed_slice(),
        };

        crate::io::write_measured_data_to_file(
            &measured,
            &cache,
            &config,
            OutputOptions {
                dir: req.output.clone(),
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

#[cfg(test)]
mod tests {
    use super::MeasureRequest;
    use crate::app::{args::OutputFormat, cli::MeasureOptions};
    use std::path::PathBuf;
    use vgn_core::io::{CompressionScheme, FileEncoding};

    /// Returns a `MeasureOptions` with a non-default value for every runtime
    /// field. CLI-shell fields (`nthreads`, `print_stats`) are also set so
    /// the test can assert they do NOT leak onto the request.
    fn fully_populated_options() -> MeasureOptions {
        MeasureOptions {
            inputs: vec![PathBuf::from("a.yaml"), PathBuf::from("b.yaml")],
            output: Some(PathBuf::from("/out/dir")),
            output_format: OutputFormat::VgmoExr,
            resolution: 1024,
            encoding: FileEncoding::Ascii,
            compression: CompressionScheme::Zlib,
            nthreads: Some(8),
            print_stats: true,
        }
    }

    fn fully_populated_request() -> MeasureRequest {
        MeasureRequest {
            inputs: vec![PathBuf::from("a.yaml"), PathBuf::from("b.yaml")],
            output: Some(PathBuf::from("/out/dir")),
            output_format: OutputFormat::VgmoExr,
            resolution: 1024,
            encoding: FileEncoding::Ascii,
            compression: CompressionScheme::Zlib,
        }
    }

    // ------------------------------------------------------------------
    // From<&MeasureOptions> for MeasureRequest: every runtime field on
    // the options struct must be mirrored onto the request, and the
    // CLI-shell-only fields must NOT leak onto the request. When a new
    // field is added to `MeasureOptions`, update both `fully_populated_*`
    // helpers and the assertions below.
    // ------------------------------------------------------------------

    #[test]
    fn measure_request_from_options_mirrors_runtime_fields() {
        let opts = fully_populated_options();
        let req = MeasureRequest::from(&opts);
        assert_eq!(req.inputs, opts.inputs);
        assert_eq!(req.output, opts.output);
        assert_eq!(req.output_format, opts.output_format);
        assert_eq!(req.resolution, opts.resolution);
        assert_eq!(req.encoding, opts.encoding);
        assert_eq!(req.compression, opts.compression);
    }

    // ------------------------------------------------------------------
    // serde round-trip: the request crosses a JSON wire on every
    // envelope, so any `#[serde(rename_all = ...)]` change on a nested
    // enum (OutputFormat, FileEncoding, CompressionScheme) that breaks
    // decode would silently break envelopes. Encode + decode + re-encode
    // and compare; structural equality via re-encoded bytes avoids
    // needing PartialEq on the request, matching the FitRequest pattern.
    // ------------------------------------------------------------------

    #[test]
    fn measure_request_serde_round_trip() {
        let req = fully_populated_request();
        let encoded = serde_json::to_string(&req).expect("MeasureRequest should serialize");
        let decoded: MeasureRequest =
            serde_json::from_str(&encoded).expect("MeasureRequest should deserialize");
        let re_encoded = serde_json::to_string(&decoded)
            .expect("decoded MeasureRequest should serialize");
        assert_eq!(encoded, re_encoded);
    }
}
