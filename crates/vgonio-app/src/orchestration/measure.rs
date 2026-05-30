//! Measurement orchestration: resolves descriptions, loads surfaces, runs the
//! measurement, and writes output. CLI shape (clap + thread-pool) stays in
//! [`crate::app::cli::cmd_measure`]; Phase 2 moves this module into a
//! capability crate.
//!
//! # Progress reporting (DIST Plan Task 1.15, done)
//!
//! This module is reporter-clean: it emits
//! [`vgn_job_api::progress::Activity`] events through `ctx.progress` rather
//! than calling `vgn_core::cli` (the one ADAPTER print, the `vgonio measure`
//! ROOT banner, lives in `cmd_measure::measure`), so it can move into a
//! capability crate where `vgn_core::cli` is unreachable.
//!
//! Each phase ("Reading measurement description files...", loading IORs /
//! surfaces, the per-kind measurement, writing output) is one
//! [`Activity::PhaseBegin`] + one [`Activity::PhaseEnd`] under a `measure`
//! root phase, keyed on a [`vgn_job_api::progress::PhaseKind`] from the
//! `measure.*` namespace (`measure.read_descriptions`, `measure.load_iors`,
//! `measure.load_surfaces`, `measure.bsdf` / `.ndf` / `.msf` / `.sdf`,
//! `measure.write_output`). Per-surface notes and BSDF receiver details are
//! [`Activity::Message`]s under the surrounding phase. The non-fatal
//! "No micro-surface to measure..." case emits [`Activity::Warning`] and
//! ends the root phase [`vgn_job_api::progress::PhaseOutcome::Skipped`].
//! Fatal errors return `Err(VgonioError)` and surface through
//! [`vgn_job_api::progress::Lifecycle::Failed`], which reconciles any open
//! phase.
//!
//! [`Activity::PhaseBegin`]: vgn_job_api::progress::Activity::PhaseBegin
//! [`Activity::PhaseEnd`]: vgn_job_api::progress::Activity::PhaseEnd
//! [`Activity::Message`]: vgn_job_api::progress::Activity::Message
//! [`Activity::Warning`]: vgn_job_api::progress::Activity::Warning

use crate::{
    app::{args::OutputFormat, cache::Cache, cli::MeasureOptions},
    io::{OutputFileFormatOption, OutputOptions},
    measure,
    measure::params::{MeasurementDescription, MeasurementParams, NdfMeasurementMode},
};
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc, time::Instant};
use vgn_core::{
    config::Config,
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
};
use vgn_job_api::{
    context::JobContext,
    progress::{Activity, PhaseKind, PhaseOutcome},
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

pub fn run(req: MeasureRequest, config: Arc<Config>, ctx: JobContext) -> Result<(), VgonioError> {
    let root_phase = ctx.next_phase_id();
    let root_started = std::time::Instant::now();
    ctx.progress.emit_activity(Activity::PhaseBegin {
        instance: root_phase,
        parent: None,
        kind: PhaseKind::measure_root(),
        label: "Starting measurement orchestration".into(),
        at: chrono::Utc::now(),
    });

    let measurements = {
        let phase = ctx.next_phase_id();
        let started = std::time::Instant::now();
        ctx.progress.emit_activity(Activity::PhaseBegin {
            instance: phase,
            parent: Some(root_phase),
            kind: PhaseKind::measure_read_descriptions(),
            label: "Reading measurement description files".into(),
            at: chrono::Utc::now(),
        });
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
        ctx.progress.emit_activity(Activity::PhaseEnd {
            instance: phase,
            outcome: PhaseOutcome::Ok,
            duration_micros: Some(started.elapsed().as_micros() as u64),
            summary: Some(format!("{} measurement(s)", measurements.len())),
            at: chrono::Utc::now(),
        });
        measurements
    };

    let cache = Cache::new(config.cache_dir());

    let (tasks, num_surfs) = cache.write(|cache| {
        // Load data files: refractive indices, spd etc. if needed.
        if measurements.iter().any(|meas| meas.params.is_bsdf()) {
            let phase = ctx.next_phase_id();
            let started = std::time::Instant::now();
            ctx.progress.emit_activity(Activity::PhaseBegin {
                instance: phase,
                parent: Some(root_phase),
                kind: PhaseKind::measure_load_iors(),
                label: "Loading data files (refractive indices, spd etc.)...".into(),
                at: chrono::Utc::now(),
            });
            cache.load_ior_database(&config);
            ctx.progress.emit_activity(Activity::PhaseEnd {
                instance: phase,
                outcome: PhaseOutcome::Ok,
                duration_micros: Some(started.elapsed().as_micros() as u64),
                summary: Some("Successfully loaded data files".into()),
                at: chrono::Utc::now(),
            });
        }

        let phase = ctx.next_phase_id();
        let started = std::time::Instant::now();
        ctx.progress.emit_activity(Activity::PhaseBegin {
            instance: phase,
            parent: Some(root_phase),
            kind: PhaseKind::measure_load_surfaces(),
            label: "Resolving and loading micro-surfaces...".into(),
            at: chrono::Utc::now(),
        });
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
        ctx.progress.emit_activity(Activity::PhaseEnd {
            instance: phase,
            outcome: PhaseOutcome::Ok,
            duration_micros: Some(started.elapsed().as_micros() as u64),
            summary: Some(format!(
                "{} micro-surface(s) loaded",
                cache.num_micro_surfaces()
            )),
            at: chrono::Utc::now(),
        });

        #[cfg(debug_assertions)]
        cache
            .loaded_micro_surface_paths()
            .unwrap()
            .iter()
            .for_each(|s| {
                ctx.progress.emit_activity(Activity::Message {
                    phase: root_phase,
                    level: 6,
                    text: s.display().to_string(),
                })
            });

        (tasks, cache.num_micro_surfaces())
    });

    if num_surfs == 0 {
        ctx.progress.emit_activity(Activity::Warning {
            phase: root_phase,
            text: "No micro-surface to measure. Exiting...".into(),
        });
        ctx.progress.emit_activity(Activity::PhaseEnd {
            instance: root_phase,
            outcome: PhaseOutcome::Skipped {
                reason: "no micro-surface to measure".into(),
            },
            duration_micros: Some(root_started.elapsed().as_micros() as u64),
            summary: None,
            at: chrono::Utc::now(),
        });
        return Ok(());
    }

    let start_time = Instant::now();
    for (desc, surfaces) in tasks {
        let measurement_start_time = std::time::SystemTime::now();
        // Each per-kind launch banner in this `match` opens a measurement
        // phase (`measure.bsdf` / `.ndf` / `.msf` / `.sdf`); the matching
        // `PhaseEnd` after the measurement carries the "finished in … secs"
        // summary. Receiver details for BSDF ride as `Message`s under the
        // phase.
        // TODO [cli-report worthy]: the actual measurement calls below
        // (measure_bsdf_rt / measure_area_distribution / …) run silently for
        // potentially minutes with no per-progress feedback. They should
        // thread `ctx.progress` in for per-incident-direction / per-surface
        // `Progress` ticks, not just a start banner + final summary.
        let meas_phase = ctx.next_phase_id();
        let meas_started = Instant::now();
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
                ctx.progress.emit_activity(Activity::PhaseBegin {
                    instance: meas_phase,
                    parent: Some(root_phase),
                    kind: PhaseKind::measure_bsdf(),
                    label: format!(
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
                    ),
                    at: chrono::Utc::now(),
                });
                for receiver in &params.receivers {
                    ctx.progress.emit_activity(Activity::Message {
                        phase: meas_phase,
                        level: 0,
                        text: format!(
                            "      + receiver:
        - domain: {}
        - scheme: {:?}
        - precision: {}",
                            receiver.domain, receiver.scheme, receiver.precision
                        ),
                    });
                }
                cache.read(|cache| measure::bsdf::measure_bsdf_rt(params, &surfaces, cache))
            },
            MeasurementParams::Ndf(measurement) => {
                let label = match &measurement.mode {
                    NdfMeasurementMode::ByPoints { azimuth, zenith } => format!(
                        "Measuring microfacet area distribution:
    • parameters:
      + mode: by points
        + azimuth: {}
        + zenith: {}",
                        azimuth.pretty_print(),
                        zenith.pretty_print()
                    ),
                    NdfMeasurementMode::ByPartition { precision } => format!(
                        "Measuring microfacet area distribution:
    • parameters:
       + mode: by partition
           + scheme: Beckers
           + precision: {}",
                        precision.prettified()
                    ),
                };
                ctx.progress.emit_activity(Activity::PhaseBegin {
                    instance: meas_phase,
                    parent: Some(root_phase),
                    kind: PhaseKind::measure_ndf(),
                    label,
                    at: chrono::Utc::now(),
                });
                cache.read(|cache| {
                    measure::mfd::measure_area_distribution(measurement, &surfaces, cache)
                })
            },
            MeasurementParams::Gaf(measurement) => {
                ctx.progress.emit_activity(Activity::PhaseBegin {
                    instance: meas_phase,
                    parent: Some(root_phase),
                    kind: PhaseKind::measure_msf(),
                    label: format!(
                        "Measuring microfacet masking-shadowing function:
    • parameters:
      + azimuth: {}
      + zenith: {}
      + resolution: {} x {}",
                        measurement.azimuth.pretty_print(),
                        measurement.zenith.pretty_print(),
                        measurement.resolution,
                        measurement.resolution
                    ),
                    at: chrono::Utc::now(),
                });

                #[cfg(debug_assertions)]
                log::warn!(
                    "Debug mode is enabled. Measuring MMSF in debug mode is not recommended."
                );
                cache.read(|cache| {
                    measure::mfd::measure_masking_shadowing_function(measurement, &surfaces, cache)
                })
            },
            MeasurementParams::Sdf(params) => {
                ctx.progress.emit_activity(Activity::PhaseBegin {
                    instance: meas_phase,
                    parent: Some(root_phase),
                    kind: PhaseKind::measure_sdf(),
                    label: "Measuring slope distribution function...".into(),
                    at: chrono::Utc::now(),
                });
                cache.read(|cache| {
                    measure::mfd::measure_slope_distribution(&surfaces, params, cache)
                })
            },
        };

        ctx.progress.emit_activity(Activity::PhaseEnd {
            instance: meas_phase,
            outcome: PhaseOutcome::Ok,
            duration_micros: Some(meas_started.elapsed().as_micros() as u64),
            summary: Some(format!(
                "Measurement finished in {} secs.",
                measurement_start_time.elapsed().unwrap().as_secs_f32()
            )),
            at: chrono::Utc::now(),
        });

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

        let write_phase = ctx.next_phase_id();
        let write_started = Instant::now();
        ctx.progress.emit_activity(Activity::PhaseBegin {
            instance: write_phase,
            parent: Some(root_phase),
            kind: PhaseKind::measure_write_output(),
            label: "Writing measurement output...".into(),
            at: chrono::Utc::now(),
        });
        crate::io::write_measured_data_to_file(
            &measured,
            &cache,
            &config,
            OutputOptions {
                dir: req.output.clone(),
                formats,
            },
        )?;
        ctx.progress.emit_activity(Activity::PhaseEnd {
            instance: write_phase,
            outcome: PhaseOutcome::Ok,
            duration_micros: Some(write_started.elapsed().as_micros() as u64),
            summary: Some("Done!".into()),
            at: chrono::Utc::now(),
        });
    }

    ctx.progress.emit_activity(Activity::PhaseEnd {
        instance: root_phase,
        outcome: PhaseOutcome::Ok,
        duration_micros: Some(root_started.elapsed().as_micros() as u64),
        summary: Some(format!("Finished in {:.2} s", start_time.elapsed().as_secs_f32())),
        at: chrono::Utc::now(),
    });

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
        let re_encoded =
            serde_json::to_string(&decoded).expect("decoded MeasureRequest should serialize");
        assert_eq!(encoded, re_encoded);
    }
}
