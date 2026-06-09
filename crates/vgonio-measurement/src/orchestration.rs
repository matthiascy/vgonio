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
    bsdf,
    cache::ComputeCache,
    io::{write_measured_data_to_file, OutputFileFormatOption, OutputOptions},
    mfd,
    params::{MeasurementDescription, MeasurementParams, NdfMeasurementMode},
    request::MeasureRequest,
};
use std::{sync::Arc, time::Instant};
use vgn_core::{config::Config, error::VgonioError, io::OutputFormat};
use vgn_job_api::{
    context::JobContext,
    progress::{Activity, PhaseKind, PhaseOutcome},
};

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

    let mut cache = ComputeCache::new(config.cache_dir());

    let (tasks, num_surfs) = {
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
    };

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
        // Catch cancellation between measurements: if the user Ctrl-C'd while
        // the previous one was writing out, drop the remaining queue rather
        // than launching another Embree run.
        ctx.cancel
            .check()
            .map_err(|e| VgonioError::new(e.message, None))?;
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
                bsdf::measure_bsdf_rt(params, &surfaces, &cache, &ctx.cancel)
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
                mfd::measure_area_distribution(measurement, &surfaces, &cache)
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
                mfd::measure_masking_shadowing_function(measurement, &surfaces, &cache)
            },
            MeasurementParams::Sdf(params) => {
                ctx.progress.emit_activity(Activity::PhaseBegin {
                    instance: meas_phase,
                    parent: Some(root_phase),
                    kind: PhaseKind::measure_sdf(),
                    label: "Measuring slope distribution function...".into(),
                    at: chrono::Utc::now(),
                });
                mfd::measure_slope_distribution(&surfaces, params, &cache)
            },
        };

        // `measure_*_rt` returns whatever it had accumulated when it observed
        // the cancel flag; if that happened mid-run the partial slice is not
        // a publishable result. Bail before the write phase so we don't land
        // truncated data on disk.
        ctx.cancel
            .check()
            .map_err(|e| VgonioError::new(e.message, None))?;

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
        write_measured_data_to_file(
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
        summary: Some(format!(
            "Finished in {:.2} s",
            start_time.elapsed().as_secs_f32()
        )),
        at: chrono::Utc::now(),
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::MeasureRequest;
    use std::path::PathBuf;
    use vgn_core::io::{CompressionScheme, FileEncoding, OutputFormat};

    // NOTE: the `From<&MeasureOptions> for MeasureRequest`
    // conversion and its `measure_request_from_options_mirrors_runtime_fields`
    // test moved app-side with `MeasureOptions` (clap-shaped). The conversion
    // lives in `vgonio-app::app::cli::cmd_measure`; its mirroring test belongs
    // in `vgonio-app`.

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

    // ------------------------------------------------------------------
    // End-to-end: the whole `measure` pipeline on a generated flat surface,
    // driven through the real local executor (the same path `cmd_measure`
    // uses). Builds a flat micro-surface + a BSDF description on disk, submits
    // a `MeasureRequest`, and asserts it drives parse -> load -> embree trace
    // -> receiver collect -> write and lands a `.vgmo`. Uses `vac`/`vac` media
    // so no IOR database is required. Mirrors the manual CLI run.
    // ------------------------------------------------------------------
    #[cfg(feature = "embree")]
    #[test]
    fn measure_pipeline_writes_vgmo_for_flat_surface() {
        use super::run;
        use std::sync::{atomic::AtomicU32, Arc};
        use vgn_core::{
            config::{Config, UserConfig},
            units::LengthUnit,
            TriangulationPattern,
        };
        use vgn_io::MicroSurface;
        use vgn_job_api::{
            artifact::{ArtifactKind, ArtifactRef},
            context::{
                ArtifactHandle, ArtifactStore, CancellationToken, JobContext, ProgressSender,
            },
            envelope::TraceContext,
            error::JobError,
            ids::{IdempotencyKey, JobId},
        };

        // The measure orchestration never touches the artifact store, so a
        // panicking stub is safe and keeps the test off the executor's plumbing.
        #[derive(Debug)]
        struct NoArtifacts;
        impl ArtifactStore for NoArtifacts {
            fn resolve(&self, _: &ArtifactRef) -> Result<ArtifactHandle, JobError> {
                unimplemented!("measure does not resolve artifacts")
            }
            fn publish(&self, _: ArtifactKind, _: bytes::Bytes) -> Result<ArtifactRef, JobError> {
                unimplemented!("measure does not publish artifacts")
            }
        }

        // The media registry is a process-wide OnceLock the real app installs
        // at startup; bootstrap it here so the description's medium tokens
        // parse. Ignore `AlreadyInitialized` if another test got there first.
        let _ = vgn_core::utils::medium::bootstrap(None, None);

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let cache_dir = root.join("cache");
        let out_dir = root.join("out");
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::fs::create_dir_all(&out_dir).unwrap();

        // Flat micro-surface on disk.
        let surf_path = root.join("flat.vgms");
        MicroSurface::new(64, 64, 0.1, 0.1, 0.0, LengthUnit::UM)
            .write_to_file(&surf_path, FileEncoding::Binary, CompressionScheme::None)
            .unwrap();

        // BSDF description on disk. `vac`/`vac` => no IOR database; `num_rays`
        // is not a multiple of the ray-stream size.
        let desc_path = root.join("brdf.yml");
        std::fs::write(
            &desc_path,
            format!(
                r#"---
type: !bsdf
  kind: brdf
  sim_kind: !geom-optics embree
  incident_medium: vac
  transmitted_medium: vac
  fresnel: false
  emitter:
    num_rays: 512
    num_sectors: 1
    max_bounces: 4
    zenith: 0deg .. =30deg / 30deg
    azimuth: 0deg .. =360deg / 120deg
    spectrum: 400 nm .. =400 nm / 100 nm
  receivers:
    - domain: upper_hemisphere
      precision:
        theta: 2.0 deg
        phi: 2.0 deg
      scheme: beckers
surfaces:
  - {surface}
"#,
                surface = surf_path.display()
            ),
        )
        .unwrap();

        let config = Config {
            sys_config_dir: root.to_path_buf(),
            sys_cache_dir: cache_dir.clone(),
            sys_data_dir: root.join("data"),
            cwd: root.to_path_buf(),
            user: UserConfig {
                cache_dir: Some(cache_dir),
                output_dir: Some(out_dir.clone()),
                data_dir: None,
                triangulation: TriangulationPattern::default(),
                excluded_ior_files: None,
            },
        };

        let request = MeasureRequest {
            inputs: vec![desc_path],
            output: Some(out_dir.clone()),
            output_format: OutputFormat::Vgmo,
            resolution: 64,
            encoding: FileEncoding::Binary,
            compression: CompressionScheme::None,
        };

        let (tx, _events) = std::sync::mpsc::channel();
        let ctx = JobContext {
            job_id: JobId::new(),
            idempotency_key: IdempotencyKey("test-measure".into()),
            progress: ProgressSender::new(tx),
            phase_counter: Arc::new(AtomicU32::new(0)),
            cancel: CancellationToken::new(),
            artifacts: Arc::new(NoArtifacts),
            trace: TraceContext::default(),
            deadline: None,
        };

        run(request, Arc::new(config), ctx).expect("measure run should succeed");

        let wrote_vgmo = std::fs::read_dir(&out_dir)
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.path().extension().is_some_and(|ext| ext == "vgmo"));
        assert!(
            wrote_vgmo,
            "expected a .vgmo output in {}",
            out_dir.display()
        );
    }
}
