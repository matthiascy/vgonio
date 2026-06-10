//! CLI shape for the `measure` subcommand: clap parsing + thread-pool wrapper.
//! Orchestration lives in [`vgn_measurement::orchestration`].

use crate::app::{args::OutputFormat, executor};
use std::{path::PathBuf, sync::Arc};
use vgn_core::{
    cli::{cli_step, Indent},
    config::Config,
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
};
use vgn_executor::{Executor, LocalExecutor, LocalStatusBridge};
use vgn_job_api::{
    artifact::ArtifactRef,
    context::ArtifactHandle,
    envelope::{JobEnvelope, PayloadEncoding, TraceContext},
    ids::{CapabilityId, IdempotencyKey, JobId},
    resources::ResourceHints,
};
use vgn_measurement::request::MeasureRequest;

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

/// Measure different metrics of the micro-surface.
pub fn measure(opts: MeasureOptions, config: Config) -> Result<(), VgonioError> {
    log::info!("{:#?}", config);

    let request = MeasureRequest::from(&opts);
    let config = Arc::new(config);

    // `--nthreads` rides on the envelope's `cpu_cores` hint; the handler
    // (in `app::executor`) installs the rayon pool around the orchestration
    // body. Doing it here would install on the CLI thread, which the
    // executor's worker thread does not inherit, so the user-requested
    // count would not reach the actual `par_iter` calls. `u32 -> u16`
    // saturates because `cpu_cores` is `u16` and any realistic count
    // fits well within that range.
    let cpu_cores = opts.nthreads.map(|n| n.min(u16::MAX as u32) as u16);
    // Banner reflects the eventual pool size: the requested count when
    // `--nthreads` is set, otherwise rayon's global default (typically all
    // available CPUs).
    let pool_size = cpu_cores
        .map(usize::from)
        .unwrap_or_else(rayon::current_num_threads);
    cli_step!(
        Indent::ROOT,
        "Executing 'vgonio measure' with a thread pool of size: {}",
        pool_size
    );
    submit(request, config, cpu_cores)
}

fn submit(
    request: MeasureRequest,
    config: Arc<Config>,
    cpu_cores: Option<u16>,
) -> Result<(), VgonioError> {
    let executor = Arc::new(executor::build_local_executor(config));
    let payload = serde_json::to_vec(&request)
        .map_err(|e| VgonioError::new(format!("Failed to serialize MeasureRequest: {e}"), None))?;
    let envelope = JobEnvelope {
        job_id: JobId::new(),
        protocol_version: vgn_job_api::PROTOCOL_VERSION,
        capability_id: CapabilityId(executor::MEASURE_BATCH_CAPABILITY.into()),
        capability_version: 1,
        payload_encoding: PayloadEncoding::Json,
        payload: payload.into(),
        resources: ResourceHints {
            cpu_cores,
            ..ResourceHints::default()
        },
        inputs: vec![],
        idempotency_key: IdempotencyKey("cli-measure".into()),
        trace: TraceContext::default(),
    };
    let handle = executor
        .submit(envelope)
        .map_err(|e| VgonioError::new(e.to_string(), None))?;
    let job_id = handle.job_id;

    // Forward Ctrl-C to the executor so a long-running measurement drains
    // through cancellation rather than dying mid-write. `ctrlc::set_handler`
    // is process-global and errors on a second registration, so we swallow
    // the error: in test runs that exercise `measure` twice in one process
    // the first registration wins, and that's the correct behaviour.
    let executor_for_signal = Arc::clone(&executor);
    let _ = ctrlc::set_handler(move || {
        let _ = executor_for_signal.cancel(job_id);
    });

    // Render events while waiting on the result. Bridge runs on its own
    // thread so it can drain `events` concurrently with `recv`; join it
    // after the recv so the terminal `Completed` line is fully rendered
    // before we return.
    let bridge = LocalStatusBridge::new();
    let result_rx = handle.result;
    let events_rx = handle.events;
    let bridge_handle = std::thread::spawn(move || bridge.run(events_rx));
    let result = result_rx
        .recv()
        .map_err(|e| VgonioError::new(format!("Failed to receive measure result: {e}"), None))?;
    let _ = bridge_handle.join();
    // Use `e.message`: `JobError`'s Display is `"{code:?}: {message}"`,
    // which would compound with the handler's own context.
    let outcome =
        result.map_err(|e| VgonioError::new(format!("Measure job failed: {}", e.message), None))?;

    // The orchestration publishes outputs through the artifact
    // store rather than writing to `--output` itself. The CLI now owns
    // placement: copy each published blob into `--output` (or report its
    // store path when no `--output` was given).
    lay_out_artifacts(&executor, request.output.as_deref(), &outcome.artifacts)
}

/// Resolves published artifacts and either copies them into `output_dir`
/// (CLI-generated filenames from the artifact kind + a short timestamp) or
/// reports their on-disk store paths when no `--output` was requested.
fn lay_out_artifacts(
    executor: &LocalExecutor,
    output_dir: Option<&std::path::Path>,
    artifacts: &[ArtifactRef],
) -> Result<(), VgonioError> {
    if artifacts.is_empty() {
        return Ok(());
    }
    let store = executor.artifact_store();
    if let Some(dir) = output_dir {
        std::fs::create_dir_all(dir).map_err(|e| {
            VgonioError::from_io_error(e, format!("Failed to create output dir {}", dir.display()))
        })?;
    }
    let stamp = vgn_core::utils::iso_timestamp_short(chrono::Local::now());
    for (i, aref) in artifacts.iter().enumerate() {
        let handle = store.resolve(aref).map_err(|e| {
            VgonioError::new(format!("Failed to resolve artifact: {}", e.message), None)
        })?;
        match output_dir {
            Some(dir) => {
                // Prefer the producer's display_name hint (e.g.
                // "ndf_aluminiummirror_<ts>"); fall back to a generated name.
                let ext = artifact_extension(&aref.kind);
                let filename = match &aref.display_name {
                    Some(name) => format!("{name}.{ext}"),
                    None => format!("measurement_{stamp}_{i:03}.{ext}"),
                };
                let dst = dir.join(&filename);
                match handle {
                    ArtifactHandle::Path(src) => {
                        std::fs::copy(&src, &dst).map_err(|e| {
                            VgonioError::from_io_error(
                                e,
                                format!("Failed to copy artifact to {}", dst.display()),
                            )
                        })?;
                    },
                    ArtifactHandle::Bytes(bytes) => {
                        std::fs::write(&dst, &bytes).map_err(|e| {
                            VgonioError::from_io_error(
                                e,
                                format!("Failed to write artifact to {}", dst.display()),
                            )
                        })?;
                    },
                }
                cli_step!(
                    Indent::DETAIL,
                    "Saved {:?} to \"{}\"",
                    aref.kind,
                    dst.display()
                );
            },
            None => match handle {
                ArtifactHandle::Path(src) => {
                    cli_step!(
                        Indent::DETAIL,
                        "Published {:?} at \"{}\"",
                        aref.kind,
                        src.display()
                    );
                },
                ArtifactHandle::Bytes(_) => {
                    cli_step!(
                        Indent::DETAIL,
                        "Published {:?} (in-memory artifact {})",
                        aref.kind,
                        aref.id.0
                    );
                },
            },
        }
    }
    Ok(())
}

/// Filename extension for a published artifact kind. Used by the CLI to name
/// files copied into `--output` (the orchestration no longer chooses names).
fn artifact_extension(kind: &vgn_job_api::artifact::ArtifactKind) -> &'static str {
    use vgn_job_api::artifact::ArtifactKind::*;
    match kind {
        Vgbsdf => "vgbsdf",
        Vgndf => "vgndf",
        Vgmsf => "vgmsf",
        Vgsdf => "vgsdf",
        Vgms => "vgms",
        Vgmo => "vgmo",
        IorRon => "ior.ron",
        Exr => "exr",
        _ => "bin",
    }
}

/// Options for the `measure` command.
#[derive(clap::Args, Debug)]
#[clap(about = "Measure different aspects of the micro-surface.")]
pub struct MeasureOptions {
    #[arg(
        short,
        long,
        required = true,
        num_args(1..),
        help = "The measurement description files or directories."
    )]
    pub inputs: Vec<PathBuf>,

    #[arg(
        short,
        long,
        help = "The path where stores the simulation data. Use // at the start of the\npath to \
                set the output path relative to the input file location.\nOutput path can also be \
                specified in configuration file."
    )]
    pub output: Option<PathBuf>,

    #[arg(
        short = 'f',
        long,
        default_value_t = OutputFormat::Vgmo,
        help = "The format of the measurement output. If not specified, the format\nwill be the \
                vgonio internal file format.",
    )]
    pub output_format: OutputFormat,

    #[arg(
        short,
        long,
        default_value_t = 512,
        help = "The resolution of the measurement output in case the output is image.\nIf not \
                specified, the resolution will be 512."
    )]
    pub resolution: u32,

    #[arg(
        short,
        long,
        required_if_eq("output_format", "vgms"),
        default_value_t = FileEncoding::Binary,
        help = "Data format for the measurement output.\nOnly used when output format is vgms."
    )]
    pub encoding: FileEncoding,

    #[arg(
    short,
    long,
    required_if_eq("output_format", "vgms"),
    default_value_t = CompressionScheme::None,
    help = "Data compression for the measurement output."
    )]
    pub compression: CompressionScheme,

    #[arg(
        short,
        long = "num-threads",
        help = "The number of threads in the thread pool"
    )]
    pub nthreads: Option<u32>,

    #[clap(
        long,
        help = "Show detailed statistics about memory and time\nusage during the measurement"
    )]
    pub print_stats: bool,
}
