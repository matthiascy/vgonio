//! CLI shape for the `measure` subcommand: clap parsing + thread-pool wrapper.
//! Orchestration lives in [`crate::orchestration::measure`].

use crate::{
    app::{args::OutputFormat, executor},
    orchestration::measure::MeasureRequest,
};
use std::{path::PathBuf, sync::Arc};
use vgn_core::{
    cli::{cli_step, Indent},
    config::Config,
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
};
use vgn_executor::{Executor, LocalStatusBridge};
use vgn_job_api::{
    envelope::{JobEnvelope, PayloadEncoding, TraceContext},
    ids::{CapabilityId, IdempotencyKey, JobId},
    resources::ResourceHints,
};

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
    let cpu_cores = opts
        .nthreads
        .map(|n| n.min(u16::MAX as u32) as u16);
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
    let outcome = result_rx
        .recv()
        .map_err(|e| VgonioError::new(format!("Failed to receive measure result: {e}"), None))?
        .map(|_| ())
        // Use `e.message`: `JobError`'s Display is `"{code:?}: {message}"`,
        // which would compound with the handler's own context.
        .map_err(|e| VgonioError::new(format!("Measure job failed: {}", e.message), None));
    let _ = bridge_handle.join();
    outcome
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
