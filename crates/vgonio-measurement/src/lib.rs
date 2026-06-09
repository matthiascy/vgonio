//! Measurement capability: orchestration, request types, per-kind dispatch,
//! and `register_handlers` for the executor registry.
#![feature(adt_const_params)]
#![feature(vec_push_within_capacity)]
#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(decl_macro)]
#![feature(portable_simd)]
#![feature(seek_stream_len)]
#![feature(slice_pattern)]
#![feature(let_chains)]
#![feature(stmt_expr_attributes)]

use std::sync::Arc;
use bytes::Bytes;
use vgn_core::config::Config;
use vgn_executor::CapabilityRegistry;
use vgn_job_api::{
    error::{JobError, JobErrorCode},
    ids::CapabilityId,
};

pub mod backend;
pub mod cache;
pub mod io;
pub mod measurement;
pub mod orchestration;
pub mod request;

pub mod bsdf;
pub mod mfd;
pub mod params;

/// Numerical-robustness gamma factor (PBRT-style), used by the ray-tracing
/// distance bounds.
pub const MACHINE_EPSILON: f32 = f32::EPSILON * 0.5;
pub const fn gamma_f32(n: f32) -> f32 { (n * MACHINE_EPSILON) / (1.0 - n * MACHINE_EPSILON) }

/// Wire ID for the `vgonio measure` batch capability.
pub const MEASURE_BATCH_CAPABILITY: &str = "measure";

/// Registers the `measure` capability handler on `reg`. The handler
/// deserializes a [`request::MeasureRequest`], installs a rayon pool sized to
/// the envelope's `cpu_cores` hint, and runs [`orchestration::run`], mapping a
/// cooperative cancel to [`JobErrorCode::Cancelled`].
pub fn register_handlers(reg: &mut CapabilityRegistry, config: Arc<Config>) {
    let config_for_measure = Arc::clone(&config);
    reg.register(
        CapabilityId(MEASURE_BATCH_CAPABILITY.into()),
        Arc::new(move |envelope, ctx| {
            let req: request::MeasureRequest =
                serde_json::from_slice(&envelope.payload).map_err(|e| JobError {
                    code: JobErrorCode::HandlerError,
                    message: format!("Failed to deserialize MeasureRequest: {e}"),
                    retriable: false,
                    details: None,
                })?;
            let config_clone = Arc::clone(&config_for_measure);
            let cancel = ctx.cancel.clone();
            let run = move || {
                orchestration::run(req, config_clone, ctx)
                    .map(|_| Bytes::new())
                    .map_err(|e| {
                        if cancel.is_cancelled() {
                            JobError {
                                code: JobErrorCode::Cancelled,
                                message: "measurement cancelled".into(),
                                retriable: false,
                                details: None,
                            }
                        } else {
                            JobError {
                                code: JobErrorCode::HandlerError,
                                message: e.to_string(),
                                retriable: false,
                                details: None,
                            }
                        }
                    })
            };
            match envelope.resources.cpu_cores {
                Some(cores) => {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(cores as usize)
                        .build()
                        .map_err(|e| JobError {
                            code: JobErrorCode::HandlerError,
                            message: format!(
                                "Failed to create measurement thread pool with {cores} threads: {e}"
                            ),
                            retriable: false,
                            details: None,
                        })?;
                    pool.install(run)
                },
                None => run(),
            }
        }),
    );
}
