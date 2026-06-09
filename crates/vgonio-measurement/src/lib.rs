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
use vgn_core::config::Config;
use vgn_executor::CapabilityRegistry;

pub mod backend;
pub mod cache;
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

pub fn register_handlers(_reg: &mut CapabilityRegistry, _config: Arc<Config>) { todo!() }
