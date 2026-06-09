//! Cache for micro-surface profiles and related data.
//!
//! **Status: DIST Plan Task 0.2 complete.** `RawCache` was split into
//! [`compute::ComputeCache`] (compute-only; no `vgn_uxgx`) and
//! [`ui::UiCache`] (renderables + file-picker state, embeds `ComputeCache`).
//! `RawCache` is kept as a migration alias for `ComputeCache`. The audit
//! table below (Task 0.1) is retained as the historical categorization that
//! drove the split.
//!
//! # RawCache UI/compute audit (DIST Plan Task 0.1 — complete)
//!
//! Every field and method of the former `RawCache` is categorized `compute`
//! or `ui` so the Task 0.2 split into `ComputeCache` + `UiCache` does not drag
//! UI deps (`vgn_uxgx::gfx::context::GpuContext`, `RenderableMesh`) into the
//! compute crate.
//!
//! ## Fields
//!
//! | Field | Type | Side | Justification |
//! |---|---|---|---|
//! | `dir` | `PathBuf` | compute | Set in `new()`; cache-directory path read by loaders/external callers. No UI use. |
//! | `iors` | `IorReg` | compute | Only touched by `load_ior_database` / `load_refractive_indices`; used by BSDF refraction compute. |
//! | `records` | `HashMap<Handle, MicroSurfaceRecord>` | compute | `MicroSurfaceRecord` split: compute keeps `{path, surf, mesh}`; surf→renderable moved to `UiCache::renderable_for`. |
//! | `msurfs` | `HashMap<Handle, MicroSurface>` | compute | Private. Core surface data; loaded/queried by compute only. |
//! | `meshes` | `HashMap<Handle, MicroSurfaceMesh>` | compute | `pub(crate)`. Core triangle-mesh data; no GPU. |
//! | `renderables` | `HashMap<Handle, RenderableMesh>` | **ui** | `RenderableMesh` is `vgn_uxgx::gfx`; only created via `GpuContext`. |
//! | `measurements` | `HashMap<Handle, Measurement>` | compute | Private. Measured-data store; loaded/queried by compute only. |
//! | `recent_opened_files` | `Option<Vec<PathBuf>>` | **ui** | GUI file-picker history only. |
//! | `last_opened_dir` | `Option<PathBuf>` | **ui** | GUI file-picker state only. |
//!
//! ## Methods
//!
//! `get_*` / `load_*` / `num_*` / measurement / IOR methods → `ComputeCache`.
//! `create_micro_surface_renderable_mesh`,
//! `get_micro_surface_renderable_mesh*` → `UiCache`.
//! `unload_micro_surface` → `UiCache` (drops its renderable, then delegates
//! the surface/mesh/record removal to `ComputeCache::unload_micro_surface`).
//!
//! ## Task 0.2 outcome
//!
//! 1. `MicroSurfaceRecord` is `{ path, surf, mesh }`; the `renderable` handle became
//!    `UiCache::renderable_for: HashMap<Handle, Handle>`.
//! 2. `UiCache` embeds `ComputeCache` by value behind the single existing `RwLock` (no second lock,
//!    so no compute/UI lock-ordering hazard) and `Deref`s to it, so compute call sites are
//!    unchanged.
//! 3. `unload_micro_surface` lives on `UiCache`, delegating to compute.

pub mod ui;

use std::{
    path::Path,
    sync::{Arc, RwLock},
};

pub use ui::UiCache;
pub use vgn_measurement::cache::ComputeCache;
/// A thread-safe cache. Wraps a [`UiCache`] (which embeds [`ComputeCache`])
/// behind a single `RwLock`. The closure-based `read`/`write` API is
/// preserved; closures receive `&UiCache` / `&mut UiCache`, and `UiCache`
/// `Deref`s to `ComputeCache`.
#[derive(Debug, Clone)]
pub struct Cache(Arc<RwLock<UiCache>>);

impl Cache {
    /// Wraps an already-built [`ComputeCache`] (e.g. one preloaded with the
    /// IOR database) into the shared cache.
    pub fn from_raw(inner: ComputeCache) -> Self {
        Self(Arc::new(RwLock::new(UiCache::new(inner))))
    }

    pub fn new(cache_dir: &Path) -> Self {
        Self(Arc::new(RwLock::new(UiCache::new(ComputeCache::new(
            cache_dir,
        )))))
    }

    pub fn read<R>(&self, reader: impl FnOnce(&UiCache) -> R) -> R {
        let cache = self.0.read().unwrap();
        reader(&cache)
    }

    pub fn write<R>(&self, writer: impl FnOnce(&mut UiCache) -> R) -> R {
        let mut cache = self.0.write().unwrap();
        writer(&mut cache)
    }
}
