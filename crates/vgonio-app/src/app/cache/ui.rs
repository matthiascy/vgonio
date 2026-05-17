//! UI-side cache: GPU-backed renderable meshes and file-picker state, wrapped
//! around the compute cache.
//!
//! [`UiCache`] embeds [`ComputeCache`] by value (not behind a second lock):
//! the whole cache lives under the single `RwLock` held by
//! [`crate::app::cache::Cache`], so there is exactly one lock and no
//! compute/UI lock-ordering hazard. `Deref`/`DerefMut` expose every
//! `ComputeCache` method/field transparently, so compute call sites are
//! unchanged by the split.

use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    path::PathBuf,
};

use vgn_core::{error::VgonioError, res::Handle};
use vgn_uxgx::gfx::{context::GpuContext, mesh::RenderableMesh};

use crate::app::cache::ComputeCache;

#[derive(Debug)]
pub struct UiCache {
    /// Compute-side cache. Owned by value; the surrounding `RwLock` (in
    /// [`crate::app::cache::Cache`]) protects both halves together.
    compute: ComputeCache,
    /// Cache for `RenderableMesh`s, indexed by renderable mesh uuid.
    renderables: HashMap<Handle, RenderableMesh>,
    /// surf uuid -> renderable mesh uuid. Replaces the old
    /// `MicroSurfaceRecord.renderable` field.
    renderable_for: HashMap<Handle, Handle>,
    // TODO: recently files
    /// Cache for recently opened files.
    pub recent_opened_files: Option<Vec<PathBuf>>,
    /// Cache for recently opened directory.
    pub last_opened_dir: Option<PathBuf>,
}

impl Deref for UiCache {
    type Target = ComputeCache;

    fn deref(&self) -> &ComputeCache { &self.compute }
}

impl DerefMut for UiCache {
    fn deref_mut(&mut self) -> &mut ComputeCache { &mut self.compute }
}

impl UiCache {
    pub fn new(compute: ComputeCache) -> Self {
        Self {
            compute,
            renderables: Default::default(),
            renderable_for: Default::default(),
            recent_opened_files: None,
            last_opened_dir: None,
        }
    }

    /// Borrows the compute-side cache.
    pub fn compute(&self) -> &ComputeCache { &self.compute }

    /// Mutably borrows the compute-side cache.
    pub fn compute_mut(&mut self) -> &mut ComputeCache { &mut self.compute }

    /// Creates (or returns the existing) renderable mesh for the given
    /// micro-surface handle.
    pub fn create_micro_surface_renderable_mesh(
        &mut self,
        ctx: &GpuContext,
        msurf: Handle,
    ) -> Result<Handle, VgonioError> {
        log::debug!("Creating renderable mesh for micro-surface: {}", msurf);
        if let Some(&existing) = self.renderable_for.get(&msurf) {
            if existing.is_valid() {
                log::debug!(
                    "Renderable mesh {} already exists for micro-surface: {}",
                    existing,
                    msurf
                );
                return Ok(existing);
            }
        }
        let mesh_hdl = self
            .compute
            .get_micro_surface_record(msurf)
            .ok_or_else(|| {
                VgonioError::new(
                    format!("[Cache] Record for surface {} doesn't exist.", msurf),
                    None,
                )
            })?
            .mesh;
        let mesh = self
            .compute
            .get_micro_surface_mesh(mesh_hdl)
            .ok_or_else(|| {
                VgonioError::new(
                    format!("[Cache] Mesh for surface {} doesn't exist.", msurf),
                    None,
                )
            })?;
        log::trace!("MicroSurfaceMesh of surface {}: {}", msurf, mesh.uuid);
        let (renderable, handle) = RenderableMesh::from_micro_surface_mesh(ctx, mesh);
        self.renderables.insert(handle, renderable);
        self.renderable_for.insert(msurf, handle);
        log::debug!(
            "Updated renderable mesh {} for micro-surface: {}",
            handle,
            msurf
        );
        Ok(handle)
    }

    /// Returns the renderable-mesh handle bound to a micro-surface, if any.
    pub fn renderable_for(&self, surf: Handle) -> Option<Handle> {
        self.renderable_for.get(&surf).copied()
    }

    /// Inserts a renderable and binds it to a surface, returning the old
    /// renderable handle (if any) so the caller can release it. Used by the
    /// subdivision path that rebuilds a surface's mesh.
    pub fn set_surface_renderable(
        &mut self,
        surf: Handle,
        renderable_hdl: Handle,
        renderable: RenderableMesh,
    ) -> Option<Handle> {
        self.renderables.insert(renderable_hdl, renderable);
        self.renderable_for.insert(surf, renderable_hdl)
    }

    /// Drops a renderable mesh by its handle.
    pub fn remove_renderable(&mut self, handle: Handle) -> Option<RenderableMesh> {
        self.renderables.remove(&handle)
    }

    /// Returns a renderable mesh from the cache given its handle.
    pub fn get_micro_surface_renderable_mesh(&self, handle: Handle) -> Option<&RenderableMesh> {
        self.renderables.get(&handle)
    }

    /// Returns the renderable mesh for a micro-surface given the surface
    /// handle.
    pub fn get_micro_surface_renderable_mesh_by_surface_id(
        &self,
        handle: Handle,
    ) -> Option<&RenderableMesh> {
        self.renderable_for
            .get(&handle)
            .and_then(|r| self.renderables.get(r))
    }

    /// Unloads a micro-surface: drops the UI-side renderable first, then
    /// delegates surface/mesh/record removal to the compute cache.
    pub fn unload_micro_surface(&mut self, handle: Handle) -> Result<(), VgonioError> {
        if let Some(renderable) = self.renderable_for.remove(&handle) {
            self.renderables.remove(&renderable);
        }
        self.compute.unload_micro_surface(handle)
    }
}
