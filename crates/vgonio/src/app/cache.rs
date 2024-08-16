use std::{
    any::TypeId,
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    str::FromStr,
};

use uuid::Uuid;

use crate::{
    app::{cli::ansi, Config},
    measure::{params::SurfacePath, Measurement},
};
use base::{
    error::VgonioError, handle::Handle, medium::Medium, optics::ior::RefractiveIndexRegistry, Asset,
};
use gxtk::{context::GpuContext, mesh::RenderableMesh};
use surf::{
    subdivision::Subdivision, HeightOffset, MicroSurface, MicroSurfaceMesh, TriangulationPattern,
};

/// A record inside the cache for a micro-surface.
#[derive(Debug, Clone)]
pub struct MicroSurfaceRecord {
    path: PathBuf,
    pub surf: Handle<MicroSurface>,
    pub mesh: Handle<MicroSurfaceMesh>,
    pub renderable: Handle<RenderableMesh>,
}

impl Hash for MicroSurfaceRecord {
    fn hash<H: Hasher>(&self, state: &mut H) { self.surf.hash(state) }
}

impl PartialEq<Self> for MicroSurfaceRecord {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.surf == other.surf
            && self.mesh == other.mesh
            && self.renderable == other.renderable
    }
}

impl Eq for MicroSurfaceRecord {}

impl MicroSurfaceRecord {
    pub fn name(&self) -> &str {
        self.path
            .file_stem()
            .map(|s| s.to_str().unwrap_or("unknown"))
            .unwrap_or("unknown")
    }

    pub fn path(&self) -> &Path { &self.path }
}

/// Structure for caching intermediate results and data.
/// Also used for managing assets.
#[derive(Debug)]
pub struct RawCache {
    /// Path to the cache directory.
    pub dir: PathBuf,

    /// Refractive index database.
    pub iors: RefractiveIndexRegistry,

    /// Micro-surface record cache, indexed by micro-surface uuid.
    pub records: HashMap<Handle<MicroSurface>, MicroSurfaceRecord>,

    /// Micro-surface cache, indexed by micro-surface uuid.
    msurfs: HashMap<Handle<MicroSurface>, MicroSurface>,

    /// Micro-surface triangle mesh cache, indexed by micro-surface mesh uuid.
    pub(crate) meshes: HashMap<Handle<MicroSurfaceMesh>, MicroSurfaceMesh>,

    /// Cache for `RenderableMesh`s, indexed by renderable mesh uuid.
    pub(crate) renderables: HashMap<Handle<RenderableMesh>, RenderableMesh>,

    /// Cache for measured data.
    measurements: HashMap<Handle<Measurement>, Measurement>,

    // TODO: recently files
    /// Cache for recently opened files.
    pub recent_opened_files: Option<Vec<PathBuf>>,

    /// Cache for recently opened directory.
    pub last_opened_dir: Option<PathBuf>,
}

impl RawCache {
    pub fn new(cache_dir: &Path) -> Self {
        Self {
            dir: cache_dir.to_path_buf(),
            msurfs: Default::default(),
            meshes: Default::default(),
            recent_opened_files: None,
            last_opened_dir: None,
            iors: RefractiveIndexRegistry::default(),
            renderables: Default::default(),
            records: Default::default(),
            measurements: Default::default(),
        }
    }

    /// Returns a micro-surface profile [`MicroSurface`] from the cache given
    /// its handle.
    pub fn get_micro_surface(&self, handle: Handle<MicroSurface>) -> Option<&MicroSurface> {
        self.msurfs.get(&handle)
    }

    /// Returns a micro-surface profile [`MicroSurfaceMesh`] from the cache
    /// given handle to the micro-surface profile.
    pub fn get_micro_surface_mesh_by_surface_id(
        &self,
        handle: Handle<MicroSurface>,
    ) -> Option<&MicroSurfaceMesh> {
        self.records
            .get(&handle)
            .and_then(|record| self.meshes.get(&record.mesh))
    }

    /// Returns a micro-surface profile [`MicroSurfaceMesh`] from the cache
    /// given its handle.
    pub fn get_micro_surface_mesh(
        &self,
        handle: Handle<MicroSurfaceMesh>,
    ) -> Option<&MicroSurfaceMesh> {
        self.meshes.get(&handle)
    }

    /// Creates a micro-surface renderable mesh for the given micro-surface
    /// handle.
    pub fn create_micro_surface_renderable_mesh(
        &mut self,
        ctx: &GpuContext,
        msurf: Handle<MicroSurface>,
    ) -> Result<Handle<RenderableMesh>, VgonioError> {
        log::debug!("Creating renderable mesh for micro-surface: {}", msurf);
        let record = self.records.get_mut(&msurf).ok_or_else(|| {
            VgonioError::new(
                format!("[Cache] Record for surface {} doesn't exist.", msurf),
                None,
            )
        })?;
        if record.renderable.is_valid() {
            log::debug!(
                "Renderable mesh {} already exists for micro-surface: {}",
                record.renderable,
                msurf
            );
            return Ok(record.renderable);
        }
        let mesh = self.meshes.get(&record.mesh).ok_or_else(|| {
            VgonioError::new(
                format!("[Cache] Mesh for surface {} doesn't exist.", msurf),
                None,
            )
        })?;
        log::trace!("MicroSurfaceMesh of surface {}: {}", msurf, mesh.uuid);
        let handle = Handle::new();
        let renderable = RenderableMesh::from_micro_surface_mesh_with_id(ctx, mesh, handle.id);
        self.renderables.insert(handle, renderable);
        record.renderable = handle;
        log::debug!(
            "Updated renderable mesh {} for micro-surface: {}",
            handle,
            msurf
        );
        log::trace!("Renderable meshes: {:#?}", self.renderables);
        Ok(handle)
    }

    pub fn get_micro_surface_record(
        &self,
        handle: Handle<MicroSurface>,
    ) -> Option<&MicroSurfaceRecord> {
        self.records.get(&handle)
    }

    pub fn get_micro_surface_records<'a, T>(&self, handles: T) -> Vec<MicroSurfaceRecord>
    where
        T: Iterator<Item = &'a Handle<MicroSurface>>,
    {
        handles
            .filter_map(|hdl| self.records.get(hdl))
            .cloned()
            .collect()
    }

    /// Returns a micro-surface profile [`RenderableMesh`] from the cache
    /// given handle to the micro-surface profile.
    pub fn get_micro_surface_renderable_mesh_by_surface_id(
        &self,
        handle: Handle<MicroSurface>,
    ) -> Option<&RenderableMesh> {
        self.records
            .get(&handle)
            .and_then(|record| self.renderables.get(&record.renderable))
    }

    /// Returns a micro-surface profile [`RenderableMesh`] from the cache
    /// given its handle.
    pub fn get_micro_surface_renderable_mesh(
        &self,
        handle: Handle<RenderableMesh>,
    ) -> Option<&RenderableMesh> {
        self.renderables.get(&handle)
    }

    /// Returns the file path to the micro-surface profile given its handle.
    pub fn get_micro_surface_filepath(&self, handle: Handle<MicroSurface>) -> Option<&Path> {
        self.records.get(&handle).map(|r| r.path.as_path())
    }

    /// Returns the number of loaded micro-surface profiles.
    pub fn num_micro_surfaces(&self) -> usize { self.msurfs.len() }

    /// Returns a list of micro-surface profiles [`MicroSurface`] from the
    /// cache, given a list of handles to the micro-surface profiles.
    pub fn get_micro_surfaces(
        &self,
        handles: &[Handle<MicroSurface>],
    ) -> Vec<Option<&MicroSurface>> {
        handles.iter().map(|h| self.msurfs.get(h)).collect()
    }

    /// Returns a list of micro-surface meshes [`MicroSurfaceMesh`] from the
    /// cache given a list of handles to the micro-surface profiles.
    pub fn get_micro_surface_meshes_by_surfaces(
        &self,
        handles: &[Handle<MicroSurface>],
    ) -> Vec<Option<&MicroSurfaceMesh>> {
        handles
            .iter()
            .map(|h| self.get_micro_surface_mesh_by_surface_id(*h))
            .collect()
    }

    /// Returns a list of micro-surface meshes [`MicroSurfaceMesh`] from the
    /// cache given its handles.
    pub fn get_micro_surface_meshes(
        &self,
        handles: &[Handle<MicroSurfaceMesh>],
    ) -> Vec<Option<&MicroSurfaceMesh>> {
        handles.iter().map(|h| self.meshes.get(h)).collect()
    }

    /// Returns a list of micro-surface profiles' file paths from the cache
    /// given a list of handles to the micro-surface profiles.
    pub fn get_micro_surface_filepaths(
        &self,
        handles: &[Handle<MicroSurface>],
    ) -> Option<Vec<&Path>> {
        handles
            .iter()
            .map(|handle| self.get_micro_surface_filepath(*handle))
            .collect()
    }

    /// Returns a list of loaded micro-surface profiles' file paths from the
    /// cache.
    pub fn loaded_micro_surface_paths(&self) -> Option<Vec<&Path>> {
        self.msurfs
            .keys()
            .map(|uuid| self.msurfs.get(uuid).unwrap().path.as_deref())
            .collect()
    }

    pub fn get_measurement(&self, handle: Handle<Measurement>) -> Option<&Measurement> {
        self.measurements.get(&handle)
    }

    /// Loads a surface from its relevant place and returns its cache handle.
    pub fn load_micro_surface(
        &mut self,
        config: &Config,
        path: &Path,
        subdiv: Option<Subdivision>,
    ) -> Result<(Handle<MicroSurface>, Handle<MicroSurfaceMesh>), VgonioError> {
        match config.resolve_path(path) {
            None => Err(VgonioError::new(
                format!(
                    "Failed to resolve micro-surface file path: \"{}\"",
                    path.display()
                ),
                None,
            )),
            Some(filepath) => {
                if let Some((msurf_id, record)) = self
                    .records
                    .iter()
                    .find(|(_, record)| record.path == filepath)
                {
                    log::debug!("-- already loaded: {}", filepath.display());
                    Ok((*msurf_id, record.mesh))
                } else {
                    log::debug!("-- loading: {}", filepath.display());
                    let msurf = MicroSurface::read_from_file(&filepath, None)?;
                    let msurf_hdl = Handle::with_id(msurf.uuid);
                    let mesh = msurf.as_micro_surface_mesh(
                        HeightOffset::Grounded,
                        config.user.triangulation,
                        subdiv,
                    );
                    let mesh_hdl = Handle::with_id(mesh.uuid);
                    self.msurfs.insert(msurf_hdl, msurf);
                    self.meshes.insert(mesh_hdl, mesh);
                    self.records.insert(
                        msurf_hdl,
                        MicroSurfaceRecord {
                            path: filepath,
                            surf: msurf_hdl,
                            mesh: mesh_hdl,
                            renderable: Handle::invalid(),
                        },
                    );
                    Ok((msurf_hdl, mesh_hdl))
                }
            }
        }
    }

    /// Unloads a micro-surface from the cache.
    ///
    /// This will remove the micro-surface from the cache, including surface
    /// mesh and renderable mesh.
    pub fn unload_micro_surface(
        &mut self,
        handle: Handle<MicroSurface>,
    ) -> Result<(), VgonioError> {
        let record = self.records.get(&handle).ok_or_else(|| {
            VgonioError::new(format!("Failed to unload micro-surface: {}", handle), None)
        })?;
        self.msurfs.remove(&handle);
        self.meshes.remove(&record.mesh);
        self.renderables.remove(&record.renderable);
        self.records.remove(&handle);
        Ok(())
    }

    /// Adds micro-surface measurement data to the cache.
    pub fn add_micro_surface_measurement(
        &mut self,
        data: Measurement,
    ) -> Result<Handle<Measurement>, VgonioError> {
        let handle = Handle::with_type_id(data.measured.kind() as u8);
        self.measurements.insert(handle, data);
        Ok(handle)
    }

    /// Loads a micro-surface measurement data from the given path and returns
    /// its cache handle.
    pub fn load_micro_surface_measurement(
        &mut self,
        config: &Config,
        path: &Path,
    ) -> Result<Handle<Measurement>, VgonioError> {
        match config.resolve_path(path) {
            None => Err(VgonioError::new(
                "Failed to resolve measurement file.",
                None,
            )),
            Some(filepath) => {
                if let Some((hdl, _)) = self
                    .measurements
                    .iter()
                    .find(|(_, d)| d.source.path() == Some(&filepath))
                {
                    log::debug!("-- already loaded: {}", filepath.display());
                    Ok(*hdl)
                } else {
                    log::debug!("-- loading: {}", filepath.display());
                    let data = Measurement::read_from_file(&filepath)?;
                    let handle = Handle::with_type_id(data.measured.kind() as u8);
                    self.measurements.insert(handle, data);
                    Ok(handle)
                }
            }
        }
    }

    /// Loads surfaces from their relevant places and returns
    /// their cache handles.
    ///
    /// Note: this function automatically triangulates the surfaces.
    ///
    /// # Arguments
    ///
    /// * `config` - The application configuration.
    ///
    /// * `paths` - Paths to the surfaces to be load. Paths may not in canonical
    ///   form.
    ///
    /// * `alignment` - The axis alignment when constructing the surface mesh.
    pub fn load_micro_surfaces(
        &mut self,
        config: &Config,
        paths: &[SurfacePath],
        pattern: TriangulationPattern,
    ) -> Result<Vec<Handle<MicroSurface>>, VgonioError> {
        log::info!("Loading micro surfaces from {:?}", paths);
        let canonical = paths
            .iter()
            .filter_map(|s| {
                config.resolve_path(&s.path).map(|path| SurfacePath {
                    path,
                    subdivision: s.subdivision,
                })
            })
            .collect::<Vec<_>>();
        log::debug!("-- canonical paths: {:?}", canonical);
        let mut loaded = vec![];
        for surf in canonical {
            if surf.path.exists() {
                let files_to_load = {
                    if surf.path.is_dir() {
                        surf.path
                            .read_dir()
                            .unwrap()
                            .map(|entry| (entry.unwrap().path(), surf.subdivision))
                            .collect::<Box<_>>()
                    } else {
                        vec![(surf.path, surf.subdivision)].into_boxed_slice()
                    }
                };
                for (filepath, subdivision) in files_to_load {
                    if let Some((msurf_id, _)) = self
                        .records
                        .iter()
                        .find(|(_, record)| record.path == filepath)
                    {
                        log::debug!("-- already loaded: {}", filepath.display());
                        loaded.push(*msurf_id);
                    } else {
                        log::debug!("-- loading: {}", filepath.display());
                        let msurf = MicroSurface::read_from_file(&filepath, None).unwrap();
                        let msurf_hdl = Handle::with_id(msurf.uuid);
                        let mesh = msurf.as_micro_surface_mesh(
                            HeightOffset::Grounded,
                            pattern,
                            subdivision,
                        );
                        let mesh_hdl = Handle::new();
                        self.msurfs.insert(msurf_hdl, msurf);
                        self.meshes.insert(mesh_hdl, mesh);
                        self.records.insert(
                            msurf_hdl,
                            MicroSurfaceRecord {
                                path: filepath,
                                surf: msurf_hdl,
                                mesh: mesh_hdl,
                                renderable: Handle::invalid(),
                            },
                        );
                        loaded.push(msurf_hdl);
                    }
                }
            } else {
                eprintln!(
                    "    {}!{} file not found: {}",
                    ansi::BRIGHT_RED,
                    ansi::RESET,
                    surf.path.display()
                );
            }
        }
        log::debug!("- loaded micro surfaces: {:?}", loaded);
        Ok(loaded)
    }

    // TODO: move to RefractiveIndexRegistry
    /// Loads the refractive index database from the paths specified in the
    /// configuration.
    ///
    /// The database is loaded from the following paths:
    /// - `VgonioConfig::sys_data_dir()`
    /// - `VgonioConfig::user_data_dir()`
    pub fn load_ior_database(&mut self, config: &Config) {
        log::debug!("Loading refractive index database ...");
        // let mut database = RefractiveIndexDatabase::new();
        let sys_path: PathBuf = config.sys_data_dir().to_path_buf().join("ior");
        let user_path = config
            .user_data_dir()
            .map(|path| path.to_path_buf().join("ior"));
        log::debug!("  -- sys_path: {:?}", sys_path);
        log::debug!("  -- user_path: {:?}", user_path);

        if !sys_path.exists() {
            log::debug!("  Refractive index database not found at {:?}", sys_path);
        }

        // First load refractive indices from `VgonioConfig::sys_data_dir()`.
        let n = Self::load_refractive_indices(&mut self.iors, &sys_path);
        log::debug!("  Loaded {} ior files from {:?}", n, sys_path);

        if !user_path.as_ref().is_some_and(|p| p.exists()) {
            log::debug!("  Refractive index database not found at {:?}", user_path);
        }

        let n = Self::load_refractive_indices(&mut self.iors, user_path.as_ref().unwrap());
        log::debug!("  Loaded {} ior files from {:?}", n, user_path);
    }

    // TODO: move to RefractiveIndexRegistry
    /// Load the refractive index database from the given path.
    /// Returns the number of files loaded.
    fn load_refractive_indices(iors: &mut RefractiveIndexRegistry, path: &Path) -> u32 {
        let mut n_files = 0;
        if path.is_file() {
            log::debug!("Loading refractive index database from {:?}", path);
            let medium = Medium::from_str(
                path.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .split('_')
                    .next()
                    .unwrap(),
            )
            .unwrap();
            // TODO: make this a method of RefractiveIndexRegistry
            let refractive_indices = RefractiveIndexRegistry::read_iors_from_file(path).unwrap();
            let iors = iors.entry(medium).or_default();
            for ior in refractive_indices.iter() {
                if !iors.contains(&ior) {
                    iors.push(*ior);
                }
            }
            iors.sort_by(|a, b| a.wavelength.partial_cmp(&b.wavelength).unwrap());
            n_files += 1;
        } else if path.is_dir() {
            for entry in path.read_dir().unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                n_files += Self::load_refractive_indices(iors, &path);
            }
        }

        n_files
    }
}

/// A thread-safe cache. This is a wrapper around `RawCache`.
#[derive(Debug, Clone)]
pub struct Cache(std::sync::Arc<std::sync::RwLock<RawCache>>);

impl Cache {
    pub fn from_raw(inner: RawCache) -> Self {
        Self(std::sync::Arc::new(std::sync::RwLock::new(inner)))
    }

    pub fn new(cache_dir: &Path) -> Self {
        Self(std::sync::Arc::new(std::sync::RwLock::new(RawCache::new(
            cache_dir,
        ))))
    }

    pub fn read<R>(&self, reader: impl FnOnce(&RawCache) -> R) -> R {
        let cache = self.0.read().unwrap();
        reader(&cache)
    }

    pub fn write<R>(&self, writer: impl FnOnce(&mut RawCache) -> R) -> R {
        let mut cache = self.0.write().unwrap();
        writer(&mut cache)
    }
}
