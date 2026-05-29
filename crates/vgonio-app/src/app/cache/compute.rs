//! Compute-side cache: micro-surface profiles, meshes, records, measured data
//! and the refractive-index database. Holds no GPU/UI state and must never
//! import `vgn_uxgx` (see DIST Plan Task 0.2).

use crate::measure::{params::SurfacePath, Measurement};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    str::FromStr,
};
use vgn_core::{
    cli::ansi, config::Config, error::VgonioError, optics::IorReg, res::Handle,
    TriangulationPattern,
};
use vgn_io::{subdivision::Subdivision, HeightOffset, MicroSurface, MicroSurfaceMesh};

/// Compute-side record of a micro-surface. The former `renderable` handle moved
/// to [`crate::app::cache::ui::UiCache::renderable_for`].
#[derive(Clone, Debug)]
pub struct MicroSurfaceRecord {
    path: PathBuf,
    pub surf: Handle,
    pub mesh: Handle,
}

impl Hash for MicroSurfaceRecord {
    fn hash<H: Hasher>(&self, state: &mut H) { self.surf.hash(state) }
}

impl PartialEq<Self> for MicroSurfaceRecord {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.surf == other.surf && self.mesh == other.mesh
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

/// Cache for compute-side intermediate results and managed assets.
#[derive(Debug)]
pub struct ComputeCache {
    /// Path to the cache directory.
    pub dir: PathBuf,

    /// Refractive index database.
    pub iors: IorReg,

    /// Micro-surface record cache, indexed by micro-surface uuid.
    pub records: HashMap<Handle, MicroSurfaceRecord>,

    /// Micro-surface cache, indexed by micro-surface uuid.
    msurfs: HashMap<Handle, MicroSurface>,

    /// Micro-surface triangle mesh cache, indexed by micro-surface mesh uuid.
    pub(crate) meshes: HashMap<Handle, MicroSurfaceMesh>,

    /// Cache for measured data.
    measurements: HashMap<Handle, Measurement>,
}

impl ComputeCache {
    pub fn new(cache_dir: &Path) -> Self {
        Self {
            dir: cache_dir.to_path_buf(),
            iors: IorReg::default(),
            records: Default::default(),
            msurfs: Default::default(),
            meshes: Default::default(),
            measurements: Default::default(),
        }
    }

    /// Returns a micro-surface profile [`MicroSurface`] from the cache given
    /// its handle.
    pub fn get_micro_surface(&self, handle: Handle) -> Option<&MicroSurface> {
        self.msurfs.get(&handle)
    }

    /// Returns a micro-surface profile [`MicroSurfaceMesh`] from the cache
    /// given handle to the micro-surface profile.
    pub fn get_micro_surface_mesh_by_surface_id(
        &self,
        handle: Handle,
    ) -> Option<&MicroSurfaceMesh> {
        self.records
            .get(&handle)
            .and_then(|record| self.meshes.get(&record.mesh))
    }

    /// Returns a micro-surface profile [`MicroSurfaceMesh`] from the cache
    /// given its handle.
    pub fn get_micro_surface_mesh(&self, handle: Handle) -> Option<&MicroSurfaceMesh> {
        self.meshes.get(&handle)
    }

    pub fn get_micro_surface_record(&self, handle: Handle) -> Option<&MicroSurfaceRecord> {
        self.records.get(&handle)
    }

    pub fn get_micro_surface_records<'a, T>(&self, handles: T) -> Vec<MicroSurfaceRecord>
    where
        T: Iterator<Item = &'a Handle>,
    {
        handles
            .filter_map(|hdl| self.records.get(hdl))
            .cloned()
            .collect()
    }

    /// Returns the file path to the micro-surface profile given its handle.
    pub fn get_micro_surface_filepath(&self, handle: Handle) -> Option<&Path> {
        self.records.get(&handle).map(|r| r.path.as_path())
    }

    /// Returns the number of loaded micro-surface profiles.
    pub fn num_micro_surfaces(&self) -> usize { self.msurfs.len() }

    /// Returns a list of micro-surface profiles [`MicroSurface`] from the
    /// cache, given a list of handles to the micro-surface profiles.
    pub fn get_micro_surfaces(&self, handles: &[Handle]) -> Vec<Option<&MicroSurface>> {
        handles.iter().map(|h| self.msurfs.get(h)).collect()
    }

    /// Returns a list of micro-surface meshes [`MicroSurfaceMesh`] from the
    /// cache given a list of handles to the micro-surface profiles.
    pub fn get_micro_surface_meshes_by_surfaces(
        &self,
        handles: &[Handle],
    ) -> Vec<Option<&MicroSurfaceMesh>> {
        handles
            .iter()
            .map(|h| self.get_micro_surface_mesh_by_surface_id(*h))
            .collect()
    }

    /// Returns a list of micro-surface meshes [`MicroSurfaceMesh`] from the
    /// cache given its handles.
    pub fn get_micro_surface_meshes(&self, handles: &[Handle]) -> Vec<Option<&MicroSurfaceMesh>> {
        handles.iter().map(|h| self.meshes.get(h)).collect()
    }

    /// Returns a list of micro-surface profiles' file paths from the cache
    /// given a list of handles to the micro-surface profiles.
    pub fn get_micro_surface_filepaths(&self, handles: &[Handle]) -> Option<Vec<&Path>> {
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

    pub fn get_measurement(&self, handle: Handle) -> Option<&Measurement> {
        self.measurements.get(&handle)
    }

    /// Loads a surface from its relevant place and returns its cache handle.
    pub fn load_micro_surface(
        &mut self,
        config: &Config,
        path: &Path,
        subdiv: Option<Subdivision>,
    ) -> Result<(Handle, Handle), VgonioError> {
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
                    let msurf_hdl = Handle::from_uuid::<MicroSurface>(msurf.uuid);
                    let mesh = msurf.as_micro_surface_mesh(
                        HeightOffset::Grounded,
                        config.user.triangulation,
                        subdiv,
                    );
                    let mesh_hdl = Handle::from_uuid::<MicroSurfaceMesh>(mesh.uuid);
                    self.msurfs.insert(msurf_hdl, msurf);
                    self.meshes.insert(mesh_hdl, mesh);
                    self.records.insert(
                        msurf_hdl,
                        MicroSurfaceRecord {
                            path: filepath,
                            surf: msurf_hdl,
                            mesh: mesh_hdl,
                        },
                    );
                    Ok((msurf_hdl, mesh_hdl))
                }
            },
        }
    }

    /// Unloads a micro-surface from the compute cache (surface + mesh +
    /// record). The UI-side renderable, if any, is dropped by
    /// [`crate::app::cache::ui::UiCache::unload_micro_surface`], which calls
    /// this after removing its own state.
    pub fn unload_micro_surface(&mut self, handle: Handle) -> Result<(), VgonioError> {
        let record = self.records.get(&handle).ok_or_else(|| {
            VgonioError::new(format!("Failed to unload micro-surface: {}", handle), None)
        })?;
        let mesh = record.mesh;
        self.msurfs.remove(&handle);
        self.meshes.remove(&mesh);
        self.records.remove(&handle);
        Ok(())
    }

    /// Adds micro-surface measurement data to the cache.
    pub fn add_micro_surface_measurement(
        &mut self,
        data: Measurement,
    ) -> Result<Handle, VgonioError> {
        let handle = Handle::new_with_variant::<Measurement>(data.measured.kind() as u8);
        self.measurements.insert(handle, data);
        Ok(handle)
    }

    /// Loads a micro-surface measurement data from the given path and returns
    /// its cache handle.
    pub fn load_micro_surface_measurement(
        &mut self,
        config: &Config,
        path: &Path,
    ) -> Result<Handle, VgonioError> {
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
                    let handle =
                        Handle::new_with_variant::<Measurement>(data.measured.kind() as u8);
                    self.measurements.insert(handle, data);
                    Ok(handle)
                }
            },
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
    /// * `paths` - Paths to the surfaces to be loaded. Paths may not in canonical form.
    ///
    /// * `pattern` - The triangulation pattern when constructing the surface mesh.
    pub fn load_micro_surfaces(
        &mut self,
        config: &Config,
        paths: &[SurfacePath],
        pattern: TriangulationPattern,
    ) -> Result<Vec<Handle>, VgonioError> {
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
                        let msurf_hdl = Handle::from_uuid::<MicroSurface>(msurf.uuid);
                        let mesh = msurf.as_micro_surface_mesh(
                            HeightOffset::Grounded,
                            pattern,
                            subdivision,
                        );
                        let mesh_hdl = Handle::from_uuid::<MicroSurfaceMesh>(mesh.uuid);
                        self.msurfs.insert(msurf_hdl, msurf);
                        self.meshes.insert(mesh_hdl, mesh);
                        self.records.insert(
                            msurf_hdl,
                            MicroSurfaceRecord {
                                path: filepath,
                                surf: msurf_hdl,
                                mesh: mesh_hdl,
                            },
                        );
                        loaded.push(msurf_hdl);
                    }
                }
            } else {
                eprintln!(
                    "    {}!{} file not found: {}",
                    ansi::Color::Red.code(),
                    ansi::RESET,
                    surf.path.display()
                );
            }
        }
        log::debug!("- loaded micro surfaces: {:?}", loaded);
        Ok(loaded)
    }

    /// Loads the refractive index database from the paths specified in the
    /// configuration.
    pub fn load_ior_database(&mut self, config: &Config) {
        use vgn_core::{optics::IorRegLoader, res::AssetLoader};

        let loader = IorRegLoader::new(
            Some(config.sys_data_dir()),
            config.user_data_dir(),
            config.user.excluded_ior_files.clone(),
        );

        match loader.load(None) {
            Ok(boxed) => match boxed.into_any().downcast::<IorReg>() {
                Ok(reg) => self.iors = *reg,
                Err(_) => log::error!("loaded IOR asset had unexpected type"),
            },
            Err(e) => log::error!("failed to load IOR database: {e}"),
        }
    }
}
