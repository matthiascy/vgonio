use super::gfx::RenderableMesh;
use crate::{
    app::{
        cli::{BRIGHT_RED, RESET},
        Config,
    },
    measure::measurement::MeasurementData,
    optics::ior::{RefractiveIndex, RefractiveIndexDatabase},
    Medium,
};
use std::{
    collections::HashMap,
    fmt::{Debug, Display, Formatter},
    hash::Hash,
    path::{Path, PathBuf},
    rc::{Rc, Weak},
    str::FromStr,
};
use uuid::Uuid;
use vgcore::error::VgonioError;
use vgsurf::{HeightOffset, MicroSurface, MicroSurfaceMesh};

pub trait Asset: Send + Sync + 'static {}

/// Handle referencing loaded assets.
pub struct Handle<T>
where
    T: Asset,
{
    id: Uuid,
    _phantom: std::marker::PhantomData<fn() -> T>,
}

impl<T: Asset> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            _phantom: Default::default(),
        }
    }
}
impl<T: Asset> Copy for Handle<T> {}

impl<T: Asset> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool { self.id == other.id }
}

impl<T: Asset> Eq for Handle<T> {}

impl<T: Asset> Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.id.hash(state) }
}

impl<T: Asset> Debug for Handle<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}<{}>",
            std::any::type_name::<T>().split(&"::").last().unwrap(),
            self.id
        )
    }
}

impl<T: Asset> Display for Handle<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{:?}", self) }
}

impl<T> Handle<T>
where
    T: Asset,
{
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_id(id: Uuid) -> Self {
        Self {
            id,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns an invalid handle.
    pub fn invalid() -> Self {
        Self {
            id: Uuid::nil(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the id of the handle.
    pub fn id(&self) -> Uuid { self.id }

    /// Returns true if the handle is valid.
    pub fn is_valid(&self) -> bool { !self.id.is_nil() }
}

/// Default handle is an invalid handle.
impl<T: Asset> Default for Handle<T> {
    fn default() -> Self { Self::with_id(Uuid::nil()) }
}

impl From<Uuid> for Handle<MicroSurface> {
    fn from(id: Uuid) -> Self { Self::with_id(id) }
}

impl Asset for MicroSurface {}

impl Asset for MicroSurfaceMesh {}

/// A record inside the cache for a micro-surface.
#[derive(Debug, Clone)]
pub struct MicroSurfaceRecord {
    path: PathBuf,
    pub surf: Handle<MicroSurface>,
    pub mesh: Handle<MicroSurfaceMesh>,
    pub renderable: Handle<RenderableMesh>,
}

impl Hash for MicroSurfaceRecord {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.surf.hash(state) }
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
pub struct Cache {
    /// Path to the cache directory.
    pub dir: PathBuf,

    /// Refractive index database.
    pub iors: RefractiveIndexDatabase,

    /// Micro-surface record cache, indexed by micro-surface uuid.
    pub records: HashMap<Handle<MicroSurface>, MicroSurfaceRecord>,

    /// Micro-surface cache, indexed by micro-surface uuid.
    msurfs: HashMap<Handle<MicroSurface>, MicroSurface>,

    /// Micro-surface triangle mesh cache, indexed by micro-surface mesh uuid.
    meshes: HashMap<Handle<MicroSurfaceMesh>, MicroSurfaceMesh>,

    /// Cache for `RenderableMesh`s, indexed by renderable mesh uuid.
    renderables: HashMap<Handle<RenderableMesh>, RenderableMesh>,

    /// Cache for measured data.
    measurements_data: HashMap<Handle<MeasurementData>, Rc<MeasurementData>>,

    // TODO: recently files
    /// Cache for recently opened files.
    pub recent_opened_files: Option<Vec<PathBuf>>,

    /// Cache for recently opened directory.
    pub last_opened_dir: Option<PathBuf>,
}

impl Cache {
    pub fn new(cache_dir: &Path) -> Self {
        Self {
            dir: cache_dir.to_path_buf(),
            msurfs: Default::default(),
            meshes: Default::default(),
            recent_opened_files: None,
            last_opened_dir: None,
            iors: RefractiveIndexDatabase::default(),
            renderables: Default::default(),
            records: Default::default(),
            measurements_data: Default::default(),
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
        device: &wgpu::Device,
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
        let renderable = RenderableMesh::from_micro_surface_mesh_with_id(device, mesh, handle.id);
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

    pub fn get_measurement_data(
        &self,
        handle: Handle<MeasurementData>,
    ) -> Option<Weak<MeasurementData>> {
        self.measurements_data.get(&handle).map(Rc::downgrade)
    }

    /// Loads a surface from its relevant place and returns its cache handle.
    pub fn load_micro_surface(
        &mut self,
        config: &Config,
        path: &Path,
    ) -> Result<(Handle<MicroSurface>, Handle<MicroSurfaceMesh>), VgonioError> {
        match self.resolve_path(path, config) {
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
                    let mesh = msurf.as_micro_surface_mesh(HeightOffset::Grounded);
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
                    Ok((msurf_hdl, mesh_hdl))
                }
            }
        }
    }

    /// Loads a micro-surface measurement data from the given path and returns
    /// its cache handle.
    pub fn load_micro_surface_measurement(
        &mut self,
        config: &Config,
        path: &Path,
    ) -> Result<Handle<MeasurementData>, VgonioError> {
        match self.resolve_path(path, config) {
            None => Err(VgonioError::new(
                "Failed to resolve measurement file.",
                None,
            )),
            Some(filepath) => {
                if let Some((hdl, _)) = self
                    .measurements_data
                    .iter()
                    .find(|(_, d)| d.source.path() == Some(&filepath))
                {
                    log::debug!("-- already loaded: {}", filepath.display());
                    Ok(*hdl)
                } else {
                    log::debug!("-- loading: {}", filepath.display());
                    let data = Rc::new(MeasurementData::read_from_file(&filepath)?);
                    let handle = Handle::new();
                    self.measurements_data.insert(handle, data);
                    Ok(handle)
                }
            }
        }
    }

    /// Resolves a path to a file.
    fn resolve_path(&self, path: &Path, config: &Config) -> Option<PathBuf> {
        if let Ok(stripped) = path.strip_prefix("usr://") {
            if config.user_data_dir().is_some() {
                config.user_data_dir().map(|p| p.join(stripped))
            } else {
                log::warn!(
                    "The file path begins with `usr://`: {}, but the user data directory is not \
                     configured.",
                    path.display()
                );
                None
            }
        } else if let Ok(stripped) = path.strip_prefix("sys://") {
            Some(config.sys_data_dir().join(stripped))
        } else {
            Some(resolve_path(&config.cwd, Some(path)))
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
        paths: &[PathBuf],
    ) -> Result<Vec<Handle<MicroSurface>>, VgonioError> {
        log::info!("Loading micro surfaces from {:?}", paths);
        let canonical_paths = paths
            .iter()
            .filter_map(|s| self.resolve_path(s, config))
            .collect::<Vec<_>>();
        log::debug!("-- canonical paths: {:?}", canonical_paths);
        let mut loaded = vec![];
        for path in canonical_paths {
            if path.exists() {
                let files_to_load = {
                    if path.is_dir() {
                        path.read_dir()
                            .unwrap()
                            .map(|entry| entry.unwrap().path())
                            .collect::<Vec<_>>()
                    } else {
                        vec![path]
                    }
                };
                for filepath in files_to_load {
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
                        let mesh = msurf.as_micro_surface_mesh(HeightOffset::Grounded);
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
                    "    {BRIGHT_RED}!{RESET} file not found: {}",
                    path.display()
                );
            }
        }
        log::debug!("- loaded micro surfaces: {:?}", loaded);
        Ok(loaded)
    }

    /// Loads the refractive index database from the paths specified in the
    /// config.
    pub fn load_ior_database(&mut self, config: &Config) {
        // let mut database = RefractiveIndexDatabase::new();
        let sys_path: PathBuf = config.sys_data_dir().to_path_buf().join("ior");
        let user_path = config
            .user_data_dir()
            .map(|path| path.to_path_buf().join("ior"));
        // First load refractive indices from `VgonioConfig::sys_data_dir()`.
        if sys_path.exists() {
            // Load one by one the files in the directory.
            let n = Self::load_refractive_indices(&mut self.iors, &sys_path);
            log::debug!("Loaded {} ior files from {:?}", n, sys_path);
        }

        // Second load refractive indices from `VgonioConfig::user_data_dir()`.
        if user_path.as_ref().is_some_and(|p| p.exists()) {
            let n = Self::load_refractive_indices(&mut self.iors, user_path.as_ref().unwrap());
            log::debug!("Loaded {} ior files from {:?}", n, user_path);
        }
    }

    /// Load the refractive index database from the given path.
    /// Returns the number of files loaded.
    fn load_refractive_indices(iors: &mut RefractiveIndexDatabase, path: &Path) -> u32 {
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
            let refractive_indices = RefractiveIndex::read_iors_from_file(path).unwrap();
            let iors = iors.0.entry(medium).or_insert(Vec::new());
            for ior in refractive_indices {
                if !iors.contains(&ior) {
                    iors.push(ior);
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

/// Resolves the path to canonical form.
///
/// # Note
///
/// Resolved path is not guaranteed to exist.
///
/// # Arguments
///
/// * `base` - Base path to resolve relative paths.
///
/// * `path` - Path to be resolved.
///
/// # Returns
///
/// A `PathBuf` indicating the resolved path. It differs according to the
/// base path and patterns inside of `path`.
///
///   1. `path` is `None`
///
///      Returns the `base` path.
///
///   2. `path` is relative
///
///      Returns the a path which is relative to `base` path, with
///      the remaining of the `path` appended.
///
///   3. `path` is absolute
///
///      Returns the `path` as is.
pub(crate) fn resolve_path(base: &Path, path: Option<&Path>) -> PathBuf {
    path.map_or(base.to_path_buf(), |path| {
        let path = if path.is_relative() {
            base.join(path)
        } else {
            path.to_path_buf()
        };

        if !path.exists() {
            normalise_path(&path)
        } else {
            path.canonicalize().unwrap()
        }
    })
}

/// Resolves the path to canonical form even if the path does not exist.
pub(crate) fn normalise_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut components = path.components().peekable();
    let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().cloned() {
        components.next();
        PathBuf::from(c.as_os_str())
    } else {
        PathBuf::new()
    };

    for component in components {
        match component {
            Component::Prefix(..) => unreachable!(),
            Component::RootDir => {
                ret.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                ret.pop();
            }
            Component::Normal(c) => {
                ret.push(c);
            }
        }
    }
    ret
}

#[test]
fn test_normalise_path() {
    let path = Path::new("/a/b/c/../../d");
    let normalised = normalise_path(path);
    assert_eq!(normalised, Path::new("/a/d"));
}
