use crate::{
    app::{
        cli::{BRIGHT_RED, RESET},
        Config,
    },
    msurf::{MicroSurface, MicroSurfaceMesh},
    optics::ior::{RefractiveIndex, RefractiveIndexDatabase},
    Error, Medium,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    str::FromStr,
};
use uuid::Uuid;

use super::gfx::RenderableMesh;

pub trait Asset: Send + Sync + 'static {}

/// Handle referencing loaded assets.
#[derive(Debug, PartialEq, Eq, Hash)]
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
impl<T> Handle<T>
where
    T: Asset,
{
    pub fn new(id: Uuid) -> Self {
        Self {
            id,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn id(&self) -> Uuid { self.id }

    pub fn is_null(&self) -> bool { self.id == Uuid::default() }
}

/// Default handle is a null handle.
impl<T: Asset> Default for Handle<T> {
    fn default() -> Self { Self::new(Uuid::default()) }
}

#[derive(Debug)]
struct MicroSurfaceRecord {
    path: PathBuf,
    mesh: Handle<MicroSurfaceMesh>,
    renderable: Option<Handle<RenderableMesh>>,
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
    records: HashMap<Uuid, MicroSurfaceRecord>,

    /// Micro-surface cache, indexed by micro-surface uuid.
    msurfs: HashMap<Uuid, MicroSurface>,

    /// Micro-surface triangle mesh cache, indexed by micro-surface mesh uuid.
    meshes: HashMap<Uuid, MicroSurfaceMesh>,

    /// Cache for `RenderableMesh`s, indexed by renderable mesh uuid.
    renderables: HashMap<Uuid, RenderableMesh>,

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
        }
    }

    /// Returns a micro-surface profile [`MicroSurface`] from the cache given
    /// its handle.
    pub fn micro_surface(&self, handle: Handle<MicroSurface>) -> Result<&MicroSurface, Error> {
        self.msurfs
            .get(&handle.id())
            .ok_or_else(|| Error::Any("Surface not exist!".to_string()))
    }

    /// Returns a micro-surface profile [`MicroSurfaceMesh`] from the cache
    /// given handle to the micro-surface profile.
    pub fn micro_surface_mesh_by_surface_id(
        &self,
        handle: Handle<MicroSurface>,
    ) -> Result<&MicroSurfaceMesh, Error> {
        let record = self
            .records
            .get(&handle.id())
            .ok_or_else(|| Error::Any("Surface record not exist!".to_string()))?;
        self.meshes
            .get(&record.mesh.id())
            .ok_or_else(|| Error::Any("Surface mesh not exist!".to_string()))
    }

    /// Returns a micro-surface profile [`MicroSurfaceMesh`] from the cache
    /// given its handle.
    pub fn micro_surface_mesh(
        &self,
        handle: Handle<MicroSurfaceMesh>,
    ) -> Result<&MicroSurfaceMesh, Error> {
        self.meshes
            .get(&handle.id())
            .ok_or_else(|| Error::Any("Surface mesh not exist!".to_string()))
    }

    /// Creates a micro-surface renderable mesh for the given micro-surface
    /// handle.
    pub fn create_micro_surface_renderable_mesh(
        &mut self,
        device: &wgpu::Device,
        msurf: Handle<MicroSurface>,
    ) -> Result<Handle<RenderableMesh>, Error> {
        let record = self
            .records
            .get(&msurf.id())
            .ok_or_else(|| Error::Any(format!("{BRIGHT_RED}Surface record not exist!{RESET}")))?;
        if record.renderable.is_some() {
            Ok(record.renderable.unwrap())
        } else {
            let mesh = self
                .meshes
                .get(&record.mesh.id)
                .ok_or_else(|| Error::Any(format!("{BRIGHT_RED}Surface mesh not exist!{RESET}")))?;
            let renderable = RenderableMesh::from_micro_surface_mesh(device, mesh);
            let handle = Handle::new(Uuid::new_v4());
            self.renderables.insert(handle.id(), renderable);
            Ok(handle)
        }
    }

    /// Returns a micro-surface profile [`RenderableMesh`] from the cache
    /// given handle to the micro-surface profile.
    pub fn micro_surface_renderable_mesh_by_surface_id(
        &self,
        handle: Handle<MicroSurface>,
    ) -> Result<&RenderableMesh, Error> {
        let record = self
            .records
            .get(&handle.id())
            .ok_or_else(|| Error::Any("Surface record not exist!".to_string()))?;
        self.renderables
            .get(&record.renderable.as_ref().unwrap().id())
            .ok_or_else(|| Error::Any("Surface renderable not exist!".to_string()))
    }

    /// Returns a micro-surface profile [`RenderableMesh`] from the cache
    /// given its handle.
    pub fn micro_surface_renderable_mesh(
        &self,
        handle: Handle<RenderableMesh>,
    ) -> Result<&RenderableMesh, Error> {
        self.renderables
            .get(&handle.id())
            .ok_or_else(|| Error::Any("Surface renderable not exist!".to_string()))
    }

    /// Returns the file path to the micro-surface profile given its handle.
    pub fn micro_surface_path(&self, handle: Handle<MicroSurface>) -> Option<&Path> {
        if self.msurfs.contains_key(&handle.id) {
            self.msurfs[&handle.id].path.as_deref()
        } else {
            None
        }
    }

    /// Returns the number of loaded micro-surface profiles.
    pub fn num_micro_surfaces(&self) -> usize { self.msurfs.len() }

    /// Returns a list of micro-surface profiles [`MicroSurface`] from the
    /// cache.
    pub fn micro_surfaces(
        &self,
        handles: &[Handle<MicroSurface>],
    ) -> Result<Vec<&MicroSurface>, Error> {
        let mut surfaces = vec![];
        for handle in handles {
            let surface = self
                .msurfs
                .get(&handle.id())
                .ok_or_else(|| Error::Any("Surface not exist!".to_string()))?;
            surfaces.push(surface)
        }

        Ok(surfaces)
    }

    /// Returns a list of micro-surface meshes [`MicroSurfaceMesh`] from the
    /// cache given a list of handles to the micro-surface profiles.
    pub fn micro_surface_meshes_by_surfaces(
        &self,
        handles: &[Handle<MicroSurface>],
    ) -> Result<Vec<&MicroSurfaceMesh>, Error> {
        let mut meshes = vec![];
        for handle in handles {
            match self.records.get(&handle.id()) {
                Some(record) => match self.meshes.get(&record.mesh.id()) {
                    Some(mesh) => {
                        meshes.push(mesh);
                    }
                    None => {
                        return Err(Error::Any("Surface mesh not exist!".to_string()));
                    }
                },
                None => {
                    return Err(Error::Any("Surface record not exist!".to_string()));
                }
            }
        }
        Ok(meshes)
    }

    /// Returns a list of micro-surface meshes [`MicroSurfaceMesh`] from the
    /// cache given its handles.
    pub fn micro_surface_meshes(
        &self,
        handles: &[Handle<MicroSurfaceMesh>],
    ) -> Vec<&MicroSurfaceMesh> {
        let mut meshes = vec![];
        for handle in handles {
            meshes.push(self.meshes.get(&handle.id()).unwrap());
        }
        meshes
    }

    /// Returns a list of micro-surface profiles' file paths from the cache
    /// given a list of handles to the micro-surface profiles.
    pub fn micro_surface_paths(&self, handles: &[Handle<MicroSurface>]) -> Option<Vec<&Path>> {
        handles
            .iter()
            .map(|handle| self.micro_surface_path(*handle))
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

    /// Loads a surface from its relevant place and returns its cache handle.
    pub fn load_micro_surface(
        &mut self,
        config: &Config,
        paths: &PathBuf,
    ) -> Result<(Handle<MicroSurface>, Handle<MicroSurfaceMesh>), Error> {
        self.resolve_path(paths, config)
            .map(|filepath| {
                if let Some((msurf_id, record)) = self
                    .records
                    .iter()
                    .find(|(_, record)| record.path == filepath)
                {
                    log::debug!("-- already loaded: {}", filepath.display());
                    (Handle::new(*msurf_id), record.mesh)
                } else {
                    log::debug!("-- loading: {}", filepath.display());
                    let msurf = MicroSurface::read_from_file(&filepath, None).unwrap();
                    let msurf_id = msurf.uuid;
                    let mesh = msurf.as_micro_surface_mesh();
                    let mesh_id = Uuid::new_v4();
                    self.msurfs.insert(msurf_id, msurf);
                    self.meshes.insert(mesh_id, mesh);
                    self.records.insert(
                        msurf_id,
                        MicroSurfaceRecord {
                            path: filepath,
                            mesh: Handle::new(mesh_id),
                            renderable: None,
                        },
                    );
                    (Handle::new(msurf_id), Handle::new(mesh_id))
                }
            })
            .ok_or(Error::Any(
                "Failed to load micro surface profile!".to_string(),
            ))
    }

    /// Resolves a path to a file.
    fn resolve_path(&self, path: &PathBuf, config: &Config) -> Option<PathBuf> {
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
    ) -> Result<Vec<Handle<MicroSurface>>, Error> {
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
                        loaded.push(Handle::new(*msurf_id));
                    } else {
                        log::debug!("-- loading: {}", filepath.display());
                        let msurf = MicroSurface::read_from_file(&filepath, None).unwrap();
                        let msurf_id = msurf.uuid;
                        let mesh = msurf.as_micro_surface_mesh();
                        let mesh_id = Uuid::new_v4();
                        self.msurfs.insert(msurf_id, msurf);
                        self.meshes.insert(mesh_id, mesh);
                        self.records.insert(
                            msurf_id,
                            MicroSurfaceRecord {
                                path: filepath,
                                mesh: Handle::new(mesh_id),
                                renderable: None,
                            },
                        );
                        loaded.push(Handle::new(msurf_id));
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
