use crate::{
    acq::{
        ior::{Ior, IorDatabase},
        Medium,
    },
    app::{
        cli::{BRIGHT_RED, RESET},
        Config,
    },
    msurf::{AxisAlignment, MicroSurface, MicroSurfaceTriMesh},
    Error,
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
}

/// Structure for caching intermediate results and data.
/// Also used for managing assets.
#[derive(Debug)]
pub struct Cache {
    /// Path to the cache directory.
    pub dir: std::path::PathBuf,

    /// Refractive index database.
    pub iors: IorDatabase,

    /// Lookup table for micro-surface's path to its corresponding uuid.
    msurf_path_to_uuid: HashMap<PathBuf, Uuid>,

    /// Micro-surface cache, indexed by micro-surface uuid.
    msurfs: HashMap<Uuid, MicroSurface>,

    /// Micro-surface triangle mesh cache, indexed by micro-surface uuid.
    msurf_meshes: HashMap<Uuid, MicroSurfaceTriMesh>,

    /// Cache for `RenderableMesh`s. TODO: usage?
    #[allow(dead_code)]
    renderables: HashMap<Uuid, RenderableMesh>,

    /// Cache for recently opened files.
    pub recent_opened_files: Option<Vec<std::path::PathBuf>>,

    /// Cache for recently opened directory.
    pub last_opened_dir: Option<std::path::PathBuf>,
}

impl Cache {
    pub fn new(cache_dir: &Path) -> Self {
        Self {
            dir: cache_dir.to_path_buf(),
            msurfs: Default::default(),
            msurf_meshes: Default::default(),
            recent_opened_files: None,
            last_opened_dir: None,
            iors: IorDatabase::default(),
            renderables: Default::default(),
            msurf_path_to_uuid: Default::default(),
        }
    }

    pub fn num_micro_surfaces(&self) -> usize { self.msurfs.len() }

    pub fn get_loaded_surface_paths(&self) -> Option<Vec<&Path>> {
        self.msurfs
            .keys()
            .map(|uuid| self.msurfs.get(uuid).unwrap().path.as_deref())
            .collect()
    }

    pub fn get_micro_surface_path(&self, handle: Handle<MicroSurface>) -> Option<&Path> {
        if self.msurfs.contains_key(&handle.id) {
            self.msurfs[&handle.id].path.as_deref()
        } else {
            None
        }
    }

    pub fn get_micro_surface_paths(&self, handles: &[Handle<MicroSurface>]) -> Option<Vec<&Path>> {
        handles
            .iter()
            .map(|handle| self.get_micro_surface_path(*handle))
            .collect()
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
        alignment: Option<AxisAlignment>,
    ) -> Result<Vec<Handle<MicroSurface>>, Error> {
        log::info!("Loading micro surfaces from {:?}", paths);
        let canonical_paths = paths
            .iter()
            .filter_map(|s| {
                if let Ok(stripped) = s.strip_prefix("usr://") {
                    if config.user_data_dir().is_some() {
                        config.user_data_dir().map(|p| p.join(stripped))
                    } else {
                        log::warn!(
                            "The file path begins with `usr://`: {}, but the user data directory \
                             is not configured.",
                            s.display()
                        );
                        None
                    }
                } else if let Ok(stripped) = s.strip_prefix("sys://") {
                    Some(config.sys_data_dir().join(stripped))
                } else {
                    Some(resolve_path(&config.cwd, Some(s)))
                }
            })
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
                    if let Some(uuid) = self.msurf_path_to_uuid.get(&filepath) {
                        log::debug!("-- already loaded: {}", filepath.display());
                        loaded.push(Handle::new(*uuid));
                    } else {
                        log::debug!("-- loading: {}", filepath.display());
                        let msurf = MicroSurface::from_file(&filepath, None, alignment)?;
                        let uuid = msurf.uuid;
                        self.msurfs.insert(uuid, msurf);
                        self.msurf_path_to_uuid.insert(filepath, uuid);
                        loaded.push(Handle::new(uuid));
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
        self.triangulate_surfaces(&loaded);
        Ok(loaded)
    }

    pub fn get_micro_surfaces(
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

    /// Triangulates the given micro-surface handles.
    fn triangulate_surfaces(&mut self, handles: &[Handle<MicroSurface>]) {
        use std::collections::hash_map::Entry;
        handles.iter().for_each(|handle| {
            let uuid = handle.id();
            // Surface must exist.
            if self.msurfs.contains_key(&uuid) {
                // Has not been triangulated.
                if let Entry::Vacant(entry) = self.msurf_meshes.entry(uuid) {
                    // Triangulate the surface.
                    entry.insert(self.msurfs.get(&uuid).unwrap().triangulate());
                }
            }
        })
    }

    pub fn get_micro_surface_meshes(
        &self,
        handles: &[Handle<MicroSurface>],
    ) -> Vec<&MicroSurfaceTriMesh> {
        let mut meshes = vec![];
        for handle in handles {
            meshes.push(self.msurf_meshes.get(&handle.id()).unwrap())
        }
        meshes
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
    fn load_refractive_indices(iors: &mut IorDatabase, path: &Path) -> u32 {
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
            let refractive_indices = Ior::read_iors_from_file(path).unwrap();
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
