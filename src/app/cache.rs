use crate::{
    acq::{
        ior::{Ior, IorDb},
        Medium,
    },
    app::{
        cli::{BRIGHT_RED, RESET},
        Config,
    },
    msurf::{
        mesh::{MicroSurfaceTriMesh, TriangulationMethod},
        AxisAlignment, MicroSurface,
    },
    Error,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    str::FromStr,
};
use uuid::Uuid;

pub enum AssetHandle {
    MicroSurface { uuid: Uuid, path: PathBuf },
}

#[derive(Debug, Clone)]
pub struct MicroSurfaceHandle {
    pub uuid: Uuid,
    pub path: PathBuf,
}

#[derive(Debug)]
pub struct VgonioDatafiles {
    /// Refractive index database.
    pub ior_db: IorDb,
}

// TODO(yang): unify HeightField with corresponding MicroSurfaceTriMesh.
// TODO(yang): maybe rename HeightField as MicroSurface.

impl Default for VgonioDatafiles {
    fn default() -> Self {
        Self {
            ior_db: IorDb::new(),
        }
    }
}

impl VgonioDatafiles {
    /// Creates a new `VgonioDatafiles` instance and loads all data files(TODO).
    pub fn new() -> Self { Self::default() }

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
            let n = Self::load_refractive_indices(&mut self.ior_db, &sys_path);
            log::debug!("Loaded {} ior files from {:?}", n, sys_path);
        }

        // Second load refractive indices from `VgonioConfig::user_data_dir()`.
        if user_path.as_ref().is_some_and(|p| p.exists()) {
            let n = Self::load_refractive_indices(&mut self.ior_db, user_path.as_ref().unwrap());
            log::debug!("Loaded {} ior files from {:?}", n, user_path);
        }
    }

    /// Load the refractive index database from the given path.
    /// Returns the number of files loaded.
    fn load_refractive_indices(ior_db: &mut IorDb, path: &Path) -> u32 {
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
            let iors = ior_db.0.entry(medium).or_insert(Vec::new());
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
                n_files += Self::load_refractive_indices(ior_db, &path);
            }
        }

        n_files
    }
}

/// Structure for caching intermediate results and data.
/// Also used for managing assets.
#[derive(Debug)]
pub struct Cache {
    /// Path to the cache directory.
    pub dir: std::path::PathBuf,

    /// Micro surface heightfield cache. (key: surface_path, value: heightfield)
    pub micro_surfaces: HashMap<String, MicroSurface>,

    /// Cache for triangulated heightfield meshes.
    pub triangle_meshes: HashMap<Uuid, MicroSurfaceTriMesh>,

    /// Cache for recently opened files.
    pub recent_opened_files: Option<Vec<std::path::PathBuf>>,

    /// Cache for recently opened directory.
    pub last_opened_dir: Option<std::path::PathBuf>,
}

impl Cache {
    pub fn new(cache_dir: &Path) -> Self {
        Self {
            dir: cache_dir.to_path_buf(),
            micro_surfaces: Default::default(),
            triangle_meshes: Default::default(),
            recent_opened_files: None,
            last_opened_dir: None,
        }
    }

    pub fn num_micro_surfaces(&self) -> usize { self.micro_surfaces.len() }

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
    ) -> Result<Vec<MicroSurfaceHandle>, Error> {
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
        use std::collections::hash_map::Entry;
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
                    let path_string = filepath.to_string_lossy().to_string();
                    if let Entry::Vacant(e) = self.micro_surfaces.entry(path_string.clone()) {
                        let heightfield = MicroSurface::read_from_file(&filepath, None, alignment)?;
                        loaded.push(MicroSurfaceHandle {
                            uuid: heightfield.uuid,
                            path: filepath.clone(),
                        });
                        e.insert(heightfield);
                    } else {
                        loaded.push(MicroSurfaceHandle {
                            uuid: self.micro_surfaces[&path_string].uuid,
                            path: filepath.clone(),
                        });
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
        surface_handles: &[MicroSurfaceHandle],
    ) -> Result<Vec<&MicroSurface>, Error> {
        let mut surfaces = vec![];
        for handle in surface_handles {
            let surface = self
                .micro_surfaces
                .get(&handle.path.to_string_lossy().to_string())
                .ok_or_else(|| Error::Any("Surface not exist!".to_string()))?;
            surfaces.push(surface)
        }

        Ok(surfaces)
    }

    /// Triangulates the given height fields and returns uuid of corresponding
    /// heightfield.
    fn triangulate_surfaces(&mut self, handles: &[MicroSurfaceHandle]) -> Vec<Uuid> {
        let mut meshes = vec![];
        for handle in handles {
            for surface in self.micro_surfaces.values() {
                if surface.uuid == handle.uuid {
                    if let std::collections::hash_map::Entry::Vacant(e) =
                        self.triangle_meshes.entry(surface.uuid)
                    {
                        let mesh = surface.triangulate(TriangulationMethod::Regular);
                        e.insert(mesh);
                        meshes.push(surface.uuid);
                    } else {
                        meshes.push(surface.uuid);
                    }
                    break;
                }
            }
        }
        meshes
    }

    pub fn get_micro_surface_meshes(
        &self,
        handles: &[MicroSurfaceHandle],
    ) -> Vec<&MicroSurfaceTriMesh> {
        let mut meshes = vec![];
        for handle in handles {
            meshes.push(self.triangle_meshes.get(&handle.uuid).unwrap())
        }
        meshes
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
    path.map_or_else(
        || base.to_path_buf(),
        |path| {
            if path.is_absolute() {
                path.canonicalize().unwrap()
            } else {
                base.join(path).canonicalize().unwrap()
            }
        },
    )
}
