use crate::{
    acq::{
        ior::{Ior, IorDb},
        Medium,
    },
    app::VgonioConfig,
    htfld::{AxisAlignment, Heightfield},
    mesh::{TriangleMesh, TriangulationMethod},
    Error,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    str::FromStr,
};
use uuid::Uuid;

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

impl VgonioDatafiles {
    pub fn new() -> Self {
        Self {
            ior_db: IorDb::new(),
        }
    }

    /// Load the refractive index database from the paths specified in the
    /// config.
    pub fn load_ior_database(&mut self, config: &VgonioConfig) {
        // let mut database = RefractiveIndexDatabase::new();
        let default_path = config.data_dir.join("ior");
        let user_path = config.user_config.data_dir.join("ior");

        // First load refractive indices from `VgonioConfig::data_files_dir`.
        if default_path.exists() {
            // Load one by one the files in the directory.
            let n = Self::load_refractive_indices(&mut self.ior_db, &default_path);
            log::debug!("Loaded {} ior files from {:?}", n, default_path);
        }

        // Second load refractive indices from
        // `VgonioConfig::user_config::data_files_dir`.
        if user_path.exists() {
            let n = Self::load_refractive_indices(&mut self.ior_db, &user_path);
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
#[derive(Debug)]
pub struct Cache {
    /// Path to the cache directory.
    pub dir: std::path::PathBuf,

    /// Surface heightfield cache. (key: surface_path, value: heightfield)
    pub surfaces: HashMap<String, Heightfield>,

    /// Cache for triangulated heightfield meshes.
    pub triangle_meshes: HashMap<Uuid, TriangleMesh>,

    /// Cache for recently opened files.
    pub recent_opened_files: Option<Vec<std::path::PathBuf>>,

    /// Cache for recently opened directory.
    pub last_opened_dir: Option<std::path::PathBuf>,
}

impl Cache {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            dir: cache_dir,
            surfaces: Default::default(),
            triangle_meshes: Default::default(),
            recent_opened_files: None,
            last_opened_dir: None,
        }
    }

    /// Loads surfaces from their relevant places and returns
    /// their handle inside the cache.
    ///
    /// # Arguments
    ///
    /// * `config` - The application configuration.
    ///
    /// * `paths` - Paths to the surfaces to be load. Paths are not in canonical
    ///   form, they may also be relative paths.
    ///
    /// * `alignment` - The axis alignment the surfaces mesh.
    pub fn load_micro_surfaces(
        &mut self,
        &config: &VgonioConfig,
        paths: &[PathBuf],
        alignment: Option<AxisAlignment>,
    ) -> Result<Vec<MicroSurfaceHandle>, Error> {
        let canonical_paths = paths
            .iter()
            .map(|s| {
                if let Ok(stripped) = s.strip_prefix("user://") {
                    config.user_config.data_dir.join(stripped)
                } else if let Ok(stripped) = s.strip_prefix("local://") {
                    config.data_dir.join(stripped)
                } else {
                    resolve_file_path(&config.cwd, Some(s))
                }
            })
            .collect::<Vec<_>>();

        use std::collections::hash_map::Entry;
        let mut loaded = vec![];
        for path in paths {
            let path_string = path.to_string_lossy().to_string();
            if let Entry::Vacant(e) = self.surfaces.entry(path_string.clone()) {
                let heightfield = Heightfield::read_from_file(path, None, alignment)?;
                loaded.push(MicroSurfaceHandle {
                    uuid: heightfield.uuid,
                    path: path.clone(),
                });
                e.insert(heightfield);
            } else {
                loaded.push(MicroSurfaceHandle {
                    uuid: self.surfaces[&path_string].uuid,
                    path: path.clone(),
                });
            }
        }
        Ok(loaded)
    }

    pub fn get_micro_surfaces(
        &self,
        surface_handles: &[MicroSurfaceHandle],
    ) -> Result<Vec<&Heightfield>, Error> {
        let mut surfaces = vec![];
        for handle in surface_handles {
            let surface = self
                .surfaces
                .get(&handle.path.to_string_lossy().to_string())
                .ok_or_else(|| Error::Any("Surface not exist!".to_string()))?;
            surfaces.push(surface)
        }

        Ok(surfaces)
    }

    /// Triangulates the given height fields and returns uuid of corresponding
    /// heightfield.
    pub fn triangulate_surfaces(&mut self, handles: &[MicroSurfaceHandle]) -> Vec<Uuid> {
        let mut meshes = vec![];
        for handle in handles {
            for surface in self.surfaces.values() {
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

    pub fn get_surface_meshes(&self, handles: &[MicroSurfaceHandle]) -> Vec<&TriangleMesh> {
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
pub(crate) fn resolve_file_path(base: &Path, path: Option<&Path>) -> PathBuf {
    path.map_or_else(
        || base.to_path_buf(),
        |path| {
            if path.is_absolute() {
                path.to_path_buf().canonicalize().unwrap()
            } else {
                base.join(path).canonicalize().unwrap()
            }
        },
    )
}
