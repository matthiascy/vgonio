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
pub struct SurfaceHandle {
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
        let default_path = config.data_files_dir.join("ior");
        let user_path = config.user_config.data_files_dir.join("ior");

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
pub struct VgonioCache {
    /// Path to the cache directory.
    pub dir: std::path::PathBuf,

    /// Surface heightfield cache. (key: surface_path, value: heightfield)
    pub surfaces: HashMap<String, Heightfield>,

    /// Cache for triangulated heightfield meshes.
    pub triangle_meshes: HashMap<Uuid, TriangleMesh>,
}

impl VgonioCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            dir: cache_dir,
            surfaces: Default::default(),
            triangle_meshes: Default::default(),
        }
    }

    /// Loads surfaces from their relevant places and returns
    /// their index of inside of the cache.
    pub fn load_surfaces_from_files(
        &mut self,
        surfaces_paths: &[PathBuf],
    ) -> Result<Vec<SurfaceHandle>, Error> {
        use std::collections::hash_map::Entry;
        let mut loaded = vec![];
        for path in surfaces_paths {
            let path_string = path.to_string_lossy().to_string();
            if let Entry::Vacant(e) = self.surfaces.entry(path_string.clone()) {
                let heightfield = Heightfield::read_from_file(path, None, Some(AxisAlignment::XZ))?;
                loaded.push(SurfaceHandle {
                    uuid: heightfield.uuid,
                    path: path.clone(),
                });
                e.insert(heightfield);
            } else {
                loaded.push(SurfaceHandle {
                    uuid: self.surfaces[&path_string].uuid,
                    path: path.clone(),
                });
            }
        }
        Ok(loaded)
    }

    pub fn get_surfaces(
        &self,
        surface_handles: &[SurfaceHandle],
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
    pub fn triangulate_surfaces(&mut self, handles: &[SurfaceHandle]) -> Vec<Uuid> {
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

    pub fn get_surface_meshes(&self, handles: &[SurfaceHandle]) -> Vec<&TriangleMesh> {
        let mut meshes = vec![];
        for handle in handles {
            meshes.push(self.triangle_meshes.get(&handle.uuid).unwrap())
        }
        meshes
    }
}
