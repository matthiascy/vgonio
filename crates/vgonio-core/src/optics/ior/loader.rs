//! Loader for the refractive-index registry.

use super::IorReg;
use crate::res::{Asset, AssetLoader, AssetTypeId, Error};
use std::path::{Path, PathBuf};

/// Loader for refractive index database.
pub struct IorRegLoader {
    /// System directory.
    sys_dir: Option<PathBuf>,
    /// User directory.
    usr_dir: Option<PathBuf>,
    /// Excluded files.
    excluded: Option<Vec<String>>,
}

impl IorRegLoader {
    /// Creates a new refractive index database loader.
    ///
    /// # Arguments
    ///
    /// * `sys_dir` - The base system directory used to resolve the path when the path is in form
    ///   `sys://`. The actual path is resolved to `sys_dir/ior/path`.
    /// * `usr_dir` - The base user directory used to resolve the path when the path is in form
    ///   `user://`. The actual path is resolved to `usr_dir/ior/path`.
    /// * `excluded` - The list of excluded files.
    pub fn new(
        sys_dir: Option<&Path>,
        usr_dir: Option<&Path>,
        excluded: Option<Vec<String>>,
    ) -> Self {
        Self {
            sys_dir: sys_dir.map(|path| path.to_path_buf().join("ior")),
            usr_dir: usr_dir.map(|path| path.to_path_buf().join("ior")),
            excluded,
        }
    }
}

impl AssetLoader for IorRegLoader {
    fn asset_type(&self) -> &'static str { IorReg::asset_type() }

    fn asset_type_id(&self) -> AssetTypeId { IorReg::asset_type_id() }

    fn load(&self, path: Option<&Path>) -> Result<Box<dyn Asset>, Error> {
        let own_excluded = self
            .excluded
            .as_ref()
            .map(|ss| ss.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        let excluded = own_excluded.as_deref().unwrap_or(&[]);
        match path {
            Some(path) => {
                let mut ior_reg = IorReg::new();
                let n = ior_reg.load_from_path(path, excluded)?;
                log::debug!("  Loaded {} ior files from {:?}", n, path);
                Ok(Box::new(ior_reg))
            },
            None => {
                log::debug!("Loading refractive index database from default paths ...");
                log::debug!("  -- sys_dir: {:?}", self.sys_dir);
                log::debug!("  -- usr_dir: {:?}", self.usr_dir);

                if self.sys_dir.is_none() {
                    log::debug!(
                        "  Refractive index database not found at sys dir: {:?}",
                        self.sys_dir
                    );
                }

                if self.usr_dir.is_none() {
                    log::debug!(
                        "  Refractive index database not found at user dir: {:?}",
                        self.usr_dir
                    );
                }

                if self.usr_dir.is_none() && self.sys_dir.is_none() {
                    log::error!("No path specified to load refractive index database");
                    todo!("return an error");
                }

                let mut ior_reg = IorReg::new();
                let n = ior_reg.load_from_path(self.sys_dir.as_ref().unwrap(), excluded)?;
                log::debug!("  Loaded {} ior files from {:?}", n, self.sys_dir);
                let n = ior_reg.load_from_path(self.usr_dir.as_ref().unwrap(), excluded)?;
                log::debug!("  Loaded {} ior files from {:?}", n, self.usr_dir);
                Ok(Box::new(ior_reg))
            },
        }
    }
}
