//! Loader for the refractive-index registry.
//!
//! A compiled-in baseline (`datafiles/ior/`, embeded via `include_dir`) is always loaded, and
//! additional files can be loaded from the system and user directories. The loader supports
//! excluding specific files by name, which is useful for testing and development.

use super::IorReg;
use crate::{
    optics::{IorDataset, ManifestDto},
    res::{Asset, AssetLoader, AssetTypeId, Error},
    utils::medium::{merge_layers, MediumId, MergePolicy, Provenance},
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    str::FromStr,
};

#[cfg(feature = "embed-datafiles")]
static EMBEDDED_IOR: include_dir::Dir<'static> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../datafiles/ior");

/// Loader for refractive index database.
pub struct IorRegLoader {
    /// System directory.
    sys_dir: Option<PathBuf>,
    /// User directory.
    usr_dir: Option<PathBuf>,
    /// Excluded files.
    excluded: Vec<String>,
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
    ///
    /// Layering order (later overrides earlier for the same medium):
    /// 1. Embedded baseline
    /// 2. System
    /// 3. User
    /// The embeded baseline is implicit (no parameter): it is the compiled-in `datafiles/ior/`
    /// when the `embed-datafiles` feature is enabled, and empty otherwise. Pass `None` for both
    /// `sys_dir` and `usr_dir` to only load the embedded baseline.
    pub fn new(
        sys_dir: Option<&Path>,
        usr_dir: Option<&Path>,
        excluded: Option<Vec<String>>,
    ) -> Self {
        Self {
            sys_dir: sys_dir.map(|path| path.to_path_buf().join("ior")),
            usr_dir: usr_dir.map(|path| path.to_path_buf().join("ior")),
            excluded: excluded.unwrap_or_default(),
        }
    }

    /// Returns `true` if `file_name` is in this loader's exclusion list
    /// (matched verbatim, by file name).
    pub fn is_excluded(&self, file_name: &str) -> bool {
        self.excluded.iter().any(|excluded| excluded == file_name)
    }

    /// The error returned when an IOR load is attempted before the medium
    /// identity registry has been bootstrapped. A dedicated variant rather than
    /// `IoError`: this is a lifecycle precondition, not an I/O failure, and
    /// callers that retry on I/O errors must not retry on this.
    fn spine_not_ready_error() -> Error { Error::SpineNotBootstrapped }
}

/// One layer's raw inputs: optional manifest text + an iterator of
/// `(file_name, ron_text)` for every file in the layer.
type LayerItems<'a> = Box<dyn Iterator<Item = Result<(String, String), Error>> + 'a>;

/// Resolve one layer (a dir, or the embedded baseline) to "the chosen
/// dataset per medium". Applies exclusion, the filename/DTO/manifest
/// consistency rule, the manifest `default` flag, and the no-silent-pick
/// ambiguity rule. `label` is used only in error messages.
///
/// # Arguments
///
/// - `label`: A label for this layer, used in error messages.
/// - `manifest_src`: Optional manifest text for this layer. If `None`, treated as empty manifest
///   (no entries).
/// - `items`: An iterator of `(file_name, ron_text)` for every file in this layer.
/// - `excluded`: A predicate for excluding files by name. Excluded files are ignored as if they
///   don't exist, and do not cause errors by themselves.
fn resolve_layer(
    label: &str,
    manifest_src: Option<&str>,
    items: LayerItems<'_>,
    excluded: &dyn Fn(&str) -> bool,
) -> Result<HashMap<MediumId, IorDataset>, Error> {
    let manifest = match manifest_src {
        Some(s) => ManifestDto::parse(s, None)?,
        None => ManifestDto::default(),
    };
    let manifest_by_file: HashMap<&str, &super::dto::DatasetEntry> = manifest
        .datasets
        .iter()
        .map(|d| (d.file.as_str(), d))
        .collect();

    // Collect candidates per medium: (file_name, dataset, is_default).
    let mut by_medium: HashMap<MediumId, Vec<(String, IorDataset, bool)>> = HashMap::new();
    for item in items {
        let (file_name, src) = item?;
        if !file_name.ends_with(".ior.ron") {
            continue; // ignore non-".ior.ron" (incl. leftover .csv during migration)
        }
        if excluded(&file_name) {
            log::debug!("  -- excluded ior file: {file_name}");
            continue;
        }

        let path = format!("{label}/iors/{file_name}");
        let ds = super::dto::IorDatasetDto::parse(&src, Some(path.clone()))?.into_runtime(&path)?;

        // Consistency rule: filename `<medium>_` prefix (if recognized) and the
        // manifest entry (if present) must agree with the DTO's `medium`/`name`.
        if let Some((prefix, _)) = file_name.split_once('_') {
            if let Some(prefix_medium) = MediumId::try_from_name(prefix) {
                if prefix_medium != ds.medium {
                    return Err(Error::IoError(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "{file_name}: filename prefix says {:?} but file body says {:?}",
                            prefix_medium, ds.medium
                        ),
                    )));
                }
            }
        }
        let is_default = if let Some(m) = manifest_by_file.get(file_name.as_str()) {
            if let Some(medium) = MediumId::try_from_name(&m.medium) {
                if medium != ds.medium {
                    return Err(Error::IoError(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "{file_name}: sources.toml says medium {:?} but file body says {:?}",
                            m.medium, ds.medium
                        ),
                    )));
                }
            } else {
                return Err(Error::IoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{file_name}: sources.toml says medium {:?} but file body says {:?}",
                        m.medium, ds.medium
                    ),
                )));
            }
            m.default
        } else {
            false
        };
        by_medium
            .entry(ds.medium)
            .or_default()
            .push((file_name, ds, is_default));
    }

    // Resolve to one per medium.
    let mut chosen = HashMap::new();
    for (medium, mut candidates) in by_medium {
        let dataset = if candidates.len() == 1 {
            candidates.pop().unwrap().1
        } else {
            let defaults: Vec<_> = candidates.iter().filter(|c| c.2).collect();
            match defaults.len() {
                1 => {
                    let name = defaults[0].0.clone();
                    candidates.into_iter().find(|c| c.0 == name).unwrap().1
                },
                0 => {
                    let names: Vec<_> = candidates.iter().map(|c| c.0.as_str()).collect();
                    return Err(Error::IoError(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "{label}: {:?} has {} datasets and no `default = true` in \
                             sources.toml (nor all-but-one in `excluded_ior_files`): {names:?}",
                            medium,
                            candidates.len()
                        ),
                    )));
                },
                _ => {
                    let names: Vec<_> = defaults.iter().map(|c| c.0.as_str()).collect();
                    return Err(Error::IoError(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "{label}: {:?} has multiple `default = true` datasets: {names:?}",
                            medium
                        ),
                    )));
                },
            }
        };
        chosen.insert(medium, dataset);
    }
    Ok(chosen)
}

/// Filesystem layer: read `<dir>/sources.toml` + every file in `<dir>`.
fn resolve_dir(
    dir: &Path,
    excluded: &dyn Fn(&str) -> bool,
) -> Result<HashMap<MediumId, IorDataset>, Error> {
    if !dir.is_dir() {
        return Ok(HashMap::new());
    }
    let manifest_src = std::fs::read_to_string(dir.join("sources.toml")).ok();
    let rd = std::fs::read_dir(dir).map_err(Error::from)?;
    let items: LayerItems = Box::new(rd.filter_map(|e| {
        let path = match e {
            Ok(e) => e.path(),
            Err(e) => return Some(Err(Error::from(e))),
        };
        let name = path.file_name()?.to_str()?.to_string();
        if !name.ends_with(".ior.ron") {
            return None;
        }
        Some(
            std::fs::read_to_string(&path)
                .map(|s| (name, s))
                .map_err(Error::from),
        )
    }));
    resolve_layer(
        &dir.display().to_string(),
        manifest_src.as_deref(),
        items,
        excluded,
    )
}

/// Embedded baseline layer: the compiled-in `datafiles/ior/`.
/// Empty when built without the `embed-datafiles` feature.
fn resolve_embedded(
    excluded: &dyn Fn(&str) -> bool,
) -> Result<HashMap<MediumId, IorDataset>, Error> {
    #[cfg(not(feature = "embed-datafiles"))]
    {
        let _ = excluded;
        Ok(HashMap::new())
    }
    #[cfg(feature = "embed-datafiles")]
    {
        let manifest_src = EMBEDDED_IOR
            .get_file("sources.toml")
            .and_then(|f| f.contents_utf8())
            .map(str::to_owned);
        let items: LayerItems = Box::new(EMBEDDED_IOR.files().filter_map(|f| {
            let name = f.path().file_name()?.to_str()?.to_string();
            if !name.ends_with(".ior.ron") {
                return None;
            }
            match f.contents_utf8() {
                Some(s) => Some(Ok((name, s.to_owned()))),
                None => Some(Err(Error::IoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Embedded IOR file {:?} is not valid UTF-8", f.path()),
                )))),
            }
        }));
        resolve_layer(
            "<embedded datafiles/ior>",
            manifest_src.as_deref(),
            items,
            excluded,
        )
    }
}

impl IorRegLoader {
    /// Merge the three layers -- embedded baseline, then system, then user;
    /// later layers override earlier per medium.
    fn load_reg(&self) -> Result<IorReg, Error> {
        let excluded = |f: &str| self.is_excluded(f);
        let system = match &self.sys_dir {
            Some(d) => resolve_dir(d, &excluded)?,
            None => HashMap::new(),
        };
        let user = match &self.usr_dir {
            Some(d) => resolve_dir(d, &excluded)?,
            None => HashMap::new(),
        };
        let layers = [
            (Provenance::Builtin, resolve_embedded(&excluded)?),
            (Provenance::System, system),
            (Provenance::User, user),
        ];
        let merged = merge_layers(layers, MergePolicy::LastWins)
            .expect("merge ior layers with LastWins never returns Err");

        let mut reg = IorReg::new();
        for (medium, (mut dataset, provenance)) in merged {
            dataset.provenance = Some(provenance);
            reg.0.insert(medium, dataset);
        }

        if reg.0.is_empty() {
            // Not an error in principle (e.g. `--no-default-features` with no
            // dirs configured), but almost always a misconfiguration.
            log::warn!("IOR registry is empty: no embedded baseline and no sys/user datasets");
        }
        log::debug!("Loaded IOR registry: {} datasets", reg.0.len());
        Ok(reg)
    }
}

impl AssetLoader for IorRegLoader {
    fn asset_type(&self) -> &'static str { IorReg::asset_type() }

    fn asset_type_id(&self) -> AssetTypeId { IorReg::asset_type_id() }

    fn load(&self, path: Option<&Path>) -> Result<Box<dyn Asset>, Error> {
        if crate::utils::medium::registry().is_none() {
            return Err(Self::spine_not_ready_error());
        }
        match path {
            // Explicit single directory: resolve it alone (no embedded baseline).
            Some(p) => {
                let excluded = |f: &str| self.is_excluded(f);
                let mut reg = IorReg::new();
                for (m, ds) in resolve_dir(p, &excluded)? {
                    reg.0.insert(m, ds);
                }
                log::debug!("Loaded IOR registry from {:?}: {} datasets", p, reg.0.len());
                Ok(Box::new(reg))
            },
            None => Ok(Box::new(self.load_reg()?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        optics::ior::{write_dataset_file, IorData, IorDataset, IorRecord},
        units::nm,
        utils::medium::bootstrap,
    };

    fn ds(medium: MediumId, name: &str) -> IorDataset {
        IorDataset {
            medium,
            name: name.into(),
            reference: String::new(),
            comments: String::new(),
            data: IorData::Tabulated(Box::from([
                IorRecord::new(nm!(400.0), 1.0, 5.0),
                IorRecord::new(nm!(500.0), 2.0, 7.0),
            ])),
            provenance: None,
        }
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("vgn-iorload-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("ior")).unwrap();
        d
    }

    #[test]
    fn single_dataset_is_chosen() {
        let _ = bootstrap(None, None);
        let root = tmpdir("single");
        write_dataset_file(&root.join("ior/al_A.ior.ron"), &ds(MediumId::AL, "A")).unwrap();
        let loader = IorRegLoader::new(Some(&root), None, None);
        let reg = match loader.load(None).unwrap().into_any().downcast::<IorReg>() {
            Ok(b) => *b,
            Err(_) => unreachable!(),
        };
        assert_eq!(reg.get(&MediumId::AL).unwrap().name, "A");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn two_datasets_no_default_is_error() {
        let _ = bootstrap(None, None);
        let root = tmpdir("ambig");
        write_dataset_file(&root.join("ior/al_A.ior.ron"), &ds(MediumId::AL, "A")).unwrap();
        write_dataset_file(&root.join("ior/al_B.ior.ron"), &ds(MediumId::AL, "B")).unwrap();
        let loader = IorRegLoader::new(Some(&root), None, None);
        // `.err().unwrap()` rather than `.unwrap_err()`: the Ok type
        // `Box<dyn Asset>` is not `Debug`, which `Result::unwrap_err` requires.
        let err = loader.load(None).err().unwrap();
        assert!(
            format!("{err}").contains("no `default = true`"),
            "got: {err}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn default_flag_picks_the_winner_and_excluded_disambiguates() {
        let _ = bootstrap(None, None);
        let root = tmpdir("default");
        write_dataset_file(&root.join("ior/al_A.ior.ron"), &ds(MediumId::AL, "A")).unwrap();
        write_dataset_file(&root.join("ior/al_B.ior.ron"), &ds(MediumId::AL, "B")).unwrap();
        // via sources.toml default:
        std::fs::write(
            root.join("ior/sources.toml"),
            "[[dataset]]\nfile=\"al_A.ior.ron\"\nmedium=\"al\"\ndefault=true\n[[dataset]]\nfile=\"\
             al_B.ior.ron\"\nmedium=\"al\"\n",
        )
        .unwrap();
        let loader = IorRegLoader::new(Some(&root), None, None);
        let reg = *loader
            .load(None)
            .unwrap()
            .into_any()
            .downcast::<IorReg>()
            .ok()
            .unwrap();
        assert_eq!(reg.get(&MediumId::AL).unwrap().name, "A");
        // via excluded:
        let loader = IorRegLoader::new(Some(&root), None, Some(vec!["al_A.ior.ron".into()]));
        // with A excluded and B not default, B is the only candidate ⇒ chosen.
        let reg = *loader
            .load(None)
            .unwrap()
            .into_any()
            .downcast::<IorReg>()
            .ok()
            .unwrap();
        assert_eq!(reg.get(&MediumId::AL).unwrap().name, "B");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn filename_body_mismatch_is_rejected() {
        let _ = bootstrap(None, None);
        let root = tmpdir("mismatch");
        // file named al_* but body says copper
        write_dataset_file(&root.join("ior/al_Bad.ior.ron"), &ds(MediumId::CU, "Bad")).unwrap();
        let loader = IorRegLoader::new(Some(&root), None, None);
        // `.err().unwrap()` rather than `.unwrap_err()`: the Ok type
        // `Box<dyn Asset>` is not `Debug`, which `Result::unwrap_err` requires.
        let err = loader.load(None).err().unwrap();
        assert!(
            format!("{err}").contains("filename prefix says"),
            "got: {err}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn prefixless_file_loads_on_its_medium_field() {
        let _ = bootstrap(None, None);
        let root = tmpdir("prefixless");
        write_dataset_file(
            &root.join("ior/copper-data.ior.ron"),
            &ds(MediumId::CU, "X"),
        )
        .unwrap();
        let loader = IorRegLoader::new(Some(&root), None, None);
        let reg = *loader
            .load(None)
            .unwrap()
            .into_any()
            .downcast::<IorReg>()
            .ok()
            .unwrap();
        assert_eq!(reg.get(&MediumId::CU).unwrap().name, "X");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn stale_manifest_entry_is_ignored() {
        let _ = bootstrap(None, None);
        let root = tmpdir("stale");
        write_dataset_file(&root.join("ior/al_A.ior.ron"), &ds(MediumId::AL, "A")).unwrap();
        std::fs::write(
            root.join("ior/sources.toml"),
            "[[dataset]]\nfile=\"al_A.ior.ron\"\nmedium=\"al\"\n[[dataset]]\nfile=\"al_GONE.ior.\
             ron\"\nmedium=\"al\"\ndefault=true\n",
        )
        .unwrap();
        // al_GONE doesn't exist on disk ⇒ ignored; al_A is the only real candidate ⇒ chosen.
        let loader = IorRegLoader::new(Some(&root), None, None);
        let reg = *loader
            .load(None)
            .unwrap()
            .into_any()
            .downcast::<IorReg>()
            .ok()
            .unwrap();
        assert_eq!(reg.get(&MediumId::AL).unwrap().name, "A");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn user_dir_overrides_system_dir_per_medium() {
        let _ = bootstrap(None, None);
        let sys = tmpdir("sys");
        let usr = tmpdir("usr");
        write_dataset_file(&sys.join("ior/al_S.ior.ron"), &ds(MediumId::AL, "S")).unwrap();
        write_dataset_file(&sys.join("ior/cu_S.ior.ron"), &ds(MediumId::CU, "Scu")).unwrap();
        write_dataset_file(&usr.join("ior/al_U.ior.ron"), &ds(MediumId::AL, "U")).unwrap();
        let loader = IorRegLoader::new(Some(&sys), Some(&usr), None);
        let reg = *loader
            .load(None)
            .unwrap()
            .into_any()
            .downcast::<IorReg>()
            .ok()
            .unwrap();
        assert_eq!(reg.get(&MediumId::AL).unwrap().name, "U"); // user wins
        assert_eq!(reg.get(&MediumId::CU).unwrap().name, "Scu"); // system kept where user is silent
        std::fs::remove_dir_all(&sys).ok();
        std::fs::remove_dir_all(&usr).ok();
    }

    #[test]
    fn embedded_layer_resolves_without_error() {
        let _ = bootstrap(None, None);
        // Content-agnostic: the embedded baseline must resolve cleanly whether
        // `embed-datafiles` is on or off, and whether or not `datafiles/ior/`
        // has been populated yet by Task 8. (Deterministic content tests for
        // the embedded layer aren't possible until Task 8 commits the data;
        // the sys/user override behaviour is covered by the dir tests above,
        // and the embedded→sys→user precedence chain is exercised end-to-end
        // by the Task 9 workspace run against the real committed data.)
        assert!(resolve_embedded(&|_| false).is_ok());
    }

    /// End-to-end check of the *real shipped* embedded data: the committed
    /// `datafiles/ior/` must embed, parse, satisfy the consistency rule, and
    /// resolve one dataset per medium honouring `sources.toml` defaults.
    /// (Aluminium ships two datasets: McPeak2015 is `default = true`.)
    #[cfg(feature = "embed-datafiles")]
    #[test]
    fn embedded_baseline_contains_expected_shipped_datasets() {
        let _ = bootstrap(None, None);
        let reg = resolve_embedded(&|_| false).expect("embedded baseline must resolve");
        assert!(
            reg.contains_key(&MediumId::AL)
                && reg.contains_key(&MediumId::CU)
                && reg.contains_key(&MediumId::AIR),
            "expected al/cu/air in the embedded baseline, got: {:?}",
            reg.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            reg.get(&MediumId::AL).unwrap().name,
            "McPeak2015",
            "Aluminium must resolve to the sources.toml default (McPeak2015), not Cheng2016"
        );
    }

    #[test]
    fn explicit_dir_path_bypasses_embedded_and_layers() {
        let _ = bootstrap(None, None);
        // `load(Some(dir))` resolves exactly that directory -- no embedded
        // baseline, no sys/user merge.
        let only = tmpdir("explicit");
        write_dataset_file(&only.join("ior/cu_Only.ior.ron"), &ds(MediumId::CU, "Only")).unwrap();
        let loader = IorRegLoader::new(None, None, None);
        let reg = *loader
            .load(Some(&only.join("ior")))
            .unwrap()
            .into_any()
            .downcast::<IorReg>()
            .ok()
            .unwrap();
        assert_eq!(reg.get(&MediumId::CU).unwrap().name, "Only");
        std::fs::remove_dir_all(&only).ok();
    }

    #[test]
    fn user_layer_dataset_records_user_provenance() {
        let _ = crate::utils::medium::bootstrap(None, None);
        let root = tmpdir("prov");
        write_dataset_file(&root.join("ior/al_U.ior.ron"), &ds(MediumId::AL, "U")).unwrap();
        let loader = IorRegLoader::new(None, Some(&root), None);
        let reg = *loader
            .load(None)
            .unwrap()
            .into_any()
            .downcast::<IorReg>()
            .ok()
            .unwrap();
        assert_eq!(
            reg.get(&MediumId::AL).unwrap().provenance,
            Some(crate::utils::medium::Provenance::User)
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn spine_check_message_is_clear() {
        // `require_spine` returns the precondition error; assert its
        // wording without depending on global bootstrap state.
        let err = IorRegLoader::spine_not_ready_error();
        assert!(format!("{err}").contains("medium::bootstrap"), "got: {err}");
    }
}
