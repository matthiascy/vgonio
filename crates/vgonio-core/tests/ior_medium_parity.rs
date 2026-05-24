//! Parity test: after the Medium -> MediumId rename, the IOR loader still
//! produces the same lookup results for each shipped `.ior.ron` dataset.

use std::path::PathBuf;
use vgn_core::{
    optics::{IorReg, IorRegLoader},
    res::AssetLoader,
    units::nm,
    utils::medium::{bootstrap, MediumId},
};

fn datafiles_ior_dir() -> PathBuf {
    // The repo's datafiles/ior/ directory. `CARGO_MANIFEST_DIR` is the crate root
    // (crates/vgonio-core); the repo root is two `..` up.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("datafiles/ior")
}

/// Load `datafiles/ior` once via the new loader, bypassing embedded/sys/user layering.
fn load_shipped_datasets() -> IorReg {
    let _ = bootstrap(None, None);
    let loader = IorRegLoader::new(None, None, None);
    let boxed = loader
        .load(Some(&datafiles_ior_dir()))
        .expect("IorRegLoader::load(Some(datafiles/ior)) should succeed");
    *boxed
        .into_any()
        .downcast::<IorReg>()
        .ok()
        .expect("loaded asset must be IorReg")
}

#[test]
fn ior_loads_all_shipped_media_via_mediumid() {
    let dir = datafiles_ior_dir();
    if !dir.exists() {
        eprintln!("skipping: {} not found", dir.display());
        return;
    }
    let reg = load_shipped_datasets();

    // Every shipped dataset's medium should be reachable as a `MediumId` key.
    for &mid in &[MediumId::AIR, MediumId::AL, MediumId::CU] {
        let dataset = reg.get(&mid);
        assert!(dataset.is_some(), "expected an IorDataset for {mid:?}");
        // One dataset per medium under post-IOR — no `Vec` of records to check empty-ness;
        // instead check the dataset can evaluate at one wavelength. The per-dataset method
        // is `ior_at(Nanometres) -> Option<IorRecord>` (defined in
        // `crates/vgonio-core/src/optics/ior/mod.rs`, around the IorDataset impl block).
        // The plural `ior_of_spectrum` exists only on `IorReg`, not on `IorDataset`.
        let ds = dataset.unwrap();
        assert!(
            ds.ior_at(nm!(550.0)).is_some(),
            "{mid:?}: dataset {:?} has no IOR @ 550 nm",
            ds.name,
        );
    }
}

#[test]
fn ior_of_spectrum_works_for_aluminium() {
    let dir = datafiles_ior_dir();
    if !dir.exists() {
        eprintln!("skipping: {} not found", dir.display());
        return;
    }
    let reg = load_shipped_datasets();
    let lambdas = [nm!(400.0), nm!(550.0), nm!(700.0)];
    let iors = reg
        .ior_of_spectrum(MediumId::AL, &lambdas)
        .expect("Al spectrum");
    assert_eq!(iors.len(), 3);
    // Al's k > 0 in the visible range — exercises the Tabulated/Dispersion code
    // path through ior_of_spectrum without inspecting IorDataset.data directly.
    assert!(iors.iter().all(|i| i.k > 0.0));
}
