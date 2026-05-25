//! Asserts the archive shape: ONE .vgbsdf per measurement with per-level
//! subdirectories. Regression test for review item "High: spec says one
//! container, plan was writing one per level".

#![cfg(not(feature = "vdbg"))]

use std::io::Read;
use vgn_core::BrdfLevel;
use vgonio_app::measure::{bsdf::BsdfMeasurement, params::BsdfMeasurementParams};
use zip::ZipArchive;

#[test]
fn vgbsdf_has_one_container_with_per_level_subdirs() {
    let measurement = build_two_level_bsdf_measurement_for_test();

    let tmp = tempfile::Builder::new()
        .suffix(".vgbsdf")
        .tempfile()
        .unwrap();
    let path = tmp.path().to_path_buf();
    drop(tmp);

    measurement
        .write_as_vgbsdf(&path, &chrono::Local::now(), 64)
        .unwrap();

    // The output MUST be a single file (not multiple per-level files).
    assert!(path.is_file(), "expected one .vgbsdf file, not a directory");
    let sibling_l1 = path.with_file_name(format!(
        "{}_l1.vgbsdf",
        path.file_stem().unwrap().to_str().unwrap()
    ));
    assert!(
        !sibling_l1.exists(),
        "found per-level sibling {sibling_l1:?} — the writer is still emitting one-per-level files"
    );

    // Open and assert the tree.
    let file = std::fs::File::open(&path).unwrap();
    let mut zip = ZipArchive::new(file).unwrap();
    let names: Vec<String> = (0..zip.len())
        .map(|i| zip.by_index(i).unwrap().name().to_string())
        .collect();
    assert!(names.iter().any(|n| n == "manifest.toml"));
    assert!(names.iter().any(|n| n == "partition.toml"));
    assert!(names.iter().any(|n| n == "incident_grid.toml"));
    assert!(names.iter().any(|n| n == "spectrum.toml"));
    for level in ["l0", "l1"] {
        for enc in ["disc.exr", "thetaphi.exr", "patches.exr"] {
            let expected = format!("{level}/{enc}");
            assert!(
                names.iter().any(|n| n == &expected),
                "missing {expected}; archive members were: {names:#?}"
            );
        }
    }

    // Manifest.bsdf.levels lists both levels (structurally, not by string match —
    // toml-pretty's array formatting varies with array length).
    let mut manifest_str = String::new();
    zip.by_name("manifest.toml")
        .unwrap()
        .read_to_string(&mut manifest_str)
        .unwrap();
    let manifest: vgn_io::vgbsdf::Manifest =
        toml::from_str(&manifest_str).expect("manifest.toml is not valid TOML");
    let levels = manifest
        .bsdf
        .as_ref()
        .expect("manifest must have [bsdf] block")
        .levels
        .clone();
    assert_eq!(
        levels,
        vec!["l0".to_string(), "l1".to_string()],
        "manifest.bsdf.levels mismatch"
    );
    assert_eq!(manifest.archive.kind, "bsdf");
    assert_eq!(manifest.vgonio.conventions, "v1");

    std::fs::remove_file(&path).ok();
}

/// Builds a `BsdfMeasurement` whose `bsdfs` map contains exactly `L0` and `L1`.
///
/// `synthesise_minimal` already produces multiple levels (L0..L3 plus L1Plus)
/// because the synthetic per-bounce energy is non-zero for every bounce in the
/// fixture. We retain only `L0` and `L1` so the regression-test assertion on
/// `manifest.bsdf.levels = ["l0", "l1"]` is deterministic.
fn build_two_level_bsdf_measurement_for_test() -> BsdfMeasurement {
    let params = BsdfMeasurementParams::default();
    let mut bsdf = BsdfMeasurement::synthesise_minimal(&params);
    bsdf.bsdfs
        .retain(|level, _| matches!(level, BrdfLevel::L0 | BrdfLevel::L1));
    assert!(
        bsdf.bsdfs.contains_key(&BrdfLevel::L0) && bsdf.bsdfs.contains_key(&BrdfLevel::L1),
        "synthesise_minimal must produce L0 + L1 — got keys {:?}",
        bsdf.bsdfs.keys().collect::<Vec<_>>()
    );
    bsdf
}
