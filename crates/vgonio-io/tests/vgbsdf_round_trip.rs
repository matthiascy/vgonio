use std::path::{Path, PathBuf};

use tempfile::NamedTempFile;
use vgn_io::vgbsdf::VgbsdfReader;

#[test]
fn rejects_unknown_conventions_version() {
    // Build a minimal valid archive, then patch manifest.toml to a bogus
    // conventions version and verify open() rejects it.
    let (path, _tmp) = build_minimal_bsdf_archive_for_test();
    patch_manifest_toml(&path, |s| {
        s.replace("conventions = \"v1\"", "conventions = \"v999\"")
    });
    let result = VgbsdfReader::open(&path);
    assert!(
        result.is_err(),
        "expected open() to fail on unknown conventions version"
    );
    let Err(err) = VgbsdfReader::open(&path) else {
        unreachable!()
    };
    assert!(
        err.message().contains("vgonio.conventions"),
        "expected conventions error, got {:?}",
        err.message()
    );
}

#[test]
fn rejects_missing_required_exr() {
    // Build a minimal archive then remove l0/disc.exr; validate_levels()
    // must complain that a declared member is missing.
    let (path, _tmp) = build_minimal_bsdf_archive_for_test();
    remove_zip_member(&path, "l0/disc.exr");
    let mut r = VgbsdfReader::open(&path).unwrap();
    let err = r.validate_levels().unwrap_err();
    assert!(
        err.message().contains("l0/disc.exr"),
        "expected missing-member error, got {}",
        err.message()
    );
}

#[test]
fn rejects_channel_count_mismatch() {
    // Build a 2-wavelength archive, patch spectrum.toml to claim 3 wavelengths,
    // and verify validate_levels() catches the inconsistency.
    let (path, _tmp) = build_minimal_bsdf_archive_for_test();
    patch_spectrum_toml(&path, vec![400.0, 500.0, 600.0]); // archive has 2 channels
    let mut r = VgbsdfReader::open(&path).unwrap();
    let err = r.validate_levels().unwrap_err();
    assert!(
        err.message().contains("channel count"),
        "expected channel-count error, got {}",
        err.message()
    );
}

#[test]
fn rejects_patches_exr_wrong_width() {
    // Patch n_patches in partition.toml so it disagrees with the actual EXR.
    let (path, _tmp) = build_minimal_bsdf_archive_for_test();
    patch_partition_n_patches(&path, 999);
    let mut r = VgbsdfReader::open(&path).unwrap();
    let err = r.validate_levels().unwrap_err();
    assert!(
        err.message().contains("patches.exr"),
        "expected patches-dim error, got {}",
        err.message()
    );
}

fn build_minimal_bsdf_archive_for_test() -> (PathBuf, NamedTempFile) { todo!() }
fn patch_manifest_toml(path: &Path, edit: impl FnOnce(String) -> String) { todo!() }
fn patch_spectrum_toml(path: &Path, wavelengths: Vec<f32>) { todo!() }
fn patch_partition_n_patches(path: &Path, n: u32) { todo!() }
fn remove_zip_member(path: &Path, name: &str) { todo!() }
