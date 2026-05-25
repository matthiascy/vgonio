//! End-to-end: build a small BSDF measurement, write it via VgbsdfWriter,
//! re-open with VgbsdfReader, verify the manifest/partition/spectrum/grid round-trip
//! and that each EXR member is non-empty.
use std::path::{Path, PathBuf};

use tempfile::NamedTempFile;

use vgn_core::{
    units::rad,
    utils::partition::{HemisphereLayer, SphericalDomain, SphericalPartition},
};
use vgn_io::vgbsdf::{
    conventions::OutgoingEncoding,
    incident_grid::IncidentGridToml,
    manifest::{ArchiveBlock, BsdfBlock, MaterialBlock, VgonioBlock},
    spectrum::SpectrumToml,
    Manifest, OutputKind, VgbsdfReader, VgbsdfWriter,
};

#[test]
fn vgbsdf_round_trip() {
    // Tiny partition: 2 rings, a handful of patches.
    let partition = SphericalPartition::new_beckers(SphericalDomain::Upper, rad!(0.3));
    let n_patches = partition.n_patches();
    let n_spectrum = 2;
    let n_wi = 3;

    // Synthetic samples: layer i, channel j, patch p → 100*i + 10*j + p.
    let make_layer = |wi_idx: usize| -> Vec<f32> {
        (0..n_spectrum * n_patches)
            .map(|k| {
                let ch = k / n_patches;
                let p = k % n_patches;
                100.0 * wi_idx as f32 + 10.0 * ch as f32 + p as f32
            })
            .collect()
    };
    let samples: Vec<Vec<f32>> = (0..n_wi).map(make_layer).collect();

    let layers: Vec<HemisphereLayer<'_>> = (0..n_wi)
        .map(|i| HemisphereLayer {
            layer_name: format!("θ{:02}_00.φ0_00", i * 15),
            channel_names: vec!["400nm".into(), "500nm".into()],
            samples: &samples[i],
        })
        .collect();

    // Write archive.
    let tmp = tempfile::NamedTempFile::with_suffix(".vgbsdf").unwrap();
    let path = tmp.path().to_path_buf();
    drop(tmp);

    let manifest = Manifest {
        vgonio: VgonioBlock {
            version: "0.2.0".into(),
            conventions: "v1".into(),
        },
        archive: ArchiveBlock {
            created: "2026-05-14T12:34:56Z".into(),
            kind: "bsdf".into(),
            description: Some("round-trip".into()),
        },
        material: Some(MaterialBlock {
            incident_medium: "air".into(),
            transmitted_medium: "al".into(),
        }),
        bsdf: Some(BsdfBlock {
            levels: vec!["l0".into()],
            encodings: vec!["disc".into(), "thetaphi".into(), "patches".into()],
        }),
        provenance: None,
    };
    let spectrum = SpectrumToml::nm(vec![400.0, 500.0]);
    let incident_grid = IncidentGridToml {
        positions: vec![[0.0, 0.0], [0.261, 0.0], [0.523, 0.0]],
    };

    let mut writer = VgbsdfWriter::create(&path).unwrap();
    writer
        .write_metadata(&manifest, &partition, Some(&incident_grid), &spectrum)
        .unwrap();
    writer
        .write_level(
            "l0",
            &partition,
            &layers,
            OutputKind::Bsdf,
            &chrono::Local::now(),
            64,
            (32, 8),
        )
        .unwrap();
    writer.finish().unwrap();

    // Read back.
    let mut reader = VgbsdfReader::open(&path).unwrap();
    assert_eq!(reader.manifest.vgonio.conventions, "v1");
    assert_eq!(reader.manifest.archive.kind, "bsdf");
    assert_eq!(reader.spectrum.wavelengths, vec![400.0, 500.0]);
    assert_eq!(reader.incident_grid.as_ref().unwrap().positions.len(), 3);

    // Partition round-trips.
    let rebuilt = reader.partition.to_partition().unwrap();
    assert_eq!(rebuilt.n_patches(), partition.n_patches());

    // All three EXR members exist and are non-empty.
    for enc in [
        OutgoingEncoding::Disc,
        OutgoingEncoding::Thetaphi,
        OutgoingEncoding::Patches,
    ] {
        let bytes = reader.read_exr_bytes("l0", enc).unwrap();
        assert!(bytes.len() > 0, "encoding {} is empty", enc.as_str());
        // OpenEXR magic bytes — first 4 bytes are 0x76, 0x2F, 0x31, 0x01.
        assert_eq!(
            &bytes[..4],
            &[0x76, 0x2F, 0x31, 0x01],
            "{} not a valid EXR",
            enc.as_str()
        );
    }

    std::fs::remove_file(&path).ok();
}

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
