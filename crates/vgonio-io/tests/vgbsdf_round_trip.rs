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
    partition_toml::PartitionToml,
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

/// Build a minimal-but-valid `.vgbsdf` BSDF archive (single level "l0", 1 wi,
/// 2 wavelengths). Returns the path plus the `NamedTempFile` whose `Drop`
/// cleans the file up after the test.
fn build_minimal_bsdf_archive_for_test() -> (PathBuf, NamedTempFile) {
    let tmp = tempfile::Builder::new()
        .suffix(".vgbsdf")
        .tempfile()
        .unwrap();
    let path = tmp.path().to_path_buf();

    let partition = SphericalPartition::new_beckers(SphericalDomain::Upper, rad!(0.3));
    let n_patches = partition.n_patches();
    let n_spectrum = 2;
    let samples: Vec<f32> = (0..n_spectrum * n_patches)
        .map(|k| {
            let ch = k / n_patches;
            let p = k % n_patches;
            10.0 * ch as f32 + p as f32
        })
        .collect();
    let layer = HemisphereLayer {
        layer_name: "θ00_00.φ0_00".into(),
        channel_names: vec!["400nm".into(), "500nm".into()],
        samples: &samples,
    };

    let manifest = Manifest {
        vgonio: VgonioBlock {
            version: "0.2.0".into(),
            conventions: "v1".into(),
        },
        archive: ArchiveBlock {
            created: "2026-05-14T12:34:56Z".into(),
            kind: "bsdf".into(),
            description: None,
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
        positions: vec![[0.0, 0.0]],
    };

    let mut writer = VgbsdfWriter::create(&path).unwrap();
    writer
        .write_metadata(&manifest, &partition, Some(&incident_grid), &spectrum)
        .unwrap();
    writer
        .write_level(
            "l0",
            &partition,
            std::slice::from_ref(&layer),
            OutputKind::Bsdf,
            &chrono::Local::now(),
            64,
            (32, 8),
        )
        .unwrap();
    writer.finish().unwrap();

    (path, tmp)
}

/// Re-pack the archive: stream every member through `transform`. Returning
/// `Some(new_bytes)` replaces the member; `None` drops it. All members are
/// written back with `CompressionMethod::Stored` (matches the writer's
/// invariant).
fn rewrite_zip(path: &Path, mut transform: impl FnMut(&str, Vec<u8>) -> Option<Vec<u8>>) {
    use std::io::{Cursor, Read, Write};
    use zip::{write::SimpleFileOptions, CompressionMethod, ZipArchive, ZipWriter};

    let bytes = std::fs::read(path).unwrap();
    let mut archive = ZipArchive::new(Cursor::new(bytes)).unwrap();

    let mut members: Vec<(String, Vec<u8>)> = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let mut f = archive.by_index(i).unwrap();
        let name = f.name().to_string();
        let mut buf = Vec::with_capacity(f.size() as usize);
        f.read_to_end(&mut buf).unwrap();
        members.push((name, buf));
    }

    let out_file = std::fs::File::create(path).unwrap();
    let mut out = ZipWriter::new(std::io::BufWriter::new(out_file));
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    for (name, bytes) in members {
        if let Some(new_bytes) = transform(&name, bytes) {
            out.start_file(&name, opts).unwrap();
            out.write_all(&new_bytes).unwrap();
        }
    }
    out.finish().unwrap();
}

fn patch_manifest_toml(path: &Path, edit: impl FnOnce(String) -> String) {
    let mut edit_opt = Some(edit);
    rewrite_zip(path, |name, bytes| {
        if name == "manifest.toml" {
            let s = String::from_utf8(bytes).unwrap();
            let new_s = (edit_opt.take().unwrap())(s);
            Some(new_s.into_bytes())
        } else {
            Some(bytes)
        }
    });
}

fn patch_spectrum_toml(path: &Path, wavelengths: Vec<f32>) {
    let new_spec = SpectrumToml::nm(wavelengths);
    let new_bytes = toml::to_string_pretty(&new_spec).unwrap().into_bytes();
    let mut nb = Some(new_bytes);
    rewrite_zip(path, |name, bytes| {
        if name == "spectrum.toml" {
            Some(nb.take().unwrap())
        } else {
            Some(bytes)
        }
    });
}

fn patch_partition_n_patches(path: &Path, n: u32) {
    rewrite_zip(path, |name, bytes| {
        if name == "partition.toml" {
            let s = std::str::from_utf8(&bytes).unwrap();
            let mut pt: PartitionToml = toml::from_str(s).unwrap();
            pt.n_patches = n;
            Some(toml::to_string_pretty(&pt).unwrap().into_bytes())
        } else {
            Some(bytes)
        }
    });
}

fn remove_zip_member(path: &Path, name: &str) {
    rewrite_zip(path, |n, bytes| if n == name { None } else { Some(bytes) });
}

/// For any pixel in `disc.exr` or `thetaphi.exr` whose mapping resolves to patch index `p`, the
/// pixel value MUST equal the value at `(p, 0)` in `patches.exr` (within floating-point
/// representation, no compression-induced drift since EXR `SMALL_LOSSLESS`).
#[test]
fn cross_encoding_consistency() {
    use exr::prelude::*;
    use vgn_core::utils::partition::{write_hemisphere_exr, HemisphereEncoding};

    let partition = SphericalPartition::new_beckers(SphericalDomain::Upper, rad!(0.25));
    let n_patches = partition.n_patches();
    let samples: Vec<f32> = (0..n_patches).map(|i| (i as f32) * 0.01).collect();
    let layer = HemisphereLayer {
        layer_name: "x".into(),
        channel_names: vec!["value".into()],
        samples: &samples,
    };
    let ts = chrono::Local::now();
    let dir = tempfile::tempdir().unwrap();
    let disc_p = dir.path().join("disc.exr");
    let thetaphi_p = dir.path().join("thetaphi.exr");
    let patches_p = dir.path().join("patches.exr");

    let resolution = 128;
    let (n_phi, n_theta) = (64, 16);

    write_hemisphere_exr(
        &partition,
        HemisphereEncoding::Disc { resolution },
        std::slice::from_ref(&layer),
        &disc_p,
        &ts,
        &[],
    )
    .unwrap();
    write_hemisphere_exr(
        &partition,
        HemisphereEncoding::Thetaphi { n_phi, n_theta },
        std::slice::from_ref(&layer),
        &thetaphi_p,
        &ts,
        &[],
    )
    .unwrap();
    write_hemisphere_exr(
        &partition,
        HemisphereEncoding::Patches,
        std::slice::from_ref(&layer),
        &patches_p,
        &ts,
        &[],
    )
    .unwrap();

    // Read patches.exr into a flat Vec keyed by patch index.
    let patches_img = read_flat_grid(&patches_p).unwrap();
    let mut by_patch = vec![0.0_f32; n_patches];
    for p in 0..n_patches {
        by_patch[p] = patches_img.sample_at(p, 0);
    }

    // Verify disc pixels match by_patch (where they map to a patch).
    let disc_img = read_flat_grid(&disc_p).unwrap();
    let mut disc_idx = vec![0i32; (resolution * resolution) as usize];
    partition.compute_pixel_patch_indices(resolution, resolution, &mut disc_idx);
    let mut mismatches = 0;
    for j in 0..resolution {
        for i in 0..resolution {
            let pi = disc_idx[(i + j * resolution) as usize];
            if pi >= 0 {
                let want = by_patch[pi as usize];
                let got = disc_img.sample_at(i as usize, j as usize);
                if (want - got).abs() > 1e-6 {
                    mismatches += 1;
                }
            }
        }
    }
    assert_eq!(mismatches, 0, "disc-encoding cross-consistency failed");

    // Verify thetaphi pixels.
    let tp_img = read_flat_grid(&thetaphi_p).unwrap();
    let mut tp_idx = vec![0i32; (n_phi * n_theta) as usize];
    partition.compute_thetaphi_patch_indices(n_phi, n_theta, &mut tp_idx);
    let mut tp_mismatches = 0;
    for j in 0..n_theta {
        for i in 0..n_phi {
            let pi = tp_idx[(i + j * n_phi) as usize];
            if pi >= 0 {
                let want = by_patch[pi as usize];
                let got = tp_img.sample_at(i as usize, j as usize);
                if (want - got).abs() > 1e-6 {
                    tp_mismatches += 1;
                }
            }
        }
    }
    assert_eq!(
        tp_mismatches, 0,
        "thetaphi-encoding cross-consistency failed"
    );
}

/// Helper: read first layer, first channel of an EXR as a 2D f32 grid.
fn read_flat_grid(path: &std::path::Path) -> Result<FlatImage2D, Box<dyn std::error::Error>> {
    use exr::prelude::*;
    let image = exr::prelude::read_first_flat_layer_from_file(path)?;
    let layer = &image.layer_data;
    let (w, h) = (layer.size.width(), layer.size.height());
    let ch = &layer.channel_data.list[0];
    let data: Vec<f32> = match &ch.sample_data {
        FlatSamples::F32(v) => v.to_vec(),
        _ => return Err("expected F32 samples".into()),
    };
    Ok(FlatImage2D {
        width: w,
        height: h,
        data,
    })
}

struct FlatImage2D {
    width: usize,
    height: usize,
    data: Vec<f32>,
}
impl FlatImage2D {
    fn sample_at(&self, i: usize, j: usize) -> f32 { self.data[i + j * self.width] }
}
