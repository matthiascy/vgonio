//! End-to-end VGMO round-trip with different compression schemes. Writes a small BSDF
//! measurement to a real .vgmo file and reads it back, asserting that samples
//! + headers round-trip byte-indentical.
//!
//! Skipped when the `vdbg` feature is active: `BsdfMeasurement` then carries
//! visual-debug fields (`trajectories`, `hit_points`) that the on-disk VGMO codec
//! doesn't serialize, so the in-memory equality check is intentionally not
//! applicable on that build.

#![cfg(not(feature = "vdbg"))]

use std::{
    fs::File,
    io::{BufReader, BufWriter},
};
use vgn_core::{
    io::{CompressionScheme, FileEncoding},
    BrdfLevel,
};

use vgonio_app::measure::{bsdf::BsdfMeasurement, params::BsdfMeasurementParams};

fn round_trip_with(scheme: CompressionScheme) {
    let params = BsdfMeasurementParams::default();
    let bsdf = BsdfMeasurement::synthesise_minimal(&params);

    let tmp = tempfile::Builder::new().suffix(".vgmo").tempfile().unwrap();
    let path = tmp.path().to_path_buf();

    drop(tmp);

    {
        let file = File::create(&path).unwrap();
        let mut w = BufWriter::new(file);
        bsdf.write_to_vgmo(&mut w, FileEncoding::Binary, scheme)
            .expect("write vgmo failed");
    }

    let size = std::fs::metadata(&path).unwrap().len();
    assert!(
        size > 64,
        "scheme={:?} produced too-small file ({} bytes)",
        scheme,
        size
    );

    {
        let file = File::open(&path).unwrap();
        let mut r = BufReader::new(file);
        let read_bsdf =
            BsdfMeasurement::read_from_vgmo(&mut r, &params, FileEncoding::Binary, scheme)
                .expect("read vgmo failed");

        assert_eq!(
            read_bsdf.raw, bsdf.raw,
            "scheme={:?}: raw payload mismatch",
            scheme
        );
        assert_eq!(
            read_bsdf.bsdfs.len(),
            bsdf.bsdfs.len(),
            "scheme={:?}: bsdf level count mismatch",
            scheme
        );
        assert_eq!(bsdf.params, read_bsdf.params);
        assert_eq!(
            bsdf.brdf_at(BrdfLevel::L0).unwrap().n_samples(),
            read_bsdf.brdf_at(BrdfLevel::L0).unwrap().n_samples()
        );
        for (s1, s2) in bsdf
            .brdf_at(BrdfLevel::L0)
            .unwrap()
            .samples
            .iter()
            .zip(read_bsdf.brdf_at(BrdfLevel::L0).unwrap().samples.iter())
        {
            assert_eq!(s1, s2);
        }

        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn round_trip_lz4() { round_trip_with(CompressionScheme::Lz4); }

#[test]
fn round_trip_none() { round_trip_with(CompressionScheme::None); }

#[test]
fn round_trip_zlib() { round_trip_with(CompressionScheme::Zlib); }

#[test]
fn round_trip_gzip() { round_trip_with(CompressionScheme::Gzip); }
