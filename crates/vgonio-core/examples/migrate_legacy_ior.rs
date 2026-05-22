//! One-time migration: converts the legacy `datafiles/ior/*.csv` files into vgonio-native
//! `*.ior.ron` files, and seeds `datafiles/ior/sources.toml`.
//!
//! Run from the workspace root: `cargo run -p vgonio-core --example migrate_legacy_ior`

use std::path::Path;
use vgn_core::{
    optics::{write_dataset_file, DatasetEntry, IorDatasetDto, ManifestDto},
    utils::medium::MediumId,
};

fn main() {
    // workspace root + `datafiles/ior`
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ior = root.join("datafiles/ior");

    // (legacy CSV file, medium, output stem, default?, manifest path, verified)
    let jobs: &[(&str, MediumId, &str, bool, Option<&str>, bool)] = &[
        (
            "air_iors_[0.23-1.69]_Ciddor1996.csv",
            MediumId::AIR,
            "Ciddor1996",
            true,
            Some("other/mixed gases/air/nk/Ciddor.yml"),
            true,
        ),
        (
            "al_iors_[0.15-1.7]_McPeak2015.csv",
            MediumId::AL,
            "McPeak2015",
            true,
            Some("main/Al/nk/McPeak.yml"),
            true,
        ),
        (
            "al_iors_[0.225-1.0]_Cheng2016.csv",
            MediumId::AL,
            "Cheng2016",
            false,
            Some("main/Al/nk/Cheng.yml"),
            true,
        ),
        (
            "cu_iors_[0.3-1.7]_McPeak2015.csv",
            MediumId::CU,
            "McPeak2015",
            true,
            Some("main/Cu/nk/McPeak.yml"),
            true,
        ),
    ];

    let mut manifest = ManifestDto {
        upstream: Some("https://github.com/polyanskiy/refractiveindex.info-database".into()),
        default_ref: None,
        datasets: Vec::new(),
    };

    for &(csv_name, medium, stem, is_default, path, verified) in jobs {
        let csv_path = ior.join(csv_name);
        let dto = IorDatasetDto::from_legacy_csv(&csv_path, medium, stem)
            .unwrap_or_else(|e| panic!("{csv_name}: {e}"));
        let runtime = dto.clone().into_runtime(csv_name).unwrap();
        let out_name = format!("{}_{}.ior.ron", medium.name(), stem);
        write_dataset_file(&ior.join(&out_name), &runtime)
            .unwrap_or_else(|e| panic!("{out_name}: {e}"));
        manifest.datasets.push(DatasetEntry {
            file: out_name,
            medium: medium.name().to_string(),
            default: is_default,
            path: path.map(str::to_string),
            git_ref: None,
            sha256: None,
            verified,
        });
        println!("migrated {csv_name} -> {}_{}.ior.ron", medium.name(), stem);
    }

    manifest.write(&ior.join("sources.toml")).unwrap();
    println!("wrote {}", ior.join("sources.toml").display());
}
