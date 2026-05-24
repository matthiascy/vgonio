//! Import a single refractiveindex.info dataset into the vgonio IOR layout.
//!
//! Reads one upstream YAML file (the `tabulated nk` variant — formula variants
//! aren't supported by this example yet; that's `cargo x ior add`'s job once
//! Plan 2 lands), writes the matching `<medium>_<name>.ior.ron`, and registers
//! it in `<ior_dir>/sources.toml`.
//!
//! Run from the workspace root:
//!
//! ```sh
//! cargo run -p vgonio-core --features cli --example add_ior -- \
//!     --yaml /tmp/Johnson.yml --medium ni --name Johnson1974 \
//!     --path "main/Ni/nk/Johnson.yml" --default
//! ```

use clap::Parser;
use serde::Deserialize;
use std::{fs, path::PathBuf};
use vgn_core::{
    optics::{DatasetEntry, IorData, IorDataset, IorRecord, ManifestDto},
    units::nanometres,
    utils::medium::MediumId,
};

#[derive(Parser, Debug)]
#[command(about = "Import a single refractiveindex.info YAML into datafiles/ior/")]
struct Args {
    /// Path to the upstream `.yml` from refractiveindex.info-database.
    #[arg(long)]
    yaml: PathBuf,
    /// Canonical medium name (must exist in builtin.toml or a loaded layer).
    #[arg(long)]
    medium: String,
    /// Source label used in the file name (`<medium>_<name>.ior.ron`) and the
    /// `name` field of the dataset.
    #[arg(long)]
    name: String,
    /// Catalog-relative upstream path recorded in sources.toml
    /// (e.g. `main/Ni/nk/Johnson.yml`).
    #[arg(long)]
    path: Option<String>,
    /// Mark this dataset as the medium's default. Required if a medium will
    /// have more than one dataset.
    #[arg(long)]
    default: bool,
    /// Record as `verified = false` in sources.toml (default is true).
    #[arg(long)]
    unverified: bool,
    /// Output directory holding `.ior.ron` files and `sources.toml`.
    #[arg(long, default_value = "datafiles/ior")]
    ior_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct UpstreamYaml {
    #[serde(rename = "REFERENCES", default)]
    references: String,
    #[serde(rename = "COMMENTS", default)]
    comments: String,
    #[serde(rename = "DATA")]
    data: Vec<UpstreamData>,
}

#[derive(Debug, Deserialize)]
struct UpstreamData {
    #[serde(rename = "type")]
    ty: String,
    data: String,
}

fn main() {
    let args = Args::parse();
    vgn_core::utils::medium::bootstrap(None, None).ok();

    let medium = MediumId::try_from_name(&args.medium)
        .unwrap_or_else(|| panic!("unknown medium {:?} — add it to builtin.toml first", args.medium));

    let yaml_text = fs::read_to_string(&args.yaml)
        .unwrap_or_else(|e| panic!("read {}: {e}", args.yaml.display()));
    let upstream: UpstreamYaml = serde_yaml::from_str(&yaml_text)
        .unwrap_or_else(|e| panic!("parse {}: {e}", args.yaml.display()));

    let tabulated_nk = upstream
        .data
        .iter()
        .find(|d| d.ty.trim() == "tabulated nk")
        .unwrap_or_else(|| {
            panic!(
                "{}: this example only supports DATA entries of type 'tabulated nk'; found {:?}",
                args.yaml.display(),
                upstream.data.iter().map(|d| d.ty.as_str()).collect::<Vec<_>>()
            )
        });

    let mut samples: Vec<IorRecord> = Vec::new();
    for (lineno, raw) in tabulated_nk.data.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split_ascii_whitespace().collect();
        if cols.len() != 3 {
            panic!(
                "{}:{}: expected 3 columns (wavelength_um, n, k), got {} ({:?})",
                args.yaml.display(),
                lineno + 1,
                cols.len(),
                line
            );
        }
        let parse = |c: &str, what: &str| -> f32 {
            c.parse::<f64>()
                .unwrap_or_else(|e| panic!("{}:{}: bad {what} {c:?}: {e}", args.yaml.display(), lineno + 1))
                as f32
        };
        let lambda_um = parse(cols[0], "wavelength");
        let n = parse(cols[1], "n");
        let k = parse(cols[2], "k");
        samples.push(IorRecord::new(nanometres!(lambda_um * 1000.0), n, k));
    }
    samples.sort_by(|a, b| a.wavelength.partial_cmp(&b.wavelength).unwrap());

    let dataset = IorDataset {
        medium,
        name: args.name.clone(),
        reference: upstream.references.trim().to_string(),
        comments: upstream.comments.trim().to_string(),
        data: IorData::Tabulated(samples.into_boxed_slice()),
        provenance: None,
    };

    let ior_dir = args.ior_dir;
    fs::create_dir_all(&ior_dir)
        .unwrap_or_else(|e| panic!("create {}: {e}", ior_dir.display()));
    let out_name = format!("{}_{}.ior.ron", medium.name(), args.name);
    let out_path = ior_dir.join(&out_name);
    dataset
        .write(&out_path)
        .unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
    println!("wrote {}", out_path.display());

    let manifest_path = ior_dir.join("sources.toml");
    let mut manifest = ManifestDto::read(&manifest_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest_path.display()));
    manifest.datasets.retain(|d| d.file != out_name);
    manifest.datasets.push(DatasetEntry {
        file: out_name.clone(),
        medium: medium.name().to_string(),
        default: args.default,
        path: args.path.clone(),
        git_ref: None,
        sha256: None,
        verified: !args.unverified,
    });
    manifest
        .write(&manifest_path)
        .unwrap_or_else(|e| panic!("write {}: {e}", manifest_path.display()));
    println!("updated {} (entry {out_name})", manifest_path.display());
}
