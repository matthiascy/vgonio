## Unreleased

### ✨ Overview & highlights

- Remove nightly channel specific date
- Fix the initial microfacet total area calculation
- Unify the surface subdivision scheme
- Enable manually setting the height offset during the wiggly surface subdivision
- Define the random height offset range for the wiggly surface subdivision
    - the base height offset is the diagonal of one height field cell
    - the actual offset is expressed as a percentage of the base offset
    - described as `/path/to/the/surface.vgmo ~~ wiggly l2 k100` in measurement description file
- Enable excluding data points during BRDF fitting
- Enable logarithmic scale for the residual calculation (non filtered) from the paper: BRDF Models for Accurate and
  Efficient Rendering of Glossy Surfaces
- Support for saving height field files in EXR image format: single channel 32-bit float with extra surface information
- Implement cargo like external subcommands
- Load BRDF data measured from RGL's BRDF database
- Enable excluding ior files in configuration
- Replace the per-medium IOR `*.csv` files with a vgonio-native, versioned `*.ior.ron`
  format (RON), described by a `sources.toml` manifest
    - the registry now holds one dataset per medium (it previously merged every CSV for
      a medium); the default Aluminium dataset is now `McPeak2015`
    - baseline datasets are compiled into the binary (default-on `embed-datafiles`
      feature) and overridden by the system then user data dirs. If present,
      `--no-default-features` drops the embedded baseline for distro packaging
    - `excluded_ior_files` entries must now name `.ior.ron` files
    - out-of-range wavelength lookups now return no value instead of panicking
    - dispersion-formula datasets (refractiveindex.info forms 1-9) are now supported
- Replace the hand-written `Medium` enum with `MediumId`, a `Copy` wrapper around a
  registry-interned canonical name (`vgn_core::utils::medium::MediumId`)
    - shipped consts: `MediumId::{VACUUM, AIR, AL, CU, NI, PVC, CR}`; `Medium::Unknown`
      is gone — use `Option<MediumId>`; `MaterialKind` and `Medium::kind()` removed as
      dead code (the answer was already derivable from `Ior::is_dielectric()` /
      `is_conductor()`)
    - a three-layer registry — embedded `builtin.toml` baseline plus optional
      system/user `media.toml` overrides — is bootstrapped once at startup via
      `medium::bootstrap(sys, user)`; lookups go through `medium::registry()` (returns
      `Option<&'static MediumRegistry>`, non-panicking pre-bootstrap)
    - `MediumId::try_from_name(s)` resolves a canonical name *or* an alias to the
      same canonical id; `Display` prints the registry's `display_name`
      post-bootstrap (e.g. `"Aluminium"`), and falls back to the canonical name
      pre-bootstrap (e.g. `"al"`)
    - the BSDF measurement file format stays at version `0.1.0`; reads/writes route
      through a private legacy 7-symbol codec
      (`crates/vgonio-app/src/io/legacy_medium.rs`) closed over the pre-`MediumId`
      symbol set so every emitted file remains migratable by the future
      `cargo x bsdf migrate` (Plan 2)
    - serde round-trips canonical names (`MediumId::AL` ↔ `"al"`); historical
      spellings continue to deserialize via aliases (`"chromium"`, `"aluminium"`, …)
      but writers always emit the canonical form
    - the GUI medium picker is now registry-driven (sorted by canonical name); MERL
      filename parsing routes through `MediumId::try_from_name` instead of a
      hard-coded match arm
    - the IOR registry is now keyed by `MediumId` (`HashMap<MediumId, IorDataset>`);
      `IorRegLoader` enforces a filename/manifest/DTO consistency rule and shares the
      `merge_layers` machinery with the medium registry. Adding a new medium becomes
      a data-only change: drop a `[[medium]]` block into `media.toml`, drop a
      `<name>_<source>.ior.ron` into the IOR dir (and register it in `sources.toml`)
    - **adding a medium previously required a `Medium` variant + match-arm edits in
      ~13 files; it is now a data-only change**
- New fitting interface
- Remove `alpha_start`, `alpha_stop`, `alpha_step` from CLI
- Rearrange the crates in the workspace into /bins and /libs
- Merge `vgonio-bxdf` into `vgonio-base`
- Rename `MeasuredData` trait to `AnyMeasured`
- Rename `MeasuredBrdfLevel` to `BrdfLevel`
- Remove `AnalyticalFit` trait
- New `AnyMeasuredBrdf` trait
- Rename `RawMeasuredBsdfData` to `RawBsdfMeasurement`
- Overhaul the project structure and module organization
- Update dependencies
- Development facilities improvements (xtask)
- Add LZ4 as a body compression scheme for cache/measurement files (`.vgmo`), alongside zlib and gzip
- Fix zlib-compressed sample data being silently truncated on large payloads (the encoder was
  sync-flushed instead of finished, leaving an incomplete stream)

## 0.3.2 - 2024-08-16

- Replace native file dialogue [rfd](https://crates.io/crates/rfd)
  with [egui-file-dialog](https://crates.io/crates/egui-file-dialog).
    - remove `rfd` dependency
- Use only WGSL shaders
    - remove build script
    - remove `shaderc` build dependency
- Enable subdivision of the surface on the fly with UI
- Support subdivision description in the measurement description file
    - `/path/to/the/surface.vgmo ~~ curved/wiggly l2`
- Implement Doubly Connected Edge List (DCEL) data structure for mesh processing
- Isolate the UI from the main application
- Rename crate `gfxkit` to `gxtk`
- Rename crate `uikit` to `uxtk`
- Adopt [catpuccin](https://catppuccin.com/) pastel theme for the UI

## 0.3.1 - 2024-07-23

First public release.
