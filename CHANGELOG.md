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
- New fitting interface
- Remove `alpha_start`, `alpha_stop`, `alpha_step` from CLI
- Rearrange the crates in the workspace into /bins and /libs
- Merge `vgonio-bxdf` into `vgonio-base`
- Rename `MeasuredData` trait to `AnyMeasured`
- Rename `MeasuredBrdfLevel` to `BrdfLevel`
- Remove `AnalyticalFit` trait
- New `AnyMeasuredBrdf` trait
- Rename `RawMeasuredBsdfData` to `RawBsdfMeasurement`

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
