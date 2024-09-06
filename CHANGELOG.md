## Unreleased

### ✨ Overview & highlights

- Remove nightly channel specific date
- Fix the initial microfacet total area calculation
- Unify the surface subdivision scheme
- Enable manually setting the height offset during the wiggly surface subdivision

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

## 0.3.1 - 2024-06-23

First public release.