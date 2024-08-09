## Unreleased

### ✨ Overview & highlights

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

## 0.3.1 - 2024-06-23

First public release.