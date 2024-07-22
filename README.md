# VGonio

VGonio is an advanced tool designed to measure the geometric properties of materials described by microfacet-based BRDF
models. By analyzing a material's microstructure, VGonio can estimate the Normal Distribution Function (NDF), the
Geometric Attenuation Function (GAF), and the Bidirectional Reflectance Distribution Function (BRDF). This tool is
specifically created to tackle the complexities associated with multiple scattering phenomena in BRDF models, offering a
thorough analysis of the material's behaviour under various lighting conditions.

## Features

* **NDF Estimation**: VGonio can estimate the Normal Distribution Function (NDF) of a material by analyzing its
  microstructure.
* **GAF Estimation**: The Geometric Attenuation Function (GAF) is another essential parameter that VGonio can estimate.
* **BRDF Estimation**: By combining the NDF and GAF, VGonio can calculate the Bidirectional Reflectance Distribution
  Function (BRDF) of a material.

## Download

You can download the latest version of VGonio from the [releases page](https://github.com/matthiascy/vgonio/releases).

## Documentation

To support latex, when generating the documentation, extra HTML header content will be added. With `cargo doc` or
`cargo rustdoc`, the `rustdocflags` defined in `./cargo/config.toml` will be used. With `cargo doc`, `rustdocflags` is
applied globally (applies to each dependent crate), it will mess up dependencies. Possible workarounds are to either add
`--no-deps` to the `cargo doc`, or use `cargo rustdoc` to generate documentation only for the root crate(this crate).
For more information, see cargo issue [#331](https://github.com/rust-lang/cargo/issues/331)
and rust pull request [#95691](https://github.com/rust-lang/rust/pull/95691).

## Building from Source

### Requirements

* [shaderc](https://github.com/google/shaderc)
* [Embree3](https://www.embree.org/)
* `zlib`/`zlib-ng`

The default cargo build uses [sccache](https://github.com/mozilla/sccache) and [mold](https://github.com/rui314/mold) to
speed up the build process. If you don't have them installed, you can disable them by commenting out the `rustc_wrapper`
and `rustflags` in `./cargo/config.toml`.

```toml
rustc_wrapper = ["sccache"] # use sccache as compiler
rustflags = ["-Clink-arg=-fuse-ld=mold"] # use mold as linker
```

## Usage

### Measurement

```shell
vgonio measure [OPTIONS]
```

#### NDF Estimation

### `vgonio fit`

```shell
fit ~/Documents/virtual-gonio/output/test-data-d2/bsdf_aluminium3bar100_2024-02-28T21-58-41.vgmo --normalize --astart 0.175 --astop 0.225 --astep 0.001 -e nlls --model beckmann
```

## Environment Variables

* `ORIGINAL_MAX`
* `DENSE`

## Contributions

Contributions are welcome!
For feature requests and bug reports, please submit an issue.
For code contributions, please submit a pull request.

## License