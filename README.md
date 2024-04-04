# vgonio

## Requirements

* [shaderc](https://github.com/google/shaderc)
* [Embree3](https://www.embree.org/)(optional)

## Documentation

To support latex, when generating the documentation extra html header content will be added. With `cargo doc` or
`cargo rustdoc`, the `rustdocflags` defined in `./cargo/config.toml` will be used. With `cargo doc`, `rustdocflags` is
applied globally (applies to each dependent crate), it will mess up dependencies. Possible workarounds are to either add
`--no-deps` to the `cargo doc`, or use `cargo rustdoc` to generate documentation only for the root crate(this crate).
For more information, see cargo issue [#331](https://github.com/rust-lang/cargo/issues/331)
and rust pull request [#95691](https://github.com/rust-lang/rust/pull/95691).

## Subcommands

### `vgonio fit`

```shell
fit ~/Documents/virtual-gonio/output/test-data-d2/bsdf_aluminium3bar100_2024-02-28T21-58-41.vgmo --normalize --astart 0.175 --astop 0.225 --astep 0.001 -e nlls --model beckmann
```