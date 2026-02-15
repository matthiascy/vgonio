//! CLI parser tests for fit command options.

use clap::Parser;
use vgonio_app::FitOptions;

#[derive(Debug, Parser)]
struct FitCli {
    #[command(flatten)]
    fit: FitOptions,
}

fn base_args(symmetry: &str) -> Vec<String> {
    vec![
        "fit-tests".into(),
        "input.vgmo".into(),
        "--kind".into(),
        "vgonio".into(),
        "--family".into(),
        "microfacet".into(),
        "--distro".into(),
        "trowbridge".into(),
        "--symmetry".into(),
        symmetry.into(),
        "--method".into(),
        "brute".into(),
        "--level".into(),
        "l0".into(),
        "--err".into(),
        "mse".into(),
    ]
}

#[test]
fn parses_valid_isotropic_roughness_range() {
    let mut args = base_args("isotropic");
    args.extend(["--a".into(), "0.1:0.5:0.01".into()]);
    let parsed = FitCli::try_parse_from(args).unwrap();
    assert_eq!(parsed.fit.a, Some([0.1, 0.5, 0.01]));
    assert!(parsed.fit.ax.is_none());
    assert!(parsed.fit.ay.is_none());
}

#[test]
fn rejects_non_numeric_roughness_value() {
    let mut args = base_args("isotropic");
    args.extend(["--a".into(), "a:b:c".into()]);
    let err = FitCli::try_parse_from(args).unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
}

#[test]
fn requires_ay_when_ax_is_provided() {
    let mut args = base_args("anisotropic");
    args.extend(["--ax".into(), "0.1:0.5:0.01".into()]);
    let err = FitCli::try_parse_from(args).unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
}

#[test]
fn rejects_conflicting_isotropic_and_anisotropic_ranges() {
    let mut args = base_args("anisotropic");
    args.extend([
        "--a".into(),
        "0.1:0.5:0.01".into(),
        "--ax".into(),
        "0.1:0.5:0.01".into(),
        "--ay".into(),
        "0.1:0.5:0.01".into(),
    ]);
    let err = FitCli::try_parse_from(args).unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
}

#[test]
fn parses_per_wavelength_anisotropic_ranges_from_paths() {
    let mut args = base_args("anisotropic");
    args.extend([
        "--per-wl".into(),
        "--per-wl-ax".into(),
        "ax_ranges.txt".into(),
        "--per-wl-ay".into(),
        "ay_ranges.txt".into(),
    ]);
    let parsed = FitCli::try_parse_from(args).unwrap();
    assert!(parsed.fit.per_wavelength);
    assert_eq!(
        parsed
            .fit
            .per_wavelength_ax
            .as_ref()
            .and_then(|p| p.to_str()),
        Some("ax_ranges.txt")
    );
    assert_eq!(
        parsed
            .fit
            .per_wavelength_ay
            .as_ref()
            .and_then(|p| p.to_str()),
        Some("ay_ranges.txt")
    );
}
