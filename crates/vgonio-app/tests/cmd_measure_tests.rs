//! CLI parser tests for measure command options.

use clap::Parser;
use vgonio_app::MeasureOptions;

#[derive(Debug, Parser)]
struct MeasureCli {
    #[command(flatten)]
    measure: MeasureOptions,
}

fn base_args() -> Vec<String> { vec!["measure-tests".into(), "--inputs".into(), "meas.yml".into()] }

#[test]
fn parses_minimal_measure_options() {
    let parsed = MeasureCli::try_parse_from(base_args()).unwrap();
    assert_eq!(parsed.measure.inputs.len(), 1);
    assert_eq!(
        parsed.measure.inputs[0].to_str(),
        Some("meas.yml"),
        "input path should be parsed as-is"
    );
    assert_eq!(parsed.measure.output_format.to_string(), "vgmo");
    assert_eq!(parsed.measure.resolution, 512);
    assert_eq!(parsed.measure.nthreads, None);
    assert!(!parsed.measure.print_stats);
}

#[test]
fn parses_multiple_inputs_and_runtime_flags() {
    let mut args = base_args();
    args.extend([
        "meas2.yml".into(),
        "--num-threads".into(),
        "4".into(),
        "--print-stats".into(),
    ]);

    let parsed = MeasureCli::try_parse_from(args).unwrap();
    assert_eq!(parsed.measure.inputs.len(), 2);
    assert_eq!(parsed.measure.nthreads, Some(4));
    assert!(parsed.measure.print_stats);
}

#[test]
fn parses_combined_output_format() {
    let mut args = base_args();
    args.extend(["--output-format".into(), "vgmo-exr".into()]);
    let parsed = MeasureCli::try_parse_from(args).unwrap();
    assert_eq!(parsed.measure.output_format.to_string(), "vgmo+exr");
}

#[test]
fn rejects_unknown_output_format() {
    let mut args = base_args();
    args.extend(["--output-format".into(), "unknown".into()]);
    let err = MeasureCli::try_parse_from(args).unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::InvalidValue);
}

#[test]
fn rejects_missing_inputs() {
    let args = vec!["measure-tests".to_string()];
    let err = MeasureCli::try_parse_from(args).unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
}
