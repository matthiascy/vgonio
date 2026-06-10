//! CLI shape for the `fit` subcommand. Orchestration lives in
//! [`crate::orchestration::fitting`]; this file is clap parsing + dispatch only.

use crate::app::executor;
use vgn_fitting::{
    orchestration::{parse_roughness_values, read_per_wavelength_roughness_values_unbounded},
    request::{
        BrdfFitRequest, BrdfSource, ClausenResample, FitRequest, FittingMethod, NdfFitRequest,
        Roughness,
    },
};
use clap::builder::ValueParser;
use std::{path::PathBuf, sync::Arc};
use vgn_bxdf::{brdf::measured::MeasuredBrdfKind, distro::MicrofacetDistroKind, BrdfFamily};
use vgn_core::{
    config::Config, error::VgonioError, units::Radians, utils::range::StepRangeIncl, BrdfLevel,
    ErrorMetric, Symmetry, Weighting,
};
use vgn_executor::{Executor, LocalStatusBridge};
use vgn_job_api::{
    envelope::{JobEnvelope, PayloadEncoding, TraceContext},
    ids::{CapabilityId, IdempotencyKey, JobId},
    resources::ResourceHints,
};

pub fn fit(opts: FitOptions, config: Config) -> Result<(), VgonioError> {
    let executor = executor::build_local_executor(Arc::new(config));
    let request = FitRequest::try_from(&opts)?;
    let payload = serde_json::to_vec(&request)
        .map_err(|e| VgonioError::new(format!("Failed to serialize FitRequest: {e}"), None))?;
    let envelope = JobEnvelope {
        job_id: JobId::new(),
        protocol_version: vgn_job_api::PROTOCOL_VERSION,
        capability_id: CapabilityId::fit(),
        capability_version: 1,
        payload_encoding: PayloadEncoding::Json,
        payload: payload.into(),
        resources: ResourceHints::default(),
        inputs: vec![],
        idempotency_key: IdempotencyKey("cli-fit".into()),
        trace: TraceContext::default(),
    };
    let handle = executor
        .submit(envelope)
        .map_err(|e| VgonioError::new(e.to_string(), None))?;

    // Render events while waiting on result.
    let bridge = LocalStatusBridge::new();
    let result_rx = handle.result;
    let events_rx = handle.events;
    let bridge_handle = std::thread::spawn(move || bridge.run(events_rx));
    let result = result_rx
        .recv()
        .map_err(|e| VgonioError::new(format!("Failed to receive fit result: {e}"), None))?;
    // Join the bridge so the terminal event (e.g. Completed) is rendered
    // before we return. `result_rx.recv()` can resolve before the bridge has
    // drained the event channel, and the bridge renders the final line at
    // default verbosity; dropping the handle would risk a missing/flaky
    // trailing line.
    let _ = bridge_handle.join();
    // Use `e.message` (not `{e}`): `JobError`'s Display is
    // `"{code:?}: {message}"`, which would compound with the handler's own
    // context and surface as "Fit job failed: HandlerError: ...".
    let outcome =
        result.map_err(|e| VgonioError::new(format!("Fit job failed: {}", e.message), None))?;

    // When `--plot` was requested the headless capability returns the plot
    // data in the payload; render it here (pyo3 + matplotlib live app-side).
    if !outcome.payload.is_empty() {
        render_fit_plots(&outcome.payload);
    }
    Ok(())
}

/// Renders the fit plot data the capability returned (see
/// [`vgn_fitting::plot::FitPlotData`]). Best-effort: a malformed payload or a
/// failed matplotlib call is logged, not fatal.
fn render_fit_plots(payload: &[u8]) {
    use vgn_fitting::plot::FitPlotData;
    let plots: Vec<FitPlotData> = match serde_json::from_slice(payload) {
        Ok(p) => p,
        Err(e) => {
            log::error!("Failed to deserialize fit plot data: {e}");
            return;
        },
    };
    for plot in &plots {
        if let Some(pw) = &plot.per_wavelength {
            crate::pyplot::plot_per_wavelength_err(
                &pw.wavelengths,
                &pw.alphas,
                &pw.errors,
                plot.n_digits,
            );
        }
        for curve in &plot.error_vs_alpha {
            if let Err(e) = crate::pyplot::plot_err(&curve.error, &curve.alpha, plot.n_digits) {
                log::error!("Failed to render error-vs-alpha plot: {e:?}");
            }
        }
    }
}

// TODO: separate the NDF fitting & the BRDF fitting

/// Options for the `fit` subcommand.
#[derive(clap::Args, Debug, Clone)]
#[clap(about = "Fits a micro-surface related measurement to a given model.")]
pub struct FitOptions {
    #[clap(num_args = 1.., value_delimiter = ' ', help = "Input files to fit the measurement to.")]
    pub inputs: Vec<PathBuf>,

    #[clap(long, short, help = "Kind of the measured BRDF data.")]
    pub kind: MeasuredBrdfKind,

    #[clap(
        long = "ax",
        value_name = "START:END:STEP",
        value_parser = ValueParser::new(parse_roughness_values),
        requires("ay"),
        help = "Anisotropic roughness in x direction (inclusive)."
    )]
    pub ax: Option<[f64; 3]>,

    #[clap(
        long = "ay",
        value_name = "START:END:STEP",
        value_parser = ValueParser::new(parse_roughness_values),
        requires("ax"),
        help = "Anisotropic roughness in y direction (inclusive)."
    )]
    pub ay: Option<[f64; 3]>,

    #[clap(
        long = "per-wl-ax",
        requires("per_wavelength_ay"),
        requires("per_wavelength"),
        conflicts_with_all(&["ax", "a", "per_wavelength_a"]),
        help = "Anisotropic roughness in y direction for each wavelength. The values are \
                specified as a list of triplets, each triplet is a range of roughness values for \
                a wavelength. The number of triplets should be equal to the number of wavelengths."
    )]
    pub per_wavelength_ax: Option<PathBuf>,

    #[clap(
        long = "per-wl-ay",
        requires("per_wavelength_ax"),
        requires("per_wavelength"),
        conflicts_with_all(&["ay", "a", "per_wavelength_a"]),
        help = "Anisotropic roughness in y direction for each wavelength. The values are \
                specified as a list of triplets, each triplet is a range of roughness values for \
                a wavelength. The number of triplets should be equal to the number of wavelengths."
    )]
    pub per_wavelength_ay: Option<PathBuf>,

    #[clap(
        long = "a",
        value_name = "START:END:STEP",
        value_parser = ValueParser::new(parse_roughness_values),
        conflicts_with_all(&["ax", "ay", "per_wavelength_ax", "per_wavelength_ay", "per_wavelength_a"]),
        help = "Isotropic roughness (inclusive)."
    )]
    pub a: Option<[f64; 3]>,

    // TODO: Add support for per-wavelength isotropic roughness
    #[clap(
        long = "per-wl-a",
        conflicts_with_all(&["per_wavelength_ax", "per_wavelength_ay", "ax", "ay", "a"]),
        help = "Isotropic roughness for each wavelength. The values are specified as a list of \
                triplets, each triplet is a range of roughness values for a wavelength. The \
                number of triplets should be equal to the number of wavelengths."
    )]
    pub per_wavelength_a: Option<PathBuf>,

    #[clap(
        long,
        help = "Whether to match the measured BRDF data to physically measured in-plane BRDF data \
                by O. Clausen. If true, the inputs should be in pairs of measured data and O. \
                Clausen's data.",
        default_value = "false"
    )]
    pub clausen: bool,

    #[clap(
        long,
        help = "Whether to use 4 times mores samples while resampling the measured data to match \
                the resolution of the Clausen's data.",
        default_value = "false"
    )]
    pub dense: bool,

    #[clap(
        long,
        short,
        help = "Specifies the output file for saving the fitted data. If omitted, the data is \
                written to standard output. Use \"auto\" to automatically determine the output \
                filename."
    )]
    pub output: Option<String>,

    #[clap(
        long,
        short,
        help = "Model to fit the measurement to. If not specified, the default model will be used."
    )]
    pub family: BrdfFamily,

    #[clap(long, help = "Symmetry of the microfacet model.")]
    pub symmetry: Symmetry,

    #[clap(
        long,
        short,
        help = "Distribution to use for the microfacet model. If not specified, the default \
                distribution will be used.",
        required_if_eq("family", "microfacet")
    )]
    pub distro: Option<MicrofacetDistroKind>,

    #[clap(
        long,
        short,
        help = "Level of the measured BRDF data to fit. Only used for the Vgonio kind.",
        required_if_eq("kind", "vgonio"),
        default_value = "l0"
    )]
    pub level: BrdfLevel,

    #[clap(long, help = "Theta limit for the fitting in degrees. Default to 90°.")]
    pub theta_limit: Option<f32>,

    #[clap(long, short, help = "Method to use for the fitting.")]
    pub method: FittingMethod,

    #[clap(
        long = "bprec",
        help = "Precision of the brute force fitting: number of digits after the decimal point.",
        default_value = "6"
    )]
    pub brute_precision: u32,

    #[clap(
        long = "err",
        help = "Error metric used during calculation of objective function (residuals).",
        default_value = "mse",
        required_if_eq("method", "brute")
    )]
    pub error_metric: Option<ErrorMetric>,

    #[clap(
        short,
        long,
        help = "The weighting to use to weight the measured data.",
        default_value = "none"
    )]
    pub weighting: Weighting,

    #[clap(
        long = "per-wl",
        help = "Whether to fit the measured data per wavelength.",
        default_value = "false"
    )]
    pub per_wavelength: bool,

    #[clap(
        long,
        help = "Whether to plot the fitted data.",
        default_value = "false"
    )]
    pub plot: bool,

    #[cfg(feature = "cuda")]
    #[clap(long, help = "Enable CUDA acceleration for the fitting.")]
    pub cuda: bool,

    // Temporary fix for adding the NDF fitting
    #[clap(long, help = "NDF fitting.")]
    pub ndf: bool,
}

impl FitOptions {
    #[inline]
    pub fn on_gpu(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    #[inline]
    pub fn on_cpu(&self) -> bool { !self.on_gpu() }
}

impl TryFrom<&FitOptions> for FitRequest {
    type Error = VgonioError;

    fn try_from(opts: &FitOptions) -> Result<Self, Self::Error> {
        if opts.inputs.is_empty() {
            return Err(VgonioError::new(
                "No input files specified or some files do not exist.",
                None,
            ));
        }

        if opts.ndf {
            let distro = opts.distro.ok_or_else(|| {
                VgonioError::new("NDF fitting requires --distro to be set.", None)
            })?;
            return Ok(FitRequest::Ndf(NdfFitRequest {
                inputs: opts.inputs.clone(),
                distro,
            }));
        }

        if opts.clausen && opts.kind != MeasuredBrdfKind::Vgonio {
            return Err(VgonioError::new(
                "--clausen is only supported with --kind vgonio.",
                None,
            ));
        }

        let source = match opts.kind {
            MeasuredBrdfKind::Vgonio => BrdfSource::Vgonio {
                level: opts.level,
                clausen_resample: opts.clausen.then(|| ClausenResample { dense: opts.dense }),
            },
            MeasuredBrdfKind::Clausen => BrdfSource::Clausen,
            MeasuredBrdfKind::Merl => BrdfSource::Merl,
            MeasuredBrdfKind::Rgl => BrdfSource::Rgl,
            MeasuredBrdfKind::Yan2018 => BrdfSource::Yan2018,
            MeasuredBrdfKind::Utia => BrdfSource::Utia,
            MeasuredBrdfKind::Unknown => BrdfSource::Unknown,
        };

        let roughness = build_roughness(opts)?;
        let theta_limit = opts.theta_limit.map(Radians::from_degrees);
        let output = resolve_output(opts)?;

        Ok(FitRequest::Brdf(BrdfFitRequest {
            inputs: opts.inputs.clone(),
            source,
            family: opts.family,
            distro: opts.distro,
            symmetry: opts.symmetry,
            roughness,
            method: opts.method,
            brute_precision: opts.brute_precision,
            error_metric: opts.error_metric,
            weighting: opts.weighting,
            theta_limit,
            per_wavelength: opts.per_wavelength,
            plot: opts.plot,
            #[cfg(feature = "cuda")]
            cuda: opts.cuda,
            output,
        }))
    }
}

fn build_roughness(opts: &FitOptions) -> Result<Roughness, VgonioError> {
    let to_range = |[s, e, t]: [f64; 3]| StepRangeIncl::new(s, e, t);

    if let (Some(ax_path), Some(ay_path)) = (
        opts.per_wavelength_ax.as_ref(),
        opts.per_wavelength_ay.as_ref(),
    ) {
        let ax = read_per_wavelength_roughness_values_unbounded(ax_path).map_err(|e| {
            VgonioError::new(&format!("Failed to read --per-wl-ax file: {}", e), None)
        })?;
        let ay = read_per_wavelength_roughness_values_unbounded(ay_path).map_err(|e| {
            VgonioError::new(&format!("Failed to read --per-wl-ay file: {}", e), None)
        })?;
        if ax.len() != ay.len() {
            return Err(VgonioError::new(
                &format!(
                    "--per-wl-ax has {} entries but --per-wl-ay has {}.",
                    ax.len(),
                    ay.len()
                ),
                None,
            ));
        }
        return Ok(Roughness::PerWavelengthAniso { ax, ay });
    }

    if opts.per_wavelength_a.is_some() {
        return Err(VgonioError::new(
            "--per-wl-a is not yet supported (per-wavelength isotropic ranges).",
            None,
        ));
    }

    if let Some(a) = opts.a {
        return Ok(Roughness::Iso(to_range(a)));
    }

    if let (Some(ax), Some(ay)) = (opts.ax, opts.ay) {
        return Ok(Roughness::Aniso {
            ax: to_range(ax),
            ay: to_range(ay),
        });
    }

    Ok(Roughness::Default)
}

fn resolve_output(opts: &FitOptions) -> Result<Option<PathBuf>, VgonioError> {
    let Some(raw) = opts.output.as_ref() else {
        return Ok(None);
    };
    if raw != "auto" {
        return Ok(Some(PathBuf::from(raw)));
    }
    let first_input = opts.inputs.first().ok_or_else(|| {
        VgonioError::new(
            "--output auto requires at least one input file to derive a stem from.",
            None,
        )
    })?;
    let stem = first_input
        .file_stem()
        .ok_or_else(|| {
            VgonioError::new(
                format!(
                    "--output auto requires the first input ({}) to have a file stem.",
                    first_input.display()
                ),
                None,
            )
        })?
        .to_string_lossy()
        .into_owned();
    let err = opts.error_metric.ok_or_else(|| {
        VgonioError::new(
            "--output auto requires --err to be set so the filename can encode it.",
            None,
        )
    })?;
    let distro = opts.distro.ok_or_else(|| {
        VgonioError::new(
            "--output auto requires --distro to be set so the filename can encode it.",
            None,
        )
    })?;
    Ok(Some(PathBuf::from(format!(
        "{}_{}_{:?}_{:?}.csv",
        stem, err, distro, opts.weighting
    ))))
}

#[cfg(test)]
mod tests {
    use super::{build_roughness, resolve_output, FitOptions};
    use vgn_fitting::{
        orchestration::{
            parse_roughness_values, read_per_wavelength_roughness_values,
            read_per_wavelength_roughness_values_unbounded,
        },
        request::{BrdfSource, FitRequest, FittingMethod, Roughness},
    };
    use proptest::prelude::*;
    use std::{io::Write, path::PathBuf};
    use vgn_bxdf::{brdf::measured::MeasuredBrdfKind, distro::MicrofacetDistroKind, BrdfFamily};
    use vgn_core::{utils::range::StepRangeIncl, BrdfLevel, ErrorMetric, Symmetry, Weighting};

    /// Builds a `FitOptions` populated with the defaults clap would produce for
    /// a minimal valid `vgonio fit` invocation. Individual tests mutate only
    /// the fields they care about, so changes in unrelated defaults stay local
    /// to this helper.
    fn default_fit_options() -> FitOptions {
        FitOptions {
            inputs: vec![PathBuf::from("input.vgmo")],
            kind: MeasuredBrdfKind::Vgonio,
            ax: None,
            ay: None,
            per_wavelength_ax: None,
            per_wavelength_ay: None,
            a: None,
            per_wavelength_a: None,
            clausen: false,
            dense: false,
            output: None,
            family: BrdfFamily::Microfacet,
            symmetry: Symmetry::Isotropic,
            distro: Some(MicrofacetDistroKind::TrowbridgeReitz),
            level: BrdfLevel::L0,
            theta_limit: None,
            method: FittingMethod::Brute,
            brute_precision: 6,
            error_metric: Some(ErrorMetric::Mse),
            weighting: Weighting::None,
            per_wavelength: false,
            plot: false,
            #[cfg(feature = "cuda")]
            cuda: false,
            ndf: false,
        }
    }

    fn write_triplets(triplets: &[[f64; 3]]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        for [s, e, t] in triplets {
            writeln!(file, "{}:{}:{}", s, e, t).unwrap();
        }
        file.flush().unwrap();
        file
    }

    // ------------------------------------------------------------------
    // parse_roughness_values
    // ------------------------------------------------------------------

    #[test]
    fn parse_roughness_values_accepts_valid_triplet() {
        let parsed = parse_roughness_values("0.1:0.5:0.01").unwrap();
        assert_eq!(parsed, [0.1, 0.5, 0.01]);
    }

    #[test]
    fn parse_roughness_values_rejects_non_numeric_values() {
        let err = parse_roughness_values("a:b:c").unwrap_err();
        assert_eq!(err, "roughness value is not a valid floating point number");
    }

    #[test]
    fn parse_roughness_values_rejects_non_positive_step() {
        let err = parse_roughness_values("0.1:0.5:0.0").unwrap_err();
        assert_eq!(err, "roughness step must be greater than 0");
    }

    #[test]
    fn parse_roughness_values_rejects_negative_step() {
        let err = parse_roughness_values("0.1:0.5:-0.01").unwrap_err();
        assert_eq!(err, "roughness step must be greater than 0");
    }

    #[test]
    fn parse_roughness_values_rejects_reversed_range() {
        let err = parse_roughness_values("0.5:0.1:0.01").unwrap_err();
        assert_eq!(err, "roughness range start must be <= end");
    }

    #[test]
    fn parse_roughness_values_accepts_degenerate_zero_width_range() {
        // start == stop is the boundary of the start <= stop check; covered
        // here explicitly because the brute-force grid degenerates to a single
        // sample, which is still a legitimate request.
        let parsed = parse_roughness_values("0.25:0.25:0.01").unwrap();
        assert_eq!(parsed, [0.25, 0.25, 0.01]);
    }

    #[test]
    fn parse_roughness_values_rejects_wrong_arity() {
        assert!(parse_roughness_values("0.1:0.5").is_err());
        assert!(parse_roughness_values("0.1:0.2:0.3:0.4").is_err());
        assert!(parse_roughness_values("").is_err());
    }

    #[test]
    fn parse_roughness_values_rejects_non_finite() {
        assert_eq!(
            parse_roughness_values("nan:0.5:0.01").unwrap_err(),
            "roughness values must be finite"
        );
        assert_eq!(
            parse_roughness_values("0.1:inf:0.01").unwrap_err(),
            "roughness values must be finite"
        );
        assert_eq!(
            parse_roughness_values("0.1:0.5:inf").unwrap_err(),
            "roughness values must be finite"
        );
    }

    proptest! {
        #[test]
        fn proptest_parse_roughness_values_round_trip(
            start in -1.0e6f64..1.0e6f64,
            extra in 0.0f64..1.0e3f64,
            step in 1.0e-6f64..1.0e3f64,
        ) {
            let stop = start + extra;
            let s = format!("{}:{}:{}", start, stop, step);
            let parsed = parse_roughness_values(&s).unwrap();
            // Use bitwise equality via to_bits because the formatter is
            // round-trip-stable for f64 in Rust.
            prop_assert_eq!(parsed[0].to_bits(), start.to_bits());
            prop_assert_eq!(parsed[1].to_bits(), stop.to_bits());
            prop_assert_eq!(parsed[2].to_bits(), step.to_bits());
        }

        #[test]
        fn proptest_parse_roughness_values_rejects_non_positive_step(
            start in -1.0e3f64..1.0e3f64,
            extra in 0.0f64..1.0e3f64,
            step in -1.0e3f64..=0.0f64,
        ) {
            let stop = start + extra;
            let s = format!("{}:{}:{}", start, stop, step);
            prop_assert_eq!(
                parse_roughness_values(&s).unwrap_err(),
                "roughness step must be greater than 0"
            );
        }

        #[test]
        fn proptest_parse_roughness_values_rejects_reversed_range(
            stop in -1.0e3f64..1.0e3f64,
            extra in 1.0e-9f64..1.0e3f64,
            step in 1.0e-6f64..1.0e3f64,
        ) {
            let start = stop + extra;
            let s = format!("{}:{}:{}", start, stop, step);
            prop_assert_eq!(
                parse_roughness_values(&s).unwrap_err(),
                "roughness range start must be <= end"
            );
        }
    }

    // ------------------------------------------------------------------
    // read_per_wavelength_roughness_values{,_unbounded}
    // ------------------------------------------------------------------

    #[test]
    fn read_per_wavelength_roughness_values_parses_valid_file() {
        let file = write_triplets(&[[0.1, 0.2, 0.01], [0.2, 0.3, 0.01], [0.3, 0.4, 0.01]]);
        let values = read_per_wavelength_roughness_values(file.path(), 3).unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values[0].start, 0.1);
        assert_eq!(values[2].stop, 0.4);
    }

    #[test]
    fn read_per_wavelength_roughness_values_rejects_length_mismatch() {
        let file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let err = read_per_wavelength_roughness_values(file.path(), 2).unwrap_err();
        assert_eq!(
            err,
            "The number of triplets should be equal to the number of wavelengths."
        );
    }

    #[test]
    fn read_per_wavelength_roughness_values_unbounded_skips_blank_lines() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "0.1:0.2:0.01").unwrap();
        writeln!(file).unwrap();
        writeln!(file, "   ").unwrap();
        writeln!(file, "0.3:0.4:0.01").unwrap();
        file.flush().unwrap();

        let values = read_per_wavelength_roughness_values_unbounded(file.path()).unwrap();
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn read_per_wavelength_roughness_values_unbounded_surfaces_parse_errors() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "0.1:0.2:0.01").unwrap();
        writeln!(file, "this is not a triplet").unwrap();
        file.flush().unwrap();

        let err = read_per_wavelength_roughness_values_unbounded(file.path()).unwrap_err();
        // Whichever validation triggered first, it should propagate as a
        // `&'static str`, not panic.
        assert!(!err.is_empty());
    }

    #[test]
    fn read_per_wavelength_roughness_values_unbounded_missing_file_errors() {
        let path = PathBuf::from("/this/path/does/not/exist_42.txt");
        let err = read_per_wavelength_roughness_values_unbounded(&path).unwrap_err();
        assert_eq!(err, "Failed to open the roughness values file.");
    }

    proptest! {
        #[test]
        fn proptest_per_wavelength_roughness_round_trip(
            triplets in proptest::collection::vec(
                (-1.0e3f64..1.0e3f64, 0.0f64..1.0e3f64, 1.0e-6f64..1.0e3f64),
                1..16,
            ),
        ) {
            let triplets: Vec<[f64; 3]> = triplets
                .into_iter()
                .map(|(s, extra, step)| [s, s + extra, step])
                .collect();
            let file = write_triplets(&triplets);
            let values = read_per_wavelength_roughness_values(file.path(), triplets.len()).unwrap();
            prop_assert_eq!(values.len(), triplets.len());
            for (got, want) in values.iter().zip(triplets.iter()) {
                prop_assert_eq!(got.start.to_bits(), want[0].to_bits());
                prop_assert_eq!(got.stop.to_bits(), want[1].to_bits());
                prop_assert_eq!(got.step_size.to_bits(), want[2].to_bits());
            }
        }
    }

    // ------------------------------------------------------------------
    // build_roughness
    // ------------------------------------------------------------------

    #[test]
    fn build_roughness_defaults_to_default_variant_when_unspecified() {
        let opts = default_fit_options();
        match build_roughness(&opts).unwrap() {
            Roughness::Default => {},
            other => panic!("expected Roughness::Default, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_isotropic_from_a() {
        let mut opts = default_fit_options();
        opts.a = Some([0.1, 0.5, 0.01]);
        match build_roughness(&opts).unwrap() {
            Roughness::Iso(r) => assert_eq!(r, StepRangeIncl::new(0.1, 0.5, 0.01)),
            other => panic!("expected Roughness::Iso, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_anisotropic_from_ax_ay() {
        let mut opts = default_fit_options();
        opts.ax = Some([0.0, 1.0, 0.05]);
        opts.ay = Some([0.1, 0.9, 0.02]);
        match build_roughness(&opts).unwrap() {
            Roughness::Aniso { ax, ay } => {
                assert_eq!(ax, StepRangeIncl::new(0.0, 1.0, 0.05));
                assert_eq!(ay, StepRangeIncl::new(0.1, 0.9, 0.02));
            },
            other => panic!("expected Roughness::Aniso, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_prefers_a_over_ax_ay_when_both_set() {
        // clap's `conflicts_with` ordinarily prevents this combination, but the
        // builder must still pick a deterministic precedence (isotropic wins)
        // if it ever sees both — codifies the early `if let Some(a)` branch.
        let mut opts = default_fit_options();
        opts.a = Some([0.2, 0.3, 0.01]);
        opts.ax = Some([0.0, 1.0, 0.05]);
        opts.ay = Some([0.1, 0.9, 0.02]);
        match build_roughness(&opts).unwrap() {
            Roughness::Iso(r) => assert_eq!(r, StepRangeIncl::new(0.2, 0.3, 0.01)),
            other => panic!("expected Roughness::Iso, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_per_wavelength_aniso_reads_paired_files() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01], [0.3, 0.4, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01], [0.7, 0.8, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());

        match build_roughness(&opts).unwrap() {
            Roughness::PerWavelengthAniso { ax, ay } => {
                assert_eq!(ax.len(), 2);
                assert_eq!(ay.len(), 2);
                assert_eq!(ax[0], StepRangeIncl::new(0.1, 0.2, 0.01));
                assert_eq!(ay[1], StepRangeIncl::new(0.7, 0.8, 0.01));
            },
            other => panic!("expected Roughness::PerWavelengthAniso, got {:?}", other),
        }
    }

    #[test]
    fn build_roughness_per_wavelength_aniso_rejects_length_mismatch() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01], [0.3, 0.4, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());

        let err = build_roughness(&opts).unwrap_err();
        assert!(err.message().contains("--per-wl-ax has 2 entries"));
    }

    #[test]
    fn build_roughness_per_wavelength_a_unsupported() {
        let a_file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength_a = Some(a_file.path().to_path_buf());

        let err = build_roughness(&opts).unwrap_err();
        assert!(err.message().contains("--per-wl-a is not yet supported"));
    }

    #[test]
    fn build_roughness_surfaces_io_error_for_missing_per_wavelength_file() {
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(PathBuf::from("/no/such/file_ax.txt"));
        opts.per_wavelength_ay = Some(PathBuf::from("/no/such/file_ay.txt"));

        let err = build_roughness(&opts).unwrap_err();
        assert!(err.message().contains("Failed to read --per-wl-ax file"));
    }

    // ------------------------------------------------------------------
    // resolve_output
    // ------------------------------------------------------------------

    #[test]
    fn resolve_output_none_when_unset() {
        let opts = default_fit_options();
        assert_eq!(resolve_output(&opts).unwrap(), None);
    }

    #[test]
    fn resolve_output_passes_explicit_path_through() {
        let mut opts = default_fit_options();
        opts.output = Some("fits/out.csv".to_string());
        assert_eq!(
            resolve_output(&opts).unwrap(),
            Some(PathBuf::from("fits/out.csv"))
        );
    }

    #[test]
    fn resolve_output_auto_builds_filename_from_inputs_and_metadata() {
        let mut opts = default_fit_options();
        opts.inputs = vec![PathBuf::from("/data/measure_xyz.vgmo")];
        opts.output = Some("auto".to_string());
        opts.error_metric = Some(ErrorMetric::Rmse);
        opts.distro = Some(MicrofacetDistroKind::Beckmann);
        opts.weighting = Weighting::LnCos;
        let out = resolve_output(&opts).unwrap().unwrap();
        // Error metric is rendered via `Display` (lowercase "rmse") to match
        // the pre-request-split filename format that downstream scripts read.
        assert_eq!(out, PathBuf::from("measure_xyz_rmse_Beckmann_LnCos.csv"));
    }

    #[test]
    fn resolve_output_auto_errors_when_inputs_empty() {
        let mut opts = default_fit_options();
        opts.inputs.clear();
        opts.output = Some("auto".to_string());
        let err = resolve_output(&opts).unwrap_err();
        assert!(
            err.message().contains("at least one input file"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn resolve_output_auto_errors_when_error_metric_missing() {
        let mut opts = default_fit_options();
        opts.output = Some("auto".to_string());
        opts.error_metric = None;
        let err = resolve_output(&opts).unwrap_err();
        assert!(
            err.message().contains("--err"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn resolve_output_auto_errors_when_distro_missing() {
        let mut opts = default_fit_options();
        opts.output = Some("auto".to_string());
        opts.distro = None;
        let err = resolve_output(&opts).unwrap_err();
        assert!(
            err.message().contains("--distro"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn resolve_output_auto_errors_when_input_has_no_file_stem() {
        let mut opts = default_fit_options();
        // Bare "/" has no file_stem; auto-derivation must fail loudly.
        opts.inputs = vec![PathBuf::from("/")];
        opts.output = Some("auto".to_string());
        let err = resolve_output(&opts).unwrap_err();
        assert!(
            err.message().contains("file stem"),
            "unexpected error: {}",
            err.message()
        );
    }

    #[test]
    fn try_from_propagates_resolve_output_error() {
        // "auto" with no distro must surface as a TryFrom error, not be
        // silently swallowed into output=None.
        let mut opts = default_fit_options();
        opts.output = Some("auto".to_string());
        opts.distro = None;
        let err = FitRequest::try_from(&opts).unwrap_err();
        assert!(err.message().contains("--distro"));
    }

    // ------------------------------------------------------------------
    // TryFrom<&FitOptions> for FitRequest
    // ------------------------------------------------------------------

    #[test]
    fn try_from_rejects_empty_inputs() {
        let mut opts = default_fit_options();
        opts.inputs.clear();
        let err = FitRequest::try_from(&opts).unwrap_err();
        assert!(err.message().contains("No input files"));
    }

    #[test]
    fn try_from_ndf_branch_requires_distro() {
        let mut opts = default_fit_options();
        opts.ndf = true;
        opts.distro = None;
        let err = FitRequest::try_from(&opts).unwrap_err();
        assert!(err.message().contains("NDF fitting requires --distro"));
    }

    #[test]
    fn try_from_ndf_branch_produces_ndf_request() {
        let mut opts = default_fit_options();
        opts.ndf = true;
        opts.distro = Some(MicrofacetDistroKind::TrowbridgeReitz);
        opts.inputs = vec![PathBuf::from("a.vgmo"), PathBuf::from("b.vgmo")];
        match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Ndf(req) => {
                assert_eq!(req.distro, MicrofacetDistroKind::TrowbridgeReitz);
                assert_eq!(req.inputs.len(), 2);
            },
            FitRequest::Brdf(_) => panic!("expected NDF branch"),
        }
    }

    #[test]
    fn try_from_clausen_flag_requires_vgonio_kind() {
        let mut opts = default_fit_options();
        opts.clausen = true;
        opts.kind = MeasuredBrdfKind::Merl;
        let err = FitRequest::try_from(&opts).unwrap_err();
        assert!(err.message().contains("--clausen is only supported"));
    }

    #[test]
    fn try_from_vgonio_kind_without_clausen_produces_plain_vgonio_source() {
        let mut opts = default_fit_options();
        opts.kind = MeasuredBrdfKind::Vgonio;
        opts.level = BrdfLevel::L2;
        opts.clausen = false;
        let req = match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        };
        match req.source {
            BrdfSource::Vgonio {
                level,
                clausen_resample: None,
            } => assert_eq!(level, BrdfLevel::L2),
            other => panic!("expected Vgonio source, got {:?}", other),
        }
    }

    #[test]
    fn try_from_vgonio_kind_with_clausen_carries_dense_flag() {
        let mut opts = default_fit_options();
        opts.kind = MeasuredBrdfKind::Vgonio;
        opts.clausen = true;
        opts.dense = true;
        let req = match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        };
        match req.source {
            BrdfSource::Vgonio {
                clausen_resample: Some(resample),
                ..
            } => assert!(resample.dense),
            other => panic!("expected Vgonio with clausen_resample, got {:?}", other),
        }
    }

    #[test]
    fn try_from_maps_each_brdf_kind_to_matching_source() {
        let cases = [
            (MeasuredBrdfKind::Clausen, "Clausen"),
            (MeasuredBrdfKind::Merl, "Merl"),
            (MeasuredBrdfKind::Rgl, "Rgl"),
            (MeasuredBrdfKind::Yan2018, "Yan2018"),
            (MeasuredBrdfKind::Utia, "Utia"),
            (MeasuredBrdfKind::Unknown, "Unknown"),
        ];
        for (kind, name) in cases {
            let mut opts = default_fit_options();
            opts.kind = kind;
            let req = match FitRequest::try_from(&opts).unwrap() {
                FitRequest::Brdf(req) => req,
                FitRequest::Ndf(_) => panic!("expected BRDF branch for {name}"),
            };
            let matched = matches!(
                (&req.source, kind),
                (BrdfSource::Clausen, MeasuredBrdfKind::Clausen)
                    | (BrdfSource::Merl, MeasuredBrdfKind::Merl)
                    | (BrdfSource::Rgl, MeasuredBrdfKind::Rgl)
                    | (BrdfSource::Yan2018, MeasuredBrdfKind::Yan2018)
                    | (BrdfSource::Utia, MeasuredBrdfKind::Utia)
                    | (BrdfSource::Unknown, MeasuredBrdfKind::Unknown)
            );
            assert!(matched, "kind {name} did not map to a matching BrdfSource");
        }
    }

    #[test]
    fn try_from_converts_theta_limit_degrees_to_radians() {
        let mut opts = default_fit_options();
        opts.theta_limit = Some(45.0);
        let req = match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        };
        let limit = req.theta_limit.expect("theta_limit should be Some");
        let expected = std::f32::consts::FRAC_PI_4;
        assert!(
            (limit.as_f32() - expected).abs() < 1.0e-6,
            "expected ~PI/4, got {}",
            limit.as_f32()
        );
    }

    #[test]
    fn try_from_propagates_brute_precision_and_metric() {
        let mut opts = default_fit_options();
        opts.brute_precision = 9;
        opts.error_metric = Some(ErrorMetric::L2);
        opts.weighting = Weighting::LnCos;
        opts.method = FittingMethod::Nllsq;
        let req = match FitRequest::try_from(&opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        };
        assert_eq!(req.brute_precision, 9);
        assert_eq!(req.error_metric, Some(ErrorMetric::L2));
        assert_eq!(req.weighting, Weighting::LnCos);
        assert_eq!(req.method, FittingMethod::Nllsq);
    }

    // ------------------------------------------------------------------
    // serde round-trip (envelopes carry FitRequest as JSON)
    // ------------------------------------------------------------------

    fn assert_json_round_trip(req: &FitRequest) {
        let encoded = serde_json::to_string(req).expect("FitRequest should serialize");
        let decoded: FitRequest =
            serde_json::from_str(&encoded).expect("FitRequest should deserialize");
        // Re-encode the decoded value; structural equality is asserted via the
        // re-encoded payload because FitRequest does not derive PartialEq.
        let re_encoded =
            serde_json::to_string(&decoded).expect("decoded FitRequest should serialize");
        assert_eq!(encoded, re_encoded);
    }

    #[test]
    fn serde_round_trip_ndf_request() {
        let mut opts = default_fit_options();
        opts.ndf = true;
        let req = FitRequest::try_from(&opts).unwrap();
        assert_json_round_trip(&req);
    }

    #[test]
    fn serde_round_trip_brdf_request_default_roughness() {
        let opts = default_fit_options();
        let req = FitRequest::try_from(&opts).unwrap();
        assert_json_round_trip(&req);
    }

    #[test]
    fn serde_round_trip_brdf_request_anisotropic_roughness() {
        let mut opts = default_fit_options();
        opts.ax = Some([0.0, 1.0, 0.05]);
        opts.ay = Some([0.1, 0.9, 0.02]);
        opts.symmetry = Symmetry::Anisotropic;
        opts.theta_limit = Some(60.0);
        let req = FitRequest::try_from(&opts).unwrap();
        assert_json_round_trip(&req);
    }

    #[test]
    fn serde_round_trip_brdf_request_per_wavelength_aniso() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01], [0.3, 0.4, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01], [0.7, 0.8, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let req = FitRequest::try_from(&opts).unwrap();
        assert_json_round_trip(&req);
    }

    // ------------------------------------------------------------------
    // validate(): structural invariants enforced at the run boundary so that
    // requests arriving via JSON envelopes can't bypass what TryFrom enforces.
    // ------------------------------------------------------------------

    fn brdf_request_from(opts: &FitOptions) -> super::BrdfFitRequest {
        match FitRequest::try_from(opts).unwrap() {
            FitRequest::Brdf(req) => req,
            FitRequest::Ndf(_) => panic!("expected BRDF branch"),
        }
    }

    #[test]
    fn validate_accepts_request_from_try_from() {
        // Every TryFrom-produced request must satisfy the invariant; this
        // guards against a future TryFrom change silently constructing
        // PerWavelengthAniso with per_wavelength=false.
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01], [0.3, 0.4, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01], [0.7, 0.8, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let req = FitRequest::try_from(&opts).unwrap();
        assert!(req.validate().is_ok());
    }

    #[test]
    fn validate_rejects_per_wavelength_aniso_without_per_wavelength_flag() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let mut req = brdf_request_from(&opts);
        // Simulate a malformed deserialized request (or a hand-built one).
        req.per_wavelength = false;
        let err = req.validate().unwrap_err();
        assert!(err.message().contains("requires per_wavelength=true"));
    }

    #[test]
    fn validate_via_top_level_dispatches_to_brdf() {
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let mut req = brdf_request_from(&opts);
        req.per_wavelength = false;
        let req = FitRequest::Brdf(req);
        assert!(req.validate().is_err());
    }

    #[test]
    fn validate_ndf_is_unconditionally_ok() {
        let mut opts = default_fit_options();
        opts.ndf = true;
        let req = FitRequest::try_from(&opts).unwrap();
        assert!(req.validate().is_ok());
    }

    #[test]
    fn validate_passes_for_default_roughness_with_per_wavelength_false() {
        // Only PerWavelengthAniso is gated; other variants combine freely
        // with either per_wavelength setting.
        let opts = default_fit_options();
        let req = brdf_request_from(&opts);
        assert!(req.validate().is_ok());
    }

    #[test]
    fn validate_rejects_brdf_request_with_fit_invoking_source_and_no_distro() {
        // Every fit-invoking source (anything except Utia/Unknown) must error
        // with the same diagnostic so an operator gets a uniform signal at
        // the request boundary instead of a per-source surprise downstream.
        let fit_sources = [
            BrdfSource::Vgonio {
                level: BrdfLevel::L0,
                clausen_resample: None,
            },
            BrdfSource::Clausen,
            BrdfSource::Merl,
            BrdfSource::Rgl,
            BrdfSource::Yan2018,
        ];
        for source in fit_sources {
            let mut opts = default_fit_options();
            opts.kind = MeasuredBrdfKind::Vgonio;
            let mut req = brdf_request_from(&opts);
            req.source = source;
            req.distro = None;
            let err = req.validate().unwrap_err();
            assert!(
                err.message().contains("`distro` to be set"),
                "expected distro-required error, got: {}",
                err.message()
            );
        }
    }

    #[test]
    fn validate_accepts_no_op_sources_without_distro() {
        // Utia and Unknown short-circuit in run_brdf without ever calling the
        // fit pipeline, so they don't need distro -- requiring it here would
        // over-constrain harmless no-op envelopes.
        for source in [BrdfSource::Utia, BrdfSource::Unknown] {
            let mut opts = default_fit_options();
            let mut req = brdf_request_from(&opts);
            req.source = source;
            req.distro = None;
            assert!(
                req.validate().is_ok(),
                "expected Ok for no-op source, got: {:?}",
                req.validate().unwrap_err().message()
            );
            // suppress unused
            let _ = &mut opts;
        }
    }

    #[test]
    fn validate_rejects_deserialized_brdf_request_missing_distro() {
        // The schema lets `distro: null` round-trip cleanly; validate() is
        // what catches it before fit downstream panics on the expect().
        let mut opts = default_fit_options();
        opts.kind = MeasuredBrdfKind::Merl;
        let req = FitRequest::try_from(&opts).unwrap();
        let encoded = serde_json::to_string(&req).unwrap();
        let mut decoded: FitRequest = serde_json::from_str(&encoded).unwrap();
        if let FitRequest::Brdf(brdf) = &mut decoded {
            brdf.distro = None;
        } else {
            panic!("expected BRDF branch");
        }
        let err = decoded.validate().unwrap_err();
        assert!(err.message().contains("`distro` to be set"));
    }

    #[test]
    fn validate_recovers_invariant_from_round_trip_then_mutation() {
        // Round-trip a valid request, then corrupt the deserialized copy in
        // the same way a hand-crafted JSON envelope might. The corruption
        // round-trips cleanly through JSON (the schema doesn't forbid it),
        // so validate() is the only line of defense.
        let ax_file = write_triplets(&[[0.1, 0.2, 0.01]]);
        let ay_file = write_triplets(&[[0.5, 0.6, 0.01]]);
        let mut opts = default_fit_options();
        opts.per_wavelength = true;
        opts.per_wavelength_ax = Some(ax_file.path().to_path_buf());
        opts.per_wavelength_ay = Some(ay_file.path().to_path_buf());
        let req = FitRequest::try_from(&opts).unwrap();
        let encoded = serde_json::to_string(&req).unwrap();
        let mut decoded: FitRequest = serde_json::from_str(&encoded).unwrap();
        if let FitRequest::Brdf(brdf) = &mut decoded {
            brdf.per_wavelength = false;
        } else {
            panic!("expected BRDF branch");
        }
        assert!(decoded.validate().is_err());
    }

    // ------------------------------------------------------------------
    // Roughness::validate_against_spectrum_len: the only spot the fit path
    // still needs a dynamic check (spectrum length is unknown until data is
    // loaded). The brute-force loop calls this and propagates the error
    // instead of panicking, so we test it directly here.
    // ------------------------------------------------------------------

    #[test]
    fn roughness_length_check_passes_for_non_per_wavelength_variants() {
        assert!(Roughness::Default.validate_against_spectrum_len(0).is_ok());
        assert!(Roughness::Default.validate_against_spectrum_len(5).is_ok());
        assert!(Roughness::Iso(StepRangeIncl::new(0.1, 0.5, 0.01))
            .validate_against_spectrum_len(3)
            .is_ok());
        assert!(Roughness::Aniso {
            ax: StepRangeIncl::new(0.0, 1.0, 0.05),
            ay: StepRangeIncl::new(0.1, 0.9, 0.02),
        }
        .validate_against_spectrum_len(3)
        .is_ok());
    }

    #[test]
    fn roughness_length_check_accepts_matching_per_wavelength() {
        let r = Roughness::PerWavelengthAniso {
            ax: vec![
                StepRangeIncl::new(0.1, 0.2, 0.01),
                StepRangeIncl::new(0.3, 0.4, 0.01),
            ]
            .into_boxed_slice(),
            ay: vec![
                StepRangeIncl::new(0.5, 0.6, 0.01),
                StepRangeIncl::new(0.7, 0.8, 0.01),
            ]
            .into_boxed_slice(),
        };
        assert!(r.validate_against_spectrum_len(2).is_ok());
    }

    #[test]
    fn roughness_length_check_rejects_mismatched_per_wavelength() {
        let r = Roughness::PerWavelengthAniso {
            ax: vec![StepRangeIncl::new(0.1, 0.2, 0.01)].into_boxed_slice(),
            ay: vec![StepRangeIncl::new(0.5, 0.6, 0.01)].into_boxed_slice(),
        };
        let err = r.validate_against_spectrum_len(3).unwrap_err();
        assert!(err.message().contains("does not match"));
        // Both ax and ay lengths plus the spectrum length should appear in
        // the message so the operator can correlate envelope vs data.
        assert!(err.message().contains("1/1"));
        assert!(err.message().contains("(3)"));
    }

    // ------------------------------------------------------------------
    // source_kind(): the CSV `kind` column reflects the user-passed
    // `--kind`, not the runtime brdf type. The Vgonio+Clausen resample
    // path is the regression-pinning row: the resampled brdf is a
    // `ClausenBrdf`, but the CSV must still record "Vgonio".
    // ------------------------------------------------------------------

    #[test]
    fn source_kind_reflects_user_kind_for_each_brdf_source() {
        let cases: &[(MeasuredBrdfKind, bool, MeasuredBrdfKind)] = &[
            // (--kind, --clausen, expected source_kind)
            (MeasuredBrdfKind::Vgonio, false, MeasuredBrdfKind::Vgonio),
            // Resampling against Clausen reference data does NOT change the
            // reported kind: the user asked to fit Vgonio simulation output.
            (MeasuredBrdfKind::Vgonio, true, MeasuredBrdfKind::Vgonio),
            (MeasuredBrdfKind::Clausen, false, MeasuredBrdfKind::Clausen),
            (MeasuredBrdfKind::Merl, false, MeasuredBrdfKind::Merl),
            (MeasuredBrdfKind::Rgl, false, MeasuredBrdfKind::Rgl),
            (MeasuredBrdfKind::Yan2018, false, MeasuredBrdfKind::Yan2018),
            (MeasuredBrdfKind::Utia, false, MeasuredBrdfKind::Utia),
            (MeasuredBrdfKind::Unknown, false, MeasuredBrdfKind::Unknown),
        ];
        for &(kind, clausen, expected) in cases {
            // Paired inputs needed for the clausen-resample TryFrom path:
            // the boundary doesn't actually open them, but the request
            // build itself doesn't enforce the parity check (the chunk-2
            // assertion runs at `run` time). One input is enough here.
            let mut opts = default_fit_options();
            opts.kind = kind;
            opts.clausen = clausen;
            let req = brdf_request_from(&opts);
            assert_eq!(
                req.source_kind(),
                expected,
                "source_kind mismatch for kind={:?} clausen={}",
                kind,
                clausen
            );
        }
    }
}
