//! Integration tests for BRDF fitting with synthetic proxy data.

use rand::{Rng, SeedableRng};
use std::borrow::Cow;
use vgn_bxdf::{
    brdf::{analytical::microfacet::MicrofacetBrdfTR, measured::MeasuredBrdfKind},
    distro::MicrofacetDistroKind,
    fitting::{
        proxy::{BrdfProxy, OutgoingDirs, ProxySource},
        FittingProblem, Roughness,
    },
    AnyMeasuredBrdf, Scattering,
};
use vgn_core::{
    math::Vec3,
    optics::{Ior, IorReg},
    units::{nm, Nanometres, Radians},
    utils::{medium::MediumId, range::StepRangeIncl},
    ErrorMetric, Symmetry, Weighting,
};
use vgn_jabr::array::{DyArr, DynArr};

struct SyntheticMeasuredBrdf {
    spectrum: Box<[Nanometres]>,
}

impl SyntheticMeasuredBrdf {
    fn new(spectrum: Box<[Nanometres]>) -> Self { Self { spectrum } }
}

impl AnyMeasuredBrdf for SyntheticMeasuredBrdf {
    fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Unknown }

    fn spectrum(&self) -> &[Nanometres] { &self.spectrum }

    fn transmitted_medium(&self) -> MediumId { MediumId::AIR }

    fn incident_medium(&self) -> MediumId { MediumId::AIR }

    fn proxy(&self, _: &IorReg) -> BrdfProxy<'_> { unreachable!("not used in synthetic tests") }

    fn as_any(&self) -> &dyn std::any::Any { self }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

fn make_proxy(alpha_x: f64, alpha_y: f64, wavelengths: &[Nanometres]) -> BrdfProxy<'static> {
    let measured = Box::leak(Box::new(SyntheticMeasuredBrdf::new(
        wavelengths.to_vec().into_boxed_slice(),
    )));
    let i_thetas = vec![
        0.05f32, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75,
    ];
    let i_phis = vec![0.0f32, 0.5, 1.0];
    let o_thetas = vec![
        0.1f32, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.6,
    ];
    let o_phis = vec![0.0f32, 0.65, 1.3, 2.6];

    let iors_i = wavelengths
        .iter()
        .map(|_| Ior::new_dielectric(1.0))
        .collect::<Vec<_>>();
    let iors_t = wavelengths
        .iter()
        .map(|_| Ior::new_dielectric(1.5))
        .collect::<Vec<_>>();

    let model = MicrofacetBrdfTR::new(alpha_x, alpha_y);
    let mut samples = Vec::with_capacity(
        i_thetas.len() * i_phis.len() * o_thetas.len() * o_phis.len() * wavelengths.len(),
    );
    for theta_i in i_thetas.iter() {
        for phi_i in i_phis.iter() {
            let vi = Vec3::new(
                theta_i.sin() * phi_i.cos(),
                theta_i.sin() * phi_i.sin(),
                theta_i.cos(),
            );
            for theta_o in o_thetas.iter() {
                for phi_o in o_phis.iter() {
                    let vo = Vec3::new(
                        theta_o.sin() * phi_o.cos(),
                        theta_o.sin() * phi_o.sin(),
                        theta_o.cos(),
                    );
                    let spectral =
                        Scattering::eval_reflectance_spectrum(&model, &vi, &vo, &iors_i, &iors_t);
                    samples.extend(spectral.iter().map(|&x| x as f32));
                }
            }
        }
    }

    let resampled = DynArr::from_vec(
        &[
            i_thetas.len(),
            i_phis.len(),
            o_thetas.len(),
            o_phis.len(),
            wavelengths.len(),
        ],
        samples,
    );

    BrdfProxy::new(
        false,
        ProxySource::Measured,
        measured,
        Cow::Owned(DyArr::from_vec_1d(i_thetas)),
        Cow::Owned(DyArr::from_vec_1d(i_phis)),
        OutgoingDirs::new_grid(
            Cow::Owned(DyArr::from_vec_1d(o_thetas)),
            Cow::Owned(DyArr::from_vec_1d(o_phis)),
        ),
        Cow::Owned(resampled),
        Cow::Owned(iors_i),
        Cow::Owned(iors_t),
    )
}

fn brute_fit(
    proxy: &BrdfProxy,
    symmetry: Symmetry,
    metric: ErrorMetric,
    weighting: Weighting,
    max_theta_i: Option<Radians>,
    max_theta_o: Option<Radians>,
    alpha: Option<Roughness>,
) -> [f64; 2] {
    #[cfg(feature = "cuda")]
    let report = proxy.brute_fit(
        MicrofacetDistroKind::TrowbridgeReitz,
        symmetry,
        metric,
        weighting,
        max_theta_i,
        max_theta_o,
        1,
        false,
        alpha,
    );

    #[cfg(not(feature = "cuda"))]
    let report = proxy.brute_fit(
        MicrofacetDistroKind::TrowbridgeReitz,
        symmetry,
        metric,
        weighting,
        max_theta_i,
        max_theta_o,
        1,
        alpha,
    );

    report.best_model().expect("expected best model").params()
}

#[test]
fn test_brute_force_fitting_recovers_known_isotropic_parameters() {
    let true_alpha = 0.25;
    let proxy = make_proxy(true_alpha, true_alpha, &[nm!(550.0)]);
    let [ax, ay] = brute_fit(
        &proxy,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        None,
    );

    assert!(
        (ax - true_alpha).abs() <= 0.01,
        "ax={}, expected={}",
        ax,
        true_alpha
    );
    assert!(
        (ay - true_alpha).abs() <= 0.01,
        "ay={}, expected={}",
        ay,
        true_alpha
    );
}

#[test]
fn test_fitting_symmetry_detection() {
    let iso_proxy = make_proxy(0.2, 0.2, &[nm!(550.0)]);
    let [iso_x, iso_y] = brute_fit(
        &iso_proxy,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        None,
    );
    assert!((iso_x - iso_y).abs() < 1.0e-12);

    let aniso_proxy = make_proxy(0.12, 0.48, &[nm!(550.0)]);
    let [ax, ay] = brute_fit(
        &aniso_proxy,
        Symmetry::Anisotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        Some(Roughness::Anisotropic {
            ax: StepRangeIncl::new(0.08, 0.2, 0.02),
            ay: StepRangeIncl::new(0.36, 0.6, 0.04),
        }),
    );
    assert!((ax - 0.12).abs() <= 0.02, "ax={}", ax);
    assert!((ay - 0.48).abs() <= 0.04, "ay={}", ay);
    assert!(
        (ax - ay).abs() > 0.1,
        "anisotropic fit collapsed to isotropic"
    );
}

#[test]
fn test_fitting_error_metrics() {
    let proxy = make_proxy(0.27, 0.27, &[nm!(550.0)]);
    let mse = brute_fit(
        &proxy,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        None,
    );
    let rmse = brute_fit(
        &proxy,
        Symmetry::Isotropic,
        ErrorMetric::Rmse,
        Weighting::None,
        None,
        None,
        None,
    );

    assert!((mse[0] - rmse[0]).abs() <= 0.01);
    assert!((mse[1] - rmse[1]).abs() <= 0.01);
}

#[test]
fn test_per_wavelength_fitting_consistency() {
    let proxy = make_proxy(0.3, 0.3, &[nm!(450.0), nm!(550.0), nm!(650.0)]);
    let mut alphas = Vec::new();
    for idx in 0..proxy.spectrum.len() {
        let wl_proxy = proxy.per_wavelength(idx);
        let best = brute_fit(
            &wl_proxy,
            Symmetry::Isotropic,
            ErrorMetric::Mse,
            Weighting::None,
            None,
            None,
            None,
        );
        alphas.push(best[0]);
    }

    let min = alphas.iter().copied().fold(f64::INFINITY, f64::min);
    let max = alphas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!((max - min) <= 0.01, "per-wavelength alpha spread too large");
}

#[test]
fn test_fitting_with_noise() {
    let clean = make_proxy(0.24, 0.24, &[nm!(550.0)]);
    let mut noisy_samples = clean.samples().clone();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    noisy_samples.as_mut_slice().iter_mut().for_each(|x| {
        *x = (*x + rng.gen_range(-0.002f32..0.002f32)).max(0.0);
    });

    let noisy = BrdfProxy::new(
        false,
        ProxySource::Measured,
        clean.brdf,
        clean.i_thetas.clone(),
        clean.i_phis.clone(),
        clean.o_dirs.clone(),
        Cow::Owned(noisy_samples),
        clean.iors_i.clone(),
        clean.iors_t.clone(),
    );
    let [ax, ay] = brute_fit(
        &noisy,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        None,
    );

    assert!((ax - 0.24).abs() <= 0.03, "ax={}", ax);
    assert!((ay - 0.24).abs() <= 0.03, "ay={}", ay);
}

#[test]
fn test_fitting_bounds_checking() {
    let proxy = make_proxy(0.01, 0.99, &[nm!(550.0)]);
    let [ax, ay] = brute_fit(
        &proxy,
        Symmetry::Anisotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        Some(Roughness::Anisotropic {
            ax: StepRangeIncl::new(0.02, 0.08, 0.02),
            ay: StepRangeIncl::new(0.9, 1.0, 0.05),
        }),
    );

    assert!(
        (0.02..=0.08).contains(&ax),
        "ax={} out of expected range",
        ax
    );
    assert!((0.9..=1.0).contains(&ay), "ay={} out of expected range", ay);
}

#[test]
fn test_weighting_schemes() {
    let proxy = make_proxy(0.31, 0.31, &[nm!(550.0)]);
    let none = brute_fit(
        &proxy,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        None,
    );
    let lncos = brute_fit(
        &proxy,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::LnCos,
        None,
        None,
        None,
    );

    assert!((none[0] - 0.31).abs() <= 0.02);
    assert!((lncos[0] - 0.31).abs() <= 0.02);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cpu_gpu_consistency() {
    if std::env::var("RUN_CUDA_TESTS").is_err() {
        eprintln!("Skipping CUDA fitting consistency test. Set RUN_CUDA_TESTS=1 to run.");
        return;
    }

    let proxy = make_proxy(0.25, 0.25, &[nm!(550.0)]);
    let cpu = proxy.brute_fit(
        MicrofacetDistroKind::TrowbridgeReitz,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        1,
        false,
        None,
    );
    let gpu = proxy.brute_fit(
        MicrofacetDistroKind::TrowbridgeReitz,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        1,
        true,
        None,
    );

    let cpu = cpu.best_model().unwrap().params();
    let gpu = gpu.best_model().unwrap().params();
    assert!((cpu[0] - gpu[0]).abs() <= 0.01);
    assert!((cpu[1] - gpu[1]).abs() <= 0.01);
}

#[test]
fn test_theta_limit_filtering() {
    let proxy = make_proxy(0.25, 0.25, &[nm!(550.0)]);
    let n_full = proxy.n_filtered_samples(None, None);
    let n_limited = proxy.n_filtered_samples(
        Some(Radians::from_degrees(35.0)),
        Some(Radians::from_degrees(45.0)),
    );
    assert!(
        n_limited < n_full,
        "theta filtering did not reduce sample count"
    );

    let full = brute_fit(
        &proxy,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        None,
    );
    let limited = brute_fit(
        &proxy,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        Some(Radians::from_degrees(35.0)),
        Some(Radians::from_degrees(45.0)),
        None,
    );

    assert!((full[0] - 0.25).abs() <= 0.02);
    assert!((limited[0] - 0.25).abs() <= 0.05);
}
