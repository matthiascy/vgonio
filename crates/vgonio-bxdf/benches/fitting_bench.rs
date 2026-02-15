//! Benchmarks for BRDF fitting on synthetic proxy data.

#![feature(test)]

extern crate test;

use std::borrow::Cow;
use test::{black_box, Bencher};
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
    units::{nm, Nanometres},
    utils::{medium::Medium, range::StepRangeIncl},
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

    fn transmitted_medium(&self) -> Medium { Medium::Air }

    fn incident_medium(&self) -> Medium { Medium::Air }

    fn proxy(&self, _: &IorReg) -> BrdfProxy<'_> { unreachable!("not used in benches") }

    fn as_any(&self) -> &dyn std::any::Any { self }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

fn make_proxy(alpha_x: f64, alpha_y: f64, wavelengths: &[Nanometres]) -> BrdfProxy<'static> {
    let measured = Box::leak(Box::new(SyntheticMeasuredBrdf::new(
        wavelengths.to_vec().into_boxed_slice(),
    )));
    let i_thetas = vec![0.2f32, 0.55f32];
    let i_phis = vec![0.0f32];
    let o_thetas = vec![0.15f32, 0.45f32];
    let o_phis = vec![0.0f32, 1.7f32];
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

fn distance_mse(a: &BrdfProxy, b: &BrdfProxy) -> f64 {
    #[cfg(feature = "cuda")]
    {
        a.distance(b, ErrorMetric::Mse, Weighting::None, None, None)
    }
    #[cfg(not(feature = "cuda"))]
    {
        a.distance(b, ErrorMetric::Mse, Weighting::None)
    }
}

fn brute_fit_isotropic(proxy: &BrdfProxy) -> [f64; 2] {
    #[cfg(feature = "cuda")]
    let report = proxy.brute_fit(
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

    #[cfg(not(feature = "cuda"))]
    let report = proxy.brute_fit(
        MicrofacetDistroKind::TrowbridgeReitz,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        1,
        None,
    );

    report.best_model().unwrap().params()
}

fn brute_fit_anisotropic(proxy: &BrdfProxy) -> [f64; 2] {
    let alpha = Some(Roughness::Anisotropic {
        ax: StepRangeIncl::new(0.1, 0.3, 0.05),
        ay: StepRangeIncl::new(0.35, 0.55, 0.05),
    });

    #[cfg(feature = "cuda")]
    let report = proxy.brute_fit(
        MicrofacetDistroKind::TrowbridgeReitz,
        Symmetry::Anisotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        1,
        false,
        alpha,
    );

    #[cfg(not(feature = "cuda"))]
    let report = proxy.brute_fit(
        MicrofacetDistroKind::TrowbridgeReitz,
        Symmetry::Anisotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        1,
        alpha,
    );

    report.best_model().unwrap().params()
}

#[bench]
fn bench_generate_analytical_proxy(b: &mut Bencher) {
    let measured = make_proxy(0.25, 0.25, &[nm!(550.0)]);
    let model = MicrofacetBrdfTR::new(0.27, 0.27);
    b.iter(|| {
        let generated = measured.generate_analytical(black_box(&model));
        black_box(generated.samples().len())
    });
}

#[bench]
fn bench_proxy_distance_mse(b: &mut Bencher) {
    let measured = make_proxy(0.25, 0.25, &[nm!(550.0)]);
    let model = MicrofacetBrdfTR::new(0.27, 0.27);
    let generated = measured.generate_analytical(&model);
    b.iter(|| black_box(distance_mse(&measured, &generated)));
}

#[bench]
fn bench_brute_force_isotropic(b: &mut Bencher) {
    let measured = make_proxy(0.25, 0.25, &[nm!(550.0)]);
    b.iter(|| black_box(brute_fit_isotropic(&measured)));
}

#[bench]
fn bench_brute_force_anisotropic_windowed(b: &mut Bencher) {
    let measured = make_proxy(0.15, 0.45, &[nm!(550.0)]);
    b.iter(|| black_box(brute_fit_anisotropic(&measured)));
}
