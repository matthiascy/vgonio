//! Benchmarks for BRDF fitting on synthetic proxy data.

#![feature(test)]

extern crate test;

use std::{borrow::Cow, sync::OnceLock};
use test::{black_box, Bencher};

static INIT: OnceLock<()> = OnceLock::new();

/// Bootstrap the process-wide medium registry exactly once before any bench
/// runs. Cargo bench shares a process across benchmarks, so a single
/// `OnceLock` is sufficient.
fn init_bench_registry() {
    INIT.get_or_init(|| {
        let _ = vgn_core::utils::medium::bootstrap(None, None);
    });
}
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
    utils::{medium::MediumId, range::StepRangeIncl},
    ErrorMetric, Symmetry, Weighting,
};
use vgn_jabr::array::{DyArr, DynArr};

#[derive(Clone, Copy)]
struct ProxyShape {
    i_theta_count: usize,
    i_phi_count: usize,
    o_theta_count: usize,
    o_phi_count: usize,
}

const SHAPE_SMALL: ProxyShape = ProxyShape {
    i_theta_count: 3,
    i_phi_count: 2,
    o_theta_count: 4,
    o_phi_count: 4,
};

const SHAPE_MEDIUM: ProxyShape = ProxyShape {
    i_theta_count: 8,
    i_phi_count: 4,
    o_theta_count: 10,
    o_phi_count: 10,
};

const SHAPE_LARGE: ProxyShape = ProxyShape {
    i_theta_count: 16,
    i_phi_count: 16,
    o_theta_count: 16,
    o_phi_count: 16,
};

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

    fn proxy(&self, _: &IorReg) -> BrdfProxy<'_> { unreachable!("not used in benches") }

    fn as_any(&self) -> &dyn std::any::Any { self }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

fn linspace(start: f32, end: f32, count: usize) -> Vec<f32> {
    debug_assert!(count > 0);
    if count == 1 {
        return vec![start];
    }
    let step = (end - start) / (count as f32 - 1.0);
    (0..count).map(|i| start + i as f32 * step).collect()
}

fn make_proxy(
    alpha_x: f64,
    alpha_y: f64,
    wavelengths: &[Nanometres],
    shape: ProxyShape,
) -> BrdfProxy<'static> {
    let measured = Box::leak(Box::new(SyntheticMeasuredBrdf::new(
        wavelengths.to_vec().into_boxed_slice(),
    )));
    let i_thetas = linspace(0.05, 0.75, shape.i_theta_count);
    let i_phis = linspace(0.0, 2.8, shape.i_phi_count);
    let o_thetas = linspace(0.05, 0.7, shape.o_theta_count);
    let o_phis = linspace(0.0, 2.8, shape.o_phi_count);
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

fn proxy_bytes(proxy: &BrdfProxy) -> u64 {
    (proxy.samples().len() * std::mem::size_of::<f32>()) as u64
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

fn brute_fit_isotropic(proxy: &BrdfProxy, precision: u32) -> [f64; 2] {
    #[cfg(feature = "cuda")]
    let report = proxy.brute_fit(
        MicrofacetDistroKind::TrowbridgeReitz,
        Symmetry::Isotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        precision,
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
        precision,
        None,
    );

    report.best_model().unwrap().params()
}

fn brute_fit_anisotropic(proxy: &BrdfProxy, precision: u32) -> [f64; 2] {
    let alpha = Some(Roughness::Anisotropic {
        ax: StepRangeIncl::new(0.08, 0.22, 0.01),
        ay: StepRangeIncl::new(0.32, 0.58, 0.01),
    });

    #[cfg(feature = "cuda")]
    let report = proxy.brute_fit(
        MicrofacetDistroKind::TrowbridgeReitz,
        Symmetry::Anisotropic,
        ErrorMetric::Mse,
        Weighting::None,
        None,
        None,
        precision,
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
        precision,
        alpha,
    );

    report.best_model().unwrap().params()
}

#[bench]
fn bench_generate_analytical_proxy_small(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(0.25, 0.25, &[nm!(550.0)], SHAPE_SMALL);
    let model = MicrofacetBrdfTR::new(0.27, 0.27);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| {
        let generated = measured.generate_analytical(black_box(&model));
        black_box(generated.samples().len())
    });
}

#[bench]
fn bench_generate_analytical_proxy_medium(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(
        0.25,
        0.25,
        &[nm!(450.0), nm!(550.0), nm!(650.0)],
        SHAPE_MEDIUM,
    );
    let model = MicrofacetBrdfTR::new(0.27, 0.27);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| {
        let generated = measured.generate_analytical(black_box(&model));
        black_box(generated.samples().len())
    });
}

#[bench]
fn bench_generate_analytical_proxy_large(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(
        0.25,
        0.25,
        &[nm!(450.0), nm!(550.0), nm!(650.0)],
        SHAPE_LARGE,
    );
    let model = MicrofacetBrdfTR::new(0.27, 0.27);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| {
        let generated = measured.generate_analytical(black_box(&model));
        black_box(generated.samples().len())
    });
}

#[bench]
fn bench_proxy_distance_mse_small(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(0.25, 0.25, &[nm!(550.0)], SHAPE_SMALL);
    let model = MicrofacetBrdfTR::new(0.27, 0.27);
    let generated = measured.generate_analytical(&model);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| black_box(distance_mse(&measured, &generated)));
}

#[bench]
fn bench_proxy_distance_mse_medium(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(
        0.25,
        0.25,
        &[nm!(450.0), nm!(550.0), nm!(650.0)],
        SHAPE_MEDIUM,
    );
    let model = MicrofacetBrdfTR::new(0.27, 0.27);
    let generated = measured.generate_analytical(&model);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| black_box(distance_mse(&measured, &generated)));
}

#[bench]
fn bench_proxy_distance_mse_large(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(
        0.25,
        0.25,
        &[nm!(450.0), nm!(550.0), nm!(650.0)],
        SHAPE_LARGE,
    );
    let model = MicrofacetBrdfTR::new(0.27, 0.27);
    let generated = measured.generate_analytical(&model);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| black_box(distance_mse(&measured, &generated)));
}

#[bench]
fn bench_brute_force_isotropic_medium(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(0.25, 0.25, &[nm!(550.0)], SHAPE_MEDIUM);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| black_box(brute_fit_isotropic(&measured, 2)));
}

#[bench]
fn bench_brute_force_isotropic_large(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(0.25, 0.25, &[nm!(550.0)], SHAPE_LARGE);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| black_box(brute_fit_isotropic(&measured, 2)));
}

#[bench]
fn bench_brute_force_anisotropic_windowed_medium(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(0.15, 0.45, &[nm!(550.0)], SHAPE_MEDIUM);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| black_box(brute_fit_anisotropic(&measured, 1)));
}

#[bench]
fn bench_brute_force_anisotropic_windowed_large(b: &mut Bencher) {
    init_bench_registry();
    let measured = make_proxy(0.15, 0.45, &[nm!(550.0)], SHAPE_LARGE);
    b.bytes = proxy_bytes(&measured);
    b.iter(|| black_box(brute_fit_anisotropic(&measured, 1)));
}
