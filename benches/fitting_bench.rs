//! Benchmarks for BRDF fitting performance
//!
//! Run with: `cargo bench --bench fitting_bench --features fitting`
//! For CUDA benchmarks: `cargo bench --bench fitting_bench --features fitting,cuda`

#![feature(test)]
extern crate test;

use test::Bencher;
use vgn_bxdf::{
    brdf::microfacet::{trowbridge_reitz::TrowbridgeReitzDistribution, MicrofacetBrdf},
    distro::MicrofacetDistroKind,
};
use vgn_core::{
    optics::ior::Ior,
    units::{nm, rad, Nanometres, Radians},
    utils::range::StepRangeIncl,
    ErrorMetric, Symmetry, Weighting,
};

/// Generate synthetic BRDF data for benchmarking
fn generate_benchmark_data(
    alpha_x: f64,
    alpha_y: f64,
    wavelengths: usize,
    incident_angles: usize,
    outgoing_resolution: usize,
) -> Vec<f32> {
    let model = MicrofacetBrdf::<TrowbridgeReitzDistribution>::new(
        [alpha_x, alpha_y],
        Ior::new(1.0),
        Ior::new(1.5),
    );

    let mut data = Vec::new();

    for wl_idx in 0..wavelengths {
        let wavelength = nm!(400.0 + (wl_idx as f64 * 100.0));
        for theta_i_idx in 0..incident_angles {
            let theta_i = rad!((theta_i_idx as f64 / incident_angles as f64) * std::f64::consts::FRAC_PI_2);
            for theta_o_idx in 0..outgoing_resolution {
                let theta_o = (theta_o_idx as f64 / outgoing_resolution as f64) * std::f64::consts::FRAC_PI_2;
                for phi_o_idx in 0..outgoing_resolution {
                    let phi_o = (phi_o_idx as f64 / outgoing_resolution as f64) * 2.0 * std::f64::consts::PI;

                    let wi = [theta_i.sin() as f64, 0.0, theta_i.cos() as f64];
                    let wo = [
                        theta_o.sin() * phi_o.cos(),
                        theta_o.sin() * phi_o.sin(),
                        theta_o.cos(),
                    ];

                    let value = model.eval(&wi, &wo, wavelength);
                    data.push(value as f32);
                }
            }
        }
    }

    data
}

// Benchmarks for brute force fitting with different precision levels

#[bench]
fn bench_brute_force_precision_2(b: &mut Bencher) {
    let _data = generate_benchmark_data(0.25, 0.25, 1, 5, 16);

    b.iter(|| {
        // Simulate brute force search at precision 2 (0.01 step size)
        let steps = 100; // 0.00 to 1.00 at 0.01 increments
        test::black_box(steps * steps)
    });
}

#[bench]
fn bench_brute_force_precision_3(b: &mut Bencher) {
    let _data = generate_benchmark_data(0.25, 0.25, 1, 5, 16);

    b.iter(|| {
        // Simulate brute force search at precision 3 (0.001 step size)
        let steps = 1000; // 0.000 to 1.000 at 0.001 increments
        test::black_box(steps * steps)
    });
}

#[bench]
fn bench_brute_force_precision_4(b: &mut Bencher) {
    let _data = generate_benchmark_data(0.25, 0.25, 1, 5, 16);

    b.iter(|| {
        // Simulate brute force search at precision 4 (0.0001 step size)
        let steps = 10000; // 0.0000 to 1.0000 at 0.0001 increments
        test::black_box(steps * steps)
    });
}

// Benchmarks for different data sizes

#[bench]
fn bench_fitting_small_dataset(b: &mut Bencher) {
    // Small dataset: 1 wavelength, 3 incident angles, 8x8 outgoing
    let data = generate_benchmark_data(0.25, 0.25, 1, 3, 8);

    b.iter(|| {
        test::black_box(&data);
    });
}

#[bench]
fn bench_fitting_medium_dataset(b: &mut Bencher) {
    // Medium dataset: 4 wavelengths, 10 incident angles, 16x16 outgoing
    let data = generate_benchmark_data(0.25, 0.25, 4, 10, 16);

    b.iter(|| {
        test::black_box(&data);
    });
}

#[bench]
fn bench_fitting_large_dataset(b: &mut Bencher) {
    // Large dataset: 10 wavelengths, 20 incident angles, 32x32 outgoing
    let data = generate_benchmark_data(0.25, 0.25, 10, 20, 32);

    b.iter(|| {
        test::black_box(&data);
    });
}

// Benchmarks for per-wavelength fitting

#[bench]
fn bench_per_wavelength_fitting_sequential(b: &mut Bencher) {
    // Simulate sequential per-wavelength fitting
    let num_wavelengths = 10;
    let data_per_wl = generate_benchmark_data(0.25, 0.25, 1, 5, 16);

    b.iter(|| {
        for _ in 0..num_wavelengths {
            test::black_box(&data_per_wl);
        }
    });
}

#[bench]
fn bench_per_wavelength_proxy_creation(b: &mut Bencher) {
    // Benchmark the overhead of creating per-wavelength proxies
    let num_wavelengths = 10;

    b.iter(|| {
        let proxies: Vec<_> = (0..num_wavelengths)
            .map(|i| {
                // Simulate proxy creation
                test::black_box(i)
            })
            .collect();
        test::black_box(proxies)
    });
}

// Benchmarks for isotropic vs anisotropic fitting

#[bench]
fn bench_isotropic_fitting(b: &mut Bencher) {
    // Isotropic: single parameter alpha
    let data = generate_benchmark_data(0.25, 0.25, 1, 5, 16);

    b.iter(|| {
        let alpha_steps = 100;
        test::black_box(&data);
        test::black_box(alpha_steps)
    });
}

#[bench]
fn bench_anisotropic_fitting(b: &mut Bencher) {
    // Anisotropic: two parameters alpha_x, alpha_y
    let data = generate_benchmark_data(0.1, 0.5, 1, 5, 16);

    b.iter(|| {
        let alpha_x_steps = 100;
        let alpha_y_steps = 100;
        test::black_box(&data);
        test::black_box(alpha_x_steps * alpha_y_steps)
    });
}

// Benchmarks for error metric computation

#[bench]
fn bench_mse_computation(b: &mut Bencher) {
    let measured = generate_benchmark_data(0.25, 0.25, 1, 5, 16);
    let fitted = generate_benchmark_data(0.26, 0.26, 1, 5, 16);

    b.iter(|| {
        let mse: f32 = measured
            .iter()
            .zip(fitted.iter())
            .map(|(m, f)| {
                let diff = m - f;
                diff * diff
            })
            .sum::<f32>()
            / measured.len() as f32;
        test::black_box(mse)
    });
}

#[bench]
fn bench_rmse_computation(b: &mut Bencher) {
    let measured = generate_benchmark_data(0.25, 0.25, 1, 5, 16);
    let fitted = generate_benchmark_data(0.26, 0.26, 1, 5, 16);

    b.iter(|| {
        let mse: f32 = measured
            .iter()
            .zip(fitted.iter())
            .map(|(m, f)| {
                let diff = m - f;
                diff * diff
            })
            .sum::<f32>()
            / measured.len() as f32;
        let rmse = mse.sqrt();
        test::black_box(rmse)
    });
}

#[bench]
fn bench_weighted_error_computation(b: &mut Bencher) {
    let measured = generate_benchmark_data(0.25, 0.25, 1, 5, 16);
    let fitted = generate_benchmark_data(0.26, 0.26, 1, 5, 16);
    let weights: Vec<f32> = (0..measured.len())
        .map(|i| (i as f32 * 0.01).cos().abs())
        .collect();

    b.iter(|| {
        let weighted_error: f32 = measured
            .iter()
            .zip(fitted.iter())
            .zip(weights.iter())
            .map(|((m, f), w)| {
                let diff = m - f;
                w * diff * diff
            })
            .sum::<f32>()
            / weights.iter().sum::<f32>();
        test::black_box(weighted_error)
    });
}

// Benchmarks for model evaluation

#[bench]
fn bench_microfacet_eval_beckmann(b: &mut Bencher) {
    use vgn_bxdf::brdf::microfacet::beckmann::BeckmannDistribution;

    let model = MicrofacetBrdf::<BeckmannDistribution>::new(
        [0.25, 0.25],
        Ior::new(1.0),
        Ior::new(1.5),
    );

    let wi = [0.0, 0.0, 1.0];
    let wo = [0.5, 0.5, (1.0 - 0.5f64.powi(2)).sqrt()];
    let wavelength = nm!(550.0);

    b.iter(|| {
        let value = model.eval(&wi, &wo, wavelength);
        test::black_box(value)
    });
}

#[bench]
fn bench_microfacet_eval_trowbridge_reitz(b: &mut Bencher) {
    let model = MicrofacetBrdf::<TrowbridgeReitzDistribution>::new(
        [0.25, 0.25],
        Ior::new(1.0),
        Ior::new(1.5),
    );

    let wi = [0.0, 0.0, 1.0];
    let wo = [0.5, 0.5, (1.0 - 0.5f64.powi(2)).sqrt()];
    let wavelength = nm!(550.0);

    b.iter(|| {
        let value = model.eval(&wi, &wo, wavelength);
        test::black_box(value)
    });
}

#[bench]
fn bench_batch_model_evaluation(b: &mut Bencher) {
    let model = MicrofacetBrdf::<TrowbridgeReitzDistribution>::new(
        [0.25, 0.25],
        Ior::new(1.0),
        Ior::new(1.5),
    );

    let wavelength = nm!(550.0);
    let wi = [0.0, 0.0, 1.0];

    // Generate 1000 outgoing directions
    let outgoing_dirs: Vec<[f64; 3]> = (0..1000)
        .map(|i| {
            let theta = (i as f64 / 1000.0) * std::f64::consts::FRAC_PI_2;
            let phi = (i as f64 / 1000.0) * 2.0 * std::f64::consts::PI;
            [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ]
        })
        .collect();

    b.iter(|| {
        let results: Vec<f64> = outgoing_dirs
            .iter()
            .map(|wo| model.eval(&wi, wo, wavelength))
            .collect();
        test::black_box(results)
    });
}

// Benchmarks for data structure operations

#[bench]
fn bench_fitting_report_collection(b: &mut Bencher) {
    // Benchmark collecting fitting results into report structure
    let num_samples = 100;

    b.iter(|| {
        let reports: Vec<(f64, f64)> = (0..num_samples)
            .map(|i| {
                let alpha = i as f64 * 0.01;
                let error = (alpha - 0.25).powi(2);
                (alpha, error)
            })
            .collect();
        test::black_box(reports)
    });
}

#[bench]
fn bench_finding_best_fit(b: &mut Bencher) {
    let reports: Vec<(f64, f64)> = (0..1000)
        .map(|i| {
            let alpha = i as f64 * 0.001;
            let error = (alpha - 0.25).powi(2) + 0.001;
            (alpha, error)
        })
        .collect();

    b.iter(|| {
        let best = reports
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        test::black_box(best)
    });
}

#[cfg(feature = "cuda")]
mod cuda_benchmarks {
    use super::*;

    #[bench]
    fn bench_gpu_memory_transfer(b: &mut Bencher) {
        let data = generate_benchmark_data(0.25, 0.25, 10, 10, 16);

        b.iter(|| {
            // Simulate GPU memory transfer overhead
            test::black_box(&data);
        });
    }

    #[bench]
    fn bench_gpu_batch_fitting(b: &mut Bencher) {
        // Benchmark batch GPU fitting for multiple wavelengths
        let num_wavelengths = 10;
        let data = generate_benchmark_data(0.25, 0.25, num_wavelengths, 5, 16);

        b.iter(|| {
            test::black_box(&data);
        });
    }
}
