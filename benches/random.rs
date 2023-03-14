#![feature(test)]

use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;
use vgonio::units::{radians, Angle, URadian};

extern crate test;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::f32::consts::PI;

const THETA_START: Angle<URadian> = radians!(0.0);
const THETA_STOP: Angle<URadian> = radians!(PI);
const PHI_START: Angle<URadian> = radians!(0.0);
const PHI_STOP: Angle<URadian> = radians!(2.0 * PI);
const NUM_SAMPLES: usize = 100_000_000;

#[bench]
fn multi_threaded_per_chunk_1024(b: &mut test::Bencher) {
    b.iter(|| {
        let range = Uniform::new(0.0f32, 1.0);
        let mut samples = Vec::with_capacity(NUM_SAMPLES);
        samples.resize(NUM_SAMPLES, glam::Vec3::ZERO);
        samples.par_chunks_mut(1024).for_each(|chunks| {
            let mut rng = rand::thread_rng();
            let mut i = 0;
            while i < chunks.len() {
                let phi = radians!(range.sample(&mut rng) * PI * 2.0);
                let theta = radians!((1.0 - 2.0f32 * range.sample(&mut rng)).acos());
                if (THETA_START..THETA_STOP).contains(&theta)
                    && (PHI_START..PHI_STOP).contains(&phi)
                {
                    chunks[i] = glam::Vec3::new(
                        theta.sin() * phi.cos(),
                        theta.sin() * phi.sin(),
                        theta.cos(),
                    );
                    i += 1;
                }
            }
        });
    });
}

#[bench]
fn multi_threaded_per_chunk_8192(b: &mut test::Bencher) {
    b.iter(|| {
        let range = Uniform::new(0.0f32, 1.0);
        let mut samples = Vec::with_capacity(NUM_SAMPLES);
        samples.resize(NUM_SAMPLES, glam::Vec3::ZERO);
        samples.par_chunks_mut(8192).for_each(|chunks| {
            let mut rng = rand::thread_rng();
            let mut i = 0;
            while i < chunks.len() {
                let phi = radians!(range.sample(&mut rng) * PI * 2.0);
                let theta = radians!((1.0 - 2.0f32 * range.sample(&mut rng)).acos());
                if (THETA_START..THETA_STOP).contains(&theta)
                    && (PHI_START..PHI_STOP).contains(&phi)
                {
                    chunks[i] = glam::Vec3::new(
                        theta.sin() * phi.cos(),
                        theta.sin() * phi.sin(),
                        theta.cos(),
                    );
                    i += 1;
                }
            }
        });
    });
}

#[bench]
fn multi_threaded_deterministic_8192(b: &mut test::Bencher) {
    const SEED: u64 = 0;

    // Initialise all RNGs from a single seed, but using multiple streams.
    b.iter(|| {
        let range = Uniform::new(0.0f32, 1.0);
        let mut samples = Vec::with_capacity(NUM_SAMPLES);
        samples.resize(NUM_SAMPLES, glam::Vec3::ZERO);

        samples
            .par_chunks_mut(8192)
            .enumerate()
            .for_each(|(i, chunks)| {
                let mut rng = ChaCha8Rng::seed_from_u64(SEED);
                rng.set_stream(i as u64);

                let mut i = 0;
                while i < chunks.len() {
                    let phi = radians!(range.sample(&mut rng) * PI * 2.0);
                    let theta = radians!((1.0 - 2.0f32 * range.sample(&mut rng)).acos());
                    if (THETA_START..THETA_STOP).contains(&theta)
                        && (PHI_START..PHI_STOP).contains(&phi)
                    {
                        chunks[i] = glam::Vec3::new(
                            theta.sin() * phi.cos(),
                            theta.sin() * phi.sin(),
                            theta.cos(),
                        );
                        i += 1;
                    }
                }
            });
    });
}

#[bench]
fn multi_threaded_deterministic_1024(b: &mut test::Bencher) {
    const SEED: u64 = 0;

    // Initialise all RNGs from a single seed, but using multiple streams.
    b.iter(|| {
        let range = Uniform::new(0.0f32, 1.0);
        let mut samples = Vec::with_capacity(NUM_SAMPLES);
        samples.resize(NUM_SAMPLES, glam::Vec3::ZERO);

        samples
            .par_chunks_mut(1024)
            .enumerate()
            .for_each(|(i, chunks)| {
                let mut rng = ChaCha8Rng::seed_from_u64(SEED);
                rng.set_stream(i as u64);

                let mut i = 0;
                while i < chunks.len() {
                    let phi = radians!(range.sample(&mut rng) * PI * 2.0);
                    let theta = radians!((1.0 - 2.0f32 * range.sample(&mut rng)).acos());
                    if (THETA_START..THETA_STOP).contains(&theta)
                        && (PHI_START..PHI_STOP).contains(&phi)
                    {
                        chunks[i] = glam::Vec3::new(
                            theta.sin() * phi.cos(),
                            theta.sin() * phi.sin(),
                            theta.cos(),
                        );
                        i += 1;
                    }
                }
            });
    });
}

// #[bench]
// fn single_thread(b: &mut test::Bencher) {
//     b.iter(|| {
//         let range = Uniform::new(0.0f32, 1.0);
//         let mut samples = Vec::with_capacity(NUM_SAMPLES);
//         samples.resize(NUM_SAMPLES, glam::Vec3::ZERO);
//         let mut rng = rand::thread_rng();
//         let mut i = 0;
//         while i < NUM_SAMPLES {
//             let phi = radians!(range.sample(&mut rng) * PI * 2.0);
//             let theta = radians!((1.0 - 2.0f32 * range.sample(&mut
// rng)).acos());             if (THETA_START..THETA_STOP).contains(&theta) &&
// (PHI_START..PHI_STOP).contains(&phi) {                 samples[i] =
// glam::Vec3::new(                     theta.sin() * phi.cos(),
//                     theta.sin() * phi.sin(),
//                     theta.cos(),
//                 );
//                 i += 1;
//             }
//         }
//     });
// }
