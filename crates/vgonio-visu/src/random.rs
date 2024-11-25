use jabr::{Pnt3, Vec3};
use rand::distributions::{Distribution, Uniform};
use std::ops::Range;

use crate::Plane;

// TODO: improve this
pub fn samples_in_unit_square_2d(samples: &mut [Pnt3]) {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    for sample in samples {
        sample.x = dist.sample(&mut rng);
        sample.y = dist.sample(&mut rng);
        sample.z = 0.0;
    }
}

pub fn random_point_in_unit_disk_xz() -> Pnt3 {
    let mut rng = rand::thread_rng();
    let dist = Uniform::<f64>::new(0.0, 1.0);
    let r = dist.sample(&mut rng).sqrt();
    let a = dist.sample(&mut rng) * std::f64::consts::TAU;
    Pnt3::new(r * a.cos(), 0.0, r * a.sin())
}

/// Generates uniformly distributed samples on the unit disk
pub fn uniform_samples_on_unit_disk(samples: &mut [Pnt3], plane: Plane) {
    let mut rng = rand::thread_rng();
    let dist = Uniform::<f64>::new(0.0, 1.0);
    samples.iter_mut().for_each(|sample| {
        let r = dist.sample(&mut rng).sqrt();
        let a = dist.sample(&mut rng) * std::f64::consts::TAU;
        let s = r * a.cos();
        let t = r * a.sin();
        match plane {
            Plane::XY => {
                sample.x = s;
                sample.y = t;
                sample.z = 0.0;
            },
            Plane::XZ => {
                sample.x = s;
                sample.y = 0.0;
                sample.z = t;
            },
            Plane::YZ => {
                sample.x = 0.0;
                sample.y = s;
                sample.z = t;
            },
        }
    });
}

/// Generate a random vector on the hemisphere with normal `n` using rejection
/// sampling.
pub fn random_vec3_on_hemisphere(n: &Vec3) -> Vec3 {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    let phi = dist.sample(&mut rng) * std::f64::consts::TAU;
    let theta = (1.0 - 2.0 * dist.sample(&mut rng)).acos();
    let v = Vec3::new(
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    );
    if v.dot(n) > 0.0 {
        v
    } else {
        -v
    }
}

pub fn random_vec3_on_unit_sphere() -> Vec3 {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    let phi = dist.sample(&mut rng) * std::f64::consts::TAU;
    let theta = (1.0 - 2.0 * dist.sample(&mut rng)).acos();
    Vec3::new(
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    )
}

pub fn uniform_spherical_sampling(theta: Range<f64>, phi: Range<f64>, samples: &mut [Pnt3]) {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    let mut i = 0;
    while i < samples.len() {
        let ph = dist.sample(&mut rng) * std::f64::consts::TAU;
        let th = (1.0 - 2.0 * dist.sample(&mut rng)).acos();
        if theta.contains(&th) && phi.contains(&ph) {
            samples[i] = Pnt3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos());
            i += 1;
        }
    }
}

pub fn uniform_sampling_on_unit_sphere(samples: &mut [Pnt3]) {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0);
    for sample in samples {
        let phi = dist.sample(&mut rng) * std::f64::consts::TAU;
        let theta = (1.0 - 2.0 * dist.sample(&mut rng)).acos();
        let phi_sin = phi.sin();
        let phi_cos = phi.cos();
        let theta_sin = theta.sin();
        let theta_cos = theta.cos();
        *sample = Pnt3::new(theta_sin * phi_cos, theta_sin * phi_sin, theta_cos);
    }
}
