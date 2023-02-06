//! Microfacet related measurement.
//!
//! This module contains the code for measuring the microfacet distribution
//! (NDF, normal distribution function) and the shadowing-masking function (G,
//! geometric attenuation function).

mod distribution;
mod visibility;

pub use distribution::*;
pub use visibility::*;

// pub trait MicroSurfaceMeasurement {
//     fn samples(&self) -> &[f32];
// }

// // fn ndf(&self, m: &Vector3<f32>) -> f32;
// // fn g(&self, m: &Vector3<f32>, v: &Vector3<f32>, l: &Vector3<f32>) -> f32;

// impl MicroSurfaceMeasurement for MicrofacetNormalDistribution {}
// impl MicroSurfaceMeasurement for MicrofacetMaskingShadowing {}

// impl MicroSurfaceMetric for Beckmann {}
// impl MicroSurfaceMetric for TrowbridgeReitz {}
// impl MicroSurfaceMetric for Blinn {}
// impl MicroSurfaceMetric for Phong {}
// impl MicroSurfaceMetric for AshikhminShirley {}
// impl MicroSurfaceMetric for Ward {}
// impl MicroSurfaceMetric for CookTorrance {}
// impl MicroSurfaceMetric for GGX {}
