use crate::MicroSurface;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use vgcore::units::LengthUnit;

/// Kind of generated surface.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
pub enum SurfGenKind {
    /// Randomly generated surface.
    Random,
    /// Surface generated from a 2D Gaussian distribution.
    #[clap(name = "gaussian2d")]
    Gaussian2D,
}

/// Possible ways to generate a random surface.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
pub enum RandomGenMethod {
    /// White noise.
    WhiteNoise,
    /// Perlin noise.
    PerlinNoise,
    /// Simplex noise.
    SimplexNoise,
    /// Value noise.
    ValueNoise,
    /// Worley noise.
    WorleyNoise,
    /// Diamond-square algorithm.
    DiamondSquare,
}

impl MicroSurface {
    /// Generates a random surface.
    pub fn from_white_noise(
        rows: usize,
        cols: usize,
        du: f32,
        dv: f32,
        height: f32,
        unit: LengthUnit,
    ) -> Self {
        use rand::distributions::{Distribution, Uniform};
        let range: Uniform<f32> = Uniform::new(0.0, height);
        let mut rng = rand::thread_rng();
        MicroSurface::new_by(rows, cols, du, dv, unit, |_, _| range.sample(&mut rng))
    }
}
