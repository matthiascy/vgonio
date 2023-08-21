use crate::MicroSurface;
use glam::{IVec2, UVec2, Vec2};
use serde::{Deserialize, Serialize};
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
    /// Simplex noise. Improved version of Perlin noise.
    SimplexNoise,
    /// Value noise.
    ValueNoise,
    /// Worley noise. Extension of the Voronoi diagram. Known as cell noise or
    /// Voronoi noise.
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

    /// Generates a surface randomly using the Worley noise.
    pub fn from_worley_noise(
        rows: usize,
        cols: usize,
        du: f32,
        dv: f32,
        height: f32,
        seeds_count: usize,
        unit: LengthUnit,
    ) -> Self {
        use rand::distributions::{Distribution, Uniform};
        let range: Uniform<f32> = Uniform::new(0.0, height * 0.5);
        let mut rng = rand::thread_rng();

        let seeds_count_sqrt = (seeds_count as f32).sqrt().ceil() as usize;
        let step_size = 1.0 / seeds_count_sqrt as f32;
        let seeds: Vec<(Vec2, f32)> = (0..seeds_count_sqrt * seeds_count_sqrt)
            .map(|i| {
                let cell = IVec2::new((i % seeds_count_sqrt) as i32, (i / seeds_count_sqrt) as i32);
                let pos = Vec2::new(
                    range.sample(&mut rng) + cell.x as f32,
                    range.sample(&mut rng) + cell.y as f32,
                ) * step_size;
                let height = range.sample(&mut rng);
                (pos, height)
            })
            .collect();

        MicroSurface::new_by(rows, cols, du, dv, unit, |row, col| {
            let x = col as f32 / cols as f32;
            let y = row as f32 / rows as f32;
            let mut min_dist = f32::MAX;
            // let mut heights = [0.0; 2];
            let cur_cell = IVec2::new(
                (x * seeds_count_sqrt as f32) as i32,
                (y * seeds_count_sqrt as f32) as i32,
            );
            for (pos, height) in &seeds {
                let dist = (Vec2::new(x, y) - *pos).length();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            min_dist
        })
    }
}
