use crate::MicroSurface;
use glam::{IVec2, Vec2};
use serde::{Deserialize, Serialize};
use vgonio_core::units::LengthUnit;

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
        let range: Uniform<f32> = Uniform::new(0.0, 1.0);
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
                let height = range.sample(&mut rng) * height;
                (pos, height)
            })
            .collect();

        MicroSurface::new_by(rows, cols, du, dv, unit, |row, col| {
            let x = col as f32 / cols as f32;
            let y = row as f32 / rows as f32;
            let mut dists = [f32::MAX; 3];
            let mut poses = [Vec2::ZERO; 3];
            let mut heights = [f32::MAX; 3];
            for (pos, height) in &seeds {
                let dist = (Vec2::new(x, y) - *pos).length();
                if dist < dists[0] {
                    dists[2] = dists[1];
                    dists[1] = dists[0];
                    dists[0] = dist;
                    poses[2] = poses[1];
                    poses[1] = poses[0];
                    poses[0] = *pos;
                    heights[2] = heights[1];
                    heights[1] = heights[0];
                    heights[0] = *height;
                } else if dist < dists[1] {
                    dists[2] = dists[1];
                    dists[1] = dist;
                    poses[2] = poses[1];
                    poses[1] = *pos;
                    heights[2] = heights[1];
                    heights[1] = *height;
                } else if dist < dists[2] {
                    dists[2] = dist;
                    poses[2] = *pos;
                    heights[2] = *height;
                }
            }
            let a = x - poses[0].x;
            let b = poses[1].x - poses[0].x;
            let c = poses[2].x - poses[0].x;
            let d = y - poses[0].y;
            let e = poses[1].y - poses[0].y;
            let f = poses[2].y - poses[0].y;
            let det = b * f - c * e;
            let u = (a * f - c * d) / det;
            let v = (b * d - a * e) / det;
            if u + v > 1.0 {
                return heights[0];
            }
            if u < 0.0 {
                return heights[1];
            }
            if v < 0.0 {
                return heights[2];
            }
            (1.0 - u - v) * heights[0] + u * heights[1] + v * heights[2]
        })
    }
}
