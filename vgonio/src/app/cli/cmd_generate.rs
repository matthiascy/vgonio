use crate::app::{
    cli::{BRIGHT_CYAN, BRIGHT_YELLOW, RESET},
    Config,
};
use std::path::PathBuf;
use vgcore::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
    units::LengthUnit,
};
use vgsurf::{MicroSurface, RandomGenMethod, SurfGenKind};

#[derive(clap::Args, Debug)]
#[clap(about = "Generate a micro-geometry level surface using Gaussian distribution.")]
pub struct GenerateOptions {
    /// Horizontal and vertical resolution of the generated micro-geometry
    /// profile.
    #[arg(long = "res", num_args(2), default_values_t = [1024; 2])]
    pub res: Vec<u32>,

    /// Horizontal and vertical spacing between micro-geometry height samples,
    /// in micrometers.
    #[arg(long = "spacing", num_args(2), default_values_t = [0.1; 2])]
    pub spacing: Vec<f32>,

    /// Highest height of the generated micro-geometry profile, in micrometers.
    /// The lowest height is always 0.
    #[arg(long = "height", default_value_t = 1.0)]
    pub max_height: f32,

    /// Type of surface to generate.
    #[arg(ignore_case = true, long = "kind")]
    pub kind: SurfGenKind,

    /// Amplitude of the 2D gaussian. Only used when `kind` is `gaussian2d`.
    #[arg(long = "amp", required_if_eq("kind", "gaussian2d"))]
    pub amplitude: Option<f32>,

    /// Standard deviation of the 2D gaussian distribution in horizontal
    /// direction. Only used when `kind` is `gaussian2d`.
    #[arg(long = "sx", required_if_eq("kind", "gaussian2d"))]
    pub sigma_x: Option<f32>,

    /// Standard deviation of the 2D gaussian distribution in vertical
    /// direction. Only used when `kind` is `gaussian2d`.
    #[arg(long = "sy", required_if_eq("kind", "gaussian2d"))]
    pub sigma_y: Option<f32>,

    /// Mean of 2D gaussian distribution on horizontal axis. Only used when
    /// `kind` is `gaussian2d`.
    #[arg(long = "mx", required_if_eq("kind", "gaussian2d"))]
    pub mean_x: Option<f32>,

    /// Mean of 2D gaussian distribution on vertical axis. Only used when
    /// `kind` is `gaussian2d`.
    #[arg(long = "my", required_if_eq("kind", "gaussian2d"))]
    pub mean_y: Option<f32>,

    /// Random method to use. Only used when `kind` is `random`.
    #[arg(ignore_case = true, long = "method", required_if_eq("kind", "random"))]
    pub method: Option<RandomGenMethod>,

    /// Number of initial seed points for Worley noise. Should be a power of 2.
    #[arg(long = "num-seeds", required_if_eq_all([("kind", "random"), ("method", "worley-noise")]))]
    pub num_seeds: Option<u32>,

    /// Data format for the surface generation output.
    #[arg(short, long, default_value_t = FileEncoding::Binary)]
    pub encoding: FileEncoding,

    /// Data compression for the surface generation output.
    #[arg(short, long, default_value_t = CompressionScheme::None)]
    pub compression: CompressionScheme,

    /// Path to the file where to save the generated micro-geometry profile.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Generates a micro-surface.
pub fn generate(opts: GenerateOptions, config: Config) -> Result<(), VgonioError> {
    let (res_x, res_y) = (opts.res[0], opts.res[1]);
    let (du, dv) = (opts.spacing[0], opts.spacing[1]);
    println!(
        "  {BRIGHT_YELLOW}>{RESET} Generating surface with resolution {}x{}...",
        res_x, res_y
    );

    let surf = match opts.kind {
        SurfGenKind::Random => match opts.method.unwrap() {
            RandomGenMethod::WhiteNoise => {
                println!("    {BRIGHT_YELLOW}>{RESET} Generating surface from white noise...");
                MicroSurface::from_white_noise(
                    res_y as usize,
                    res_x as usize,
                    du,
                    dv,
                    opts.max_height,
                    LengthUnit::UM,
                )
            }
            RandomGenMethod::WorleyNoise => {
                println!("    {BRIGHT_YELLOW}>{RESET} Generating surface from Worley noise...");
                MicroSurface::from_worley_noise(
                    res_y as usize,
                    res_x as usize,
                    du,
                    dv,
                    opts.max_height,
                    opts.num_seeds.unwrap() as usize,
                    LengthUnit::UM,
                )
            }
            _ => {
                todo!()
            }
        },
        SurfGenKind::Gaussian2D => {
            println!(
                "  {BRIGHT_YELLOW}>{RESET} Generating surface from 2D gaussian distribution..."
            );
            let (sigma_x, sigma_y) = (opts.sigma_x.unwrap(), opts.sigma_y.unwrap());
            let (mean_x, mean_y) = (opts.mean_x.unwrap(), opts.mean_y.unwrap());
            let amp = opts.amplitude.unwrap();
            MicroSurface::new_by(
                res_y as usize,
                res_x as usize,
                du,
                dv,
                LengthUnit::UM,
                |r, c| {
                    let x = ((c as f32 / res_x as f32) * 2.0 - 1.0) * sigma_x * 4.0;
                    let y = ((r as f32 / res_y as f32) * 2.0 - 1.0) * sigma_y * 4.0;
                    amp * (-(x - mean_x) * (x - mean_x) / (2.0 * sigma_x * sigma_x)
                        - (y - mean_y) * (y - mean_y) / (2.0 * sigma_y * sigma_y))
                        .exp()
                },
            )
        }
    };

    let filename = if opts.kind == SurfGenKind::Random {
        format!(
            "msurf_{:?}_{:?}_{}.vgms",
            opts.kind,
            opts.method.unwrap(),
            vgcore::utils::iso_timestamp_short(),
        )
    } else {
        format!(
            "msurf_{:?}_{}.vgms",
            opts.kind,
            vgcore::utils::iso_timestamp_short(),
        )
    };

    let path = config
        .resolve_output_dir(opts.output.as_deref())?
        .join(filename);

    println!("    {BRIGHT_CYAN}✓{RESET} Surface generated");
    println!(
        "    {BRIGHT_YELLOW}>{RESET} Saving to \"{}\"",
        path.display()
    );
    surf.write_to_file(&path, opts.encoding, opts.compression)
}
