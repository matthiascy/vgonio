use crate::{app::cache::Cache, measure::params::SurfacePath};
use egui::ahash::HashSet;
use std::path::PathBuf;
use surf::{subdivision::Subdivision, HeightOffset};
use vgonio_core::{config::Config, error::VgonioError};

#[derive(Debug, Clone, clap::Args)]
pub struct SubdivideOptions {
    /// The input files to subdivide.
    #[clap(short, long, num_args = 1.., required = true, value_delimiter = ' ', help = "Surfaces to subdivide.")]
    pub inputs: Vec<PathBuf>,

    /// The subdivision level.
    #[clap(short, long, num_args = 1.., required = true, value_delimiter = ',', help = "Subdivision levels.")]
    pub levels: Vec<u32>,

    /// Curved subdivision.
    #[clap(short, long, help = "Use curved subdivision.")]
    pub curved: bool,

    /// Wiggly subdivision.
    #[clap(short, long, help = "Use wiggly subdivision.")]
    pub wiggly: bool,

    /// Height offset percentage when using wiggly subdivision.
    #[clap(
        short,
        long,
        num_args = 1..,
        value_delimiter = ' ',
        required_if_eq("wiggly", "true"),
        help = "Height offset percentage."
    )]
    pub offsets: Vec<u32>,
}

pub fn subdivide(opts: SubdivideOptions, config: Config) -> Result<(), VgonioError> {
    if !opts.curved && !opts.wiggly {
        return Err(VgonioError::new(
            "Either curved or wiggly subdivision must be enabled.",
            None,
        ));
    }
    let offsets = opts.offsets.into_iter().collect::<HashSet<u32>>();
    let cache = Cache::new(config.cache_dir());
    let surfaces = opts
        .inputs
        .iter()
        .map(SurfacePath::from)
        .collect::<Box<_>>();
    let subdivisions: Box<_> = opts
        .levels
        .iter()
        .flat_map(|lvl| {
            if opts.curved {
                return vec![Subdivision::Curved(*lvl)];
            }
            if opts.wiggly {
                return offsets
                    .iter()
                    .map(|offset| Subdivision::Wiggly {
                        level: *lvl,
                        offset: *offset,
                    })
                    .collect();
            }
            unreachable!();
        })
        .collect();

    println!("{:?}", subdivisions);

    let loaded = cache
        .write(|cache| cache.load_micro_surfaces(&config, &surfaces, config.user.triangulation))?;

    cache.read(|cache| {
        let info = loaded
            .into_iter()
            .map(|hdl| {
                let surface = cache.get_micro_surface(hdl).unwrap();

                subdivisions
                    .iter()
                    .clone()
                    .map(|subdiv| {
                        let mesh = surface.as_micro_surface_mesh(
                            HeightOffset::Grounded,
                            config.user.triangulation,
                            Some(*subdiv),
                        );
                        (*subdiv, mesh.facet_total_area)
                    })
                    .collect::<Box<_>>()
            })
            .collect::<Box<_>>();

        opts.inputs
            .iter()
            .zip(info.iter())
            .for_each(|(input, info)| {
                print!("{}: ", input.display());
                println!("{:?}", info);
            });
    });

    Ok(())
}
