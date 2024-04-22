use crate::{
    app::{cache::Cache, Config},
    plotting::plot_brdf,
};
use base::{error::VgonioError, units::Rads};
use std::path::PathBuf;

/// Kind of plot to generate.
#[derive(clap::ValueEnum, Debug, Clone)]
pub enum PlotKind {
    SampledBrdf,
}

/// Options for the `plot` command.
#[derive(clap::Args, Debug, Clone)]
pub struct PlotOptions {
    #[clap(short, long, help = "Input files to plot.")]
    pub inputs: Vec<PathBuf>,

    #[clap(short, long, help = "The kind of plot to generate.")]
    pub kind: PlotKind,
}

pub fn plot(opts: PlotOptions, config: Config) -> Result<(), VgonioError> {
    let cache = Cache::new(config.cache_dir());
    cache.write(|cache| {
        cache.load_ior_database(&config);
        match opts.kind {
            PlotKind::SampledBrdf => {
                if opts.inputs.len() % 2 != 0 {
                    return Err(VgonioError::new(
                        "The number of input files must be even.",
                        None,
                    ));
                }
                for input in opts.inputs.chunks(2) {
                    let simulated_hdl = cache.load_micro_surface_measurement(&config, &input[0])?;
                    // Measured by Olaf
                    let measured_hdl = cache.load_micro_surface_measurement(&config, &input[1])?;
                    let simulated = cache
                        .get_measurement_data(simulated_hdl)
                        .unwrap()
                        .measured
                        .as_bsdf()
                        .expect("Expected BSDF measured by Vgonio");
                    let olaf = cache
                        .get_measurement_data(measured_hdl)
                        .unwrap()
                        .measured
                        .as_sampled_brdf()
                        .expect("Expected BSDF measured by Olaf");
                    let dense = if std::env::var("DENSE")
                        .ok()
                        .map(|s| s == "1")
                        .unwrap_or(false)
                    {
                        true
                    } else {
                        false
                    };
                    let phi_offset = std::env::var("PHI_OFFSET")
                        .ok()
                        .map(|s| s.parse::<f32>().unwrap())
                        .unwrap_or(0.0)
                        .to_radians();
                    let itrp = simulated.sampled_brdf(&olaf, dense, Rads::new(phi_offset));
                    plot_brdf(&itrp, olaf, dense).unwrap();
                }
                Ok(())
            }
        }
    })
}
