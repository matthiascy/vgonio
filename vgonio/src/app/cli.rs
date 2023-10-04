use std::path::PathBuf;
use vgcore::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
};
use vgsurf::MicroSurface;

use crate::{
    app::{
        args::SubCommand,
        cache::{Handle, InnerCache},
        Config,
    },
    measure::{data::MeasurementData, params::MeasurementKind},
};

pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
pub const BRIGHT_RED: &str = "\u{001b}[31m";
pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
pub const RESET: &str = "\u{001b}[0m";

mod cmd_convert;

#[cfg(feature = "surf-gen")]
mod cmd_generate;
mod cmd_info;
mod cmd_measure;

use crate::app::args::OutputFormat;
pub use cmd_convert::{ConvertKind, ConvertOptions};
#[cfg(feature = "surf-gen")]
pub use cmd_generate::GenerateOptions;

/// Entry point of vgonio CLI.
pub fn run(cmd: SubCommand, config: Config) -> Result<(), VgonioError> {
    match cmd {
        SubCommand::Measure(opts) => cmd_measure::measure(opts, config),
        SubCommand::PrintInfo(opts) => cmd_info::print_info(opts, config),

        #[cfg(feature = "surf-gen")]
        SubCommand::Generate(opts) => cmd_generate::generate(opts, config),

        SubCommand::Convert(opts) => cmd_convert::convert(opts, config),
    }
}

/// Writes the measured data to a file.
fn write_measured_data_to_file(
    data: &[MeasurementData],
    surfaces: &[Handle<MicroSurface>],
    cache: &InnerCache,
    config: &Config,
    _format: OutputFormat,
    encoding: FileEncoding,
    compression: CompressionScheme,
    output: &Option<PathBuf>,
) -> Result<(), VgonioError> {
    let output_dir = config.resolve_output_dir(output)?;
    println!("    {BRIGHT_YELLOW}>{RESET} Saving measurement data...");
    // TODO: Add support for other formats.
    for (measurement, surface) in data.iter().zip(surfaces.iter()) {
        let date_string = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
        let filename = match measurement.kind() {
            MeasurementKind::Adf => {
                format!(
                    "ndf_{}_{}.vgmo",
                    cache
                        .get_micro_surface_filepath(*surface)
                        .unwrap()
                        .file_stem()
                        .unwrap()
                        .to_ascii_lowercase()
                        .to_str()
                        .unwrap(),
                    date_string
                )
            }
            MeasurementKind::Msf => {
                format!(
                    "microfacet-masking-shadowing-{}.vgmo",
                    cache
                        .get_micro_surface_filepath(*surface)
                        .unwrap()
                        .file_stem()
                        .unwrap()
                        .to_ascii_lowercase()
                        .to_str()
                        .unwrap()
                )
            }
            MeasurementKind::Bsdf => {
                format!(
                    "bsdf-{}.vgmo",
                    cache
                        .get_micro_surface_filepath(*surface)
                        .unwrap()
                        .file_stem()
                        .unwrap()
                        .to_ascii_lowercase()
                        .to_str()
                        .unwrap()
                )
            }
        };
        let filepath = output_dir.join(filename);
        println!(
            "      {BRIGHT_CYAN}-{RESET} Saving to \"{}\"",
            filepath.display()
        );
        measurement
            .write_to_file(&filepath, encoding, compression)
            .unwrap_or_else(|err| {
                eprintln!(
                    "        {BRIGHT_RED}!{RESET} Failed to save to \"{}\": {}",
                    filepath.display(),
                    err
                );
            });
        println!(
            "      {BRIGHT_CYAN}✓{RESET} Successfully saved to \"{}\"",
            output_dir.display()
        );
    }
    Ok(())
}
