use std::path::PathBuf;
use vgcore::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
};
use vgsurf::MicroSurface;

use crate::{
    app::{
        args::SubCommand,
        cache::{resolve_path, Cache, Handle},
        Config,
    },
    error::RuntimeError,
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
    cache: &Cache,
    config: &Config,
    encoding: FileEncoding,
    compression: CompressionScheme,
    output: &Option<PathBuf>,
) -> Result<(), VgonioError> {
    let output_dir = resolve_output_dir(config, output)?;
    println!("    {BRIGHT_YELLOW}>{RESET} Saving measurement data...");
    for (measurement, surface) in data.iter().zip(surfaces.iter()) {
        let filename = match measurement.kind() {
            MeasurementKind::Adf => {
                format!(
                    "microfacet-area-distribution-{}.vgmo",
                    cache
                        .get_micro_surface_filepath(*surface)
                        .unwrap()
                        .file_stem()
                        .unwrap()
                        .to_ascii_lowercase()
                        .to_str()
                        .unwrap(),
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

/// Returns the output directory in canonical form.
/// If the output directory is not specified, returns config's output directory.
///
/// # Arguments
///
/// * `config` - The configuration of the current Vgonio session.
/// * `output` - The output directory specified by the user.
fn resolve_output_dir(
    config: &Config,
    output_dir: &Option<PathBuf>,
) -> Result<PathBuf, VgonioError> {
    match output_dir {
        Some(dir) => {
            let path = resolve_path(config.cwd(), Some(dir));
            if !path.is_dir() {
                return Err(VgonioError::new(
                    format!("{} is not a directory", path.display()),
                    Some(Box::new(RuntimeError::InvalidOutputDir)),
                ));
            }
            Ok(path)
        }
        None => Ok(config.output_dir().to_path_buf()),
    }
}
