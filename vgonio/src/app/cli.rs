use std::path::PathBuf;
use vgcore::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
};
use vgsurf::MicroSurface;

use crate::{
    app::{args::SubCommand, cache::Handle, Config},
    measure::data::MeasurementData,
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

use crate::app::{args::OutputFormat, cache::Cache};
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
pub fn write_measured_data_to_file(
    data: &[MeasurementData],
    surfaces: &[Handle<MicroSurface>],
    cache: &Cache,
    config: &Config,
    format: OutputFormat,
    encoding: FileEncoding,
    compression: CompressionScheme,
    output: &Option<PathBuf>,
) -> Result<(), VgonioError> {
    println!("    {BRIGHT_YELLOW}>{RESET} Saving measurement data...");
    let output_dir = config.resolve_output_dir(output)?;
    for (measurement, surface) in data.iter().zip(surfaces.iter()) {
        let datetime = vgcore::utils::iso_timestamp_short();
        let filepath = cache.read(|cache| {
            let surf_name = cache
                .get_micro_surface_filepath(*surface)
                .unwrap()
                .file_stem()
                .unwrap()
                .to_ascii_lowercase();
            output_dir.join(format!(
                "{}_{}_{}",
                measurement.kind().ascii_str(),
                surf_name.to_str().unwrap(),
                datetime
            ))
        });
        println!(
            "      {BRIGHT_CYAN}-{RESET} Saving to \"{}\"",
            filepath.display()
        );

        match measurement.write_to_file(&filepath, format, encoding, compression) {
            Ok(_) => {
                println!(
                    "      {BRIGHT_CYAN}✓{RESET} Successfully saved to \"{}\"",
                    output_dir.display()
                );
            }
            Err(err) => {
                eprintln!(
                    "        {BRIGHT_RED}!{RESET} Failed to save to \"{}\": {}",
                    filepath.display(),
                    err
                );
            }
        }
    }
    Ok(())
}
