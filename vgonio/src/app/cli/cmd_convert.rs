#[derive(clap::Args, Debug)]
#[clap(
    about = "Converts non-vgonio surface files to vgonio files or resizes vgonio surface files."
)]
pub struct ConvertOptions {
    /// Filepaths to files to be converted.
    #[arg(
    short,
    long,
    num_args(1..),
    required = true,
    )]
    pub inputs: Vec<PathBuf>,

    /// Path to store converted files. If not specified, the files will be
    /// stored in the current working directory.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Data encoding for the output file.
    #[arg(
    short,
    long,
    default_value_t = FileEncoding::Binary,
    )]
    pub encoding: FileEncoding,

    /// Data compression for the output file.
    #[arg(
    short,
    long,
    default_value_t = CompressionScheme::None,
    )]
    pub compression: CompressionScheme,

    #[clap(
        short,
        long,
        required = true,
        help = "Type of conversion to perform. If not specified, the\nconversion will be inferred \
                from the file extension."
    )]
    pub kind: ConvertKind,

    #[clap(
        long,
        value_name = "WIDTH HEIGHT",
        num_args(2),
        help = "Resize the micro-surface profile to the given resolution. The \n resolution \
                should be samller than the original."
    )]
    pub resize: Option<Vec<u32>>,

    #[clap(
        long,
        help = "Resize the micro-surface profile to make it square. The\nresolution will be the \
                minimum of the width and height."
    )]
    pub squaring: bool,

    /// Handedness of the coordinate system, which is used to determine the
    /// height axis of the micro-surface profile only when the input is a 3D
    /// mesh file.
    #[arg(long)]
    pub handedness: Option<Handedness>,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertKind {
    #[clap(
        name = "ms",
        help = "Convert a micro-surface profile to .vgms file. Accepts files coming \
                from\n\"Predicting Appearance from Measured Microgeometry of Metal \
                Surfaces\",\nplain text data coming from µsurf confocal microscope system, \
                and\n3D mesh files (*.obj) of the micro-surface."
    )]
    MicroSurfaceProfile,
}

use crate::app::{
    cache::resolve_path,
    cli::{resolve_output_dir, BRIGHT_CYAN, BRIGHT_RED, BRIGHT_YELLOW, RESET},
    Config,
};
use std::path::PathBuf;
use vgcore::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
    math::Handedness,
    units::LengthUnit,
};
use vgsurf::MicroSurface;

pub fn convert(opts: ConvertOptions, config: Config) -> Result<(), VgonioError> {
    log::info!("Converting files...");
    let output_dir = resolve_output_dir(&config, &opts.output)?;
    for input in opts.inputs {
        let resolved = resolve_path(&config.cwd, Some(&input));
        log::info!("Converting {:?}", resolved);
        match opts.kind {
            ConvertKind::MicroSurfaceProfile => {
                let (profile, filename) = {
                    #[cfg(feature = "wavefront")]
                    let loaded = match resolved.extension() {
                        None => MicroSurface::read_from_file(&resolved, None)?,
                        Some(ext) => {
                            if ext == "obj" {
                                MicroSurface::read_from_wavefront(
                                    &resolved,
                                    opts.handedness.unwrap_or(Handedness::RightHandedZUp),
                                    LengthUnit::UM,
                                )?
                            } else {
                                MicroSurface::read_from_file(&resolved, None)?
                            }
                        }
                    };
                    #[cfg(not(feature = "wavefront"))]
                    let loaded = MicroSurface::read_from_file(&resolved, None)?;

                    let (w, h) = if let Some(new_size) = opts.resize.as_ref() {
                        let (w, h) = (new_size[0] as usize, new_size[1] as usize);
                        println!("  {BRIGHT_YELLOW}>{RESET} Resizing to {}x{}...", w, h);
                        (w, h)
                    } else {
                        (loaded.cols, loaded.rows)
                    };

                    let (w, h) = if opts.squaring {
                        let s = w.min(h);
                        println!("  {BRIGHT_YELLOW}>{RESET} Squaring to {}x{}...", s, s);
                        (s, s)
                    } else {
                        (w, h)
                    };

                    let filename = format!(
                        "{}_converted.vgms",
                        resolved
                            .file_stem()
                            .unwrap()
                            .to_ascii_lowercase()
                            .to_str()
                            .unwrap()
                    );
                    (loaded.resize(h, w), filename)
                };
                println!(
                    "{BRIGHT_YELLOW}>{RESET} Converting {:?} to {:?}...",
                    resolved, output_dir
                );

                profile
                    .write_to_file(&output_dir.join(filename), opts.encoding, opts.compression)
                    .unwrap_or_else(|err| {
                        eprintln!(
                            "  {BRIGHT_RED}!{RESET} Failed to save to \"{}\": {}",
                            resolved.display(),
                            err
                        );
                    });
                println!("{BRIGHT_CYAN}✓{RESET} Done!",);
            }
        }
    }
    Ok(())
}
