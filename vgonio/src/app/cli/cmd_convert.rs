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

    /// The height axis of the micro-surface profile only when the input is a 3D
    /// mesh file.
    #[arg(long)]
    pub axis: Option<Axis>,
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
    cli::{BRIGHT_CYAN, BRIGHT_RED, BRIGHT_YELLOW, RESET},
    Config,
};
use std::path::PathBuf;
#[cfg(feature = "surf-obj")]
use vgcore::units::LengthUnit;
use vgcore::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
    math::Axis,
};

use vgsurf::MicroSurface;

pub fn convert(opts: ConvertOptions, config: Config) -> Result<(), VgonioError> {
    let output_dir = config.resolve_output_dir(&opts.output)?;
    for input in opts.inputs {
        let resolved = {
            let path = config.resolve_path(&input);
            if path.is_none() {
                continue;
            } else {
                path.unwrap()
            }
        };
        log::debug!("Resolved path: {:?}", resolved);
        let files = if resolved.is_dir() {
            let mut files = Vec::new();
            let dir_entry = std::fs::read_dir(&resolved);
            if let Err(err) = dir_entry {
                eprintln!(
                    "  {BRIGHT_RED}!{RESET} Failed to read directory \"{}\": {}",
                    resolved.display(),
                    err
                );
                continue;
            }
            for entry in dir_entry.unwrap() {
                if let Err(err) = entry {
                    eprintln!(
                        "  {BRIGHT_RED}!{RESET} Failed to read directory \"{}\": {}",
                        resolved.display(),
                        err
                    );
                    continue;
                }
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_file() {
                    files.push(path);
                } else {
                    continue;
                }
            }
            files
        } else {
            vec![resolved.clone()]
        };
        match opts.kind {
            ConvertKind::MicroSurfaceProfile => {
                use rayon::prelude::*;
                let errors = files
                    .par_iter()
                    .filter_map(|filepath| {
                        log::info!("Converting {:?}", filepath);
                        let result: Result<(MicroSurface, String), VgonioError> = {
                            #[cfg(feature = "surf-obj")]
                            let loaded = match filepath.extension() {
                                None => MicroSurface::read_from_file(&filepath, None),
                                Some(ext) => {
                                    if ext == "obj" {
                                        MicroSurface::read_from_wavefront(
                                            &filepath,
                                            opts.axis.unwrap_or(Axis::Z),
                                            LengthUnit::UM,
                                        )
                                    } else {
                                        MicroSurface::read_from_file(&filepath, None)
                                    }
                                }
                            };
                            #[cfg(not(feature = "surf-obj"))]
                            let loaded = MicroSurface::read_from_file(&filepath, None);

                            if let Ok(loaded) = loaded {
                                let (w, h) = if let Some(new_size) = opts.resize.as_ref() {
                                    let (w, h) = (new_size[0] as usize, new_size[1] as usize);
                                    println!(
                                        "  {BRIGHT_YELLOW}>{RESET} Resizing to {}x{}...",
                                        w, h
                                    );
                                    (w, h)
                                } else {
                                    (loaded.cols, loaded.rows)
                                };

                                let (w, h) = if opts.squaring {
                                    let s = w.min(h);
                                    println!(
                                        "  {BRIGHT_YELLOW}>{RESET} Squaring to {}x{}...",
                                        s, s
                                    );
                                    (s, s)
                                } else {
                                    (w, h)
                                };

                                let filename = format!(
                                    "{}_converted.vgms",
                                    loaded.file_stem().unwrap().to_ascii_lowercase().as_str()
                                );
                                Ok((loaded.resize(h, w), filename))
                            } else {
                                Err(loaded.err().unwrap())
                            }
                        };

                        if let Ok((ref profile, ref filename)) = result {
                            println!(
                                "{BRIGHT_YELLOW}>{RESET} Converting {:?} to {:?}...",
                                filepath, output_dir
                            );

                            profile
                                .write_to_file(
                                    &output_dir.join(filename),
                                    opts.encoding,
                                    opts.compression,
                                )
                                .unwrap_or_else(|err| {
                                    eprintln!(
                                        "  {BRIGHT_RED}!{RESET} Failed to save to \"{}\": {}",
                                        resolved.display(),
                                        err
                                    );
                                });
                        }
                        result.err()
                    })
                    .collect::<Vec<_>>();
                for err in errors {
                    eprintln!(
                        "  {BRIGHT_RED}!{RESET} Failed to convert \"{}\": {}",
                        resolved.display(),
                        err
                    )
                }
                println!("{BRIGHT_CYAN}✓{RESET} Done!",);
            }
        }
    }
    Ok(())
}
