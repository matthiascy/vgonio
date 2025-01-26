use crate::app::{cli::ansi, Config};
#[cfg(feature = "surf-obj")]
use vgcore::units::LengthUnit;
use vgcore::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
    math::Axis,
};
use std::path::PathBuf;
use surf::{HeightOffset, MicroSurface};

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

    #[clap(short, long, required = true, help = "The kind of the source file.")]
    pub src_kind: ConvertKind,

    #[clap(short, long, required = true, help = "The kind of the output file.")]
    pub dst_kind: ConvertKind,

    #[clap(long, help = "The offset to apply to the height.")]
    pub offset: Option<HeightOffset>,

    #[clap(
        long,
        value_name = "WIDTH HEIGHT",
        num_args(2),
        help = "Resize the micro-surface profile to the given resolution. The \n resolution \
                should be smaller than the original."
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
        help = "Any micro-surface profile file. Accepts files coming from \
                VGonio(vgms),\"Predicting Appearance from Measured Microgeometry of Metal \
                Surfaces\", plain text data coming from µsurf confocal microscope system, and 3D \
                mesh files (*.obj) of the micro-surface."
    )]
    MicroSurface,

    #[clap(
        name = "ms-vgms",
        help = "Micro-surface profile file in VGonio format."
    )]
    MicroSurfaceVgms,

    #[clap(
        name = "ms-obj",
        help = "Micro-surface profile file in 3D mesh format."
    )]
    MicroSurfaceObj,

    #[clap(
        name = "ms-cms",
        help = "Micro-surface profile file in plain text format coming from µsurf confocal \
                microscope system."
    )]
    MicroSurfaceCms,

    #[clap(name = "exr", help = "EXR image file.")]
    Exr,
}

pub fn convert(opts: ConvertOptions, config: Config) -> Result<(), VgonioError> {
    let output_dir = config.resolve_output_dir(opts.output.as_deref())?;
    for input in opts.inputs {
        let resolved = {
            let path = config.resolve_path(&input);
            if path.is_none() {
                continue;
            }
            path.unwrap()
        };
        log::debug!("Resolved path: {:?}", resolved);
        let files = if resolved.is_dir() {
            let mut files = Vec::new();
            let dir_entry = std::fs::read_dir(&resolved);
            if let Err(err) = dir_entry {
                eprintln!(
                    "  {}!{} Failed to read directory \"{}\": {}",
                    ansi::BRIGHT_RED,
                    ansi::RESET,
                    resolved.display(),
                    err
                );
                continue;
            }
            for entry in dir_entry.unwrap() {
                if let Err(err) = entry {
                    eprintln!(
                        "  {}!{} Failed to read directory \"{}\": {}",
                        ansi::BRIGHT_RED,
                        ansi::RESET,
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
                        },
                    };
                    #[cfg(not(feature = "surf-obj"))]
                    let loaded = MicroSurface::read_from_file(filepath, None);

                    if let Ok(loaded) = loaded {
                        let (w, h) = if let Some(new_size) = opts.resize.as_ref() {
                            let (w, h) = (new_size[0] as usize, new_size[1] as usize);
                            println!(
                                "  {}>{} Resizing to {}x{}...",
                                ansi::BRIGHT_YELLOW,
                                ansi::RESET,
                                w,
                                h
                            );
                            (w, h)
                        } else {
                            (loaded.cols, loaded.rows)
                        };

                        let (w, h) = if opts.squaring {
                            let s = w.min(h);
                            println!(
                                "  {}>{} Squaring to {}x{}...",
                                ansi::BRIGHT_YELLOW,
                                ansi::RESET,
                                s,
                                s
                            );
                            (s, s)
                        } else {
                            (w, h)
                        };

                        let filename = match opts.dst_kind {
                            ConvertKind::Exr => {
                                format!(
                                    "{}_converted.exr",
                                    loaded.file_stem().unwrap().to_ascii_lowercase().as_str()
                                )
                            },
                            _ => {
                                format!(
                                    "{}_converted.vgms",
                                    loaded.file_stem().unwrap().to_ascii_lowercase().as_str()
                                )
                            },
                        };
                        Ok((loaded.resize(h, w), filename))
                    } else {
                        Err(loaded.err().unwrap())
                    }
                };

                if let Ok((ref profile, ref filename)) = result {
                    println!(
                        "{}>{} Converting {:?} to {:?}...",
                        ansi::BRIGHT_YELLOW,
                        ansi::RESET,
                        filepath,
                        output_dir
                    );

                    if opts.dst_kind == ConvertKind::Exr {
                        profile
                            .write_to_exr(
                                &output_dir.join(filename),
                                opts.offset.unwrap_or(HeightOffset::None),
                            )
                            .unwrap_or_else(|err| {
                                eprintln!(
                                    "  {}!{} Failed to save to \"{}\": {}",
                                    ansi::BRIGHT_RED,
                                    ansi::RESET,
                                    resolved.display(),
                                    err
                                );
                            });
                    } else {
                        profile
                            .write_to_file(
                                &output_dir.join(filename),
                                opts.encoding,
                                opts.compression,
                            )
                            .unwrap_or_else(|err| {
                                eprintln!(
                                    "  {}!{} Failed to save to \"{}\": {}",
                                    ansi::BRIGHT_RED,
                                    ansi::RESET,
                                    resolved.display(),
                                    err
                                );
                            });
                    }
                }
                result.err()
            })
            .collect::<Vec<_>>();
        for err in errors {
            eprintln!(
                "  {}!{} Failed to convert \"{}\": {}",
                ansi::BRIGHT_RED,
                ansi::RESET,
                resolved.display(),
                err
            )
        }
        println!("{}✓{} Done!", ansi::BRIGHT_CYAN, ansi::RESET);
    }
    Ok(())
}
