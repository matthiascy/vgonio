//! Reading and writing of surface files.

use byteorder::{LittleEndian, ReadBytesExt};
use std::{
    io::{BufRead, BufReader},
    path::Path,
};
#[cfg(feature = "surf-obj")]
use vgcore::math::Axis;
use vgcore::{
    error::VgonioError,
    io::{FileEncoding, ReadFileError, ReadFileErrorKind},
};

use crate::MicroSurface;
use vgcore::units::LengthUnit;

/// Micro-surface file format.
pub mod vgms {
    use super::*;
    use std::io::{BufWriter, Read, Seek, Write};
    use vgcore::{
        io::{Header, HeaderExt, ReadFileErrorKind, VgonioFileVariant, WriteFileErrorKind},
        units::LengthUnit,
        Version,
    };

    /// Header of the VGMS file.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct VgmsHeaderExt {
        /// The unit used for the micro-surface.
        pub unit: LengthUnit,
        /// Number of rows of the micro-surface.
        pub rows: u32,
        /// Number of columns of the micro-surface.
        pub cols: u32,
        /// The horizontal spacing between samples.
        pub du: f32,
        /// The vertical spacing between samples.
        pub dv: f32,
    }

    impl HeaderExt for VgmsHeaderExt {
        const MAGIC: &'static [u8; 4] = b"VGMS";

        fn variant() -> VgonioFileVariant { VgonioFileVariant::Vgms }

        fn write<W: Write>(
            &self,
            version: Version,
            writer: &mut BufWriter<W>,
        ) -> std::io::Result<()> {
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    writer.write_all(&(self.unit as u32).to_le_bytes())?;
                    writer.write_all(&self.cols.to_le_bytes())?;
                    writer.write_all(&self.rows.to_le_bytes())?;
                    writer.write_all(&self.du.to_le_bytes())?;
                    writer.write_all(&self.dv.to_le_bytes())?;
                }
                _ => {
                    log::error!("Unsupported VGMS version: {}", version.as_string());
                }
            }
            Ok(())
        }

        fn read<R: Read>(version: Version, reader: &mut BufReader<R>) -> std::io::Result<Self> {
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    let mut buf = [0u8; 4];
                    let unit = {
                        reader.read_exact(&mut buf)?;
                        LengthUnit::from(u32::from_le_bytes(buf) as u8)
                    };

                    let cols = {
                        reader.read_exact(&mut buf)?;
                        u32::from_le_bytes(buf)
                    };

                    let rows = {
                        reader.read_exact(&mut buf)?;
                        u32::from_le_bytes(buf)
                    };

                    let du = {
                        reader.read_exact(&mut buf)?;
                        f32::from_le_bytes(buf)
                    };

                    let dv = {
                        reader.read_exact(&mut buf)?;
                        f32::from_le_bytes(buf)
                    };
                    Ok(Self {
                        rows,
                        cols,
                        du,
                        dv,
                        unit,
                    })
                }
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Unsupported VGMS[VgmsHeaderExt] version: {}",
                        version.as_string()
                    ),
                )),
            }
        }
    }

    impl VgmsHeaderExt {
        /// Returns the number of samples in the file.
        pub const fn sample_count(&self) -> u32 { self.rows * self.cols }
    }

    /// Reads the VGMS file from the given reader.
    pub fn read<R: Read + Seek>(
        reader: &mut BufReader<R>,
    ) -> Result<(Header<VgmsHeaderExt>, Vec<f32>), ReadFileErrorKind> {
        let header = Header::<VgmsHeaderExt>::read(reader)?;
        log::debug!("Reading VGMS file of length: {}", header.meta.length);
        // TODO: file length
        // TODO: match header.meta.sample_size
        let samples = vgcore::io::read_f32_data_samples(
            reader,
            header.extra.sample_count() as usize,
            header.meta.encoding,
            header.meta.compression,
        )?;
        Ok((header, samples))
    }

    /// Writes the VGMS file to the given writer.
    pub fn write<W: Write + Seek>(
        writer: &mut BufWriter<W>,
        header: Header<VgmsHeaderExt>,
        samples: &[f32],
    ) -> Result<(), WriteFileErrorKind> {
        let init_size = writer.stream_len().unwrap();
        header.write(writer)?;

        // TODO: match header.meta.sample_size
        match header.meta.encoding {
            FileEncoding::Ascii => {
                vgcore::io::write_data_samples_ascii(writer, samples, header.extra.cols)
            }
            FileEncoding::Binary => {
                vgcore::io::write_f32_data_samples_binary(writer, header.meta.compression, samples)
            }
        }
        .map_err(WriteFileErrorKind::Write)?;

        let length = writer.stream_len().unwrap() - init_size;
        writer.seek(std::io::SeekFrom::Start(
            Header::<VgmsHeaderExt>::length_pos() as u64,
        ))?;
        writer.write_all(&(length as u32).to_le_bytes())?;

        log::debug!("Wrote {} bytes of data to VGMS file", length);
        Ok(())
    }
}

/// Read micro-surface height field following the convention specified in
/// the paper:
///
/// [`Predicting Appearance from Measured Microgeometry of Metal Surfaces. Zhao Dong, Bruce Walter, Steve Marschner, and Donald P. Greenberg. 2016.`](https://dl.acm.org/doi/10.1145/2815618)
///
/// All the data are stored as 2D matrices in a simple ascii format (or
/// equivalently as single channel floating point images).  The first line
/// gives the dimensions of the image and then the remaining lines give the
/// image data, one scanline per text line (from left to right and top to
/// bottom).  In our convention the x dimension increases along a scanline
/// and the y dimension increases with each successive scanline.  An example
/// file with small 2x3 checkerboard with a white square in the upper left
/// is:
///
/// AsciiMatrix 3 2
/// 1.0 0.1 1.0
/// 0.1 1.0 0.1
///
/// Unit used during the measurement is micrometre.
pub fn read_ascii_dong2015<R: BufRead>(
    reader: &mut R,
    filepath: &Path,
) -> Result<MicroSurface, VgonioError> {
    let mut buf = [0_u8; 4];
    reader.read_exact(&mut buf).map_err(|err| {
        VgonioError::new("Failed to read magic number from file", Some(Box::new(err)))
    })?;
    let magic_number = std::str::from_utf8(&buf)
        .map_err(|err| VgonioError::from_utf8_error(err, "Invalid magic number in file"))?;
    if magic_number != "Asci" {
        return Err(VgonioError::new(
            format!(
                "Invalid magic number \"{}\" for file: {}",
                magic_number,
                filepath.display()
            ),
            None,
        ));
    }

    let mut reader = BufReader::new(reader);

    let mut line = String::new();
    reader.read_line(&mut line).map_err(|err| {
        VgonioError::from_io_error(
            err,
            format!("Failed to read lines from file {}", filepath.display()),
        )
    })?;
    let (cols, rows, du, dv) = {
        let first_line = line.trim().split_ascii_whitespace().collect::<Vec<_>>();

        let cols = first_line[1].parse::<usize>().unwrap();
        let rows = first_line[2].parse::<usize>().unwrap();

        if first_line.len() == 3 {
            (cols, rows, 0.11, 0.11)
        } else if first_line.len() == 5 {
            let du = first_line[3].parse::<f32>().unwrap();
            let dv = first_line[4].parse::<f32>().unwrap();
            (cols, rows, du, dv)
        } else {
            panic!("Invalid first line: {line:?}");
        }
    };

    let mut samples = vec![0.0; rows * cols];

    vgcore::io::read_ascii_samples(reader, rows * cols, &mut samples).map_err(|err| {
        VgonioError::new(
            format!("Failed to read samples from file {}", filepath.display()),
            Some(Box::new(ReadFileError::from_parse_error(filepath, err))),
        )
    })?;

    Ok(MicroSurface::from_samples(
        rows,
        cols,
        (du, dv),
        LengthUnit::UM,
        samples,
        filepath
            .file_name()
            .and_then(|name| name.to_str().map(|name| name.to_owned())),
        Some(filepath.into()),
    ))
}

/// Read micro-surface height field issued from µsurf confocal microscope.
pub fn read_ascii_usurf<R: BufRead>(
    reader: &mut R,
    filepath: &Path,
) -> Result<MicroSurface, VgonioError> {
    let mut line = String::new();

    loop {
        reader.read_line(&mut line).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!("Failed to read lines from file {}", filepath.display()),
            )
        })?;
        if !line.is_empty() {
            break;
        }
    }

    let magic_number = line.trim();

    if line.trim() != "DATA" {
        return Err(VgonioError::new(
            format!(
                "Invalid magic number \"{}\" for file: {}",
                magic_number,
                filepath.display()
            ),
            None,
        ));
    }

    line.clear();

    // Read horizontal coordinates
    reader.read_line(&mut line).map_err(|err| {
        VgonioError::from_io_error(
            err,
            format!("Failed to read lines from file {}", filepath.display()),
        )
    })?;
    let x_coords: Vec<f32> = line
        .trim()
        .split_ascii_whitespace()
        .map(|x_coord| {
            x_coord
                .parse::<f32>()
                .unwrap_or_else(|_| panic!("Read f32 error! {}", x_coord))
        })
        .collect();

    let (y_coords, values): (Vec<f32>, Vec<Vec<f32>>) = reader
        .lines()
        .map(|line| {
            let mut values = read_line_ascii_usurf(&line.unwrap());
            let head = values.remove(0);
            (head, values)
        })
        .unzip();

    // Assume that the spacing between two consecutive coordinates is uniform.
    let du = x_coords[1] - x_coords[0];
    let dv = y_coords[1] - y_coords[0];
    let samples: Vec<f32> = values.into_iter().flatten().collect();

    Ok(MicroSurface::from_samples(
        y_coords.len(),
        x_coords.len(),
        (du, dv),
        LengthUnit::UM,
        samples,
        filepath
            .file_name()
            .and_then(|name| name.to_str().map(|name| name.to_owned())),
        Some(filepath.into()),
    ))
}

/// Read a line of usurf file. Height values are separated by tab character.
/// Consecutive tabs signifies that the height value at this point is
/// missing.
fn read_line_ascii_usurf(line: &str) -> Vec<f32> {
    assert!(line.is_ascii());
    line.chars()
        .enumerate()
        .filter_map(|(index, byte)| if byte == '\t' { Some(index) } else { None }) // find tab positions
        .scan((0, false), |(last, last_word_is_tab), curr| {
            // cut string into pieces: floating points string and tab character
            if *last != curr - 1 {
                let val_str = if *last == 0 {
                    &line[*last..curr]
                } else {
                    &line[(*last + 1)..curr]
                };
                *last = curr;
                *last_word_is_tab = false;
                Some(val_str)
            } else {
                *last = curr;
                *last_word_is_tab = true;
                if *last_word_is_tab {
                    if curr != line.len() - 2 {
                        Some("\t")
                    } else {
                        Some("")
                    }
                } else {
                    Some("")
                }
            }
        })
        .filter_map(|s| {
            // parse float string into floating point value
            if s.is_empty() {
                None
            } else if s == "\t" {
                Some(f32::NAN)
            } else {
                Some(s.parse::<f32>().unwrap())
            }
        })
        .collect()
}

/// Read micro-surface height field from OmniSurf3D data file.
/// The file format is described in the following link:
/// https://digitalmetrology.com/omnisurf3d-file-format/
pub fn read_omni_surf_3d<R: BufRead>(
    reader: &mut R,
    filepath: &Path,
) -> Result<MicroSurface, VgonioError> {
    let mut file_type = [0u8; 10];
    reader.read_exact(&mut file_type).map_err(|err| {
        VgonioError::from_io_error(
            err,
            format!(
                "Invalid magic number for OmniSurf3D file {}",
                filepath.display()
            ),
        )
    })?;
    if &file_type != b"OmniSurf3D" {
        return Err(VgonioError::new(
            format!(
                "Invalid file type \"{:?}\" for file: {}",
                std::str::from_utf8(&file_type).unwrap(),
                filepath.display()
            ),
            None,
        ));
    }
    let (ver_major, ver_minor) = {
        let major = reader.read_i32::<LittleEndian>().map_err(|err| {
            VgonioError::from_io_error(err, "Failed to read major version number!")
        })?;
        let minor = reader.read_i32::<LittleEndian>().map_err(|err| {
            VgonioError::from_io_error(err, "Failed to read minor version number!")
        })?;
        (major, minor)
    };

    if ver_major < 1 {
        return Err(VgonioError::new(
            format!(
                "Invalid version number \"{}.{}\" for file: {}",
                ver_major,
                ver_minor,
                filepath.display()
            ),
            None,
        ));
    }

    let ident_str_len = {
        let mut ident_str_len = [0u8; 4];
        reader.read_exact(&mut ident_str_len).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!(
                    "Failed to read ident string length from file {}",
                    filepath.display()
                ),
            )
        })?;
        i32::from_le_bytes(ident_str_len)
    };

    let _ident_str = {
        let mut ident_str = vec![0u8; ident_str_len as usize];
        reader.read_exact(&mut ident_str).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!(
                    "Failed to read ident string from file {}",
                    filepath.display()
                ),
            )
        })?;
        String::from_utf8_lossy(ident_str.trim_ascii()).into_owned()
    };
    // Check for availability of date and time (later than version 1.01)
    let _date = if ver_major >= 0 && ver_minor >= 1 {
        let date_str_len = reader.read_i32::<LittleEndian>().unwrap();
        let mut date_str = vec![0u8; date_str_len as usize];
        reader.read_exact(&mut date_str).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!(
                    "Failed to read date string from file {}",
                    filepath.display()
                ),
            )
        })?;
        String::from_utf8_lossy(date_str.trim_ascii()).into_owned()
    } else {
        String::from("unknown")
    };
    // Read the number of rows and columns
    let cols = reader.read_i32::<LittleEndian>().map_err(|err| {
        VgonioError::from_io_error(
            err,
            format!(
                "Failed to read number of columns from file {}",
                filepath.display()
            ),
        )
    })?;
    let rows = reader.read_i32::<LittleEndian>().map_err(|err| {
        VgonioError::from_io_error(
            err,
            format!(
                "Failed to read number of rows from file {}",
                filepath.display()
            ),
        )
    })?;
    // Read the horizontal and vertical spacing
    let du = reader.read_f64::<LittleEndian>().map_err(|err| {
        VgonioError::from_io_error(
            err,
            format!(
                "Failed to read horizontal spacing from file {}",
                filepath.display()
            ),
        )
    })?;
    let dv = reader.read_f64::<LittleEndian>().map_err(|err| {
        VgonioError::from_io_error(
            err,
            format!(
                "Failed to read vertical spacing from file {}",
                filepath.display()
            ),
        )
    })?;
    // Read the origin
    let _origin_x = reader.read_f64::<LittleEndian>().map_err(|err| {
        VgonioError::from_io_error(
            err,
            format!("Failed to read origin x from file {}", filepath.display()),
        )
    })?;
    let _origin_y = reader.read_f64::<LittleEndian>().map_err(|err| {
        VgonioError::from_io_error(
            err,
            format!("Failed to read origin y from file {}", filepath.display()),
        )
    })?;
    // Read the surface heights
    let mut samples = vec![0.0; (rows * cols) as usize];

    vgcore::io::read_binary_samples(reader, (rows * cols) as usize, &mut samples).map_err(
        |err| {
            VgonioError::from_read_file_error(
                ReadFileError {
                    path: filepath.to_owned().into_boxed_path(),
                    kind: ReadFileErrorKind::Parse(err),
                },
                "Failed to read OmniSurf3D file.",
            )
        },
    )?;

    // Ignore the rest of the file

    Ok(MicroSurface::from_samples(
        rows as usize,
        cols as usize,
        (du as f32, dv as f32),
        LengthUnit::UM,
        samples,
        filepath
            .file_name()
            .and_then(|name| name.to_str().map(|name| name.to_owned())),
        Some(filepath.into()),
    ))
}

#[cfg(feature = "surf-obj")]
/// Reads a Wavefront OBJ file and returns a micro-surface.
///
/// The OBJ file must contain a single mesh.
///
/// # Arguments
///
/// * `reader` - The reader to read the OBJ file from.
///
/// * `filepath` - The path to the OBJ file.
///
/// * `axis` - The axis along which the surface is oriented.
///
/// * `unit` - The unit used for the surface.
pub fn read_wavefront<R: BufRead>(
    reader: &mut R,
    filepath: &Path,
    axis: Axis,
    unit: LengthUnit,
) -> Result<MicroSurface, VgonioError> {
    let result = tobj::load_obj_buf(reader, &tobj::GPU_LOAD_OPTIONS, |_| {
        Ok((vec![], ahash::AHashMap::new()))
    });
    let models = result
        .map_err(|err| {
            VgonioError::new(
                format!("Failed to read Wavefront file {}", filepath.display()),
                Some(Box::new(err)),
            )
        })?
        .0;
    let mesh = &models[0].mesh;
    let (min_x, max_x, min_y, max_y, min_z, max_z) = mesh.positions.chunks_exact(3).fold(
        (f32::MAX, f32::MIN, f32::MAX, f32::MIN, f32::MAX, f32::MIN),
        |(min_x, max_x, min_y, max_y, min_z, max_z), pos| {
            let x = pos[0];
            let y = pos[1];
            let z = pos[2];
            (
                min_x.min(x),
                max_x.max(x),
                min_y.min(y),
                max_y.max(y),
                min_z.min(z),
                max_z.max(z),
            )
        },
    );
    log::debug!(
        "min_x: {}, max_x: {}, min_y: {}, max_y: {}, min_z: {}, max_z: {}, number of vertices: {}",
        min_x,
        max_x,
        min_y,
        max_y,
        min_z,
        max_z,
        mesh.positions.len() / 3
    );
    let (iu, iv, ih) = match axis {
        Axis::X => (1, 2, 0),
        Axis::Y => (0, 2, 1),
        Axis::Z => (0, 1, 2),
    };
    let p0 = &mesh.positions[0..3];
    let p1 = &mesh.positions[3..6];
    let p2 = &mesh.positions[6..9];
    let du = (p0[iu] - p1[iu]).abs().max((p0[iu] - p2[iu]).abs());
    let dv = (p0[iv] - p1[iv]).abs().max((p0[iv] - p2[iv]).abs());
    let rows = ((max_y - min_y) / dv) as usize + 1;
    let cols = ((max_x - min_x) / du) as usize + 1;
    let heights = mesh
        .positions
        .chunks_exact(3)
        .map(|chunk| chunk[ih])
        .collect::<Vec<_>>();
    log::debug!("du: {}, dv: {}, cols: {}, rows: {}", du, dv, cols, rows);
    Ok(MicroSurface::from_samples(
        rows,
        cols,
        (du, dv),
        unit,
        heights,
        filepath
            .file_name()
            .and_then(|name| name.to_str().map(|name| name.to_owned())),
        Some(filepath.into()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[rustfmt::skip]
    fn test_read_line_ascii_surf0() {
        let lines = [
            "0.00\t12.65\t\t12.63\t\t\t\t12.70\t12.73\t\t\t\t\t\t12.85\t\t\t\n",
            "0.00\t12.65\t\t\t12.63\t\t\t\t\t\t12.70\t12.73\t\t\t\t\t\t\t\t\t12.85\t\t\t\t\n",
            "0.00\t12.65\t\t\t\t12.63\t\t\t\t\t\t\t\t12.70\t12.73\t\t\t\t\t\t\t\t\t\t\t\t12.85\t\t\t\t\t\n",
        ];

        assert_eq!(read_line_ascii_usurf(lines[0]).len(), 16);
        assert_eq!(read_line_ascii_usurf(lines[1]).len(), 23);
        assert_eq!(read_line_ascii_usurf(lines[2]).len(), 30);
    }

    #[test]
    #[rustfmt::skip]
    fn test_read_line_ascii_surf1() {
        let lines = [
            "0.00\t12.65\t\t12.63\t12.70\t12.73\t\t12.85\t\n",
            "0.00\t12.65\t\t12.63\t\t\t\t12.70\t12.73\t\t\t\t\t\t12.85\t\t\t\n",
            "0.00\t12.65\t\t\t12.63\t\t\t\t\t\t12.70\t12.73\t\t\t\t\t\t\t\t\t12.85\t\t\t\t\n",
            "0.00\t12.65\t\t\t\t12.63\t\t\t\t\t\t\t\t12.70\t12.73\t\t\t\t\t\t\t\t\t\t\t\t12.85\t\t\t\t\t\n",
        ];

        fn _read_line(line: &str) -> Vec<&str> {
            let tabs = line
                .chars()
                .enumerate()
                .filter_map(|(index, byte)| if byte == '\t' { Some(index) } else { None })
                .collect::<Vec<usize>>();

            let pieces = tabs
                .iter()
                .scan((0, false), |(last, last_word_is_tab), curr| {
                    // cut string into pieces: floating points string and tab character
                    if *last != curr - 1 {
                        let val_str = if *last == 0 {
                            &line[*last..*curr]
                        } else {
                            &line[(*last + 1)..*curr]
                        };
                        *last = *curr;
                        *last_word_is_tab = false;
                        Some(val_str)
                    } else {
                        *last = *curr;
                        *last_word_is_tab = true;
                        if *last_word_is_tab {
                            if *curr != tabs[tabs.len() - 1] {
                                Some(&"\t")
                            } else {
                                Some(&"")
                            }
                        } else {
                            Some(&"")
                        }
                    }
                })
                .filter(|piece| !piece.is_empty())
                .collect::<Vec<&str>>();
            pieces
        }

        let mut results = vec![];

        for &line in &lines {
            let pieces = _read_line(line);
            println!("pieces: {:?}", pieces);
            results.push(pieces.len());
        }

        assert_eq!(results[0], 8);
        assert_eq!(results[1], 16);
        assert_eq!(results[2], 23);
        assert_eq!(results[3], 30);
    }
}
