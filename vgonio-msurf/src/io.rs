use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    fmt::Display,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};
use vgonio_core::{
    error::VgonioError,
    io::{
        CompressionScheme, FileEncoding, ParseError, ParseErrorKind, ReadFileError,
        ReadFileErrorKind,
    },
};

/// Write the data samples to the given writer.
fn write_f32_data_samples<W: Write>(
    writer: &mut BufWriter<W>,
    encoding: FileEncoding,
    compression: CompressionScheme,
    samples: &[f32],
    cols: u32,
) -> Result<(), std::io::Error> {
    let mut zlib_encoder;
    let mut gzip_encoder;

    let mut encoder: Box<&mut dyn Write> = match compression {
        CompressionScheme::None => Box::new(writer),
        CompressionScheme::Zlib => {
            zlib_encoder = flate2::write::ZlibEncoder::new(writer, flate2::Compression::default());
            Box::new(&mut zlib_encoder)
        }
        CompressionScheme::Gzip => {
            gzip_encoder = flate2::write::GzEncoder::new(writer, flate2::Compression::default());
            Box::new(&mut gzip_encoder)
        }
        _ => Box::new(writer),
    };

    match encoding {
        FileEncoding::Ascii => {
            for (i, s) in samples.iter().enumerate() {
                let val = if i as u32 % cols == cols - 1 {
                    format!("{s}\n")
                } else {
                    format!("{s} ")
                };
                encoder.write_all(val.as_bytes())?;
            }
        }
        FileEncoding::Binary => {
            for s in samples.iter() {
                encoder.write_all(&s.to_le_bytes())?;
            }
        }
    }

    Ok(())
}

/// Read the data samples from the given reader.
fn read_f32_data_samples<R: Read>(
    reader: &mut BufReader<R>,
    count: usize,
    encoding: FileEncoding,
    compression: CompressionScheme,
) -> Result<Vec<f32>, ParseError> {
    let mut zlib_decoder;
    let mut gzip_decoder;

    let decoder: Box<&mut dyn Read> = match compression {
        CompressionScheme::None => Box::new(reader),
        CompressionScheme::Zlib => {
            zlib_decoder = flate2::bufread::ZlibDecoder::new(reader);
            Box::new(&mut zlib_decoder)
        }
        CompressionScheme::Gzip => {
            gzip_decoder = flate2::bufread::GzDecoder::new(reader);
            Box::new(&mut gzip_decoder)
        }
        _ => Box::new(reader),
    };

    let mut samples = vec![0.0; count];
    match encoding {
        FileEncoding::Ascii => read_ascii_samples(decoder, count, &mut samples)?,
        FileEncoding::Binary => read_binary_samples(decoder, count, &mut samples)?,
    }

    Ok(samples)
}

/// Reads sample values separated by whitespace line by line.
fn read_ascii_samples<R>(reader: R, count: usize, samples: &mut [f32]) -> Result<(), ParseError>
where
    R: Read,
{
    debug_assert!(
        samples.len() >= count,
        "Samples' container must be large enough to hold all samples"
    );
    let reader = BufReader::new(reader);

    let mut loaded = 0;
    for (n, line) in reader.lines().enumerate() {
        let line = line.map_err(|_| ParseError {
            line: n as u32,
            position: 0,
            kind: ParseErrorKind::InvalidLine,
            encoding: FileEncoding::Ascii,
        })?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        for (i, x) in line.split_ascii_whitespace().enumerate() {
            let x = x.parse().map_err(|_| ParseError {
                line: n as u32,
                position: i as u32,
                kind: ParseErrorKind::ParseFloat,
                encoding: FileEncoding::Ascii,
            })?;
            samples[loaded] = x;
            loaded += 1;
        }
    }
    (count == loaded).then_some(()).ok_or(ParseError {
        line: u32::MAX,
        position: u32::MAX,
        kind: ParseErrorKind::NotEnoughData,
        encoding: FileEncoding::Ascii,
    })
}

/// Reads sample values as binary data.
fn read_binary_samples<R>(
    mut reader: R,
    count: usize,
    samples: &mut [f32],
) -> Result<(), ParseError>
where
    R: Read,
{
    debug_assert!(
        samples.len() >= count,
        "Samples' container must be large enough to hold all samples"
    );

    use byteorder::{LittleEndian, ReadBytesExt};

    let results = samples
        .iter_mut()
        .enumerate()
        .map(|(i, sample)| {
            let parsed = reader.read_f32::<LittleEndian>().map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    ParseError {
                        line: u32::MAX,
                        position: i as u32 * 4,
                        kind: ParseErrorKind::NotEnoughData,
                        encoding: FileEncoding::Binary,
                    }
                } else {
                    ParseError {
                        line: u32::MAX,
                        position: i as u32 * 4,
                        kind: ParseErrorKind::ParseFloat,
                        encoding: FileEncoding::Binary,
                    }
                }
            });
            match parsed {
                Ok(parsed) => {
                    *sample = parsed;
                    Ok(())
                }
                Err(err) => Err(err),
            }
        })
        .collect::<Result<Vec<_>, ParseError>>();

    results?;
    Ok(())
}

use crate::MicroSurface;
use vgonio_core::units::LengthUnit;

/// Micro-surface file format.
pub mod vgms {
    use super::*;
    use std::io::{BufWriter, Read, Write};
    use vgonio_core::{
        io::{ReadFileErrorKind, WriteFileErrorKind},
        units::LengthUnit,
    };

    /// Header of the VGMS file.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Header {
        pub rows: u32,
        pub cols: u32,
        pub du: f32,
        pub dv: f32,
        pub unit: LengthUnit,
        pub sample_data_size: u8,
        pub encoding: FileEncoding,
        pub compression: CompressionScheme,
    }

    impl Header {
        /// The magic number of a VGMS file.
        pub const MAGIC: &'static [u8] = b"VGMS";

        /// Returns the number of samples in the file.
        pub const fn sample_count(&self) -> u32 { self.rows * self.cols }

        /// Writes the header to the given writer.
        pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> std::io::Result<()> {
            let mut header = [0x20u8; 32];
            header[0..4].copy_from_slice(Self::MAGIC);
            header[4..8].copy_from_slice(&self.rows.to_le_bytes());
            header[8..12].copy_from_slice(&self.cols.to_le_bytes());
            header[12..16].copy_from_slice(&self.du.to_le_bytes());
            header[16..20].copy_from_slice(&self.dv.to_le_bytes());
            header[20] = self.unit as u8;
            header[21] = self.sample_data_size;
            header[22] = self.encoding as u8;
            header[23] = self.compression as u8;
            header[31] = 0x0A; // LF
            writer.write_all(&header)
        }

        /// Reads the header from the given reader.
        ///
        /// The reader must be positioned at the start of the header.
        pub fn read<R: Read>(reader: &mut BufReader<R>) -> Result<Self, std::io::Error> {
            let mut buf = [0u8; 32];
            reader.read_exact(&mut buf)?;

            if &buf[0..4] != Self::MAGIC {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid VGMS header {:?}", &buf[0..4]),
                ));
            }

            let rows = u32::from_le_bytes(buf[4..8].try_into().unwrap());
            let cols = u32::from_le_bytes(buf[8..12].try_into().unwrap());
            let du = f32::from_le_bytes(buf[12..16].try_into().unwrap());
            let dv = f32::from_le_bytes(buf[16..20].try_into().unwrap());
            let unit = LengthUnit::from(buf[20]);
            let sample_data_size = buf[21];
            let encoding = FileEncoding::from(buf[22]);
            let compression = CompressionScheme::from(buf[23]);
            Ok(Self {
                rows,
                cols,
                du,
                dv,
                unit,
                sample_data_size,
                encoding,
                compression,
            })
        }
    }

    /// Reads the VGMS file from the given reader.
    pub fn read<R: Read>(
        reader: &mut BufReader<R>,
    ) -> Result<(Header, Vec<f32>), ReadFileErrorKind> {
        let header = Header::read(reader)?;
        let samples = read_f32_data_samples(
            reader,
            header.sample_count() as usize,
            header.encoding,
            header.compression,
        )?;
        Ok((header, samples))
    }

    /// Writes the VGMS file to the given writer.
    pub fn write<W: Write>(
        writer: &mut BufWriter<W>,
        header: Header,
        samples: &[f32],
    ) -> Result<(), WriteFileErrorKind> {
        header.write(writer)?;
        write_f32_data_samples(
            writer,
            header.encoding,
            header.compression,
            samples,
            header.cols,
        )
        .map_err(|err| err.into())
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

    read_ascii_samples(reader, rows * cols, &mut samples).map_err(|err| {
        VgonioError::new(
            format!("Failed to read samples from file {}", filepath.display()),
            Some(Box::new(ReadFileError::from_parse_error(
                filepath.clone(),
                err,
            ))),
        )
    })?;

    Ok(MicroSurface::from_samples(
        rows,
        cols,
        du,
        dv,
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
                .expect(&format!("Read f32 error! {}", x_coord))
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
        du,
        dv,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        measure::{
            bsdf::{BsdfKind, BsdfMeasurementDataPoint, BsdfMeasurementStatsPoint, PerWavelength},
            collector::BounceAndEnergy,
            emitter::RegionShape,
            measurement::{BsdfMeasurementParams, Radius::Auto, SimulationKind},
            Collector, CollectorScheme, Emitter, RtcMethod,
        },
        units::{mm, nm, rad, Radians},
        Medium, RangeByStepSizeInclusive, SphericalDomain, SphericalPartition,
    };
    use vgonio_core::units::{mm, nm, rad, Radians};

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
