use crate::error::Error;
use crate::htfld::{AxisAlignment, Heightfield};
use crate::io::{CacheHeader, CacheKind, MsHeader};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum HeightFieldOrigin {
    Dong2015,
    Usurf,
}

impl Heightfield {
    /// Creates micro-geometry height field by reading the samples stored in
    /// different file format. Supported formats are
    ///
    /// 1. Ascii Matrix file (plain text) coming from
    ///    Predicting Appearance from Measured Microgeometry of Metal Surfaces.
    /// 2. Plain text data coming from µsurf confocal microscope system.
    /// 2. Micro-surface height field file (binary format, ends with *.dcms).
    /// 3. Micro-surface height field cache file (binary format, ends with
    /// *.dccc).
    pub fn read_from_file(
        path: &Path,
        origin: Option<HeightFieldOrigin>,
        alignment: Option<AxisAlignment>,
    ) -> Result<Heightfield, Error> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        if let Some(origin) = origin {
            // If origin is specified, call directly corresponding loading function.
            match origin {
                HeightFieldOrigin::Dong2015 => read_ascii_dong2015(reader, true, alignment),
                HeightFieldOrigin::Usurf => read_ascii_usurf(reader, true, alignment),
            }
        } else {
            // Otherwise, try to figure out the file format by reading first several bytes.
            let mut buf = [0_u8; 4];
            reader.read_exact(&mut buf)?;

            match std::str::from_utf8(&buf)? {
                "Asci" => read_ascii_dong2015(reader, false, alignment),
                "DATA" => read_ascii_usurf(reader, false, alignment),
                "DCCC" => {
                    let header = {
                        let mut buf = [0_u8; 6];
                        reader.read_exact(&mut buf)?;
                        CacheHeader::new(buf)
                    };

                    if header.kind != CacheKind::HeightField {
                        Err(Error::FileError("Not a valid height field cache!"))
                    } else if header.binary {
                        Ok(bincode::deserialize_from(reader)?)
                    } else {
                        Ok(serde_yaml::from_reader(reader)?)
                    }
                }
                "DCMS" => {
                    let header = {
                        let mut buf = [0_u8; 23];
                        reader.read_exact(&mut buf)?;
                        MsHeader::new(buf)
                    };

                    let samples = if header.binary {
                        read_binary_samples(reader, (header.size / 4) as usize)
                    } else {
                        read_ascii_samples(reader)
                    };

                    Ok(Heightfield::from_samples(
                        header.extent[0] as usize,
                        header.extent[1] as usize,
                        header.spacing[0],
                        header.spacing[1],
                        samples,
                        alignment.unwrap_or_default(),
                    ))
                }
                _ => Err(Error::UnrecognizedFile),
            }
        }
    }
}

/// Read micro-surface height field following the convention specified in the
/// paper: > Zhao Dong, Bruce Walter, Steve Marschner, and Donald P. Greenberg.
/// 2016. > Predicting Appearance from Measured Microgeometry of Metal Surfaces.
/// > <i>ACM Trans. Graph.</i> 35, 1, Article 9 (December 2015), 13 pages.
/// > DOI:https://doi-org.tudelft.idm.oclc.org/10.1145/2815618
fn read_ascii_dong2015<R: BufRead>(
    mut reader: R,
    read_first_4_bytes: bool,
    orientation: Option<AxisAlignment>,
) -> Result<Heightfield, Error> {
    if read_first_4_bytes {
        let mut buf = [0_u8; 4];
        reader.read_exact(&mut buf)?;

        if std::str::from_utf8(&buf)? != "Asci" {
            return Err(Error::UnrecognizedFile);
        }
    }

    let mut line = String::new();
    reader.read_line(&mut line)?;
    let extent: Vec<_> = line
        .trim()
        .split_ascii_whitespace()
        .rev()
        .take(2)
        .map(|dim| -> u32 { dim.parse().unwrap() })
        .collect();
    let samples = read_ascii_samples(reader);
    Ok(Heightfield::from_samples(
        extent[1] as usize,
        extent[0] as usize,
        0.11,
        0.11,
        samples,
        orientation.unwrap_or_default(),
    ))
}

/// Read micro-surface height field issued from µsurf confocal microscope.
fn read_ascii_usurf<R: BufRead>(
    mut reader: R,
    read_first_4_bytes: bool,
    orientation: Option<AxisAlignment>,
) -> Result<Heightfield, Error> {
    let mut line = String::new();
    reader.read_line(&mut line)?;

    if read_first_4_bytes && line.trim() != "DATA" {
        return Err(Error::UnrecognizedFile);
    }

    // Read horizontal coordinates
    reader.read_line(&mut line)?;
    let x_coords: Vec<f32> = line
        .trim()
        .split_ascii_whitespace()
        .map(|x_coord| x_coord.parse::<f32>().expect("Read f32 error!"))
        .collect();

    let (y_coords, values): (Vec<f32>, Vec<Vec<f32>>) = reader
        .lines()
        .map(|line| {
            let mut values = read_line_ascii_usurf(&line.unwrap());
            let head = values.remove(0);
            (head, values)
        })
        .unzip();

    // TODO: deal with case when coordinates are not uniform.
    let du = x_coords[1] - x_coords[0];
    let dv = y_coords[1] - y_coords[0];
    let samples: Vec<f32> = values.into_iter().flatten().collect();

    Ok(Heightfield::from_samples(
        x_coords.len(),
        y_coords.len(),
        du,
        dv,
        samples,
        orientation.unwrap_or_default(),
    ))
}

/// Read a line of usurf file. Height values are separated by tab character.
/// Consecutive tabs signifies that the height value at this point is missing.
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

/// Read sample values separated by whitespace line by line.
fn read_ascii_samples<R: BufRead>(reader: R) -> Vec<f32> {
    reader
        .lines()
        .enumerate()
        .flat_map(|(n, line)| {
            let l = line.unwrap_or_else(|_| panic!("Bad line at {}", n));
            l.trim()
                .split_ascii_whitespace()
                .enumerate()
                .map(|(i, x)| {
                    x.parse()
                        .unwrap_or_else(|_| panic!("Parse float error at line {} pos {}", n, i))
                })
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn read_binary_samples<R: Read>(mut reader: R, count: usize) -> Vec<f32> {
    use byteorder::{LittleEndian, ReadBytesExt};

    let mut samples = vec![0.0; count];

    (0..count).for_each(|i| {
        samples[i] = reader.read_f32::<LittleEndian>().expect("read f32 error");
    });

    samples
}

#[cfg(test)]
mod tests {
    use crate::height_field::io::read_line_ascii_usurf;

    #[test]
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
