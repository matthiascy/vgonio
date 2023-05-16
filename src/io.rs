use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    fmt::Display,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::Path,
};

#[derive(Debug)]
#[non_exhaustive]
pub struct ReadFileError {
    pub path: Box<Path>,
    pub kind: ReadFileErrorKind,
}

impl Display for ReadFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error while reading file: {}", self.path.display())
    }
}

impl std::error::Error for ReadFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ReadFileErrorKind::Read(err) => Some(err),
            ReadFileErrorKind::Parse(err) => Some(err),
        }
    }
}

impl ReadFileError {
    /// Creates a new `ReadFileError` from a `ParseError`.
    pub fn from_parse_error(path: impl AsRef<Path>, err: ParseError) -> Self {
        Self {
            path: path.as_ref().to_path_buf().into_boxed_path(),
            kind: ReadFileErrorKind::Parse(err),
        }
    }

    /// Creates a new `ReadFileError` from a `std::io::Error`.
    pub fn from_std_io_error(path: impl AsRef<Path>, err: std::io::Error) -> Self {
        Self {
            path: path.as_ref().to_path_buf().into_boxed_path(),
            kind: ReadFileErrorKind::Read(err),
        }
    }
}

#[derive(Debug)]
pub enum ReadFileErrorKind {
    Read(std::io::Error),
    Parse(ParseError),
}

impl From<std::io::Error> for ReadFileErrorKind {
    fn from(value: std::io::Error) -> Self { Self::Read(value) }
}

impl From<ParseError> for ReadFileErrorKind {
    fn from(value: ParseError) -> Self { Self::Parse(value) }
}

#[derive(Debug)]
pub struct ParseError {
    pub line: u32,
    pub position: u32,
    pub kind: ParseErrorKind,
    pub encoding: FileEncoding,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.encoding {
            FileEncoding::Ascii => {
                write!(
                    f,
                    "error while parsing line {} at position {}: ",
                    self.line, self.position
                )?;
            }
            FileEncoding::Binary => {
                write!(
                    f,
                    "error while parsing byte at position {}: ",
                    self.position
                )?;
            }
        }
        match &self.kind {
            ParseErrorKind::InvalidMagicNumber => write!(f, "invalid magic number"),
            ParseErrorKind::InvalidEncoding => write!(f, "invalid encoding"),
            ParseErrorKind::InvalidCompression => write!(f, "invalid compression"),
            ParseErrorKind::InvalidLine => write!(f, "invalid line"),
            ParseErrorKind::ParseFloat => write!(f, "invalid float"),
            ParseErrorKind::NotEnoughData => write!(f, "not enough data"),
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug)]
pub enum ParseErrorKind {
    InvalidMagicNumber,
    InvalidEncoding,
    InvalidCompression,
    InvalidLine,
    ParseFloat,
    NotEnoughData,
}

#[derive(Debug)]
pub struct WriteFileError {
    pub path: Box<Path>,
    pub kind: WriteFileErrorKind,
}

impl Display for WriteFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error while writing file: {}", self.path.display())
    }
}

impl std::error::Error for WriteFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            WriteFileErrorKind::Write(err) => Some(err),
        }
    }
}

impl WriteFileError {
    /// Creates a new `WriteFileError` from a `std::io::Error`.
    pub fn from_std_io_error(path: impl AsRef<Path>, err: std::io::Error) -> Self {
        Self {
            path: path.as_ref().to_path_buf().into_boxed_path(),
            kind: WriteFileErrorKind::Write(err),
        }
    }
}

#[derive(Debug)]
pub enum WriteFileErrorKind {
    Write(std::io::Error),
}

impl From<std::io::Error> for WriteFileErrorKind {
    fn from(value: std::io::Error) -> Self { Self::Write(value) }
}

/// Data encoding while storing the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum, Serialize, Deserialize)]
#[repr(u8)]
pub enum FileEncoding {
    /// The data is encoded as ascii text (plain text).
    Ascii = 0x23, // '#'
    /// The data is encoded as binary data.
    Binary = 0x21, // '!'
}

impl From<u8> for FileEncoding {
    fn from(value: u8) -> Self {
        match value {
            0x23 => FileEncoding::Ascii,
            0x21 => FileEncoding::Binary,
            _ => panic!("Invalid data encoding: {}", value),
        }
    }
}

impl Display for FileEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileEncoding::Ascii => write!(f, "ascii"),
            FileEncoding::Binary => write!(f, "binary"),
        }
    }
}

impl FileEncoding {
    /// Returns true if the data is encoded as ascii text.
    pub fn is_ascii(&self) -> bool {
        match self {
            FileEncoding::Ascii => true,
            FileEncoding::Binary => false,
        }
    }

    /// Returns true if the data is encoded as binary data.
    pub fn is_binary(&self) -> bool {
        match self {
            FileEncoding::Ascii => false,
            FileEncoding::Binary => true,
        }
    }
}

/// Data compression scheme while storing the data.
#[repr(u8)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum, Serialize, Deserialize)]
pub enum CompressionScheme {
    /// No compression.
    None = 0x00,
    /// Zlib compression.
    Zlib = 0x01,
    /// Gzip compression.
    Gzip = 0x02,
}

impl From<u8> for CompressionScheme {
    fn from(value: u8) -> Self {
        match value {
            0x00 => CompressionScheme::None,
            0x01 => CompressionScheme::Zlib,
            0x02 => CompressionScheme::Gzip,
            _ => panic!("Invalid data compression: {}", value),
        }
    }
}

impl Display for CompressionScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionScheme::None => write!(f, "none"),
            CompressionScheme::Zlib => write!(f, "zlib"),
            CompressionScheme::Gzip => write!(f, "gzip"),
        }
    }
}

impl CompressionScheme {
    /// Returns true if the data is not compressed.
    pub fn is_none(&self) -> bool { matches!(self, CompressionScheme::None) }

    /// Returns true if the data is compressed with zlib.
    pub fn is_zlib(&self) -> bool { matches!(self, CompressionScheme::Zlib) }

    /// Returns true if the data is compressed with gzip.
    pub fn is_gzip(&self) -> bool { matches!(self, CompressionScheme::Gzip) }
}

/// Write the data samples to the given writer.
fn write_data_samples<W: Write>(
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
fn read_data_samples<R: Read>(
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

    for i in 0..count {
        samples[i] = reader.read_f32::<LittleEndian>().map_err(|e| {
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
        })?;
    }

    Ok(())
}

use crate::{error::Error, msurf::MicroSurface, units::LengthUnit};

/// Micro-surface file format.
pub mod vgms {
    use super::*;
    use crate::units::LengthUnit;
    use std::io::{BufWriter, Read, Write};

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
        let samples = read_data_samples(
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
        write_data_samples(
            writer,
            header.encoding,
            header.compression,
            samples,
            header.cols,
        )
        .map_err(|err| err.into())
    }
}

pub mod vgmo {
    use super::*;
    use crate::measure::measurement::MeasurementKind;
    use std::io::BufWriter;

    /// The range of the angle in the measurement, in radians.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct AngleRange {
        /// The start angle of the measurement, in radians.
        pub start: f32,
        /// The end angle of the measurement, in radians.
        pub end: f32,
        /// The number of angle bins of the measurement.
        pub bin_count: u32,
        /// The width of each angle bin, in radians.
        pub bin_width: f32,
    }

    impl AngleRange {
        /// Returns all the possible angles as an iterator.
        pub fn angles(&self) -> impl Iterator<Item = f32> {
            let start = self.start;
            let end = self.end;
            let bin_width = self.bin_width;
            (0..self.bin_count).map(move |i| (start + bin_width * i as f32).min(end))
        }

        /// Returns negative values of all the possible angles as an iterator in
        /// reverse order.
        pub fn rev_negative_angles(&self) -> impl Iterator<Item = f32> {
            let start = self.start;
            let end = self.end;
            let bin_width = self.bin_width;
            (0..self.bin_count)
                .map(move |i| (start - bin_width * i as f32).max(-end))
                .rev()
        }
    }

    #[test]
    fn angle_range() {
        let range = AngleRange {
            start: 0.0,
            end: 90.0,
            bin_count: 19,
            bin_width: 5.0,
        };
        let angles: Vec<_> = range.angles().collect();
        assert_eq!(angles.len(), 19);
        assert_eq!(angles[0], 0.0);
        assert_eq!(angles[1], 5.0);
        assert_eq!(angles.last(), Some(&90.0));

        let angles: Vec<_> = range.negative_angles().collect();
        assert_eq!(angles.len(), 19);
        assert_eq!(angles[0], -90.0);
        assert_eq!(angles[1], -85.0);
        assert_eq!(angles.last(), Some(&0.0));
    }

    /// Header of the VGMO file.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Header {
        pub kind: MeasurementKind,
        pub encoding: FileEncoding,
        pub compression: CompressionScheme,
        pub azimuth_range: AngleRange,
        pub zenith_range: AngleRange,
        pub sample_count: u32,
    }

    impl Header {
        pub const MAGIC: &'static [u8] = b"VGMO";

        pub fn read<R: Read>(reader: &mut BufReader<R>) -> Result<Self, std::io::Error> {
            let mut buf = [0u8; 48];
            reader.read_exact(&mut buf)?;

            if &buf[0..4] != Self::MAGIC {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Invalid VGMO header {:?}", &buf[0..4]),
                ));
            }

            let kind = MeasurementKind::from(buf[4]);
            let encoding = FileEncoding::from(buf[5]);
            let compression = CompressionScheme::from(buf[6]);
            let azimuth_range = AngleRange {
                start: f32::from_le_bytes(buf[8..12].try_into().unwrap()),
                end: f32::from_le_bytes(buf[12..16].try_into().unwrap()),
                bin_width: f32::from_le_bytes(buf[16..20].try_into().unwrap()),
                bin_count: u32::from_le_bytes(buf[20..24].try_into().unwrap()),
            };
            let zenith_range = AngleRange {
                start: f32::from_le_bytes(buf[24..28].try_into().unwrap()),
                end: f32::from_le_bytes(buf[28..32].try_into().unwrap()),
                bin_width: f32::from_le_bytes(buf[32..36].try_into().unwrap()),
                bin_count: u32::from_le_bytes(buf[36..40].try_into().unwrap()),
            };
            let sample_count = u32::from_le_bytes(buf[40..44].try_into().unwrap());
            Ok(Self {
                kind,
                encoding,
                compression,
                azimuth_range,
                zenith_range,
                sample_count,
            })
        }

        pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> Result<(), WriteFileErrorKind> {
            let mut header = [0x20; 48];
            header[0..4].copy_from_slice(Self::MAGIC);
            header[4] = self.kind as u8;
            header[5] = self.encoding as u8;
            header[6] = self.compression as u8;
            header[8..12].copy_from_slice(&self.azimuth_range.start.to_le_bytes());
            header[12..16].copy_from_slice(&self.azimuth_range.end.to_le_bytes());
            header[16..20].copy_from_slice(&self.azimuth_range.bin_width.to_le_bytes());
            header[20..24].copy_from_slice(&self.azimuth_range.bin_count.to_le_bytes());
            header[24..28].copy_from_slice(&self.zenith_range.start.to_le_bytes());
            header[28..32].copy_from_slice(&self.zenith_range.end.to_le_bytes());
            header[32..36].copy_from_slice(&self.zenith_range.bin_width.to_le_bytes());
            header[36..40].copy_from_slice(&self.zenith_range.bin_count.to_le_bytes());
            header[40..44].copy_from_slice(&self.sample_count.to_le_bytes());
            header[47] = 0x0A; // LF
            writer.write_all(&header).map_err(|err| err.into())
        }
    }

    /// Writes the VGMO file to the given writer.
    pub fn write<W: Write>(
        writer: &mut BufWriter<W>,
        header: Header,
        samples: &[f32],
    ) -> Result<(), WriteFileErrorKind> {
        header.write(writer)?;
        write_data_samples(
            writer,
            header.encoding,
            header.compression,
            samples,
            header.zenith_range.bin_count,
        )
        .map_err(|err| err.into())
    }

    /// Reads the VGMO file from the given reader.
    ///
    /// Returns the header and the measurement samples.
    pub fn read<R: Read>(
        reader: &mut BufReader<R>,
    ) -> Result<(Header, Vec<f32>), ReadFileErrorKind> {
        let header = Header::read(reader)?;
        let samples = read_data_samples(
            reader,
            header.sample_count as usize,
            header.encoding,
            header.compression,
        )?;
        Ok((header, samples))
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
) -> Result<MicroSurface, Error> {
    let mut buf = [0_u8; 4];
    reader.read_exact(&mut buf)?;
    if std::str::from_utf8(&buf)? != "Asci" {
        return Err(Error::UnrecognizedFile);
    }

    let mut reader = BufReader::new(reader);

    let mut line = String::new();
    reader.read_line(&mut line)?;
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
        Error::ReadFile(ReadFileError {
            path: filepath.to_owned().into_boxed_path(),
            kind: ReadFileErrorKind::Parse(err),
        })
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
) -> Result<MicroSurface, Error> {
    let mut line = String::new();

    loop {
        reader.read_line(&mut line)?;
        if !line.is_empty() {
            break;
        }
    }

    if line.trim() != "DATA" {
        return Err(Error::UnrecognizedFile);
    }

    line.clear();

    // Read horizontal coordinates
    reader.read_line(&mut line)?;
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
