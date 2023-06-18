use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    fmt::Display,
    fs::File,
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

use crate::{
    error::Error,
    measure::{
        bsdf::MeasuredBsdfData,
        measurement::{MeasuredData, MeasurementData, MeasurementDataSource},
        microfacet::{MeasuredMadfData, MeasuredMmsfData},
    },
    msurf::MicroSurface,
    units::LengthUnit,
};

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

pub mod vgmo {
    use super::*;
    use crate::{
        measure::{
            bsdf::{BsdfKind, BsdfMeasurementDataPoint, BsdfMeasurementStatsPoint, PerWavelength},
            collector::BounceAndEnergy,
            emitter::RegionShape,
            measurement::{
                BsdfMeasurementParams, MadfMeasurementParams, MeasurementKind,
                MmsfMeasurementParams, Radius, SimulationKind,
            },
            Collector, CollectorScheme, Emitter,
        },
        ulp_eq,
        units::{
            mm, rad, solid_angle_of_region, solid_angle_of_spherical_cap, steradians, Nanometres,
            Radians,
        },
        Medium, RangeByStepCountInclusive, RangeByStepSizeInclusive, SphericalPartition,
    };
    use std::io::BufWriter;

    macro_rules! impl_range_by_step_size_inclusive_read_write {
        ($($T:ty, $step_count:ident);*) => {
            $(paste::paste! {
                impl RangeByStepSizeInclusive<$T> {
                    #[doc = "Writes the RangeByStepSizeInclusive<`" $T "`> into the given buffer, following the order: start, stop, step_size, step_count."]
                    pub fn write_to_buf(&self, buf: &mut [u8]) {
                        debug_assert!(
                            buf.len() >= 16,
                            "RangeByStepSizeInclusive needs at least 16 bytes of space"
                        );
                        buf[0..4].copy_from_slice(&self.start.value.to_le_bytes());
                        buf[4..8].copy_from_slice(&self.stop.value.to_le_bytes());
                        buf[8..12].copy_from_slice(&self.step_size.value.to_le_bytes());
                        buf[12..16].copy_from_slice(&(self.$step_count() as u32).to_le_bytes());
                    }

                    #[doc = "Reads the RangeByStepSizeInclusive<`" $T "`> from the given buffer, checking that the step count matches the expected value."]
                    pub fn read_from_buf(buf: &[u8]) -> Self {
                        debug_assert!(
                            buf.len() >= 16,
                            "RangeByStepSizeInclusive needs at least 16 bytes of space"
                        );
                        let start = <$T>::new(f32::from_le_bytes(buf[0..4].try_into().unwrap()));
                        let end = <$T>::new(f32::from_le_bytes(buf[4..8].try_into().unwrap()));
                        let step_size = <$T>::new(f32::from_le_bytes(buf[8..12].try_into().unwrap()));
                        let step_count = u32::from_le_bytes(buf[12..16].try_into().unwrap());
                        let range = Self::new(start, end, step_size);
                        assert_eq!(
                            step_count,
                            range.$step_count() as u32,
                            "RangeByStepSizeInclusive: step count mismatch"
                        );
                        range
                    }
                }
            })*
        };
    }

    impl_range_by_step_size_inclusive_read_write!(
        Radians, step_count_wrapped;
        Nanometres, step_count
    );

    impl RangeByStepCountInclusive<Radians> {
        /// Writes the RangeByStepCountInclusive<Radians> into the given buffer,
        /// following the order: start, stop, step_size, step_count.
        pub fn write_to_buf(&self, buf: &mut [u8]) {
            debug_assert!(
                buf.len() >= 16,
                "RangeByStepCountInclusive<Radians> needs at least 16 bytes of space"
            );
            buf[0..4].copy_from_slice(&self.start.value.to_le_bytes());
            buf[4..8].copy_from_slice(&self.stop.value.to_le_bytes());
            buf[8..12].copy_from_slice(&self.step_size().value.to_le_bytes());
            buf[12..16].copy_from_slice(&(self.step_count as u32).to_le_bytes());
        }

        /// Reads the RangeByStepCountInclusive<Radians> from the given buffer,
        /// checking that the step size matches the expected value.
        pub fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(
                buf.len() >= 16,
                "RangeByStepCountInclusive needs at least 16 bytes of space"
            );
            let start = Radians::new(f32::from_le_bytes(buf[0..4].try_into().unwrap()));
            let end = Radians::new(f32::from_le_bytes(buf[4..8].try_into().unwrap()));
            let step_size = Radians::new(f32::from_le_bytes(buf[8..12].try_into().unwrap()));
            let step_count = u32::from_le_bytes(buf[12..16].try_into().unwrap());
            let range = Self::new(start, end, step_count as usize);
            assert!(
                ulp_eq(range.step_size().value, step_size.value),
                "RangeByStepCountInclusive<Radians>: step size mismatch"
            );
            range
        }
    }

    /// First 8 bytes of the header.
    ///
    /// It contains the magic number, file encoding, compression scheme, and
    /// measurement kind.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct HeaderMeta {
        pub kind: MeasurementKind,
        pub encoding: FileEncoding,
        pub compression: CompressionScheme,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Header {
        Bsdf {
            meta: HeaderMeta,
            bsdf: BsdfMeasurementParams,
        },
        Madf {
            meta: HeaderMeta,
            madf: MadfMeasurementParams,
        },
        Mmsf {
            meta: HeaderMeta,
            mmsf: MmsfMeasurementParams,
        },
    }

    impl Header {
        pub fn meta(&self) -> &HeaderMeta {
            match self {
                Self::Bsdf { meta, .. } => meta,
                Self::Madf { meta, .. } => meta,
                Self::Mmsf { meta, .. } => meta,
            }
        }

        pub fn sample_count(&self) -> usize {
            match self {
                Self::Bsdf { bsdf, .. } => {
                    bsdf.emitter.azimuth.step_count_wrapped()
                        * bsdf.emitter.zenith.step_count_wrapped()
                        * bsdf.emitter.spectrum.step_count()
                        * bsdf.collector.scheme.total_sample_count()
                }
                Self::Madf { madf, .. } => {
                    madf.zenith.step_count_wrapped() * madf.azimuth.step_count_wrapped()
                }
                Self::Mmsf { mmsf, .. } => {
                    mmsf.zenith.step_count_wrapped()
                        * mmsf.azimuth.step_count_wrapped()
                        * mmsf.zenith.step_count_wrapped()
                        * mmsf.azimuth.step_count_wrapped()
                }
            }
        }

        pub fn read<R: Read>(reader: &mut BufReader<R>) -> Result<Self, std::io::Error> {
            let meta = HeaderMeta::read(reader)?;
            match meta.kind {
                MeasurementKind::Bsdf => Ok(Self::Bsdf {
                    meta,
                    bsdf: BsdfMeasurementParams::read_from_vgmo(reader)?,
                }),
                MeasurementKind::MicrofacetAreaDistribution => Ok(Self::Madf {
                    meta,
                    madf: MadfMeasurementParams::read_from_vgmo(reader)?,
                }),
                MeasurementKind::MicrofacetMaskingShadowing => Ok(Self::Mmsf {
                    meta,
                    mmsf: MmsfMeasurementParams::read_from_vgmo(reader)?,
                }),
            }
        }

        pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> Result<(), WriteFileErrorKind> {
            match self {
                Self::Bsdf { meta, bsdf } => {
                    meta.write(writer).and_then(|_| bsdf.write_to_vgmo(writer))
                }
                Self::Madf { meta, madf } => {
                    meta.write(writer).and_then(|_| madf.write_to_vgmo(writer))
                }
                Self::Mmsf { meta, mmsf } => {
                    meta.write(writer).and_then(|_| mmsf.write_to_vgmo(writer))
                }
            }
        }
    }

    impl HeaderMeta {
        pub const MAGIC: &'static [u8] = b"VGMO";

        pub fn read<R: Read>(reader: &mut BufReader<R>) -> Result<Self, std::io::Error> {
            let mut buf = [0u8; 8];
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
            Ok(Self {
                kind,
                encoding,
                compression,
            })
        }

        pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> Result<(), WriteFileErrorKind> {
            let mut meta = [0x20; 8];
            meta[0..4].copy_from_slice(Self::MAGIC);
            meta[4] = self.kind as u8;
            meta[5] = self.encoding as u8;
            meta[6] = self.compression as u8;
            writer.write_all(&meta).map_err(|err| err.into())
        }
    }

    fn madf_or_mmsf_samples_count(
        zenith: &RangeByStepSizeInclusive<Radians>,
        azimuth: &RangeByStepSizeInclusive<Radians>,
        is_madf: bool,
    ) -> usize {
        let zenith_step_count = zenith.step_count_wrapped();
        let azimuth_step_count = azimuth.step_count_wrapped();
        if is_madf {
            zenith_step_count * azimuth_step_count
        } else {
            zenith_step_count * azimuth_step_count * zenith_step_count * azimuth_step_count
        }
    }

    fn read_madf_mmsf_params_from_vgmo<R: Read>(
        reader: &mut BufReader<R>,
        #[cfg(debug_assertions)] is_madf: bool,
    ) -> Result<
        (
            RangeByStepSizeInclusive<Radians>,
            RangeByStepSizeInclusive<Radians>,
        ),
        std::io::Error,
    > {
        let mut buf = [0u8; 40];
        reader.read_exact(&mut buf)?;
        let azimuth = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[0..16]);
        let zenith = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[16..32]);
        let sample_count = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            sample_count as usize,
            madf_or_mmsf_samples_count(&azimuth, &zenith, is_madf)
        );
        Ok((azimuth, zenith))
    }

    pub fn write_madf_mmsf_params_to_vgmo<W: Write>(
        azimuth: &RangeByStepSizeInclusive<Radians>,
        zenith: &RangeByStepSizeInclusive<Radians>,
        writer: &mut BufWriter<W>,
        is_madf: bool,
    ) -> Result<(), WriteFileErrorKind> {
        let mut header = [0x20; 40];
        azimuth.write_to_buf(&mut header[0..16]);
        zenith.write_to_buf(&mut header[16..32]);
        header[32..36].copy_from_slice(
            &(madf_or_mmsf_samples_count(zenith, azimuth, is_madf) as u32).to_le_bytes(),
        );
        header[39] = 0x0A; // LF
        writer.write_all(&header).map_err(|err| err.into())
    }

    impl MadfMeasurementParams {
        /// Reads the measurement parameters from the VGMO file.
        pub fn read_from_vgmo<R: Read>(reader: &mut BufReader<R>) -> Result<Self, std::io::Error> {
            let (azimuth, zenith) = read_madf_mmsf_params_from_vgmo(
                reader,
                #[cfg(debug_assertions)]
                true,
            )?;
            Ok(Self { azimuth, zenith })
        }

        /// Writes the measurement parameters to the VGMO file.
        pub fn write_to_vgmo<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
        ) -> Result<(), WriteFileErrorKind> {
            write_madf_mmsf_params_to_vgmo(&self.azimuth, &self.zenith, writer, true)
        }
    }

    impl MmsfMeasurementParams {
        /// Reads the measurement parameters from the VGMO file.
        pub fn read_from_vgmo<R: Read>(reader: &mut BufReader<R>) -> Result<Self, std::io::Error> {
            let (azimuth, zenith) = read_madf_mmsf_params_from_vgmo(
                reader,
                #[cfg(debug_assertions)]
                false,
            )?;
            Ok(Self {
                azimuth,
                zenith,
                resolution: 512,
            })
        }

        /// Writes the measurement parameters to the VGMO file.
        pub fn write_to_vgmo<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
        ) -> Result<(), WriteFileErrorKind> {
            // TODO: write resolution for MMSF
            write_madf_mmsf_params_to_vgmo(&self.azimuth, &self.zenith, writer, false)
        }
    }

    impl RegionShape {
        pub const REQUIRED_SIZE: usize = 20;

        pub fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "RegionShape needs at least 20 bytes of space"
            );
            match u32::from_le_bytes(buf[0..4].try_into().unwrap()) {
                0x00 => {
                    // Spherical cap
                    let zenith = rad!(f32::from_le_bytes(buf[4..8].try_into().unwrap()));
                    RegionShape::SphericalCap { zenith }
                }
                0x01 => {
                    // Spherical rectangle
                    let zenith_start = rad!(f32::from_le_bytes(buf[4..8].try_into().unwrap()));
                    let zenith_stop = rad!(f32::from_le_bytes(buf[8..12].try_into().unwrap()));
                    let azimuth_start = rad!(f32::from_le_bytes(buf[12..16].try_into().unwrap()));
                    let azimuth_stop = rad!(f32::from_le_bytes(buf[16..20].try_into().unwrap()));
                    RegionShape::SphericalRect {
                        zenith: (zenith_start, zenith_stop),
                        azimuth: (azimuth_start, azimuth_stop),
                    }
                }
                0x02 => {
                    // Disk
                    RegionShape::Disk {
                        radius: Radius::Auto(mm!(0.0)),
                    }
                }
                _ => {
                    panic!("Invalid region shape");
                }
            }
        }

        pub fn write_to_buf(&self, buf: &mut [u8]) {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "RegionShape needs at least 20 bytes of space"
            );
            match self {
                RegionShape::SphericalCap { zenith } => {
                    buf[0..4].fill(0x00);
                    buf[4..8].copy_from_slice(&zenith.value.to_le_bytes());
                }
                RegionShape::SphericalRect { zenith, azimuth } => {
                    buf[0..4].copy_from_slice(&0x01u32.to_le_bytes());
                    buf[4..8].copy_from_slice(&zenith.0.value().to_le_bytes());
                    buf[8..12].copy_from_slice(&zenith.1.value().to_le_bytes());
                    buf[12..16].copy_from_slice(&azimuth.0.value().to_le_bytes());
                    buf[16..20].copy_from_slice(&azimuth.1.value().to_le_bytes());
                }
                RegionShape::Disk { .. } => {
                    buf[0..4].copy_from_slice(&0x02u32.to_le_bytes());
                }
            }
        }
    }

    impl Emitter {
        /// The required size of the buffer to read or write an emitter.
        pub const REQUIRED_SIZE: usize = 80;

        /// Reads an emitter from the given buffer.
        pub fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "Emitter needs at least 80 bytes of space"
            );
            let num_rays = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            let max_bounces = u32::from_le_bytes(buf[4..8].try_into().unwrap());
            let radius = mm!(f32::from_le_bytes(buf[8..12].try_into().unwrap()));
            let zenith = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[12..12 + 16]);
            let azimuth = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[28..28 + 16]);
            let shape = RegionShape::read_from_buf(&buf[44..44 + 20]);
            let spectrum = RangeByStepSizeInclusive::<Nanometres>::read_from_buf(&buf[64..64 + 16]);
            let solid_angle = match shape {
                RegionShape::SphericalCap { zenith } => solid_angle_of_spherical_cap(zenith),
                RegionShape::SphericalRect { zenith, azimuth } => {
                    solid_angle_of_region(zenith, azimuth)
                }
                RegionShape::Disk { .. } => {
                    log::warn!("[TO BE SUPPRESSED] Solid angle of disk emitter is not implemented");
                    steradians!(0.0)
                }
            };
            Self {
                num_rays,
                max_bounces,
                radius: Radius::Fixed(radius),
                zenith,
                azimuth,
                shape,
                spectrum,
                solid_angle,
            }
        }

        /// Writes the emitter to the given buffer.
        pub fn write_to_buf(&self, buf: &mut [u8]) {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "Emitter needs at least 80 bytes of space"
            );
            buf[0..4].copy_from_slice(&self.num_rays.to_le_bytes());
            buf[4..8].copy_from_slice(&self.max_bounces.to_le_bytes());
            buf[8..12].copy_from_slice(&self.radius.value().value.to_le_bytes());
            self.zenith.write_to_buf(&mut buf[12..12 + 16]);
            self.azimuth.write_to_buf(&mut buf[28..28 + 16]);
            self.shape.write_to_buf(&mut buf[44..44 + 20]);
            self.spectrum.write_to_buf(&mut buf[64..64 + 16]);
        }
    }

    impl SphericalPartition {
        /// The required size of the buffer to write the partition to.
        pub const REQUIRED_SIZE: usize = 36;

        /// Reads a partition from a buffer.
        pub fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "SphericalPartition needs at least 36 bytes of space"
            );
            let partition_type = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            match partition_type {
                0x00 => {
                    // EqualAngle
                    let zenith =
                        RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[4..4 + 16]);
                    let azimuth =
                        RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[20..20 + 16]);
                    SphericalPartition::EqualAngle { zenith, azimuth }
                }
                0x01 => {
                    // EqualArea
                    let zenith =
                        RangeByStepCountInclusive::<Radians>::read_from_buf(&buf[4..4 + 16]);
                    let azimuth =
                        RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[20..20 + 16]);
                    SphericalPartition::EqualArea { zenith, azimuth }
                }
                0x02 => {
                    // EqualProjectedArea
                    let zenith = RangeByStepCountInclusive::read_from_buf(&buf[4..4 + 16]);
                    let azimuth =
                        RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[20..20 + 16]);
                    SphericalPartition::EqualProjectedArea { zenith, azimuth }
                }
                _ => panic!("Invalid partition type: {}", partition_type),
            }
        }

        /// Writes the partition to a buffer.
        pub fn write_to_buf(&self, buf: &mut [u8]) {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "SphericalPartition needs at least 36 bytes of space"
            );
            match self {
                SphericalPartition::EqualAngle { zenith, azimuth } => {
                    buf[0..4].copy_from_slice(&(0x00u32).to_le_bytes());
                    zenith.write_to_buf(&mut buf[4..4 + 16]);
                    azimuth.write_to_buf(&mut buf[20..20 + 16]);
                }
                SphericalPartition::EqualArea { zenith, azimuth } => {
                    buf[0..4].copy_from_slice(&(0x01u32).to_le_bytes());
                    zenith.write_to_buf(&mut buf[4..4 + 16]);
                    azimuth.write_to_buf(&mut buf[20..20 + 16]);
                }
                SphericalPartition::EqualProjectedArea { zenith, azimuth } => {
                    buf[0..4].copy_from_slice(&(0x02u32).to_le_bytes());
                    zenith.write_to_buf(&mut buf[4..4 + 16]);
                    azimuth.write_to_buf(&mut buf[20..20 + 16]);
                }
            }
        }
    }

    impl CollectorScheme {
        /// The size of the buffer required to write the collector scheme.
        pub const REQUIRED_SIZE: usize = 60;

        /// The size of the buffer required to write the partitioned region.
        pub const PARTITIONED_REGION_SIZE: usize = 44;

        /// The size of the buffer required to write the single region.
        pub const SINGLE_REGION_SIZE: usize = 60;

        /// Reads the collector scheme from a buffer.
        pub fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "CollectorScheme needs at least 60 bytes of space"
            );
            match u32::from_le_bytes(buf[0..4].try_into().unwrap()) {
                0x00 => {
                    // Partitioned
                    let domain = (u32::from_le_bytes(buf[4..8].try_into().unwrap()) as u8)
                        .try_into()
                        .unwrap();
                    let partition = SphericalPartition::read_from_buf(
                        &buf[8..8 + SphericalPartition::REQUIRED_SIZE],
                    );
                    CollectorScheme::Partitioned { domain, partition }
                }
                0x01 => {
                    // Single Region
                    let domain = (u32::from_le_bytes(buf[4..8].try_into().unwrap()) as u8)
                        .try_into()
                        .unwrap();
                    let shape = RegionShape::read_from_buf(&buf[8..8 + RegionShape::REQUIRED_SIZE]);
                    let zenith = RangeByStepSizeInclusive::<Radians>::read_from_buf(
                        &buf[8 + RegionShape::REQUIRED_SIZE..8 + RegionShape::REQUIRED_SIZE + 16],
                    );
                    let azimuth = RangeByStepSizeInclusive::<Radians>::read_from_buf(
                        &buf[8 + RegionShape::REQUIRED_SIZE + 16
                            ..8 + RegionShape::REQUIRED_SIZE + 32],
                    );
                    CollectorScheme::SingleRegion {
                        domain,
                        shape,
                        zenith,
                        azimuth,
                    }
                }
                _ => panic!("Invalid collector scheme type"),
            }
        }

        /// Writes the collector scheme to a buffer.
        pub fn write_to_buf(&self, buf: &mut [u8]) {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "CollectorScheme needs at least 60 bytes of space"
            );
            match self {
                CollectorScheme::Partitioned { domain, partition } => {
                    buf[0..4].copy_from_slice(&(0x00u32).to_le_bytes());
                    buf[4..8].copy_from_slice(&(*domain as u32).to_le_bytes());
                    partition.write_to_buf(&mut buf[8..8 + SphericalPartition::REQUIRED_SIZE]);
                }
                CollectorScheme::SingleRegion {
                    domain,
                    shape,
                    zenith,
                    azimuth,
                } => {
                    buf[0..4].copy_from_slice(&(0x01u32).to_le_bytes());
                    buf[4..8].copy_from_slice(&(*domain as u32).to_le_bytes());
                    shape.write_to_buf(&mut buf[8..8 + RegionShape::REQUIRED_SIZE]);
                    zenith.write_to_buf(&mut buf[28..28 + 16]);
                    azimuth.write_to_buf(&mut buf[44..44 + 16]);
                }
            }
        }
    }

    impl Collector {
        /// The size of the buffer required to write the collector.
        pub const REQUIRED_SIZE: usize = 4 + CollectorScheme::REQUIRED_SIZE;

        /// Reads the collector from a buffer.
        pub fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "Collector needs at least 64 bytes of space"
            );
            let radius = {
                let val = f32::from_le_bytes(buf[0..4].try_into().unwrap());
                if val.is_infinite() {
                    Radius::Auto(mm!(0.0))
                } else {
                    Radius::Fixed(mm!(val))
                }
            };
            let scheme =
                CollectorScheme::read_from_buf(&buf[4..4 + CollectorScheme::REQUIRED_SIZE]);
            Self { radius, scheme }
        }

        /// Writes the collector to a buffer.
        pub fn write_to_buf(&self, buf: &mut [u8]) {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "Collector needs at least 64 bytes of space"
            );
            match self.radius {
                Radius::Auto(_) => {
                    buf[0..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
                }
                Radius::Fixed(r) => {
                    buf[0..4].copy_from_slice(&r.value().to_le_bytes());
                }
            }
            buf[0..4].copy_from_slice(&self.radius.value().value.to_le_bytes());
            self.scheme
                .write_to_buf(&mut buf[4..4 + CollectorScheme::REQUIRED_SIZE]);
        }
    }

    impl BsdfMeasurementParams {
        /// Reads the BSDF measurement parameters from the given reader.
        pub fn read_from_vgmo<R: Read>(reader: &mut BufReader<R>) -> Result<Self, std::io::Error> {
            let mut buf = [0u8; Collector::REQUIRED_SIZE + Emitter::REQUIRED_SIZE + 4];
            reader.read_exact(&mut buf)?;
            let kind = BsdfKind::from(buf[0]);
            let incident_medium = Medium::from(buf[1]);
            let transmitted_medium = Medium::from(buf[2]);
            let sim_kind = SimulationKind::try_from(buf[3]).unwrap();
            let emitter = Emitter::read_from_buf(&buf[4..4 + Emitter::REQUIRED_SIZE]);
            let collector = Collector::read_from_buf(&buf[4 + Emitter::REQUIRED_SIZE..]);
            Ok(Self {
                kind,
                incident_medium,
                transmitted_medium,
                sim_kind,
                emitter,
                collector,
            })
        }

        /// Writes the BSDF measurement parameters to the given writer.
        pub fn write_to_vgmo<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
        ) -> Result<(), WriteFileErrorKind> {
            let mut buf = [0u8; Collector::REQUIRED_SIZE + Emitter::REQUIRED_SIZE + 4];
            buf[0] = self.kind as u8;
            buf[1] = self.incident_medium as u8;
            buf[2] = self.transmitted_medium as u8;
            buf[3] = match self.sim_kind {
                SimulationKind::GeomOptics(method) => method as u8,
                SimulationKind::WaveOptics => 0x03,
            };
            self.emitter
                .write_to_buf(&mut buf[4..4 + Emitter::REQUIRED_SIZE]);
            self.collector
                .write_to_buf(&mut buf[4 + Emitter::REQUIRED_SIZE..]);
            writer.write_all(&buf).map_err(|err| err.into())
        }
    }

    impl MeasuredMadfData {
        /// Reads the measured MADF data from the given reader.
        pub fn read<R: Read>(
            reader: &mut BufReader<R>,
            meta: HeaderMeta,
            params: MadfMeasurementParams,
        ) -> Result<Self, ReadFileErrorKind> {
            debug_assert!(
                meta.kind == MeasurementKind::MicrofacetAreaDistribution,
                "Measurement kind mismatch"
            );
            let samples = read_f32_data_samples(
                reader,
                params.samples_count(),
                meta.encoding,
                meta.compression,
            )?;
            Ok(MeasuredMadfData { params, samples })
        }

        /// Writes the measured MADF data to the given writer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<(), WriteFileErrorKind> {
            write_f32_data_samples(
                writer,
                encoding,
                compression,
                &self.samples,
                self.params.zenith.step_count_wrapped() as u32,
            )
            .map_err(|err| err.into())
        }
    }

    impl MeasuredMmsfData {
        /// Reads the measured MMSF data from the given reader.
        pub fn read<R: Read>(
            reader: &mut BufReader<R>,
            meta: HeaderMeta,
            params: MmsfMeasurementParams,
        ) -> Result<Self, ReadFileErrorKind> {
            debug_assert!(
                meta.kind == MeasurementKind::MicrofacetMaskingShadowing,
                "Measurement kind mismatch"
            );
            let samples = read_f32_data_samples(
                reader,
                params.samples_count(),
                meta.encoding,
                meta.compression,
            )?;
            Ok(MeasuredMmsfData { params, samples })
        }

        /// Writes the measured MMSF data to the given writer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<(), WriteFileErrorKind> {
            write_f32_data_samples(
                writer,
                encoding,
                compression,
                &self.samples,
                self.params.zenith.step_count_wrapped() as u32,
            )
            .map_err(|err| err.into())
        }
    }

    macro_rules! impl_per_wavelength_data_io {
        ($($t:ty)*) => {
            $(
                impl PerWavelength<$t> {
                    /// The size of a single element in bytes.
                    pub const ELEM_SIZE: usize = std::mem::size_of::<$t>();

                    /// Writes the data to the given buffer.
                    pub fn write_to_buf(&self, buf: &mut [u8]) {
                        debug_assert!(buf.len() >= self.len() * Self::ELEM_SIZE, "Buffer too small");
                        for i in 0..self.len() {
                            buf[i * Self::ELEM_SIZE..(i + 1) * Self::ELEM_SIZE].copy_from_slice(&self[i].to_le_bytes());
                        }
                    }

                    /// Reads the data from the given buffer.
                    pub fn read_from_buf(buf: &[u8], len: usize) -> Self {
                        debug_assert!(buf.len() >= len * Self::ELEM_SIZE, "Buffer too small");
                        let mut data = vec![0 as $t; len];
                        for i in 0..len {
                            data[i] = <$t>::from_le_bytes(buf[i * Self::ELEM_SIZE..(i + 1) * Self::ELEM_SIZE].try_into().unwrap());
                        }
                        Self(data)
                    }
                }
            )*
        };
    }

    impl_per_wavelength_data_io!(f32 u32);

    impl BsdfMeasurementStatsPoint {
        /// Writes the BSDF measurement statistics at a single point to the
        /// buffer.
        pub fn write_to_buf(&self, buf: &mut [u8], n_wavelength: usize, bounces: usize) {
            let size = Self::calc_size_in_bytes(n_wavelength, bounces);
            debug_assert!(buf.len() >= size, "Buffer too small");
            let mut offset = 0;
            buf[offset..offset + 4].copy_from_slice(&self.n_received.to_le_bytes());
            offset += 4;
            self.n_absorbed
                .write_to_buf(&mut buf[offset..offset + n_wavelength * 4]);
            offset += n_wavelength * 4;
            self.n_reflected
                .write_to_buf(&mut buf[offset..offset + n_wavelength * 4]);
            offset += n_wavelength * 4;
            self.n_captured
                .write_to_buf(&mut buf[offset..offset + n_wavelength * 4]);
            offset += n_wavelength * 4;
            self.e_captured
                .write_to_buf(&mut buf[offset..offset + n_wavelength * 4]);
            offset += n_wavelength * 4;
            for i in 0..n_wavelength {
                for j in 0..bounces {
                    buf[offset + i * bounces * 4 + j * 4..offset + i * bounces * 4 + (j + 1) * 4]
                        .copy_from_slice(&self.num_rays_per_bounce[i][j].to_le_bytes());
                }
            }
            offset += n_wavelength * bounces * 4;
            for i in 0..n_wavelength {
                for j in 0..bounces {
                    buf[offset + i * bounces * 4 + j * 4..offset + i * bounces * 4 + (j + 1) * 4]
                        .copy_from_slice(&self.energy_per_bounce[i][j].to_le_bytes());
                }
            }
            offset += n_wavelength * bounces * 4;
            debug_assert_eq!(offset, size, "Buffer size mismatch");
        }

        /// Reads the BSDF measurement statistics at a single point from the
        /// buffer.
        pub fn read_from_buf(buf: &[u8], n_wavelength: usize, max_bounce: usize) -> Option<Self> {
            let mut offset = 0;
            let n_received = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
            offset += 4;
            let n_absorbed = PerWavelength::<u32>::read_from_buf(
                &buf[offset..offset + n_wavelength * 4],
                n_wavelength,
            );
            offset += n_wavelength * 4;

            let n_reflected = PerWavelength::<u32>::read_from_buf(
                &buf[offset..offset + n_wavelength * 4],
                n_wavelength,
            );
            offset += n_wavelength * 4;

            let n_captured = PerWavelength::<u32>::read_from_buf(
                &buf[offset..offset + n_wavelength * 4],
                n_wavelength,
            );
            offset += n_wavelength * 4;

            let captured_energy = PerWavelength::<f32>::read_from_buf(
                &buf[offset..offset + n_wavelength * 4],
                n_wavelength,
            );
            offset += n_wavelength * 4;

            let mut num_rays_per_bounce = vec![vec![0u32; max_bounce]; n_wavelength];
            for i in 0..n_wavelength {
                for j in 0..max_bounce {
                    num_rays_per_bounce[i][j] = u32::from_le_bytes(
                        buf[offset + i * max_bounce * 4 + j * 4
                            ..offset + i * max_bounce * 4 + (j + 1) * 4]
                            .try_into()
                            .unwrap(),
                    );
                }
            }
            offset += n_wavelength * max_bounce * 4;

            let mut energy_per_bounce = vec![vec![0f32; max_bounce]; n_wavelength];
            for i in 0..n_wavelength {
                for j in 0..max_bounce {
                    energy_per_bounce[i][j] = f32::from_le_bytes(
                        buf[offset + i * max_bounce * 4 + j * 4
                            ..offset + i * max_bounce * 4 + (j + 1) * 4]
                            .try_into()
                            .unwrap(),
                    );
                }
            }

            Some(Self {
                n_received,
                n_absorbed,
                n_reflected,
                n_captured,
                e_captured: captured_energy,
                num_rays_per_bounce: PerWavelength(num_rays_per_bounce),
                energy_per_bounce: PerWavelength(energy_per_bounce),
            })
        }
    }

    impl BounceAndEnergy {
        pub fn read_from_buf(buf: &[u8], bounces: usize) -> Option<Self> {
            let size = Self::calc_size_in_bytes(bounces);
            debug_assert_eq!(buf.len(), size, "Buffer size mismatch");
            let total_rays = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            let total_energy = f32::from_le_bytes(buf[4..8].try_into().unwrap());
            let mut offset = 8;
            let mut num_rays_per_bounce = vec![0u32; bounces];
            for i in 0..bounces {
                num_rays_per_bounce[i] =
                    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            let mut energy_per_bounce = vec![0f32; bounces];
            for i in 0..bounces {
                energy_per_bounce[i] =
                    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            debug_assert_eq!(offset, size, "Buffer size mismatch");
            Some(Self {
                total_rays,
                total_energy,
                num_rays_per_bounce,
                energy_per_bounce,
            })
        }

        pub fn write_to_buf(&self, buf: &mut [u8], bounces: usize) {
            debug_assert_eq!(self.energy_per_bounce.len(), bounces);
            debug_assert_eq!(self.num_rays_per_bounce.len(), bounces);
            let size = Self::calc_size_in_bytes(bounces);
            debug_assert!(buf.len() >= size, "Buffer size mismatch");
            buf[0..4].copy_from_slice(&self.total_rays.to_le_bytes());
            buf[4..8].copy_from_slice(&self.total_energy.to_le_bytes());
            let mut offset = 8;
            for i in 0..bounces {
                buf[offset..offset + 4].copy_from_slice(&self.num_rays_per_bounce[i].to_le_bytes());
                offset += 4;
            }
            for i in 0..bounces {
                buf[offset..offset + 4].copy_from_slice(&self.energy_per_bounce[i].to_le_bytes());
                offset += 4;
            }
            debug_assert_eq!(offset, size, "Buffer size mismatch");
        }
    }

    impl BsdfMeasurementDataPoint<BounceAndEnergy> {
        /// Calculates the size of a single data point in bytes.
        pub fn calc_size_in_bytes(
            n_wavelength: usize,
            bounces: usize,
            collector_sample_count: usize,
        ) -> usize {
            BsdfMeasurementStatsPoint::calc_size_in_bytes(n_wavelength, bounces)
                + BounceAndEnergy::calc_size_in_bytes(bounces)
                    * collector_sample_count
                    * n_wavelength
        }

        /// Reads a single data point from a buffer.
        pub fn read_from_buf(buf: &[u8], params: &BsdfMeasurementParams) -> Self {
            let n_wavelength = params.emitter.spectrum.step_count();
            let bounces = params.emitter.max_bounces as usize;
            let collector_sample_count = params.collector.scheme.total_sample_count();
            let size = Self::calc_size_in_bytes(n_wavelength, bounces, collector_sample_count);
            let bounce_and_energy_size = BounceAndEnergy::calc_size_in_bytes(bounces);
            debug_assert_eq!(buf.len(), size, "Buffer size mismatch");
            let stats_size = BsdfMeasurementStatsPoint::calc_size_in_bytes(n_wavelength, bounces);
            let stats = BsdfMeasurementStatsPoint::read_from_buf(
                &buf[0..stats_size],
                n_wavelength,
                bounces,
            )
            .unwrap();
            let mut data: Vec<PerWavelength<BounceAndEnergy>> =
                Vec::with_capacity(collector_sample_count);
            for i in 0..collector_sample_count {
                let mut per_wavelength: Vec<BounceAndEnergy> = Vec::with_capacity(n_wavelength);
                for j in 0..n_wavelength {
                    let offset = stats_size
                        + i * n_wavelength * bounce_and_energy_size
                        + j * bounce_and_energy_size;
                    let bounce_and_energy = BounceAndEnergy::read_from_buf(
                        &buf[offset..offset + bounce_and_energy_size],
                        bounces,
                    )
                    .unwrap();
                    per_wavelength.push(bounce_and_energy);
                }
                data.push(PerWavelength(per_wavelength));
            }

            Self { stats, data }
        }

        /// Writes a single data point to a buffer.
        pub fn write_to_buf(&self, buf: &mut [u8], n_wavelength: usize, bounces: usize) {
            // Write stats.
            self.stats.write_to_buf(buf, n_wavelength, bounces);
            let mut offset = BsdfMeasurementStatsPoint::calc_size_in_bytes(n_wavelength, bounces);
            let bounce_and_energy_size = BounceAndEnergy::calc_size_in_bytes(bounces);
            // Write collector's per wavelength patch data.
            for per_wavelength_patch_data in &self.data {
                for bounce_and_energy in per_wavelength_patch_data.iter() {
                    bounce_and_energy.write_to_buf(&mut buf[offset..], bounces);
                    offset += bounce_and_energy_size;
                }
            }
        }
    }

    impl MeasuredBsdfData {
        /// Reads the measured BSDF data from the given reader.
        pub fn read<R: Read>(
            reader: &mut BufReader<R>,
            meta: HeaderMeta,
            params: BsdfMeasurementParams,
        ) -> Result<Self, ReadFileErrorKind> {
            debug_assert!(
                meta.kind == MeasurementKind::Bsdf,
                "Measurement kind mismatch"
            );

            let mut zlib_decoder;
            let mut gzip_decoder;

            let mut decoder: Box<&mut dyn Read> = match meta.compression {
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

            match meta.encoding {
                FileEncoding::Ascii => {
                    todo!("Ascii encoding is not supported yet")
                }
                FileEncoding::Binary => {
                    let n_wavelength = params.emitter.spectrum.step_count();
                    let bounces = params.emitter.max_bounces as usize;
                    let collector_sample_count = params.collector.scheme.total_sample_count();
                    let sample_size =
                        BsdfMeasurementDataPoint::<BounceAndEnergy>::calc_size_in_bytes(
                            n_wavelength,
                            bounces,
                            collector_sample_count,
                        );
                    let sample_count = params.emitter.azimuth.step_count_wrapped()
                        * params.emitter.zenith.step_count_wrapped();
                    let mut buf = vec![0u8; sample_size];
                    let mut samples = Vec::with_capacity(sample_count);
                    (0..sample_count).for_each(|_| {
                        decoder.read_exact(&mut buf).unwrap();
                        samples.push(BsdfMeasurementDataPoint::read_from_buf(&buf, &params));
                    });

                    Ok(Self { params, samples })
                }
            }
        }

        /// Writes the measured BSDF data to the given writer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<(), WriteFileErrorKind> {
            let mut zlib;
            let mut gzip;

            let mut encoder: Box<&mut dyn Write> = match compression {
                CompressionScheme::None => Box::new(writer),
                CompressionScheme::Zlib => {
                    zlib = flate2::write::ZlibEncoder::new(writer, flate2::Compression::default());
                    Box::new(&mut zlib)
                }
                CompressionScheme::Gzip => {
                    gzip = flate2::write::GzEncoder::new(writer, flate2::Compression::default());
                    Box::new(&mut gzip)
                }
            };

            match encoding {
                FileEncoding::Ascii => {
                    todo!("Ascii encoding is not supported yet")
                }
                FileEncoding::Binary => {
                    for sample in &self.samples {
                        let n_wavelength = self.params.emitter.spectrum.step_count();
                        let bounces = self.params.emitter.max_bounces as usize;
                        let collector_sample_count =
                            self.params.collector.scheme.total_sample_count();
                        let sample_size =
                            BsdfMeasurementDataPoint::<BounceAndEnergy>::calc_size_in_bytes(
                                n_wavelength,
                                bounces,
                                collector_sample_count,
                            );
                        let mut buf = vec![0u8; sample_size];
                        sample.write_to_buf(&mut buf, n_wavelength, bounces);
                        encoder.write_all(&buf)?;
                    }
                    Ok(())
                }
            }
        }
    }

    impl MeasurementData {
        /// Loads the measurement data from a file.
        pub fn read_from_file(filepath: &Path) -> Result<Self, Error> {
            let file = File::open(filepath)?;
            let mut reader = BufReader::new(file);
            let header = Header::read(&mut reader)?;
            let path = filepath.to_path_buf();
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("invalid file stem")
                .to_string();
            match header {
                Header::Bsdf { meta, bsdf } => {
                    let measured =
                        MeasuredBsdfData::read(&mut reader, meta, bsdf).map_err(|err| {
                            Error::ReadFile(ReadFileError {
                                path: filepath.to_owned().into_boxed_path(),
                                kind: err,
                            })
                        })?;
                    Ok(MeasurementData {
                        name,
                        source: MeasurementDataSource::Loaded(path),
                        measured: MeasuredData::Bsdf(measured),
                    })
                }
                Header::Madf { meta, madf } => {
                    let measured =
                        MeasuredMadfData::read(&mut reader, meta, madf).map_err(|err| {
                            Error::ReadFile(ReadFileError {
                                path: filepath.to_owned().into_boxed_path(),
                                kind: err,
                            })
                        })?;
                    Ok(MeasurementData {
                        name,
                        source: MeasurementDataSource::Loaded(path),
                        measured: MeasuredData::Madf(measured),
                    })
                }
                Header::Mmsf { meta, mmsf } => {
                    let measured =
                        MeasuredMmsfData::read(&mut reader, meta, mmsf).map_err(|err| {
                            Error::ReadFile(ReadFileError {
                                path: filepath.to_owned().into_boxed_path(),
                                kind: err,
                            })
                        })?;
                    Ok(MeasurementData {
                        name,
                        source: MeasurementDataSource::Loaded(path),
                        measured: MeasuredData::Mmsf(measured),
                    })
                }
            }
        }

        /// Writes the measurement data to a file in VGMO format.
        pub fn write_to_file(
            &self,
            filepath: &Path,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<(), Error> {
            let header = match &self.measured {
                MeasuredData::Madf(adf) => {
                    assert_eq!(
                        adf.samples.len(),
                        adf.params.samples_count(),
                        "Writing a ADF requires the number of samples to match the number of bins."
                    );
                    Header::Madf {
                        meta: HeaderMeta {
                            kind: MeasurementKind::MicrofacetAreaDistribution,
                            encoding,
                            compression,
                        },
                        madf: adf.params,
                    }
                }
                MeasuredData::Mmsf(msf) => {
                    assert_eq!(
                        msf.samples.len(),
                        msf.params.samples_count(),
                        "Writing a MSF requires the number of samples to match the number of bins."
                    );
                    Header::Mmsf {
                        meta: HeaderMeta {
                            kind: MeasurementKind::MicrofacetMaskingShadowing,
                            encoding,
                            compression,
                        },
                        mmsf: msf.params,
                    }
                }
                MeasuredData::Bsdf(bsdf) => {
                    assert_eq!(
                        bsdf.samples.len(),
                        bsdf.params.bsdf_data_samples_count(),
                        "Writing a BSDF requires the number of samples to match the number of \
                         bins."
                    );
                    Header::Bsdf {
                        meta: HeaderMeta {
                            kind: MeasurementKind::Bsdf,
                            encoding,
                            compression,
                        },
                        bsdf: bsdf.params,
                    }
                }
            };
            let file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(filepath)?;
            let mut writer = BufWriter::new(file);
            header.write(&mut writer).map_err(|err| {
                Error::WriteFile(WriteFileError {
                    path: filepath.to_path_buf().into_boxed_path(),
                    kind: err,
                })
            })?;

            match &self.measured {
                MeasuredData::Madf(madf) => {
                    madf.write(&mut writer, encoding, compression)
                        .map_err(|err| {
                            Error::WriteFile(WriteFileError {
                                path: filepath.to_path_buf().into_boxed_path(),
                                kind: err,
                            })
                        })
                }
                MeasuredData::Mmsf(mmsf) => {
                    mmsf.write(&mut writer, encoding, compression)
                        .map_err(|err| {
                            Error::WriteFile(WriteFileError {
                                path: filepath.to_path_buf().into_boxed_path(),
                                kind: err,
                            })
                        })
                }
                MeasuredData::Bsdf(bsdf) => {
                    bsdf.write(&mut writer, encoding, compression)
                        .map_err(|err| {
                            Error::WriteFile(WriteFileError {
                                path: filepath.to_path_buf().into_boxed_path(),
                                kind: err,
                            })
                        })
                }
            }
        }
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

    #[test]
    fn test_bsdf_measurement_stats_point_read_write() {
        let data = BsdfMeasurementStatsPoint {
            n_received: 1234567,
            n_absorbed: PerWavelength(vec![1, 2, 3, 4]),
            n_reflected: PerWavelength(vec![5, 6, 7, 8]),
            n_captured: PerWavelength(vec![9, 10, 11, 12]),
            e_captured: PerWavelength(vec![13.0, 14.0, 15.0, 16.0]),
            num_rays_per_bounce: PerWavelength(vec![
                vec![17, 18, 19],
                vec![22, 23, 24],
                vec![26, 27, 28],
                vec![30, 31, 32],
            ]),
            energy_per_bounce: PerWavelength(vec![
                vec![1.0, 2.0, 4.0],
                vec![5.0, 6.0, 7.0],
                vec![8.0, 9.0, 10.0],
                vec![11.0, 12.0, 13.0],
            ]),
        };
        let size = BsdfMeasurementStatsPoint::calc_size_in_bytes(4, 3);
        let mut buf = vec![0; size];
        data.write_to_buf(&mut buf, 4, 3);
        let data2 = BsdfMeasurementStatsPoint::read_from_buf(&buf, 4, 3).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_bounce_and_energy_read_write() {
        let data = BounceAndEnergy {
            total_rays: 33468,
            total_energy: 1349534.0,
            num_rays_per_bounce: vec![210, 40, 60, 70, 80, 90, 100, 110, 120, 130, 0],
            energy_per_bounce: vec![
                20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90., 100., 110., 120.,
            ],
        };
        let size = BounceAndEnergy::calc_size_in_bytes(11);
        let mut buf = vec![0; size];
        data.write_to_buf(&mut buf, 11);
        let data2 = BounceAndEnergy::read_from_buf(&buf, 11).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_bsdf_measurement_data_point() {
        let data = BsdfMeasurementDataPoint::<BounceAndEnergy> {
            stats: BsdfMeasurementStatsPoint {
                n_received: 0,
                n_absorbed: PerWavelength(vec![1, 2, 3, 4]),
                n_reflected: PerWavelength(vec![5, 6, 7, 8]),
                n_captured: PerWavelength(vec![9, 10, 11, 12]),
                e_captured: PerWavelength(vec![13.0, 14.0, 15.0, 16.0]),
                num_rays_per_bounce: PerWavelength(vec![
                    vec![17, 18, 19],
                    vec![22, 23, 24],
                    vec![26, 27, 28],
                    vec![30, 31, 32],
                ]),
                energy_per_bounce: PerWavelength(vec![
                    vec![1.0, 2.0, 4.0],
                    vec![5.0, 6.0, 7.0],
                    vec![8.0, 9.0, 10.0],
                    vec![11.0, 12.0, 13.0],
                ]),
            },
            data: vec![
                PerWavelength(vec![
                    BounceAndEnergy {
                        total_rays: 33468,
                        total_energy: 1349534.0,
                        num_rays_per_bounce: vec![210, 40, 60],
                        energy_per_bounce: vec![20.0, 30.0, 40.0],
                    },
                    BounceAndEnergy {
                        total_rays: 33,
                        total_energy: 14.0,
                        num_rays_per_bounce: vec![10, 4, 0],
                        energy_per_bounce: vec![0.0, 3.0, 4.0],
                    },
                    BounceAndEnergy {
                        total_rays: 33468,
                        total_energy: 1349534.0,
                        num_rays_per_bounce: vec![210, 40, 60],
                        energy_per_bounce: vec![20.0, 30.0, 40.0],
                    },
                    BounceAndEnergy {
                        total_rays: 33,
                        total_energy: 14.0,
                        num_rays_per_bounce: vec![10, 4, 0],
                        energy_per_bounce: vec![0.0, 3.0, 4.0],
                    },
                ]),
                PerWavelength(vec![
                    BounceAndEnergy {
                        total_rays: 33468,
                        total_energy: 1349534.0,
                        num_rays_per_bounce: vec![210, 40, 60],
                        energy_per_bounce: vec![20.0, 30.0, 40.0],
                    },
                    BounceAndEnergy {
                        total_rays: 33,
                        total_energy: 14.0,
                        num_rays_per_bounce: vec![10, 4, 0],
                        energy_per_bounce: vec![0.0, 3.0, 4.0],
                    },
                    BounceAndEnergy {
                        total_rays: 33468,
                        total_energy: 1349534.0,
                        num_rays_per_bounce: vec![210, 40, 60],
                        energy_per_bounce: vec![20.0, 30.0, 40.0],
                    },
                    BounceAndEnergy {
                        total_rays: 33,
                        total_energy: 14.0,
                        num_rays_per_bounce: vec![10, 4, 0],
                        energy_per_bounce: vec![0.0, 3.0, 4.0],
                    },
                ]),
            ],
        };

        let size = BsdfMeasurementDataPoint::<BounceAndEnergy>::calc_size_in_bytes(4, 3, 2);
        let mut buf = vec![0; size];
        data.write_to_buf(&mut buf, 4, 3);
        let params = BsdfMeasurementParams {
            kind: BsdfKind::Brdf,
            sim_kind: SimulationKind::GeomOptics(RtcMethod::Grid),
            incident_medium: Medium::Air,
            transmitted_medium: Medium::Aluminium,
            emitter: Emitter {
                num_rays: 0,
                max_bounces: 3,
                radius: Auto(mm!(0.0)),
                zenith: RangeByStepSizeInclusive::zero_to_half_pi(rad!(0.2)),
                azimuth: RangeByStepSizeInclusive::zero_to_tau(rad!(0.4)),
                shape: RegionShape::SphericalCap { zenith: rad!(0.2) },
                spectrum: RangeByStepSizeInclusive::new(nm!(100.0), nm!(400.0), nm!(100.0)),
                solid_angle: Default::default(),
            },
            collector: Collector {
                radius: Auto(mm!(10.0)),
                scheme: CollectorScheme::Partitioned {
                    domain: SphericalDomain::Upper,
                    partition: SphericalPartition::EqualAngle {
                        zenith: RangeByStepSizeInclusive::zero_to_half_pi(Radians::HALF_PI),
                        azimuth: RangeByStepSizeInclusive::zero_to_tau(Radians::TAU),
                    },
                },
            },
        };
        let data2 = BsdfMeasurementDataPoint::<BounceAndEnergy>::read_from_buf(&buf, &params);
        assert_eq!(data, data2);
    }
}
