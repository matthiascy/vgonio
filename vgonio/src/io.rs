use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::Path,
};
use vgcore::{
    io::{CompressionScheme, FileEncoding},
    math,
};

use crate::measure::{
    bsdf::MeasuredBsdfData,
    microfacet::{MeasuredAdfData, MeasuredMsfData},
};

pub struct MeasurementOutputHeader {}

pub mod vgmo {
    use super::*;
    use crate::{
        measure::{
            bsdf::{
                emitter::EmitterParams,
                receiver::{BounceAndEnergy, ReceiverParams, ReceiverScheme},
                BsdfKind, BsdfMeasurementStatsPoint, BsdfSnapshotRaw, PerWavelength,
            },
            data::{MeasuredData, MeasurementData, MeasurementDataSource},
            params::{
                AdfMeasurementParams, BsdfMeasurementParams, MeasurementKind, MsfMeasurementParams,
                SimulationKind,
            },
        },
        Medium, RangeByStepCountInclusive, RangeByStepSizeInclusive,
    };
    use std::io::BufWriter;
    use vgcore::{
        error::VgonioError,
        io::{
            CompressionScheme, FileEncoding, ReadFileError, ReadFileErrorKind, WriteFileError,
            WriteFileErrorKind,
        },
        math::Sph2,
        units::{mm, rad, Nanometres, Radians},
    };

    macro_rules! impl_range_by_step_size_inclusive_read_write {
        ($($T:ty, $step_count:ident);*) => {
            $(paste::paste! {
                impl RangeByStepSizeInclusive<$T> {
                    #[doc = "Writes the RangeByStepSizeInclusive<`" $T "`> into the given buffer, following the order: start, stop, step_size, step_count."]
                    pub fn write_to_buf(&self, buf: &mut [u8]) {
                        debug_assert!(buf.len() >= 16, "RangeByStepSizeInclusive needs at least 16 bytes of space");
                        buf[0..4].copy_from_slice(&self.start.value().to_le_bytes());
                        buf[4..8].copy_from_slice(&self.stop.value().to_le_bytes());
                        buf[8..12].copy_from_slice(&self.step_size.value().to_le_bytes());
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
            buf[0..4].copy_from_slice(&self.start.value().to_le_bytes());
            buf[4..8].copy_from_slice(&self.stop.value().to_le_bytes());

            buf[8..12].copy_from_slice(&self.step_size().value().to_le_bytes());
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
                math::ulp_eq(range.step_size().value(), step_size.value()),
                "RangeByStepCountInclusive<Radians> step size mismatch: expected {}, got {}",
                range.step_size().value(),
                step_size.value()
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
            madf: AdfMeasurementParams,
        },
        Mmsf {
            meta: HeaderMeta,
            mmsf: MsfMeasurementParams,
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
                    bsdf.emitter.measurement_points_count()
                        * bsdf.emitter.spectrum.step_count()
                        * bsdf.receiver.patches_count()
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
                MeasurementKind::Adf => Ok(Self::Madf {
                    meta,
                    madf: AdfMeasurementParams::read_from_vgmo(reader)?,
                }),
                MeasurementKind::Msf => Ok(Self::Mmsf {
                    meta,
                    mmsf: MsfMeasurementParams::read_from_vgmo(reader)?,
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

    impl AdfMeasurementParams {
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

    impl MsfMeasurementParams {
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
                strict: true,
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

    impl EmitterParams {
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
            // NOTE: because of removal of RegionShape which takes 20 bytes, the offset is
            // 64 instead of 44
            let spectrum = RangeByStepSizeInclusive::<Nanometres>::read_from_buf(&buf[64..64 + 16]);
            Self {
                num_rays,
                max_bounces,
                zenith,
                azimuth,
                spectrum,
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

            // TODO: (radius removed) -- write other fields
            // buf[8..12].copy_from_slice(&self.radius.value().as_f32().to_le_bytes());
            self.zenith.write_to_buf(&mut buf[12..12 + 16]);
            self.azimuth.write_to_buf(&mut buf[28..28 + 16]);
            // TODO: (region shape removed) -- write other fields
            // self.shape.write_to_buf(&mut buf[44..44 + 20]);
            self.spectrum.write_to_buf(&mut buf[64..64 + 16]);
        }
    }

    impl ReceiverScheme {
        /// The size of the buffer required to write the collector scheme.
        pub const REQUIRED_SIZE: usize = 56; // TODO: shrink this

        /// Reads the collector scheme from a buffer.
        pub fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "CollectorScheme needs at least 60 bytes of space"
            );
            match u32::from_le_bytes(buf[0..4].try_into().unwrap()) {
                0x00 => ReceiverScheme::Beckers,
                0x01 => ReceiverScheme::Tregenza,
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
                ReceiverScheme::Beckers => {
                    buf[0..4].copy_from_slice(&(0x00u32).to_le_bytes());
                }
                ReceiverScheme::Tregenza => {
                    buf[0..4].copy_from_slice(&(0x01u32).to_le_bytes());
                }
            }
        }
    }

    impl ReceiverParams {
        /// The size of the buffer required to write the collector.
        pub const REQUIRED_SIZE: usize = 4 + ReceiverScheme::REQUIRED_SIZE;

        /// Reads the collector from a buffer.
        pub fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "Collector needs at least 64 bytes of space"
            );
            let precision = rad!(f32::from_le_bytes(buf[0..4].try_into().unwrap()));
            let scheme = ReceiverScheme::read_from_buf(&buf[4..4 + ReceiverScheme::REQUIRED_SIZE]);
            Self {
                domain: Default::default(),
                scheme,
                precision,
            }
        }

        /// Writes the collector to a buffer.
        pub fn write_to_buf(&self, buf: &mut [u8]) {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "Collector needs at least 64 bytes of space"
            );
            // TODO: (radius removed) -- write other fields
            self.scheme
                .write_to_buf(&mut buf[4..4 + ReceiverScheme::REQUIRED_SIZE]);
        }
    }

    impl BsdfMeasurementParams {
        /// Reads the BSDF measurement parameters from the given reader.
        pub fn read_from_vgmo<R: Read>(reader: &mut BufReader<R>) -> Result<Self, std::io::Error> {
            let mut buf =
                [0u8; ReceiverParams::REQUIRED_SIZE + EmitterParams::REQUIRED_SIZE + 4 + 12];
            reader.read_exact(&mut buf)?;
            let kind = BsdfKind::from(buf[0]);
            let incident_medium = Medium::from(buf[1]);
            let transmitted_medium = Medium::from(buf[2]);
            let sim_kind = SimulationKind::try_from(buf[3]).unwrap();
            let emitter = EmitterParams::read_from_buf(&buf[4..4 + EmitterParams::REQUIRED_SIZE]);
            let collector = ReceiverParams::read_from_buf(&buf[4 + EmitterParams::REQUIRED_SIZE..]);
            let samples_count = u32::from_le_bytes(
                buf[ReceiverParams::REQUIRED_SIZE + EmitterParams::REQUIRED_SIZE + 4
                    ..ReceiverParams::REQUIRED_SIZE + EmitterParams::REQUIRED_SIZE + 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(
                samples_count,
                collector.patches_count() as u32,
                "The number of samples in the VGMO file does not match the number of samples in \
                 the collector scheme"
            );
            Ok(Self {
                kind,
                incident_medium,
                transmitted_medium,
                sim_kind,
                emitter,
                receiver: collector,
            })
        }

        /// Writes the BSDF measurement parameters to the given writer.
        pub fn write_to_vgmo<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
        ) -> Result<(), WriteFileErrorKind> {
            let mut buf =
                [0x20; ReceiverParams::REQUIRED_SIZE + EmitterParams::REQUIRED_SIZE + 4 + 12];
            buf[0] = self.kind as u8;
            buf[1] = self.incident_medium as u8;
            buf[2] = self.transmitted_medium as u8;
            buf[3] = match self.sim_kind {
                SimulationKind::GeomOptics(method) => method as u8,
                SimulationKind::WaveOptics => 0x03,
            };
            self.emitter
                .write_to_buf(&mut buf[4..4 + EmitterParams::REQUIRED_SIZE]);
            self.receiver
                .write_to_buf(&mut buf[4 + EmitterParams::REQUIRED_SIZE..]);
            buf[4 + EmitterParams::REQUIRED_SIZE + ReceiverParams::REQUIRED_SIZE
                ..4 + EmitterParams::REQUIRED_SIZE + ReceiverParams::REQUIRED_SIZE + 4]
                .copy_from_slice(&(self.receiver.patches_count() as u32).to_le_bytes());
            buf[155] = 0x0A;
            writer.write_all(&buf).map_err(|err| err.into())
        }
    }

    impl MeasuredAdfData {
        /// Reads the measured MADF data from the given reader.
        pub fn read<R: Read>(
            reader: &mut BufReader<R>,
            meta: HeaderMeta,
            params: AdfMeasurementParams,
        ) -> Result<Self, ReadFileErrorKind> {
            debug_assert!(
                meta.kind == MeasurementKind::Adf,
                "Measurement kind
          mismatch"
            );
            let samples = vgsurf::io::read_f32_data_samples(
                reader,
                params.samples_count(),
                meta.encoding,
                meta.compression,
            )?;
            Ok(MeasuredAdfData { params, samples })
        }

        /// Writes the measured MADF data to the given writer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<(), WriteFileErrorKind> {
            // vgsurf::io::write_f32_data_samples(
            //     writer,
            //     encoding,
            //     compression,
            //     &self.samples,
            //     self.params.zenith.step_count_wrapped() as u32,
            // )
            // .map_err(|err| err.into())
            todo!("write MADF data")
        }
    }

    impl MeasuredMsfData {
        /// Reads the measured MMSF data from the given reader.
        pub fn read<R: Read>(
            reader: &mut BufReader<R>,
            meta: HeaderMeta,
            params: MsfMeasurementParams,
        ) -> Result<Self, ReadFileErrorKind> {
            debug_assert!(
                meta.kind == MeasurementKind::Msf,
                "Measurement kind
          mismatch"
            );
            let samples = vgsurf::io::read_f32_data_samples(
                reader,
                params.samples_count(),
                meta.encoding,
                meta.compression,
            )?;
            Ok(MeasuredMsfData { params, samples })
        }

        /// Writes the measured MMSF data to the given writer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<(), WriteFileErrorKind> {
            // vgsurf::io::write_f32_data_samples(
            //     writer,
            //     encoding,
            //     compression,
            //     &self.samples,
            //     self.params.zenith.step_count_wrapped() as u32,
            // )
            // .map_err(|err| err.into())
            todo!("write MMSF data")
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
                        Self::from_vec(data)
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
                num_rays_per_bounce: PerWavelength::from_vec(num_rays_per_bounce),
                energy_per_bounce: PerWavelength::from_vec(energy_per_bounce),
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
            for bounces in num_rays_per_bounce.iter_mut() {
                *bounces = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            let mut energy_per_bounce = vec![0f32; bounces];
            for energy in energy_per_bounce.iter_mut() {
                *energy = f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
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

    impl BsdfSnapshotRaw<BounceAndEnergy> {
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
            let detector_patches_count = params.receiver.patches_count();
            let size = Self::calc_size_in_bytes(n_wavelength, bounces, detector_patches_count);
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
                Vec::with_capacity(detector_patches_count);
            for i in 0..detector_patches_count {
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
                data.push(PerWavelength::from_vec(per_wavelength));
            }

            Self {
                w_i: Sph2::zero(), // TODO:
                stats,
                records: data,
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                trajectories: vec![],
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                hit_points: vec![],
            }
        }

        /// Writes a single data point to a buffer.
        pub fn write_to_buf(&self, buf: &mut [u8], n_wavelength: usize, bounces: usize) {
            // Write stats.
            self.stats.write_to_buf(buf, n_wavelength, bounces);
            let mut offset = BsdfMeasurementStatsPoint::calc_size_in_bytes(n_wavelength, bounces);
            let bounce_and_energy_size = BounceAndEnergy::calc_size_in_bytes(bounces);
            // Write collector's per wavelength patch data.
            for per_wavelength_patch_data in &self.records {
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
                _ => Box::new(reader),
            };

            match meta.encoding {
                FileEncoding::Ascii => {
                    todo!("Ascii encoding is not supported yet")
                }
                FileEncoding::Binary => {
                    let n_wavelength = params.emitter.spectrum.step_count();
                    let bounces = params.emitter.max_bounces as usize;
                    let detector_patches_count = params.receiver.patches_count();
                    let sample_size = BsdfSnapshotRaw::<BounceAndEnergy>::calc_size_in_bytes(
                        n_wavelength,
                        bounces,
                        detector_patches_count,
                    );
                    let sample_count = params.emitter.azimuth.step_count_wrapped()
                        * params.emitter.zenith.step_count_wrapped();
                    let mut buf = vec![0u8; sample_size];
                    let mut samples = Vec::with_capacity(sample_count);
                    (0..sample_count).for_each(|_| {
                        decoder.read_exact(&mut buf).unwrap();

                        samples.push(BsdfSnapshotRaw::read_from_buf(&buf, &params));
                    });

                    // Ok(Self { params, samples })
                    todo!("Fix this")
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
                _ => Box::new(writer),
            };

            match encoding {
                FileEncoding::Ascii => {
                    todo!("Ascii encoding is not supported yet")
                }
                FileEncoding::Binary => {
                    for sample in &self.snapshots {
                        let n_wavelength = self.params.emitter.spectrum.step_count();
                        let bounces = self.params.emitter.max_bounces as usize;
                        let detector_patches_count = self.params.receiver.patches_count();
                        let sample_size = BsdfSnapshotRaw::<BounceAndEnergy>::calc_size_in_bytes(
                            n_wavelength,
                            bounces,
                            detector_patches_count,
                        );
                        let mut buf = vec![0u8; sample_size];
                        // sample.write_to_buf(&mut buf, n_wavelength, bounces);
                        // encoder.write_all(&buf)?;
                        todo!("Fix this")
                    }
                    Ok(())
                }
            }
        }
    }

    impl MeasurementData {
        /// Loads the measurement data from a file.
        pub fn read_from_file(filepath: &Path) -> Result<Self, VgonioError> {
            let file = File::open(filepath).map_err(|err| {
                VgonioError::from_io_error(err, "Failed to open measurement file.")
            })?;
            let mut reader = BufReader::new(file);
            let header = Header::read(&mut reader).map_err(|err| {
                VgonioError::from_io_error(err, "Failed to read header from file.")
            })?;
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
                            VgonioError::from_read_file_error(
                                ReadFileError {
                                    path: filepath.to_owned().into_boxed_path(),
                                    kind: err,
                                },
                                "Failed to read measured data from file.",
                            )
                        })?;
                    Ok(MeasurementData {
                        name,
                        source: MeasurementDataSource::Loaded(path),
                        measured: MeasuredData::Bsdf(measured),
                    })
                }
                Header::Madf { meta, madf } => {
                    let measured =
                        MeasuredAdfData::read(&mut reader, meta, madf).map_err(|err| {
                            VgonioError::from_read_file_error(
                                ReadFileError {
                                    path: filepath.to_owned().into_boxed_path(),
                                    kind: err,
                                },
                                "Failed to read measured area distribution data from file.",
                            )
                        })?;
                    Ok(MeasurementData {
                        name,
                        source: MeasurementDataSource::Loaded(path),
                        measured: MeasuredData::Adf(measured),
                    })
                }
                Header::Mmsf { meta, mmsf } => {
                    let measured =
                        MeasuredMsfData::read(&mut reader, meta, mmsf).map_err(|err| {
                            VgonioError::from_read_file_error(
                                ReadFileError {
                                    path: filepath.to_owned().into_boxed_path(),
                                    kind: err,
                                },
                                "Failed to read measured masking/shadowing data from file.",
                            )
                        })?;
                    Ok(MeasurementData {
                        name,
                        source: MeasurementDataSource::Loaded(path),
                        measured: MeasuredData::Msf(measured),
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
        ) -> Result<(), VgonioError> {
            let header = match &self.measured {
                MeasuredData::Adf(adf) => {
                    assert_eq!(
                        adf.samples.len(),
                        adf.params.samples_count(),
                        "Writing a ADF requires the number of
          samples to match the number of bins."
                    );
                    Header::Madf {
                        meta: HeaderMeta {
                            kind: MeasurementKind::Adf,
                            encoding,
                            compression,
                        },
                        madf: adf.params,
                    }
                }
                MeasuredData::Msf(msf) => {
                    assert_eq!(
                        msf.samples.len(),
                        msf.params.samples_count(),
                        "Writing a MSF requires the number of
          samples to match the number of bins."
                    );
                    Header::Mmsf {
                        meta: HeaderMeta {
                            kind: MeasurementKind::Msf,
                            encoding,
                            compression,
                        },
                        mmsf: msf.params,
                    }
                }
                MeasuredData::Bsdf(bsdf) => {
                    assert_eq!(
                        bsdf.snapshots.len(),
                        bsdf.params.samples_count(),
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
                .open(filepath)
                .map_err(|err| {
                    VgonioError::from_io_error(err, "Failed to open measurement file.")
                })?;
            let mut writer = BufWriter::new(file);
            header.write(&mut writer).map_err(|err| {
                VgonioError::from_write_file_error(
                    WriteFileError {
                        path: filepath.to_owned().into_boxed_path(),
                        kind: err,
                    },
                    "Failed to write measurement data header to file.",
                )
            })?;

            match &self.measured {
                MeasuredData::Adf(madf) => {
                    madf.write(&mut writer, encoding, compression)
                        .map_err(|err| {
                            VgonioError::from_write_file_error(
                                WriteFileError {
                                    path: filepath.to_owned().into_boxed_path(),
                                    kind: err,
                                },
                                "Failed to write measured area distribution data to file.",
                            )
                        })
                }
                MeasuredData::Msf(mmsf) => {
                    mmsf.write(&mut writer, encoding, compression)
                        .map_err(|err| {
                            VgonioError::from_write_file_error(
                                WriteFileError {
                                    path: filepath.to_owned().into_boxed_path(),
                                    kind: err,
                                },
                                "Failed to write measured masking/shadowing data to file.",
                            )
                        })
                }
                MeasuredData::Bsdf(bsdf) => {
                    bsdf.write(&mut writer, encoding, compression)
                        .map_err(|err| {
                            VgonioError::from_write_file_error(
                                WriteFileError {
                                    path: filepath.to_owned().into_boxed_path(),
                                    kind: err,
                                },
                                "Failed to write measured BSDF data to file.",
                            )
                        })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        measure::{
            bsdf::{
                emitter::{Emitter, EmitterParams},
                receiver::{
                    BounceAndEnergy, Receiver, ReceiverParams, ReceiverPatches, ReceiverScheme,
                },
                rtc::RtcMethod,
                BsdfKind, BsdfMeasurementStatsPoint, BsdfSnapshotRaw, PerWavelength,
            },
            params::{BsdfMeasurementParams, SimulationKind},
        },
        Medium, RangeByStepSizeInclusive,
    };
    use vgcore::{
        math::Sph2,
        units::{mm, nm, rad, Radians},
    };

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
        let data = BsdfSnapshotRaw::<BounceAndEnergy> {
            w_i: Sph2::zero(),
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
            records: vec![
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
            #[cfg(debug_assertions)]
            trajectories: vec![],
            #[cfg(debug_assertions)]
            hit_points: vec![],
        };

        let size = BsdfSnapshotRaw::<BounceAndEnergy>::calc_size_in_bytes(4, 3, 2);
        let mut buf = vec![0; size];
        data.write_to_buf(&mut buf, 4, 3);
        let params = BsdfMeasurementParams {
            kind: BsdfKind::Brdf,
            sim_kind: SimulationKind::GeomOptics(RtcMethod::Grid),
            incident_medium: Medium::Air,
            transmitted_medium: Medium::Aluminium,
            emitter: EmitterParams {
                num_rays: 0,
                max_bounces: 3,
                zenith: RangeByStepSizeInclusive::zero_to_half_pi(rad!(0.2)),
                azimuth: RangeByStepSizeInclusive::zero_to_tau(rad!(0.4)),
                spectrum: RangeByStepSizeInclusive::new(nm!(100.0), nm!(400.0), nm!(100.0)),
            },
            receiver: ReceiverParams {
                domain: Default::default(),
                precision: rad!(0.1),
                scheme: ReceiverScheme::Beckers,
            },
        };
        let data2 = BsdfSnapshotRaw::<BounceAndEnergy>::read_from_buf(&buf, &params);
        assert_eq!(data, data2);
    }
}
