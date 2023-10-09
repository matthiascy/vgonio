use std::io::{BufReader, Read, Write};
use vgcore::math;

use crate::measure::{
    bsdf::MeasuredBsdfData,
    microfacet::{MeasuredAdfData, MeasuredMsfData},
};

pub mod vgmo {
    use super::*;
    use crate::{
        measure::{
            bsdf::{
                emitter::EmitterParams,
                receiver::{
                    BounceAndEnergy, DataRetrievalMode, ReceiverParams, ReceiverScheme, Ring,
                },
                BsdfKind, BsdfMeasurementStatsPoint, BsdfSnapshot, BsdfSnapshotRaw,
                SpectralSamples,
            },
            data::MeasuredData,
            params::{
                AdfMeasurementParams, BsdfMeasurementParams, MeasurementKind, MsfMeasurementParams,
                SimulationKind,
            },
        },
        Medium, RangeByStepCountInclusive, RangeByStepSizeInclusive, SphericalDomain,
    };
    use std::io::{BufWriter, Seek};
    use vgcore::{
        io::{
            CompressionScheme, FileEncoding, Header, HeaderExt, ParseError, ReadFileError,
            ReadFileErrorKind, VgonioFileVariant, WriteFileErrorKind,
        },
        math::Sph2,
        units::{rad, Nanometres, Radians},
        Version,
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

    /// The VGMO header extension.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum VgmoHeaderExt {
        Bsdf { params: BsdfMeasurementParams },
        Adf { params: AdfMeasurementParams },
        Msf { params: MsfMeasurementParams },
    }

    impl MeasuredData {
        /// Returns the [`VgmoHeaderExt`] variant corresponding to the
        /// measurement data.
        pub fn as_vgmo_header_ext(&self) -> VgmoHeaderExt {
            match self {
                Self::Bsdf(bsdf) => VgmoHeaderExt::Bsdf {
                    params: bsdf.params.clone(),
                },
                Self::Adf(adf) => VgmoHeaderExt::Adf {
                    params: adf.params.clone(),
                },
                Self::Msf(msf) => VgmoHeaderExt::Msf {
                    params: msf.params.clone(),
                },
            }
        }
    }

    impl HeaderExt for VgmoHeaderExt {
        const MAGIC: &'static [u8; 4] = &b"VGMO";

        fn variant() -> VgonioFileVariant { VgonioFileVariant::Vgmo }

        /// Writes the VGMO header extension to the given writer.
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
                } => match self {
                    Self::Adf { params } => {
                        writer.write_all(&[MeasurementKind::Adf as u8])?;
                        write_adf_or_msf_params_to_vgmo(
                            &params.azimuth,
                            &params.zenith,
                            writer,
                            true,
                        )?;
                    }
                    Self::Msf { params } => {
                        writer.write_all(&[MeasurementKind::Msf as u8])?;
                        write_adf_or_msf_params_to_vgmo(
                            &params.azimuth,
                            &params.zenith,
                            writer,
                            false,
                        )?;
                    }
                    Self::Bsdf { params } => {
                        writer.write_all(&[MeasurementKind::Bsdf as u8])?;
                        params.write_to_vgmo(version, writer)?;
                    }
                },
                _ => {
                    log::error!("Unsupported VGMO version: {}", version.as_string());
                }
            }
            Ok(())
        }

        /// Reads the VGMO header extension from the given reader.
        /// The reader must be positioned at the start of the header extension.
        fn read<R: Read + Seek>(
            version: Version,
            reader: &mut BufReader<R>,
        ) -> std::io::Result<Self> {
            let mut kind = [0u8; 1];
            reader.read_exact(&mut kind)?;
            match MeasurementKind::from(kind[0]) {
                MeasurementKind::Bsdf => {
                    let params = BsdfMeasurementParams::read_from_vgmo(version, reader)?;
                    Ok(Self::Bsdf { params })
                }
                MeasurementKind::Adf => {
                    let (azimuth, zenith) = read_adf_or_msf_params_from_vgmo(reader, true)?;
                    Ok(Self::Adf {
                        params: AdfMeasurementParams { azimuth, zenith },
                    })
                }
                MeasurementKind::Msf => {
                    let (azimuth, zenith) = read_adf_or_msf_params_from_vgmo(reader, false)?;
                    Ok(Self::Msf {
                        params: MsfMeasurementParams {
                            azimuth,
                            zenith,
                            resolution: 512,
                            strict: true,
                        },
                    })
                }
            }
        }
    }

    /// Reads the VGMO file from the given reader.
    pub fn read<R: Read>(
        reader: &mut BufReader<R>,
        header: &Header<VgmoHeaderExt>,
    ) -> Result<MeasuredData, ReadFileErrorKind> {
        match header.extra {
            VgmoHeaderExt::Bsdf { params } => {
                log::debug!(
                    "Reading BSDF data of {} measurement points {} samples from VGMO file",
                    params.emitter.measurement_points_count(),
                    params.samples_count()
                );
                Ok(MeasuredData::Bsdf(MeasuredBsdfData::read_from_vgmo(
                    reader,
                    &params,
                    header.meta.encoding,
                    header.meta.compression,
                )?))
            }
            VgmoHeaderExt::Adf { params } => {
                log::debug!(
                    "Reading ADF data of {} samples from VGMO file",
                    params.samples_count()
                );
                let samples = vgcore::io::read_f32_data_samples(
                    reader,
                    params.samples_count(),
                    header.meta.encoding,
                    header.meta.compression,
                )
                .map_err(|err| ReadFileErrorKind::Parse(err))?;
                Ok(MeasuredData::Adf(MeasuredAdfData { params, samples }))
            }
            VgmoHeaderExt::Msf { params } => {
                log::debug!(
                    "Reading MSF data of {} samples from VGMO file",
                    params.samples_count()
                );
                let samples = vgcore::io::read_f32_data_samples(
                    reader,
                    params.samples_count(),
                    header.meta.encoding,
                    header.meta.compression,
                )
                .map_err(|err| ReadFileErrorKind::Parse(err))?;
                Ok(MeasuredData::Msf(MeasuredMsfData { params, samples }))
            }
        }
    }

    /// Writes the given measurement data to the given writer.
    pub fn write<W: Write + Seek>(
        writer: &mut BufWriter<W>,
        header: Header<VgmoHeaderExt>,
        measured: &MeasuredData,
    ) -> Result<(), WriteFileErrorKind> {
        let init_size = writer.stream_len().unwrap();
        header.write(writer)?;

        match measured {
            MeasuredData::Adf(_) | MeasuredData::Msf(_) => {
                let samples = measured.adf_or_msf_samples().unwrap();
                log::debug!("Writing {} samples to VGMO file", samples.len());
                match header.meta.encoding {
                    FileEncoding::Ascii => {
                        let cols = if let MeasuredData::Adf(adf) = &measured {
                            adf.params.zenith.step_count_wrapped()
                        } else {
                            measured.msf().unwrap().params.zenith.step_count_wrapped()
                        };
                        vgcore::io::write_data_samples_ascii(writer, samples, cols as u32)
                    }
                    FileEncoding::Binary => vgcore::io::write_f32_data_samples_binary(
                        writer,
                        header.meta.compression,
                        &samples,
                    ),
                }
                .map_err(|err| WriteFileErrorKind::Write(err))?;
            }
            MeasuredData::Bsdf(bsdf) => {
                bsdf.write_to_vgmo(writer, header.meta.encoding, header.meta.compression)?;
            }
        }

        let length = writer.stream_len().unwrap() - init_size;
        writer.seek(std::io::SeekFrom::Start(
            Header::<VgmoHeaderExt>::length_pos() as u64,
        ))?;
        writer.write_all(&(length as u32).to_le_bytes())?;
        log::debug!("Wrote {} bytes of data to VGMO file", length);
        Ok(())
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

    fn read_adf_or_msf_params_from_vgmo<R: Read>(
        reader: &mut BufReader<R>,
        is_adf: bool,
    ) -> Result<
        (
            RangeByStepSizeInclusive<Radians>,
            RangeByStepSizeInclusive<Radians>,
        ),
        std::io::Error,
    > {
        let mut buf = [0u8; 36];
        reader.read_exact(&mut buf)?;
        let azimuth = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[0..16]);
        let zenith = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[16..32]);
        let sample_count = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        debug_assert_eq!(
            sample_count as usize,
            madf_or_mmsf_samples_count(&azimuth, &zenith, is_adf)
        );
        Ok((azimuth, zenith))
    }

    /// Writes the ADF and MSF measurement parameters to the VGMO file.
    fn write_adf_or_msf_params_to_vgmo<W: Write>(
        azimuth: &RangeByStepSizeInclusive<Radians>,
        zenith: &RangeByStepSizeInclusive<Radians>,
        writer: &mut BufWriter<W>,
        is_madf: bool,
    ) -> Result<(), std::io::Error> {
        let mut header = [0u8; 36];
        azimuth.write_to_buf(&mut header[0..16]);
        zenith.write_to_buf(&mut header[16..32]);
        header[32..36].copy_from_slice(
            &(madf_or_mmsf_samples_count(zenith, azimuth, is_madf) as u32).to_le_bytes(),
        );
        writer.write_all(&header)
    }

    impl EmitterParams {
        /// The size of the buffer required to read or write the parameters.
        pub const fn required_size(version: Version) -> Option<usize> {
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => Some(56),
                _ => None,
            }
        }

        /// Reads an emitter from the given buffer.
        pub fn read_from_buf(version: Version, buf: &[u8]) -> Self {
            let required_size = Self::required_size(version).unwrap();
            debug_assert!(
                buf.len() >= required_size,
                "Emitter {} needs at least {} bytes of space",
                version,
                required_size
            );
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    let num_rays = u32::from_le_bytes(buf[0..4].try_into().unwrap());
                    let max_bounces = u32::from_le_bytes(buf[4..8].try_into().unwrap());
                    let azimuth = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[8..24]);
                    let zenith = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[24..40]);
                    let spectrum =
                        RangeByStepSizeInclusive::<Nanometres>::read_from_buf(&buf[40..56]);
                    Self {
                        num_rays,
                        max_bounces,
                        zenith,
                        azimuth,
                        spectrum,
                    }
                }
                _ => {
                    panic!("Unsupported VGMO[EmitterPrams] version: {}", version);
                }
            }
        }

        /// Writes the emitter to the given buffer and returns the number of
        /// bytes written.
        pub fn write_to_buf(&self, version: Version, buf: &mut [u8]) -> usize {
            let required_size = Self::required_size(version).unwrap();
            debug_assert!(
                buf.len() >= required_size,
                "Emitter {} needs at least {} bytes of space",
                version,
                required_size
            );
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    buf[0..4].copy_from_slice(&self.num_rays.to_le_bytes());
                    buf[4..8].copy_from_slice(&self.max_bounces.to_le_bytes());
                    self.azimuth.write_to_buf(&mut buf[8..24]);
                    self.zenith.write_to_buf(&mut buf[24..40]);
                    self.spectrum.write_to_buf(&mut buf[40..56]);
                    required_size
                }
                _ => {
                    panic!("Unsupported VGMO[EmitterParams] version: {}", version);
                }
            }
        }
    }

    impl Ring {
        /// The size of the buffer required to read or write the parameters.
        const REQUIRED_SIZE: usize = 20;

        /// Writes the ring to the given buffer.
        pub fn write_to_buf(&self, buf: &mut [u8]) {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "Ring needs at least 20 bytes of space"
            );
            buf[0..4].copy_from_slice(&(self.theta_inner).to_le_bytes());
            buf[4..8].copy_from_slice(&(self.theta_outer).to_le_bytes());
            buf[8..12].copy_from_slice(&(self.phi_step).to_le_bytes());
            buf[12..16].copy_from_slice(&(self.patch_count as u32).to_le_bytes());
            buf[16..20].copy_from_slice(&(self.base_index as u32).to_le_bytes());
        }

        /// Reads the ring from the given buffer.
        pub fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(
                buf.len() >= Self::REQUIRED_SIZE,
                "Ring needs at least 20 bytes of space"
            );
            let theta_inner = f32::from_le_bytes(buf[0..4].try_into().unwrap());
            let theta_outer = f32::from_le_bytes(buf[4..8].try_into().unwrap());
            let phi_step = f32::from_le_bytes(buf[8..12].try_into().unwrap());
            let patch_count = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
            let base_index = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
            Self {
                theta_inner,
                theta_outer,
                phi_step,
                patch_count,
                base_index,
            }
        }
    }

    impl ReceiverParams {
        /// The size of the buffer required to read or write the parameters.
        pub const fn required_size(version: Version, num_rings: usize) -> Option<usize> {
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => Some(num_rings * Ring::REQUIRED_SIZE + 24),
                _ => None,
            }
        }

        /// Reads the receiver.
        ///
        /// Because the receiver's partition is dependent on the precision, the
        /// we need to know read the precision to know the actual size of the
        /// receiver.
        pub fn read<R: Read + Seek>(version: Version, reader: &mut BufReader<R>) -> Self {
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    let mut buf = [0u8; 24];
                    reader.read_exact(&mut buf).unwrap();
                    let domain = match u32::from_le_bytes(buf[0..4].try_into().unwrap()) {
                        0 => SphericalDomain::Whole,
                        1 => SphericalDomain::Upper,
                        2 => SphericalDomain::Lower,
                        _ => panic!("Invalid domain kind"),
                    };
                    let scheme = match u32::from_le_bytes(buf[4..8].try_into().unwrap()) {
                        0 => ReceiverScheme::Beckers,
                        1 => ReceiverScheme::Tregenza,
                        _ => panic!("Invalid scheme kind"),
                    };
                    let precision = rad!(f32::from_le_bytes(buf[8..12].try_into().unwrap()));
                    let retrieval_mode = DataRetrievalMode::from(u32::from_le_bytes(
                        buf[20..24].try_into().unwrap(),
                    ) as u8);
                    let params = ReceiverParams {
                        domain,
                        precision,
                        scheme,
                        retrieval_mode,
                    };
                    let expected_num_rings = params.num_rings();
                    let num_rings = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
                    debug_assert!(
                        num_rings == expected_num_rings,
                        "Receiver's partition ring count does not match the precision",
                    );
                    let expected_num_patches = params.num_patches();
                    let num_patches = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
                    debug_assert!(
                        num_patches == expected_num_patches,
                        "Receiver's partition patch count does not match the precision",
                    );
                    // Skip reading the rings
                    reader
                        .seek_relative((num_rings * Ring::REQUIRED_SIZE) as i64)
                        .unwrap();
                    params
                }
                _ => {
                    panic!("Unsupported VGMO version: {}", version);
                }
            }
        }

        /// Writes the collector to a buffer.
        pub fn write_to_buf(&self, version: Version, buf: &mut [u8]) {
            let required_size = Self::required_size(version, self.num_rings()).unwrap();
            debug_assert!(
                buf.len() >= required_size,
                "Receiver {} needs at least {} bytes of space",
                version,
                required_size
            );
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    let partition = self.generate_patches();
                    buf[0..4].copy_from_slice(&(self.domain as u32).to_le_bytes());
                    buf[4..8].copy_from_slice(&(self.scheme as u32).to_le_bytes());
                    buf[8..12].copy_from_slice(&self.precision.value().to_le_bytes());
                    buf[12..16].copy_from_slice(&(partition.num_rings() as u32).to_le_bytes());
                    buf[16..20].copy_from_slice(&(partition.num_patches() as u32).to_le_bytes());
                    buf[20..24].copy_from_slice(&(self.retrieval_mode as u32).to_le_bytes());
                    let offset = 24;
                    for (i, ring) in partition.rings.iter().enumerate() {
                        ring.write_to_buf(&mut buf[offset + i * Ring::REQUIRED_SIZE..]);
                    }
                }
                _ => {
                    log::error!("Unsupported VGMO[ReceiverParams] version: {version}");
                }
            }
        }
    }

    impl Medium {
        fn write_to_buf(&self, buf: &mut [u8]) {
            debug_assert!(buf.len() >= 3, "Medium needs at least 3 bytes of space");
            match self {
                Self::Vacuum => buf[0..3].copy_from_slice(b"vac"),
                Self::Air => buf[0..3].copy_from_slice(b"air"),
                Self::Aluminium => buf[0..3].copy_from_slice(b"al\0"),
                Self::Copper => buf[0..3].copy_from_slice(b"cu\0"),
            }
        }

        fn read_from_buf(buf: &[u8]) -> Self {
            debug_assert!(buf.len() >= 3, "Medium needs at least 3 bytes of space");
            match &buf[0..3] {
                b"vac" => Self::Vacuum,
                b"air" => Self::Air,
                b"al\0" => Self::Aluminium,
                b"cu\0" => Self::Copper,
                _ => panic!("Invalid medium kind {:?}", &buf[0..3]),
            }
        }
    }

    impl BsdfMeasurementParams {
        /// Reads the BSDF measurement parameters from the given reader.
        pub fn read_from_vgmo<R: Read + Seek>(
            version: Version,
            reader: &mut BufReader<R>,
        ) -> Result<Self, std::io::Error> {
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    let mut buf = vec![0u8; 8 + EmitterParams::required_size(version).unwrap()]
                        .into_boxed_slice();
                    reader.read_exact(&mut buf)?;
                    let kind = BsdfKind::from(buf[0]);
                    let incident_medium = Medium::read_from_buf(&buf[1..4]);
                    let transmitted_medium = Medium::read_from_buf(&buf[4..7]);
                    let sim_kind = SimulationKind::try_from(buf[7]).unwrap();
                    let emitter = EmitterParams::read_from_buf(version, &buf[8..]);
                    let receiver = ReceiverParams::read(version, reader);
                    Ok(Self {
                        kind,
                        incident_medium,
                        transmitted_medium,
                        sim_kind,
                        emitter,
                        receiver,
                    })
                }
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Unsupported VGMO[BsdfMeasurementParams] version {}",
                        version
                    ),
                )),
            }
        }

        /// Writes the BSDF measurement parameters to the given writer.
        pub fn write_to_vgmo<W: Write>(
            &self,
            version: Version,
            writer: &mut BufWriter<W>,
        ) -> Result<(), std::io::Error> {
            let buf_size = match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    8 + EmitterParams::required_size(version).unwrap()
                        + ReceiverParams::required_size(version, self.receiver.num_rings()).unwrap()
                }
                _ => {
                    log::error!("Unsupported VGMO version: {}", version.as_string());
                    return Ok(());
                }
            };
            let mut buf = vec![0u8; buf_size].into_boxed_slice();
            buf[0] = self.kind as u8;
            self.incident_medium.write_to_buf(&mut buf[1..4]);
            self.transmitted_medium.write_to_buf(&mut buf[4..7]);
            buf[7] = self.sim_kind.as_u8();
            let n_written = self.emitter.write_to_buf(version, &mut buf[8..]);
            self.receiver
                .write_to_buf(version, &mut buf[n_written + 8..]);
            writer.write_all(&buf)
        }
    }

    macro_rules! impl_per_wavelength_data_io {
        ($($t:ty),*) => {
            $(
                impl SpectralSamples<$t> {
                    /// The size of a single element in bytes.
                    pub const ELEM_SIZE: usize = std::mem::size_of::<$t>();

                    /// Writes the data to the given buffer.
                    pub fn write_to_buf(&self, buf: &mut [u8]) {
                        debug_assert!(buf.len() >= self.len() * Self::ELEM_SIZE, "Buffer too small desired {}, got {}", self.len() * Self::ELEM_SIZE, buf.len());
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

    impl_per_wavelength_data_io!(f32, u32);

    impl BsdfMeasurementStatsPoint {
        /// The size of the buffer required to read/write the BSDF measurement
        /// stats point.
        pub fn size_in_bytes(n_wavelengths: usize, n_bounces: usize) -> usize {
            4 * 2 + n_wavelengths * 4 * 4 + n_wavelengths * n_bounces * 4 * 2
        }

        /// Writes the BSDF measurement statistics at a single point to the
        /// writer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            n_wavelengths: usize,
        ) -> Result<(), std::io::Error> {
            let size = Self::size_in_bytes(n_wavelengths, self.n_bounces as usize);
            let mut buf = vec![0u8; size].into_boxed_slice();
            let mut offset = 0;
            buf[offset..offset + 4].copy_from_slice(&self.n_bounces.to_le_bytes());
            offset += 4;
            buf[offset..offset + 4].copy_from_slice(&self.n_received.to_le_bytes());
            offset += 4;

            self.n_absorbed
                .write_to_buf(&mut buf[offset..offset + 4 * n_wavelengths]);
            offset += 4 * n_wavelengths;

            self.n_reflected
                .write_to_buf(&mut buf[offset..offset + 4 * n_wavelengths]);
            offset += 4 * n_wavelengths;

            self.n_captured
                .write_to_buf(&mut buf[offset..offset + 4 * n_wavelengths]);
            offset += 4 * n_wavelengths;

            self.e_captured
                .write_to_buf(&mut buf[offset..offset + 4 * n_wavelengths]);
            offset += 4 * n_wavelengths;

            for i in 0..n_wavelengths {
                for j in 0..self.n_bounces as usize {
                    buf[offset..offset + 4]
                        .copy_from_slice(&self.num_rays_per_bounce[i][j].to_le_bytes());
                    offset += 4;
                }
            }
            for i in 0..n_wavelengths {
                for j in 0..self.n_bounces as usize {
                    buf[offset..offset + 4]
                        .copy_from_slice(&self.energy_per_bounce[i][j].to_le_bytes());
                    offset += 4;
                }
            }
            writer.write_all(&buf)
        }

        /// Reads the BSDF measurement statistics at a single point from the
        /// buffer.
        pub fn read<R: Read>(reader: &mut R, n_wavelengths: usize) -> Result<Self, std::io::Error> {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            let n_bounces = u32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            let n_received = u32::from_le_bytes(buf);

            let mut buf =
                vec![0u8; 4 * n_wavelengths * 4 + 4 * n_wavelengths * n_bounces as usize * 2];
            reader.read_exact(&mut buf)?;

            let mut offset = 0;
            let n_absorbed = SpectralSamples::<u32>::read_from_buf(
                &buf[offset..offset + 4 * n_wavelengths],
                n_wavelengths,
            );
            offset += 4 * n_wavelengths;
            let n_reflected = SpectralSamples::<u32>::read_from_buf(
                &buf[offset..offset + 4 * n_wavelengths],
                n_wavelengths,
            );
            offset += 4 * n_wavelengths;
            let n_captured = SpectralSamples::<u32>::read_from_buf(
                &buf[offset..offset + 4 * n_wavelengths],
                n_wavelengths,
            );
            offset += 4 * n_wavelengths;

            let e_captured = SpectralSamples::<f32>::read_from_buf(
                &buf[offset..offset + 4 * n_wavelengths],
                n_wavelengths,
            );
            offset += 4 * n_wavelengths;

            let mut num_rays_per_bounce = vec![vec![0u32; n_bounces as usize]; n_wavelengths];
            for i in 0..n_wavelengths {
                for j in 0..n_bounces as usize {
                    num_rays_per_bounce[i][j] =
                        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                    offset += 4;
                }
            }

            let mut energy_per_bounce = vec![vec![0f32; n_bounces as usize]; n_wavelengths];
            for i in 0..n_wavelengths {
                for j in 0..n_bounces as usize {
                    energy_per_bounce[i][j] =
                        f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                    offset += 4;
                }
            }

            Ok(Self {
                n_bounces,
                n_received,
                n_absorbed,
                n_reflected,
                n_captured,
                e_captured,
                num_rays_per_bounce: SpectralSamples::from_vec(num_rays_per_bounce),
                energy_per_bounce: SpectralSamples::from_vec(energy_per_bounce),
            })
        }
    }

    impl BounceAndEnergy {
        /// Calculates the size of a single data point in bytes.
        pub fn size_in_bytes(n_bounces: usize) -> usize { 4 * 3 + n_bounces * 4 * 2 }

        /// Reads the data for one single patch.
        pub fn read<R: Read>(reader: &mut R) -> Result<Self, std::io::Error> {
            let n_bounces = {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                u32::from_le_bytes(buf.try_into().unwrap())
            };
            let size_without_n_bounces = 4 * 2 + n_bounces as usize * 4 * 2;
            let mut buf = vec![0u8; size_without_n_bounces].into_boxed_slice();
            reader.read_exact(&mut buf)?;
            let total_rays = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            let total_energy = f32::from_le_bytes(buf[4..8].try_into().unwrap());
            let mut offset = 8;
            let mut num_rays_per_bounce = vec![0u32; n_bounces as usize];
            for i in 0..n_bounces as usize {
                num_rays_per_bounce[i] =
                    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            let mut energy_per_bounce = vec![0f32; n_bounces as usize];
            for i in 0..n_bounces as usize {
                energy_per_bounce[i] =
                    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            Ok(Self {
                n_bounces,
                total_rays,
                total_energy,
                num_rays_per_bounce,
                energy_per_bounce,
            })
        }

        /// Writes the data for one single patch.
        pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> Result<(), std::io::Error> {
            let mut buf =
                vec![0u8; Self::size_in_bytes(self.n_bounces as usize)].into_boxed_slice();
            buf[0..4].copy_from_slice(&self.n_bounces.to_le_bytes());
            buf[4..8].copy_from_slice(&self.total_rays.to_le_bytes());
            buf[8..12].copy_from_slice(&self.total_energy.to_le_bytes());
            let mut offset = 12;
            for i in 0..self.n_bounces as usize {
                buf[offset..offset + 4].copy_from_slice(&self.num_rays_per_bounce[i].to_le_bytes());
                offset += 4;
            }
            for i in 0..self.n_bounces as usize {
                buf[offset..offset + 4].copy_from_slice(&self.energy_per_bounce[i].to_le_bytes());
                offset += 4;
            }
            debug_assert!(
                offset == buf.len(),
                "Wrong offset: {}, buf len: {}",
                offset,
                buf.len()
            );
            writer.write_all(&buf)
        }
    }

    fn read_sph2_from_buf(buf: &[u8]) -> Sph2 {
        let theta = rad!(f32::from_le_bytes(buf[0..4].try_into().unwrap()));
        let phi = rad!(f32::from_le_bytes(buf[4..8].try_into().unwrap()));
        Sph2::new(theta, phi)
    }

    fn write_sph2_to_buf(sph2: Sph2, buf: &mut [u8]) {
        buf[0..4].copy_from_slice(&sph2.theta.as_f32().to_le_bytes());
        buf[4..8].copy_from_slice(&sph2.phi.as_f32().to_le_bytes());
    }

    impl BsdfSnapshotRaw<BounceAndEnergy> {
        /// Reads a single data point from the reader.
        pub fn read<R: Read>(
            reader: &mut R,
            n_wavelengths: usize,
            n_patches: usize,
        ) -> Result<Self, std::io::Error> {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            let w_i = read_sph2_from_buf(&buf);
            let stats = BsdfMeasurementStatsPoint::read(reader, n_wavelengths).unwrap();
            let mut records = Vec::with_capacity(n_patches);
            let mut samples = vec![BounceAndEnergy::default(); n_wavelengths];
            for _ in 0..n_patches {
                for j in 0..n_wavelengths {
                    samples[j] = BounceAndEnergy::read(reader)?;
                }
                records.push(SpectralSamples::from_vec(samples.clone()));
            }

            Ok(Self {
                w_i,
                stats,
                records,
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                trajectories: vec![],
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                hit_points: vec![],
            })
        }

        /// Writes a single data point to a buffer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            n_wavelengths: usize,
        ) -> Result<(), std::io::Error> {
            let mut buf = [0u8; 8];
            write_sph2_to_buf(self.w_i, &mut buf);
            writer.write_all(&buf)?;
            // Writes the stats for the current measurement point.
            self.stats.write(writer, n_wavelengths)?;
            // Writes the collected data for each patch.
            for samples in &self.records {
                // Writes the data for each wavelength.
                for s in samples.iter() {
                    s.write(writer)?;
                }
            }
            Ok(())
        }
    }

    impl BsdfSnapshot {
        /// Reads a BSDF snapshot from the given reader.
        ///
        /// The `samples_buf` is a buffer that is used to read the spectral
        /// samples for each patch. It must be at least `n_wavelengths * 4`
        pub fn read<R: Read>(
            reader: &mut R,
            n_wavelengths: usize,
            n_patches: usize,
            samples_buf: &mut [u8],
        ) -> Result<Self, std::io::Error> {
            assert_eq!(samples_buf.len(), n_wavelengths * 4, "Buffer too small");
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            let w_i = read_sph2_from_buf(&buf);
            let mut samples = vec![SpectralSamples::<f32>::new(); n_patches];
            for i in 0..n_patches {
                reader.read_exact(samples_buf)?;
                samples[i] = SpectralSamples::<f32>::read_from_buf(samples_buf, n_wavelengths);
            }
            Ok(Self {
                w_i,
                samples,
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                trajectories: vec![],
                #[cfg(any(feature = "visu-dbg", debug_assertions))]
                hit_points: vec![],
            })
        }

        /// Writes a BSDF snapshot to the given writer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            n_wavelengths: usize,
        ) -> Result<(), std::io::Error> {
            let mut buf = [0u8; 8];
            buf[0..4].copy_from_slice(&self.w_i.theta.as_f32().to_le_bytes());
            buf[4..8].copy_from_slice(&self.w_i.phi.as_f32().to_le_bytes());
            writer.write_all(&buf)?;
            let mut buf = vec![0u8; 4 * n_wavelengths].into_boxed_slice();
            for samples in &self.samples {
                samples.write_to_buf(&mut buf);
                writer.write_all(&buf)?;
            }
            Ok(())
        }
    }

    impl MeasuredBsdfData {
        /// Writes the BSDF data to the given writer.
        fn write_bsdf_snapshots<W: Write>(
            writer: &mut W,
            snapshots: &[BsdfSnapshot],
            n_wavelengths: usize,
        ) -> Result<(), std::io::Error> {
            let mut buf = vec![0u8; 4 * n_wavelengths].into_boxed_slice();
            for snapshot in snapshots {
                for samples in &snapshot.samples {
                    samples.write_to_buf(&mut buf);
                    writer.write_all(&buf)?;
                }
            }
            Ok(())
        }

        fn read_bsdf_snapshots<R: Read>(
            reader: &mut R,
            n_wavelengths: usize,
            n_patches: usize,
            n_snapshots: usize,
        ) -> Result<Vec<BsdfSnapshot>, std::io::Error> {
            let mut snapshots = Vec::with_capacity(n_snapshots);
            let mut spectral_samples_buf = vec![0u8; 4 * n_wavelengths].into_boxed_slice();
            for _ in 0..n_snapshots {
                snapshots.push(BsdfSnapshot::read(
                    reader,
                    n_wavelengths,
                    n_patches,
                    &mut spectral_samples_buf,
                )?);
            }
            Ok(snapshots)
        }

        /// Reads the measured BSDF data from the given reader.
        pub fn read_from_vgmo<R: Read>(
            reader: &mut BufReader<R>,
            params: &BsdfMeasurementParams,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<Self, ReadFileErrorKind> {
            match encoding {
                FileEncoding::Ascii => Err(ReadFileErrorKind::UnsupportedEncoding),
                FileEncoding::Binary => {
                    let mut zlib_decoder;
                    let mut gzip_decoder;
                    let mut decoder: Box<&mut dyn Read> = match compression {
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

                    let snapshots = Self::read_bsdf_snapshots(
                        &mut decoder,
                        params.emitter.spectrum.step_count(),
                        params.receiver.num_patches(),
                        params.emitter.measurement_points_count(),
                    )?;
                    let mut raw_snapshots = None;
                    if params.receiver.retrieval_mode == DataRetrievalMode::FullData {
                        let n_patches = params.receiver.num_patches();
                        let mut snapshots = Vec::with_capacity(n_patches);
                        for _ in 0..n_patches {
                            let snapshot = BsdfSnapshotRaw::<BounceAndEnergy>::read(
                                &mut decoder,
                                params.emitter.spectrum.step_count(),
                                params.receiver.num_patches(),
                            )?;
                            snapshots.push(snapshot);
                        }
                        raw_snapshots = Some(snapshots);
                    };

                    Ok(Self {
                        params: *params,
                        snapshots,
                        raw_snapshots,
                    })
                }
            }
        }

        /// Writes the measured BSDF data to the given writer.
        pub fn write_to_vgmo<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<(), WriteFileErrorKind> {
            if encoding == FileEncoding::Ascii {
                return Err(WriteFileErrorKind::UnsupportedEncoding);
            }
            let n_wavelengths = self.params.emitter.spectrum.step_count();
            match compression {
                CompressionScheme::None => {
                    Self::write_bsdf_snapshots(writer, &self.snapshots, n_wavelengths)?;
                    if self.params.receiver.retrieval_mode == DataRetrievalMode::FullData {
                        for snapshot in self.raw_snapshots.as_ref().unwrap() {
                            snapshot.write(writer, n_wavelengths)?;
                        }
                    }
                }
                CompressionScheme::Zlib => {
                    let mut zlib_encoder =
                        flate2::write::ZlibEncoder::new(vec![], flate2::Compression::default());
                    Self::write_bsdf_snapshots(&mut zlib_encoder, &self.snapshots, n_wavelengths)?;
                    writer.write_all(&zlib_encoder.flush_finish()?)?
                }
                CompressionScheme::Gzip => {
                    let mut gzip_encoder =
                        flate2::write::GzEncoder::new(vec![], flate2::Compression::default());
                    Self::write_bsdf_snapshots(&mut gzip_encoder, &self.snapshots, n_wavelengths)?;
                    writer.write_all(&gzip_encoder.finish()?)?
                }
                _ => {}
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::measure::bsdf::{
        receiver::BounceAndEnergy, BsdfMeasurementStatsPoint, BsdfSnapshot, BsdfSnapshotRaw,
        SpectralSamples,
    };
    use std::io::{BufReader, BufWriter, Cursor, Write};
    use vgcore::math::Sph2;

    #[test]
    fn test_bsdf_measurement_stats_point_read_write() {
        let data = BsdfMeasurementStatsPoint {
            n_bounces: 3,
            n_received: 1234567,
            n_absorbed: SpectralSamples::from_vec(vec![1, 2, 3, 4]),
            n_reflected: SpectralSamples::from_vec(vec![5, 6, 7, 8]),
            n_captured: SpectralSamples::from_vec(vec![9, 10, 11, 12]),
            e_captured: SpectralSamples::from_vec(vec![13.0, 14.0, 15.0, 16.0]),
            num_rays_per_bounce: SpectralSamples::from_vec(vec![
                vec![17, 18, 19],
                vec![22, 23, 24],
                vec![26, 27, 28],
                vec![30, 31, 32],
            ]),
            energy_per_bounce: SpectralSamples::from_vec(vec![
                vec![1.0, 2.0, 4.0],
                vec![5.0, 6.0, 7.0],
                vec![8.0, 9.0, 10.0],
                vec![11.0, 12.0, 13.0],
            ]),
        };
        let size = BsdfMeasurementStatsPoint::size_in_bytes(4, 3);
        let mut writer = BufWriter::with_capacity(size, vec![]);
        data.write(&mut writer, 4).unwrap();
        let buf = writer.into_inner().unwrap();
        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 = BsdfMeasurementStatsPoint::read(&mut reader, 4).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_bounce_and_energy_read_write() {
        let data = BounceAndEnergy {
            n_bounces: 11,
            total_rays: 33468,
            total_energy: 1349534.0,
            num_rays_per_bounce: vec![210, 40, 60, 70, 80, 90, 100, 110, 120, 130, 0],
            energy_per_bounce: vec![
                20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90., 100., 110., 120.,
            ],
        };
        let size = BounceAndEnergy::size_in_bytes(data.n_bounces as usize);
        let mut writer = BufWriter::with_capacity(size, vec![]);
        data.write(&mut writer).unwrap();
        let buf = writer.into_inner().unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 = BounceAndEnergy::read(&mut reader).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_bsdf_measurement_raw_snapshot() {
        let n_wavelengths = 4;
        let n_patches = 2;
        let data = BsdfSnapshotRaw::<BounceAndEnergy> {
            w_i: Sph2::zero(),
            stats: BsdfMeasurementStatsPoint {
                n_bounces: 3,
                n_received: 1111,
                n_absorbed: SpectralSamples::from_vec(vec![1, 2, 3, 4]),
                n_reflected: SpectralSamples::from_vec(vec![5, 6, 7, 8]),
                n_captured: SpectralSamples::from_vec(vec![9, 10, 11, 12]),
                e_captured: SpectralSamples::from_vec(vec![13.0, 14.0, 15.0, 16.0]),
                num_rays_per_bounce: SpectralSamples::from_vec(vec![
                    vec![17, 18, 19],
                    vec![22, 23, 24],
                    vec![26, 27, 28],
                    vec![30, 31, 32],
                ]),
                energy_per_bounce: SpectralSamples::from_vec(vec![
                    vec![1.0, 2.0, 4.0],
                    vec![5.0, 6.0, 7.0],
                    vec![8.0, 9.0, 10.0],
                    vec![11.0, 12.0, 13.0],
                ]),
            },
            records: vec![
                SpectralSamples::from_vec(vec![
                    BounceAndEnergy {
                        n_bounces: 3,
                        total_rays: 33468,
                        total_energy: 1349534.0,
                        num_rays_per_bounce: vec![210, 40, 60],
                        energy_per_bounce: vec![20.0, 30.0, 40.0],
                    },
                    BounceAndEnergy {
                        n_bounces: 2,
                        total_rays: 33,
                        total_energy: 14.0,
                        num_rays_per_bounce: vec![10, 4],
                        energy_per_bounce: vec![0.0, 3.0],
                    },
                    BounceAndEnergy {
                        n_bounces: 3,
                        total_rays: 33468,
                        total_energy: 1349534.0,
                        num_rays_per_bounce: vec![210, 40, 60],
                        energy_per_bounce: vec![20.0, 30.0, 40.0],
                    },
                    BounceAndEnergy {
                        n_bounces: 1,
                        total_rays: 33,
                        total_energy: 14.0,
                        num_rays_per_bounce: vec![10],
                        energy_per_bounce: vec![0.0],
                    },
                ]),
                SpectralSamples::from_vec(vec![
                    BounceAndEnergy {
                        n_bounces: 3,
                        total_rays: 33468,
                        total_energy: 1349534.0,
                        num_rays_per_bounce: vec![210, 40, 60],
                        energy_per_bounce: vec![20.0, 30.0, 40.0],
                    },
                    BounceAndEnergy {
                        n_bounces: 3,
                        total_rays: 33,
                        total_energy: 14.0,
                        num_rays_per_bounce: vec![10, 4, 0],
                        energy_per_bounce: vec![0.0, 3.0, 4.0],
                    },
                    BounceAndEnergy {
                        n_bounces: 2,
                        total_rays: 33468,
                        total_energy: 1349534.0,
                        num_rays_per_bounce: vec![210, 40],
                        energy_per_bounce: vec![20.0, 30.0],
                    },
                    BounceAndEnergy {
                        n_bounces: 2,
                        total_rays: 33,
                        total_energy: 14.0,
                        num_rays_per_bounce: vec![10, 4],
                        energy_per_bounce: vec![0.0, 3.0],
                    },
                ]),
            ],
            #[cfg(debug_assertions)]
            trajectories: vec![],
            #[cfg(debug_assertions)]
            hit_points: vec![],
        };

        let mut writer = BufWriter::new(vec![]);
        data.write(&mut writer, n_wavelengths).unwrap();
        let buf = writer.into_inner().unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 =
            BsdfSnapshotRaw::<BounceAndEnergy>::read(&mut reader, n_wavelengths, 2).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_bsdf_measurement_snapshot() {
        let n_wavelengths = 4;
        let n_patches = 4;
        let snapshot = BsdfSnapshot {
            w_i: Sph2::zero(),
            samples: vec![SpectralSamples::splat(11.0, n_wavelengths); n_patches],
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            trajectories: vec![],
            #[cfg(any(feature = "visu-dbg", debug_assertions))]
            hit_points: vec![],
        };
        let mut writer = BufWriter::new(vec![]);
        snapshot.write(&mut writer, n_wavelengths).unwrap();

        let mut reader = BufReader::new(Cursor::new(writer.into_inner().unwrap()));
        let mut samples_buf = vec![0u8; 4 * n_wavelengths].into_boxed_slice();
        let snapshot2 =
            BsdfSnapshot::read(&mut reader, n_wavelengths, n_patches, &mut samples_buf).unwrap();
        assert_eq!(snapshot, snapshot2);
    }
}
