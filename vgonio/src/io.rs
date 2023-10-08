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
                receiver::{BounceAndEnergy, ReceiverParams, ReceiverScheme, Ring},
                BsdfKind, BsdfMeasurementStatsPoint, BsdfSnapshotRaw, PerWavelength,
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
            CompressionScheme, FileEncoding, Header, HeaderExt, ParseError, ReadFileErrorKind,
            VgonioFileVariant, WriteFileErrorKind,
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

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum VgmoHeaderExt {
        Bsdf { params: BsdfMeasurementParams },
        Adf { params: AdfMeasurementParams },
        Msf { params: MsfMeasurementParams },
    }

    impl MeasuredData {
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
    ) -> Result<MeasuredData, ParseError> {
        match header.extra {
            VgmoHeaderExt::Bsdf { .. } => {
                todo!("BSDF data reading not implemented yet")
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
                )?;
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
                )?;
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

    fn receiver_num_rings(domain: SphericalDomain, precision: Radians) -> usize {
        (domain.zenith_angle_diff() / precision).round() as usize
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
                    let bsdf_only = u32::from_le_bytes(buf[20..24].try_into().unwrap()) != 0;
                    let params = ReceiverParams {
                        domain,
                        precision,
                        scheme,
                        bsdf_only,
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
                    buf[20..24].copy_from_slice(&(self.bsdf_only as u32).to_le_bytes());
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
                b"al" => Self::Aluminium,
                b"cu" => Self::Copper,
                _ => panic!("Invalid medium kind"),
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
            buf[7] = match self.sim_kind {
                SimulationKind::GeomOptics(method) => method as u8,
                SimulationKind::WaveOptics => 0x03,
            };
            let n_written = self.emitter.write_to_buf(version, &mut buf[8..]);
            self.receiver
                .write_to_buf(version, &mut buf[n_written + 8..]);
            writer.write_all(&buf)
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
            let detector_patches_count = params.receiver.num_patches();
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
        pub fn read_from_vgmo<R: Read>(
            reader: &mut BufReader<R>,
            params: BsdfMeasurementParams,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<Self, ReadFileErrorKind> {
            // TODO: check if the meta is correct
            let mut zlib_decoder;
            let mut gzip_decoder;

            let mut decoder: Box<&mut dyn Read> = match compression {
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

            match encoding {
                FileEncoding::Ascii => {
                    todo!("Ascii encoding is not supported yet")
                }
                FileEncoding::Binary => {
                    let n_wavelength = params.emitter.spectrum.step_count();
                    let bounces = params.emitter.max_bounces as usize;
                    let detector_patches_count = params.receiver.num_patches();
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
        pub fn write_to_vgmo<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            encoding: FileEncoding,
            compression: CompressionScheme,
        ) -> Result<(), WriteFileErrorKind> {
            match compression {
                CompressionScheme::None => {
                    for snapshot in self.snapshots {
                        for samples in snapshot.samples {
                            for s in &samples {
                                writer.write_all(&s.to_le_bytes())?;
                            }
                        }
                    }
                }
                CompressionScheme::Zlib => {}
                CompressionScheme::Gzip => {}
            }

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
                        let detector_patches_count = self.params.receiver.num_patches();
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
}

#[cfg(test)]
mod tests {
    use crate::{
        measure::{
            bsdf::{
                emitter::{Emitter, EmitterParams},
                receiver::{
                    BounceAndEnergy, Receiver, ReceiverParams, ReceiverPartition, ReceiverScheme,
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
