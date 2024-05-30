use crate::{
    app::{
        cache::{Cache, Handle},
        cli::ansi,
        Config,
    },
    measure::{
        bsdf::MeasuredBsdfData,
        mfd::{MeasuredMsfData, MeasuredNdfData},
        Measurement,
    },
};
use base::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
    math,
};
use std::{
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};
use surf::MicroSurface;

pub mod vgmo {
    use super::*;
    use crate::measure::{
        bsdf::{
            emitter::EmitterParams,
            receiver::{BounceAndEnergy, ReceiverParams},
            BsdfKind, MeasuredBrdfLevel, RawMeasuredBsdfData, SingleBsdfMeasurementStats,
        },
        mfd::MeasuredSdfData,
        params::{
            BsdfMeasurementParams, MsfMeasurementParams, NdfMeasurementMode, NdfMeasurementParams,
            SdfMeasurementParams, SimulationKind,
        },
    };
    use base::{
        io,
        io::{
            CompressionScheme, FileEncoding, Header, HeaderExt, ReadFileErrorKind,
            VgonioFileVariant, WriteFileErrorKind,
        },
        math::Sph2,
        medium::Medium,
        partition::{PartitionScheme, Ring, SphericalDomain},
        range::RangeByStepSizeInclusive,
        units::{rad, Nanometres, Radians},
        MeasuredData, MeasurementKind, Version,
    };
    use bxdf::brdf::measured::{Origin, VgonioBrdf, VgonioBrdfParameterisation};
    use jabr::array::DyArr;
    use std::{
        collections::HashMap,
        io::{BufWriter, Seek},
        mem,
        mem::MaybeUninit,
        ptr,
    };

    /// The VGMO header extension.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum VgmoHeaderExt {
        Bsdf { params: BsdfMeasurementParams },
        Ndf { params: NdfMeasurementParams },
        Gaf { params: MsfMeasurementParams },
        Sdf,
    }

    /// Trait for structs that can be written to and read from VGMO files.
    pub trait VgmoFile {
        /// The required size of the buffer to read or write the data without
        /// any compression.
        const REQUIRED_SIZE: usize;
    }

    /// Returns the corresponding [`VgmoHeaderExt`] variant for the given
    /// measurement data.
    pub fn vgmo_header_ext_from_data(data: &Box<dyn MeasuredData>) -> VgmoHeaderExt {
        match data.kind() {
            MeasurementKind::Bsdf => {
                let bsdf = data.downcast_ref::<MeasuredBsdfData>().unwrap();
                VgmoHeaderExt::Bsdf {
                    params: bsdf.params.clone(),
                }
            }
            MeasurementKind::Ndf => {
                let ndf = data.downcast_ref::<MeasuredNdfData>().unwrap();
                VgmoHeaderExt::Ndf {
                    params: ndf.params.clone(),
                }
            }
            MeasurementKind::Msf => {
                let msf = data.downcast_ref::<MeasuredMsfData>().unwrap();
                VgmoHeaderExt::Gaf {
                    params: msf.params.clone(),
                }
            }
            MeasurementKind::Sdf => VgmoHeaderExt::Sdf,
            _ => {
                unreachable!("Unsupported measurement kind: {}", data.kind())
            }
        }
    }

    // TODO: deal with ADF measurement params
    impl HeaderExt for VgmoHeaderExt {
        const MAGIC: &'static [u8; 4] = b"VGMO";

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
                    Self::Ndf { params } => match &params.mode {
                        NdfMeasurementMode::ByPoints { azimuth, zenith } => {
                            writer.write_all(&[MeasurementKind::Ndf as u8])?;
                            write_adf_or_msf_params_to_vgmo(azimuth, zenith, writer, true)?;
                        }
                        NdfMeasurementMode::ByPartition { .. } => {
                            // TODO: implement
                        }
                    },
                    Self::Gaf { params } => {
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
                    VgmoHeaderExt::Sdf => {
                        writer.write_all(&[MeasurementKind::Sdf as u8])?;
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
                MeasurementKind::Bsdf => BsdfMeasurementParams::read_from_vgmo(version, reader)
                    .map(|params| Ok(Self::Bsdf { params }))?,
                MeasurementKind::Ndf => NdfMeasurementParams::read_from_vgmo(version, reader)
                    .map(|params| Ok(Self::Ndf { params }))?,
                MeasurementKind::Msf => MsfMeasurementParams::read_from_vgmo(version, reader)
                    .map(|params| Ok(Self::Gaf { params }))?,
                MeasurementKind::Sdf => Ok(Self::Sdf),
                _ => {
                    unreachable!("Unsupported measurement kind: {}", kind[0])
                }
            }
        }
    }

    /// Reads the VGMO file from the given reader.
    pub fn read<R: Read>(
        reader: &mut BufReader<R>,
        header: &Header<VgmoHeaderExt>,
    ) -> Result<Box<dyn MeasuredData>, ReadFileErrorKind> {
        match header.extra {
            VgmoHeaderExt::Bsdf { params } => {
                log::debug!(
                    "Reading BSDF data of {} measurement points {} samples from VGMO file",
                    params.emitter.measurement_points_count(),
                    params.samples_count()
                );
                Ok(Box::new(MeasuredBsdfData::read_from_vgmo(
                    reader,
                    &params,
                    header.meta.encoding,
                    header.meta.compression,
                )?))
            }
            VgmoHeaderExt::Ndf { params } => {
                log::debug!(
                    "Reading ADF data of {} samples from VGMO file",
                    params.samples_count()
                );
                let samples = base::io::read_f32_data_samples(
                    reader,
                    params.samples_count(),
                    header.meta.encoding,
                    header.meta.compression,
                )
                .map_err(ReadFileErrorKind::Parse)?;
                Ok(Box::new(MeasuredNdfData { params, samples }))
            }
            VgmoHeaderExt::Gaf { params } => {
                log::debug!(
                    "Reading MSF data of {} samples from VGMO file",
                    params.samples_count()
                );
                let samples = base::io::read_f32_data_samples(
                    reader,
                    params.samples_count(),
                    header.meta.encoding,
                    header.meta.compression,
                )
                .map_err(ReadFileErrorKind::Parse)?;
                Ok(Box::new(MeasuredMsfData { params, samples }))
            }
            VgmoHeaderExt::Sdf => {
                log::debug!("Reading SDF data from VGMO file");
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                let num_slopes = u32::from_le_bytes(buf);
                let samples = base::io::read_f32_data_samples(
                    reader,
                    num_slopes as usize * 2,
                    header.meta.encoding,
                    header.meta.compression,
                )?;
                let slopes = samples
                    .chunks(2)
                    .map(|s| math::Vec2::new(s[0], s[1]))
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                Ok(Box::new(MeasuredSdfData {
                    params: SdfMeasurementParams {
                        max_slope: std::f32::consts::PI,
                    },
                    slopes,
                }))
            }
        }
    }

    /// Writes the given measurement data to the given writer.
    pub fn write<W: Write + Seek>(
        writer: &mut BufWriter<W>,
        header: Header<VgmoHeaderExt>,
        measured: &Box<dyn MeasuredData>,
    ) -> Result<(), WriteFileErrorKind> {
        let init_size = writer.stream_len().unwrap();
        log::debug!("Writing VGMO file with writer at position {}", init_size);
        header.write(writer)?;
        log::debug!(
            "Wrote VGMO header to file, writer at position {}",
            writer.stream_len().unwrap()
        );

        #[cfg(feature = "bench")]
        let start = std::time::Instant::now();

        match measured.kind() {
            mfd @ (MeasurementKind::Ndf | MeasurementKind::Msf) => {
                let (samples, cols) = match mfd {
                    MeasurementKind::Ndf => {
                        let ndf = measured.downcast_ref::<MeasuredNdfData>().unwrap();
                        let cols = match &ndf.params.mode {
                            NdfMeasurementMode::ByPoints { zenith, .. } => {
                                zenith.step_count_wrapped()
                            }
                            NdfMeasurementMode::ByPartition { .. } => {
                                todo!("Partitioned ADF data")
                            }
                        };
                        (&ndf.samples, cols)
                    }
                    MeasurementKind::Msf => {
                        let msf = measured.downcast_ref::<MeasuredMsfData>().unwrap();
                        (&msf.samples, msf.params.zenith.step_count_wrapped())
                    }
                    _ => {
                        unreachable!("Unsupported measurement kind: {}", mfd)
                    }
                };
                match header.meta.encoding {
                    FileEncoding::Ascii => {
                        base::io::write_data_samples_ascii(writer, samples, cols as u32)
                    }
                    FileEncoding::Binary => base::io::write_f32_data_samples_binary(
                        writer,
                        header.meta.compression,
                        samples,
                    ),
                }
                .map_err(WriteFileErrorKind::Write)?;
            }
            MeasurementKind::Bsdf => {
                let bsdf = measured.downcast_ref::<MeasuredBsdfData>().unwrap();
                bsdf.write_to_vgmo(writer, header.meta.encoding, header.meta.compression)?;
            }
            MeasurementKind::Sdf => {
                let sdf = measured.downcast_ref::<MeasuredSdfData>().unwrap();
                writer.write(&(sdf.slopes.len() as u32).to_le_bytes())?;
                let samples = unsafe {
                    std::slice::from_raw_parts(
                        sdf.slopes.as_ptr() as *const f32,
                        sdf.slopes.len() * 2,
                    )
                };
                base::io::write_f32_data_samples_binary(writer, header.meta.compression, samples)
                    .map_err(WriteFileErrorKind::Write)?;
            }
            _ => {}
        }

        #[cfg(feature = "bench")]
        {
            let elapsed = start.elapsed();
            log::debug!("Write samples took {} ms", elapsed.as_millis());
        }

        let length = writer.stream_len().unwrap() - init_size;
        writer.seek(std::io::SeekFrom::Start(
            Header::<VgmoHeaderExt>::length_pos() as u64,
        ))?;
        writer.write_all(&(length as u32).to_le_bytes())?;

        #[cfg(feature = "bench")]
        {
            let elapsed = start.elapsed();
            log::debug!(
                "Wrote {} bytes of data{}to VGMO file in {} ms",
                length,
                if header.meta.compression == CompressionScheme::None {
                    " "
                } else {
                    " (compressed) "
                },
                elapsed.as_millis()
            );
        }
        #[cfg(not(feature = "bench"))]
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
        /// Because the receiver's partition is dependent on the precision,
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
                        0 => PartitionScheme::Beckers,
                        1 => PartitionScheme::EqualAngle,
                        _ => panic!("Invalid scheme kind"),
                    };
                    let precision_theta = rad!(f32::from_le_bytes(buf[8..12].try_into().unwrap()));
                    let precision_phi = rad!(f32::from_le_bytes(buf[12..16].try_into().unwrap()));
                    let params = ReceiverParams {
                        domain,
                        precision: Sph2::new(precision_theta, precision_phi),
                        scheme,
                    };
                    let expected_num_rings = params.num_rings();
                    let num_rings = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
                    debug_assert!(
                        num_rings == expected_num_rings,
                        "Receiver's partition ring count does not match the precision",
                    );
                    let expected_num_patches = params.num_patches();
                    let num_patches = u32::from_le_bytes(buf[20..24].try_into().unwrap()) as usize;
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
                    let partition = self.partitioning();
                    buf[0..4].copy_from_slice(&(self.domain as u32).to_le_bytes());
                    buf[4..8].copy_from_slice(&(self.scheme as u32).to_le_bytes());
                    buf[8..12].copy_from_slice(&self.precision.theta.value().to_le_bytes());
                    buf[12..16].copy_from_slice(&self.precision.phi.value().to_le_bytes());
                    buf[16..20].copy_from_slice(&(partition.n_rings() as u32).to_le_bytes());
                    buf[20..24].copy_from_slice(&(partition.n_patches() as u32).to_le_bytes());
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

    impl NdfMeasurementParams {
        // TODO: resolve `crop_to_disk`
        // TODO: read partitioned ADF data
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
                    let mut buf = [0u8; 36];
                    reader.read_exact(&mut buf)?;
                    let azimuth = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[0..16]);
                    let zenith = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[16..32]);
                    let sample_count = u32::from_le_bytes(buf[32..36].try_into().unwrap());
                    debug_assert_eq!(
                        sample_count as usize,
                        NdfMeasurementParams::expected_samples_count_by_points(&azimuth, &zenith)
                    );
                    Ok(Self {
                        mode: NdfMeasurementMode::ByPoints { azimuth, zenith },
                        crop_to_disk: false,
                        use_facet_area: true,
                    })
                }
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unsupported VGMO[AdfMeasurementParams] version {}", version),
                )),
            }
        }
    }

    impl MsfMeasurementParams {
        // TODO: resolve resolution and strict
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
                    let mut buf = [0u8; 36];
                    reader.read_exact(&mut buf)?;
                    let azimuth = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[0..16]);
                    let zenith = RangeByStepSizeInclusive::<Radians>::read_from_buf(&buf[16..32]);
                    let sample_count = u32::from_le_bytes(buf[32..36].try_into().unwrap());
                    debug_assert_eq!(
                        sample_count as usize,
                        MsfMeasurementParams::expected_samples_count(&azimuth, &zenith)
                    );
                    Ok(Self {
                        azimuth,
                        zenith,
                        resolution: 512,
                        strict: true,
                    })
                }
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unsupported VGMO[MsfMeasurementParams] version {}", version),
                )),
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
                        fresnel: false, // TODO: read/write from file
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
                    11 + EmitterParams::required_size(version).unwrap()
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
            buf[8] = self.fresnel as u8;
            // buf[9/10] are padding bytes

            let n_written = self.emitter.write_to_buf(version, &mut buf[11..]);
            log::debug!("Written {} bytes for emitter", n_written);
            self.receiver
                .write_to_buf(version, &mut buf[n_written + 11..]);
            writer.write_all(&buf)
        }
    }

    pub trait EndianRead: Sized {
        fn read_le_bytes(src: &[u8]) -> Self;
        fn read_be_bytes(src: &[u8]) -> Self;
    }

    pub trait EndianWrite: Sized {
        fn write_le_bytes(&self, dst: &mut [u8]);
        fn write_be_bytes(&self, dst: &mut [u8]);
    }

    macro_rules! impl_endian_read_write {
        ($($t:ty),*) => {
            $(
                impl EndianRead for $t {
                    fn read_le_bytes(src: &[u8]) -> Self {
                        Self::from_le_bytes(src.try_into().expect("Invalid byte slice, expected 4 bytes"))
                    }

                    fn read_be_bytes(src: &[u8]) -> Self {
                        Self::from_be_bytes(src.try_into().expect("Invalid byte slice, expected 4 bytes"))
                    }
                }

                impl EndianWrite for $t {
                    fn write_le_bytes(&self, dst: &mut [u8]) {
                        dst.copy_from_slice(&self.to_le_bytes())
                    }

                    fn write_be_bytes(&self, dst: &mut [u8]) {
                        dst.copy_from_slice(&self.to_be_bytes())
                    }
                }
            )*
        };
    }

    impl_endian_read_write!(u32, f32);

    /// Writes the given slice to the buffer in little-endian format.
    pub fn write_slice_to_buf<T: EndianWrite>(src: &[T], dst: &mut [u8]) {
        let size: usize = mem::size_of::<T>();
        debug_assert!(
            dst.len() >= src.len() * size,
            "Write array to buffer: desired size {}, got {}",
            src.len() * size,
            dst.len()
        );
        for i in 0..src.len() {
            src[i].write_le_bytes(&mut dst[i * size..(i + 1) * size]);
        }
    }

    /// Reads the given slice from the buffer in little-endian format.
    ///
    /// # Arguments
    ///
    /// * `src` - The buffer to read from.
    /// * `len` - The number of elements to read.
    /// * `dst` - The destination slice to write to.
    #[track_caller]
    pub fn read_slice_from_buf<T: EndianRead>(src: &[u8], dst: &mut [T], len: usize) {
        let size: usize = mem::size_of::<T>();
        assert_eq!(
            src.len(),
            len * size,
            "Buffer size mismatch, expected {} bytes, got {}",
            len * size,
            src.len()
        );
        for i in 0..len {
            dst[i] = <T>::read_le_bytes(&src[i * size..(i + 1) * size]);
        }
    }

    impl SingleBsdfMeasurementStats {
        /// The size of the buffer required to read/write the BSDF measurement
        /// stats point.
        pub fn size_in_bytes(n_spectrum: usize, n_bounce: usize) -> usize {
            3 * mem::size_of::<u32>()
                + Self::N_STATS * n_spectrum * mem::size_of::<u32>()
                + n_spectrum * mem::size_of::<f32>()
                + n_spectrum * n_bounce * 2 * mem::size_of::<f32>()
        }

        /// Writes the BSDF measurement statistics at a single point to the
        /// writer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            n_spectrum: usize,
        ) -> Result<(), std::io::Error> {
            let size = Self::size_in_bytes(n_spectrum, self.n_bounce as usize);
            let mut buf = vec![0u8; size].into_boxed_slice();
            let mut offset_in_bytes = 0;
            buf[offset_in_bytes..offset_in_bytes + 4].copy_from_slice(&self.n_bounce.to_le_bytes());
            offset_in_bytes += 4;
            buf[offset_in_bytes..offset_in_bytes + 4]
                .copy_from_slice(&self.n_received.to_le_bytes());
            offset_in_bytes += 4;
            buf[offset_in_bytes..offset_in_bytes + 4].copy_from_slice(&self.n_missed.to_le_bytes());
            offset_in_bytes += 4;

            // Write n_absorbed, n_reflected, n_captured and n_escaped per wavelength
            write_slice_to_buf::<u32>(
                &self.n_ray_stats,
                &mut buf[offset_in_bytes..offset_in_bytes + 4 * n_spectrum * 4],
            );
            offset_in_bytes += 4 * n_spectrum * 4;

            // Write the energy captured per wavelength
            write_slice_to_buf(
                &self.e_captured,
                &mut buf[offset_in_bytes..offset_in_bytes + 4 * n_spectrum],
            );
            offset_in_bytes += 4 * n_spectrum;

            // Write the number of rays per bounce for each wavelength
            self.n_ray_per_bounce.iter().for_each(|val| {
                buf[offset_in_bytes..offset_in_bytes + 4].copy_from_slice(&val.to_le_bytes());
                offset_in_bytes += 4;
            });

            // Write the energy per bounce for each wavelength
            self.energy_per_bounce.iter().for_each(|val| {
                buf[offset_in_bytes..offset_in_bytes + 4].copy_from_slice(&val.to_le_bytes());
                offset_in_bytes += 4;
            });
            writer.write_all(&buf)
        }

        /// Reads the BSDF measurement statistics at a single point from the
        /// buffer.
        pub fn read<R: Read>(reader: &mut R, n_spectrum: usize) -> Result<Self, std::io::Error> {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            let n_bounce = u32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            let n_received = u32::from_le_bytes(buf);
            reader.read_exact(&mut buf)?;
            let n_missed = u32::from_le_bytes(buf);

            let mut buf = vec![0u8; 4 * n_spectrum * 4 + 4 * n_spectrum * n_bounce as usize * 2];
            reader.read_exact(&mut buf)?;

            let mut offset = 0;
            let mut n_ray_stats = vec![0u32; n_spectrum * 4].into_boxed_slice();
            let n_ray_stats_size = 4 * n_ray_stats.len();
            read_slice_from_buf::<u32>(
                &buf[offset..n_ray_stats_size],
                &mut n_ray_stats,
                n_spectrum * 3,
            );
            offset += n_ray_stats_size;

            let mut e_captured = vec![0f32; n_spectrum].into_boxed_slice();
            let e_captured_size = 4 * e_captured.len();
            read_slice_from_buf::<f32>(
                &buf[offset..offset + e_captured_size],
                &mut e_captured,
                n_spectrum,
            );
            offset += e_captured_size;

            let mut n_ray_per_bounce =
                vec![0u32; n_spectrum * n_bounce as usize].into_boxed_slice();
            let n_rays_per_bounce_size = 4 * n_ray_per_bounce.len();
            read_slice_from_buf::<u32>(
                &buf[offset..offset + n_rays_per_bounce_size],
                &mut n_ray_per_bounce,
                n_spectrum * n_bounce as usize,
            );
            offset += n_rays_per_bounce_size;

            let mut energy_per_bounce =
                vec![0f32; n_spectrum * n_bounce as usize].into_boxed_slice();
            read_slice_from_buf::<f32>(
                &buf[offset..offset + 4 * n_spectrum * n_bounce as usize],
                &mut energy_per_bounce,
                n_spectrum * n_bounce as usize,
            );

            Ok(Self {
                n_bounce,
                n_received,
                n_missed,
                n_spectrum,
                n_ray_stats,
                e_captured,
                n_ray_per_bounce,
                energy_per_bounce,
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
                u32::from_le_bytes(buf)
            };
            let size_without_n_bounces = 4 * 2 + n_bounces as usize * 4 * 2;
            let mut buf = vec![0u8; size_without_n_bounces].into_boxed_slice();
            reader.read_exact(&mut buf)?;
            let total_rays = u32::from_le_bytes(buf[0..4].try_into().unwrap());
            let total_energy = f32::from_le_bytes(buf[4..8].try_into().unwrap());
            let mut offset = 8;
            let mut n_ray_per_bounce = vec![0u32; n_bounces as usize].into_boxed_slice();
            for i in 0..n_bounces as usize {
                n_ray_per_bounce[i] =
                    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            let mut energy_per_bounce = vec![0f32; n_bounces as usize].into_boxed_slice();
            for i in 0..n_bounces as usize {
                energy_per_bounce[i] =
                    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                offset += 4;
            }
            Ok(Self {
                n_bounce: n_bounces,
                total_rays,
                total_energy,
                n_ray_per_bounce,
                energy_per_bounce,
            })
        }

        /// Writes the data for one single patch.
        pub fn write<W: Write>(&self, writer: &mut BufWriter<W>) -> Result<(), std::io::Error> {
            let mut buf = vec![0u8; Self::size_in_bytes(self.n_bounce as usize)].into_boxed_slice();
            buf[0..4].copy_from_slice(&self.n_bounce.to_le_bytes());
            buf[4..8].copy_from_slice(&self.total_rays.to_le_bytes());
            buf[8..12].copy_from_slice(&self.total_energy.to_le_bytes());
            let mut offset = 12;
            for i in 0..self.n_bounce as usize {
                buf[offset..offset + 4].copy_from_slice(&self.n_ray_per_bounce[i].to_le_bytes());
                offset += 4;
            }
            for i in 0..self.n_bounce as usize {
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

    impl MeasuredBsdfData {
        pub(crate) fn write_vgonio_brdf<W: Write>(
            writer: &mut W,
            level: MeasuredBrdfLevel,
            brdf: &VgonioBrdf,
        ) -> Result<(), std::io::Error> {
            writer.write_all(&level.0.to_le_bytes())?;
            io::write_binary_samples(writer, brdf.samples.as_slice())
        }

        pub(crate) fn read_vgonio_brdf<R: Read>(
            reader: &mut R,
            n_wi: usize,
            n_wo: usize,
            n_spectrum: usize,
            brdf: *mut VgonioBrdf,
        ) -> Result<MeasuredBrdfLevel, ReadFileErrorKind> {
            let level = {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                MeasuredBrdfLevel(u32::from_le_bytes(buf))
            };
            let mut samples = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
            io::read_binary_samples(reader, samples.len(), samples.as_mut_slice())
                .map_err(ReadFileErrorKind::Parse)?;

            unsafe {
                ptr::addr_of_mut!((*brdf).samples).write(samples);
            }
            Ok(level)
        }

        pub(crate) fn write_measured_bsdf_data<
            'a,
            W: Write,
            F: Iterator<Item = (&'a MeasuredBrdfLevel, &'a VgonioBrdf)>,
        >(
            writer: &'a mut W,
            brdfs: F,
        ) -> Result<(), std::io::Error> {
            for (level, brdf) in brdfs {
                Self::write_vgonio_brdf(writer, *level, brdf)?;
            }
            Ok(())
        }

        pub(crate) fn read_measured_bsdf_data<R: Read>(
            reader: &mut R,
            n_wi: usize,
            n_wo: usize,
            n_spectrum: usize,
        ) -> Result<HashMap<MeasuredBrdfLevel, MaybeUninit<VgonioBrdf>>, std::io::Error> {
            let mut brdfs = HashMap::new();
            loop {
                let mut brdf = MaybeUninit::<VgonioBrdf>::uninit();
                match Self::read_vgonio_brdf(reader, n_wi, n_wo, n_spectrum, brdf.as_mut_ptr()) {
                    Ok(level) => {
                        brdfs.insert(level, brdf);
                    }
                    Err(_) => {
                        break;
                    }
                }
            }
            Ok(brdfs)
        }

        pub(crate) fn write_raw_measured_data<W: Write>(
            writer: &mut W,
            raw: &RawMeasuredBsdfData,
        ) -> Result<(), std::io::Error> {
            let mut writer = BufWriter::new(writer);
            // Collected data of the receiver per incident direction per
            // outgoing(patch) direction and per wavelength: `ωi, ωo, λ`.
            for record in raw.records.iter() {
                record.write(&mut writer)?;
            }
            // Writes the statistics of the measurement per incident direction
            for stats in raw.stats.iter() {
                stats.write(&mut writer, raw.spectrum.len())?;
            }
            writer.flush()
        }

        pub(crate) fn read_raw_measured_data<R: Read>(
            reader: &mut R,
            params: &BsdfMeasurementParams,
        ) -> Result<RawMeasuredBsdfData, std::io::Error> {
            let n_wi = params.n_wi();
            let n_wo = params.n_wo();
            let n_spectrum = params.n_spectrum();
            let mut records = Box::new_uninit_slice(n_wi * n_wo * n_spectrum);
            for record in records.iter_mut() {
                record.write(BounceAndEnergy::read(reader)?);
            }

            let mut stats = Box::new_uninit_slice(n_wi);
            for stat in stats.iter_mut() {
                stat.write(SingleBsdfMeasurementStats::read(reader, n_spectrum)?);
            }

            Ok(unsafe {
                RawMeasuredBsdfData {
                    spectrum: DyArr::from_iterator(
                        [n_spectrum as isize],
                        params.emitter.spectrum.values(),
                    ),
                    incoming: DyArr::from_boxed_slice_1d(
                        params.emitter.generate_measurement_points().0,
                    ),
                    outgoing: params.receiver.partitioning(),
                    records: DyArr::from_boxed_slice(
                        [n_wi, n_wo, n_spectrum],
                        records.assume_init(),
                    ),
                    stats: DyArr::from_boxed_slice_1d(stats.assume_init()),
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    trajectories: Box::new([]),
                    #[cfg(any(feature = "visu-dbg", debug_assertions))]
                    hit_points: Box::new([]),
                }
            })
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
                    let raw = Self::read_raw_measured_data(&mut decoder, params)?;
                    let mut bsdfs = Self::read_measured_bsdf_data(
                        &mut decoder,
                        params.n_wi(),
                        params.n_wo(),
                        params.n_spectrum(),
                    )?;

                    let parameterisation = VgonioBrdfParameterisation {
                        incoming: DyArr::from_boxed_slice_1d(
                            params.emitter.generate_measurement_points().0,
                        ),
                        outgoing: params.receiver.partitioning(),
                    };

                    for bsdf in bsdfs.values_mut() {
                        let bsdf_ptr = bsdf.as_mut_ptr();
                        unsafe {
                            ptr::addr_of_mut!((*bsdf_ptr).origin).write(Origin::Simulated);
                            ptr::addr_of_mut!((*bsdf_ptr).incident_medium)
                                .write(params.incident_medium);
                            ptr::addr_of_mut!((*bsdf_ptr).transmitted_medium)
                                .write(params.transmitted_medium);
                            ptr::addr_of_mut!((*bsdf_ptr).params)
                                .write(Box::new(parameterisation.clone()));
                            ptr::addr_of_mut!((*bsdf_ptr).spectrum).write(DyArr::from_iterator(
                                [params.n_spectrum() as isize],
                                params.emitter.spectrum.values(),
                            ));
                        }
                    }

                    Ok(Self {
                        params: *params,
                        raw,
                        bsdfs: unsafe {
                            mem::transmute::<
                                HashMap<MeasuredBrdfLevel, MaybeUninit<VgonioBrdf>>,
                                HashMap<MeasuredBrdfLevel, VgonioBrdf>,
                            >(bsdfs)
                        },
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
            log::debug!("Writing BSDF data...");
            if encoding == FileEncoding::Ascii {
                return Err(WriteFileErrorKind::UnsupportedEncoding);
            }
            match compression {
                CompressionScheme::None => {
                    Self::write_raw_measured_data(writer, &self.raw)?;
                    Self::write_measured_bsdf_data(writer, self.bsdfs.iter())?;
                }
                CompressionScheme::Zlib => {
                    let mut zlib_encoder =
                        flate2::write::ZlibEncoder::new(vec![], flate2::Compression::default());
                    Self::write_raw_measured_data(&mut zlib_encoder, &self.raw)?;
                    Self::write_measured_bsdf_data(&mut zlib_encoder, self.bsdfs.iter())?;
                    writer.write_all(&zlib_encoder.flush_finish()?)?
                }
                CompressionScheme::Gzip => {
                    let mut gzip_encoder =
                        flate2::write::GzEncoder::new(vec![], flate2::Compression::default());
                    Self::write_raw_measured_data(&mut gzip_encoder, &self.raw)?;
                    Self::write_measured_bsdf_data(&mut gzip_encoder, self.bsdfs.iter())?;
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
        receiver::BounceAndEnergy, MeasuredBsdfData, RawMeasuredBsdfData,
        SingleBsdfMeasurementStats,
    };
    use base::{math::Sph2, units::nm};
    use jabr::array::DyArr;
    use std::io::{BufReader, BufWriter, Cursor, Write};

    #[test]
    fn test_bsdf_measurement_stats_point_read_write() {
        let data = SingleBsdfMeasurementStats {
            n_bounce: 3,
            n_received: 1234567,
            n_missed: 0,
            n_spectrum: 4,
            n_ray_stats: vec![
                1, 2, 3, 4, // n_absorbed
                5, 6, 7, 8, // n_reflected
                9, 10, 11, 12, // n_captured
            ]
            .into_boxed_slice(),
            e_captured: vec![13.0, 14.0, 15.0, 16.0].into_boxed_slice(),
            n_ray_per_bounce: vec![
                17, 18, 19, //
                22, 23, 24, //
                26, 27, 28, //
                30, 31, 32, //
            ]
            .into_boxed_slice(),
            energy_per_bounce: vec![
                1.0, 2.0, 4.0, //
                5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, //
                11.0, 12.0, 13.0, //
            ]
            .into_boxed_slice(),
        };
        let size = SingleBsdfMeasurementStats::size_in_bytes(4, 3);
        let mut writer = BufWriter::with_capacity(size, vec![]);
        data.write(&mut writer, 4).unwrap();
        let buf = writer.into_inner().unwrap();
        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 = SingleBsdfMeasurementStats::read(&mut reader, 4).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_bounce_and_energy_read_write() {
        let data = BounceAndEnergy {
            n_bounce: 11,
            total_rays: 33468,
            total_energy: 1349534.0,
            n_ray_per_bounce: vec![210, 40, 60, 70, 80, 90, 100, 110, 120, 130, 0]
                .into_boxed_slice(),
            energy_per_bounce: vec![
                20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90., 100., 110., 120.,
            ]
            .into_boxed_slice(),
        };
        let size = BounceAndEnergy::size_in_bytes(data.n_bounce as usize);
        let mut writer = BufWriter::with_capacity(size, vec![]);
        data.write(&mut writer).unwrap();
        let buf = writer.into_inner().unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 = BounceAndEnergy::read(&mut reader).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_bsdf_measurement_raw_snapshot() {
        let n_wavelength = 4;
        let n_patches = 2;
        let raw = RawMeasuredBsdfData {
            spectrum: DyArr::from_iterator(
                [n_wavelength as isize],
                vec![nm!(400.0), nm!(500.0), nm!(600.0), nm!(700.0)],
            ),
            incoming: (),
            outgoing: SphericalPartition {},
            records: (),
            stats: (),
            trajectories: Box::new([]),
            hit_points: Box::new([]),
        };
        let data = BsdfSnapshotRaw::<BounceAndEnergy> {
            wi: Sph2::zero(),
            stats: SingleBsdfMeasurementStats {
                n_bounce: 3,
                n_received: 1111,
                n_spectrum: n_wavelength,
                n_ray_stats: vec![
                    1, 2, 3, 4, // n_absorbed
                    5, 6, 7, 8, // n_reflected
                    9, 10, 11, 12, // n_captured
                ]
                .into_boxed_slice(),
                e_captured: vec![13.0, 14.0, 15.0, 16.0].into_boxed_slice(),
                n_ray_per_bounce: vec![
                    17, 18, 19, // wavelength 1
                    22, 23, 24, // wavelength 2
                    26, 27, 28, // wavelength 3
                    30, 31, 32, // wavelength 4
                ]
                .into_boxed_slice(),
                energy_per_bounce: vec![
                    1.0, 2.0, 4.0, // bounce 1
                    5.0, 6.0, 7.0, // bounce 2
                    8.0, 9.0, 10.0, // bounce 3
                    11.0, 12.0, 13.0, // bounce 4
                ]
                .into_boxed_slice(),
            },
            records: vec![
                BounceAndEnergy {
                    n_bounce: 3,
                    total_rays: 33468,
                    total_energy: 1349534.0,
                    n_ray_per_bounce: vec![210, 40, 60].into_boxed_slice(),
                    energy_per_bounce: vec![20.0, 30.0, 40.0].into_boxed_slice(),
                }, //  patch 0, wavelength 0
                BounceAndEnergy {
                    n_bounce: 2,
                    total_rays: 33,
                    total_energy: 14.0,
                    n_ray_per_bounce: vec![10, 4].into_boxed_slice(),
                    energy_per_bounce: vec![0.0, 3.0].into_boxed_slice(),
                }, // patch 0, wavelength 1
                BounceAndEnergy {
                    n_bounce: 3,
                    total_rays: 33468,
                    total_energy: 1349534.0,
                    n_ray_per_bounce: vec![210, 40, 60].into_boxed_slice(),
                    energy_per_bounce: vec![20.0, 30.0, 40.0].into_boxed_slice(),
                }, // patch 0, wavelength 2
                BounceAndEnergy {
                    n_bounce: 1,
                    total_rays: 33,
                    total_energy: 14.0,
                    n_ray_per_bounce: vec![10].into_boxed_slice(),
                    energy_per_bounce: vec![0.0].into_boxed_slice(),
                }, // patch 0, wavelength 3
                BounceAndEnergy {
                    n_bounce: 3,
                    total_rays: 33468,
                    total_energy: 1349534.0,
                    n_ray_per_bounce: vec![210, 40, 60].into_boxed_slice(),
                    energy_per_bounce: vec![20.0, 30.0, 40.0].into_boxed_slice(),
                }, // patch 1, wavelength 0
                BounceAndEnergy {
                    n_bounce: 3,
                    total_rays: 33,
                    total_energy: 14.0,
                    n_ray_per_bounce: vec![10, 4, 0].into_boxed_slice(),
                    energy_per_bounce: vec![0.0, 3.0, 4.0].into_boxed_slice(),
                }, // patch 1, wavelength 1
                BounceAndEnergy {
                    n_bounce: 2,
                    total_rays: 33468,
                    total_energy: 1349534.0,
                    n_ray_per_bounce: vec![210, 40].into_boxed_slice(),
                    energy_per_bounce: vec![20.0, 30.0].into_boxed_slice(),
                }, // patch 1, wavelength 2
                BounceAndEnergy {
                    n_bounce: 2,
                    total_rays: 33,
                    total_energy: 14.0,
                    n_ray_per_bounce: vec![10, 4].into_boxed_slice(),
                    energy_per_bounce: vec![0.0, 3.0].into_boxed_slice(),
                }, // patch 1, wavelength 3
            ]
            .into_boxed_slice(),
            #[cfg(debug_assertions)]
            trajectories: vec![],
            #[cfg(debug_assertions)]
            hit_points: vec![].into_boxed_slice(),
        };

        let mut writer = BufWriter::new(vec![]);
        data.write(&mut writer, n_wavelength).unwrap();
        let buf = writer.into_inner().unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 = BsdfSnapshotRaw::<BounceAndEnergy>::read(&mut reader, n_wavelength, 2).unwrap();
        assert_eq!(data, data2);
    }
}

/// Output options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputOptions {
    /// Output directory.
    pub dir: Option<PathBuf>,
    /// Output file format.
    pub formats: Box<[OutputFileFormatOption]>,
}

/// Output file format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFileFormatOption {
    Vgmo {
        encoding: FileEncoding,
        compression: CompressionScheme,
    },
    Exr {
        resolution: u32,
    },
}

/// Writes the measured data to a file.
pub fn write_measured_data_to_file(
    data: &[Measurement],
    surfaces: &[Handle<MicroSurface>],
    cache: &Cache,
    config: &Config,
    output: OutputOptions,
) -> Result<(), VgonioError> {
    println!(
        "    {}>{} Saving measurement data...",
        ansi::BRIGHT_YELLOW,
        ansi::RESET
    );
    let output_dir = config.resolve_output_dir(output.dir.as_deref())?;
    for (measurement, surface) in data.iter().zip(surfaces.iter()) {
        let datetime = base::utils::iso_timestamp_short();
        let filepath = cache.read(|cache| {
            let surf_name = cache
                .get_micro_surface_filepath(*surface)
                .unwrap()
                .file_stem()
                .unwrap()
                .to_ascii_lowercase();
            output_dir.join(format!(
                "{}_{}_{}",
                measurement.kind().ascii_str(),
                surf_name.to_str().unwrap(),
                datetime
            ))
        });
        println!(
            "      {}-{} Saving to \"{}\"",
            ansi::BRIGHT_CYAN,
            ansi::RESET,
            filepath.display()
        );

        // // TODO: to be removed
        // if let MeasuredData::Bsdf(brdf) = &measurement.measured {
        //     for snapshot in brdf.snapshots.iter() {
        //         print!("        {:?}: ", snapshot.wi);
        //         for sample in snapshot.samples.iter() {
        //             print!("{:?} ", sample[0]);
        //         }
        //         println!();
        //     }
        // }

        for format in output.formats.iter() {
            match measurement.write_to_file(&filepath, format) {
                Ok(_) => {
                    println!(
                        "      {} Successfully saved to \"{}\"",
                        ansi::CYAN_CHECK,
                        output_dir.display()
                    );
                }
                Err(err) => {
                    eprintln!(
                        "        {} Failed to save to \"{}\": {}",
                        ansi::RED_EXCLAMATION,
                        filepath.display(),
                        err
                    );
                }
            }
        }
    }
    Ok(())
}

/// Writes a single measurement data to a file.
///
/// Depending on the file extension, the data is saved as either a VGMO file or
/// an EXR file.
pub fn write_single_measured_data_to_file(
    measured: &Measurement,
    encoding: FileEncoding,
    compression: CompressionScheme,
    resolution: Option<u32>,
    filepath: &Path,
) -> Result<(), VgonioError> {
    use std::ffi::OsStr;

    println!("    {} Saving measurement data...", ansi::YELLOW_GT);
    println!(
        "      {} Saving as \"{}\"",
        ansi::CYAN_MINUS,
        filepath.display()
    );

    let ext = filepath
        .extension()
        .unwrap_or(OsStr::new("vgmo"))
        .to_str()
        .unwrap();

    if ext != "vgmo" && ext != "exr" {
        return Err(VgonioError::new("Unknown file format", None));
    }

    let format = if ext == "exr" {
        OutputFileFormatOption::Exr {
            resolution: resolution.unwrap_or(512),
        }
    } else {
        OutputFileFormatOption::Vgmo {
            encoding,
            compression,
        }
    };

    match measured.write_to_file(&filepath, &format) {
        Ok(_) => {
            println!(
                "      {} Successfully saved as \"{}\"",
                ansi::CYAN_CHECK,
                filepath.display()
            );
        }
        Err(err) => {
            eprintln!(
                "        {} Failed to save as \"{}\": {}",
                ansi::RED_EXCLAMATION,
                filepath.display(),
                err
            );
        }
    }
    Ok(())
}
