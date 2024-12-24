use crate::{
    app::{cache::Cache, cli::ansi, Config},
    measure::{
        bsdf::MeasuredBsdfData,
        mfd::{MeasuredGafData, MeasuredNdfData},
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
            BsdfMeasurementParams, GafMeasurementParams, NdfMeasurementMode, NdfMeasurementParams,
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
        partition::{PartitionScheme, Ring, SphericalDomain, SphericalPartition},
        range::StepRangeIncl,
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
    #[derive(Debug, Clone, PartialEq)]
    pub enum VgmoHeaderExt {
        Bsdf { params: BsdfMeasurementParams },
        Ndf { params: NdfMeasurementParams },
        Gaf { params: GafMeasurementParams },
        Sdf,
    }

    /// Returns the corresponding [`VgmoHeaderExt`] variant for the given
    /// measurement data.
    pub fn vgmo_header_ext_from_data(data: &(dyn MeasuredData + 'static)) -> VgmoHeaderExt {
        match data.kind() {
            MeasurementKind::Bsdf => {
                let bsdf = data.downcast_ref::<MeasuredBsdfData>().unwrap();
                VgmoHeaderExt::Bsdf {
                    params: bsdf.params.clone(),
                }
            },
            MeasurementKind::Ndf => {
                let ndf = data.downcast_ref::<MeasuredNdfData>().unwrap();
                VgmoHeaderExt::Ndf { params: ndf.params }
            },
            MeasurementKind::Gaf => {
                let msf = data.downcast_ref::<MeasuredGafData>().unwrap();
                VgmoHeaderExt::Gaf { params: msf.params }
            },
            MeasurementKind::Sdf => VgmoHeaderExt::Sdf,
            _ => {
                unreachable!("Unsupported measurement kind: {}", data.kind())
            },
        }
    }

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
                    Self::Ndf { params } => {
                        match &params.mode {
                            NdfMeasurementMode::ByPoints { azimuth, zenith } => {
                                log::debug!("Writing NDF header ext (by points) to VGMO file");
                                writer.write_all(&[
                                    MeasurementKind::Ndf as u8,
                                    0, // by points
                                    params.crop_to_disk as u8,
                                    params.use_facet_area as u8,
                                ])?;
                                write_adf_or_msf_params_to_vgmo(azimuth, zenith, writer, true)?;
                            },
                            NdfMeasurementMode::ByPartition { precision } => {
                                log::debug!("Writing NDF header ext (by partition) to VGMO file");
                                writer.write_all(&[
                                    MeasurementKind::Ndf as u8,
                                    1, // by partition
                                    params.crop_to_disk as u8,
                                    params.use_facet_area as u8,
                                ])?;
                                let partition = SphericalPartition::new(
                                    PartitionScheme::Beckers,
                                    SphericalDomain::Upper,
                                    Sph2::new(*precision, rad!(0.0)),
                                );
                                let mut buf = vec![0u8; partition.total_required_size()];
                                partition.write_to_buf(&mut buf);
                                writer.write_all(&buf)?;
                            },
                        }
                    },
                    Self::Gaf { params } => {
                        writer.write_all(&[MeasurementKind::Gaf as u8])?;
                        write_adf_or_msf_params_to_vgmo(
                            &params.azimuth,
                            &params.zenith,
                            writer,
                            false,
                        )?;
                    },
                    Self::Bsdf { params } => {
                        writer.write_all(&[MeasurementKind::Bsdf as u8])?;
                        params.write_to_vgmo(version, writer)?;
                    },
                    VgmoHeaderExt::Sdf => {
                        writer.write_all(&[MeasurementKind::Sdf as u8])?;
                    },
                },
                _ => {
                    log::error!("Unsupported VGMO version: {}", version.as_string());
                },
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
                MeasurementKind::Gaf => GafMeasurementParams::read_from_vgmo(version, reader)
                    .map(|params| Ok(Self::Gaf { params }))?,
                MeasurementKind::Sdf => Ok(Self::Sdf),
                _ => {
                    unreachable!("Unsupported measurement kind: {}", kind[0])
                },
            }
        }
    }

    /// Reads the VGMO file from the given reader.
    pub fn read<R: Read>(
        reader: &mut BufReader<R>,
        header: &Header<VgmoHeaderExt>,
    ) -> Result<Box<dyn MeasuredData>, ReadFileErrorKind> {
        // TODO: Handle multiple receivers
        match &header.extra {
            VgmoHeaderExt::Bsdf { params } => {
                log::debug!(
                    "Reading BSDF data of {} measurement points {} samples from VGMO file",
                    params.emitter.measurement_points_count(),
                    params.samples_count(0).unwrap()
                );
                Ok(Box::new(MeasuredBsdfData::read_from_vgmo(
                    reader,
                    params,
                    header.meta.encoding,
                    header.meta.compression,
                )?))
            },
            VgmoHeaderExt::Ndf { params } => {
                log::debug!(
                    "Reading ADF data of {} samples from VGMO file",
                    params.samples_count()
                );
                let samples = io::read_f32_data_samples(
                    reader,
                    params.samples_count(),
                    header.meta.encoding,
                    header.meta.compression,
                )
                .map_err(ReadFileErrorKind::Parse)?;
                Ok(Box::new(MeasuredNdfData {
                    params: *params,
                    samples,
                }))
            },
            VgmoHeaderExt::Gaf { params } => {
                log::debug!(
                    "Reading MSF data of {} samples from VGMO file",
                    params.samples_count()
                );
                let samples = io::read_f32_data_samples(
                    reader,
                    params.samples_count(),
                    header.meta.encoding,
                    header.meta.compression,
                )
                .map_err(ReadFileErrorKind::Parse)?;
                Ok(Box::new(MeasuredGafData {
                    params: *params,
                    samples,
                }))
            },
            VgmoHeaderExt::Sdf => {
                log::debug!("Reading SDF data from VGMO file");
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                let num_slopes = u32::from_le_bytes(buf);
                let samples = io::read_f32_data_samples(
                    reader,
                    num_slopes as usize * 2,
                    header.meta.encoding,
                    header.meta.compression,
                )?;
                let slopes = samples
                    .chunks(2)
                    .map(|s| math::Vec2::new(s[0], s[1]))
                    .collect::<Box<_>>();
                Ok(Box::new(MeasuredSdfData {
                    params: SdfMeasurementParams {
                        max_slope: std::f32::consts::PI,
                    },
                    slopes,
                }))
            },
        }
    }

    /// Writes the given measurement data to the given writer.
    pub fn write<W: Write + Seek>(
        writer: &mut BufWriter<W>,
        header: Header<VgmoHeaderExt>,
        measured: &(dyn MeasuredData + 'static),
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
            mfd @ (MeasurementKind::Ndf | MeasurementKind::Gaf) => {
                let (samples, cols) = match mfd {
                    MeasurementKind::Ndf => {
                        let ndf = measured.downcast_ref::<MeasuredNdfData>().unwrap();
                        let cols = match &ndf.params.mode {
                            NdfMeasurementMode::ByPoints { zenith, .. } => {
                                zenith.step_count_wrapped()
                            },
                            NdfMeasurementMode::ByPartition { .. } => ndf.samples.len(),
                        };
                        (&ndf.samples, cols)
                    },
                    MeasurementKind::Gaf => {
                        let msf = measured.downcast_ref::<MeasuredGafData>().unwrap();
                        (&msf.samples, msf.params.zenith.step_count_wrapped())
                    },
                    _ => {
                        unreachable!("Unsupported measurement kind: {}", mfd)
                    },
                };
                match header.meta.encoding {
                    FileEncoding::Ascii => {
                        io::write_data_samples_ascii(writer, samples, cols as u32)
                    },
                    FileEncoding::Binary => {
                        io::write_f32_data_samples_binary(writer, header.meta.compression, samples)
                    },
                }
                .map_err(WriteFileErrorKind::Write)?;
            },
            MeasurementKind::Bsdf => {
                let bsdf = measured.downcast_ref::<MeasuredBsdfData>().unwrap();
                bsdf.write_to_vgmo(writer, header.meta.encoding, header.meta.compression)?;
            },
            MeasurementKind::Sdf => {
                let sdf = measured.downcast_ref::<MeasuredSdfData>().unwrap();
                let _ = writer.write(&(sdf.slopes.len() as u32).to_le_bytes())?;
                let samples = unsafe {
                    std::slice::from_raw_parts(
                        sdf.slopes.as_ptr() as *const f32,
                        sdf.slopes.len() * 2,
                    )
                };
                io::write_f32_data_samples_binary(writer, header.meta.compression, samples)
                    .map_err(WriteFileErrorKind::Write)?;
            },
            _ => {},
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
        zenith: &StepRangeIncl<Radians>,
        azimuth: &StepRangeIncl<Radians>,
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
        azimuth: &StepRangeIncl<Radians>,
        zenith: &StepRangeIncl<Radians>,
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
        pub const fn required_size(version: Version, nrays64: bool) -> Option<usize> {
            match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    if nrays64 {
                        Some(60)
                    } else {
                        Some(56)
                    }
                },
                _ => None,
            }
        }

        /// Reads an emitter from the given buffer.
        pub fn read_from_buf(version: Version, buf: &[u8], nrays64: bool) -> Self {
            let required_size = Self::required_size(version, nrays64).unwrap();
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
                    log::debug!("Reading emitter from buffer, nrays64: {}", nrays64);
                    let (mut offset, num_rays) = if nrays64 {
                        (8, u64::from_le_bytes(buf[0..8].try_into().unwrap()))
                    } else {
                        (4, u32::from_le_bytes(buf[0..4].try_into().unwrap()) as u64)
                    };

                    log::debug!("Reading emitter with {} rays", num_rays);

                    let max_bounces =
                        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
                    offset += 4;

                    let azimuth =
                        StepRangeIncl::<Radians>::read_from_buf(&buf[offset..offset + 16]);
                    offset += 16;
                    let zenith = StepRangeIncl::<Radians>::read_from_buf(&buf[offset..offset + 16]);
                    offset += 16;
                    let spectrum =
                        StepRangeIncl::<Nanometres>::read_from_buf(&buf[offset..offset + 16]);
                    Self {
                        num_rays,
                        num_sectors: 1,
                        max_bounces,
                        zenith,
                        azimuth,
                        spectrum,
                    }
                },
                _ => {
                    panic!("Unsupported VGMO[EmitterPrams] version: {}", version);
                },
            }
        }

        /// Writes the emitter to the given buffer and returns the number of
        /// bytes written.
        pub fn write_to_buf(&self, version: Version, buf: &mut [u8]) -> usize {
            let nrays64 = self.num_rays > u32::MAX as u64;
            let required_size = Self::required_size(version, nrays64).unwrap();
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
                    let mut offset = if nrays64 {
                        buf[0..8].copy_from_slice(&self.num_rays.to_le_bytes());
                        8
                    } else {
                        buf[0..4].copy_from_slice(&(self.num_rays as u32).to_le_bytes());
                        4
                    };
                    buf[offset..offset + 4].copy_from_slice(&self.max_bounces.to_le_bytes());
                    offset += 4;
                    self.azimuth.write_to_buf(&mut buf[offset..offset + 16]);
                    offset += 16;
                    self.zenith.write_to_buf(&mut buf[offset..offset + 16]);
                    offset += 16;
                    self.spectrum.write_to_buf(&mut buf[offset..offset + 16]);
                    required_size
                },
                _ => {
                    panic!("Unsupported VGMO[EmitterParams] version: {}", version);
                },
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
                    let (domain, scheme, precision, n_rings, n_patches) =
                        SphericalPartition::read_skipping_rings(reader);
                    let params = ReceiverParams {
                        domain,
                        precision,
                        scheme,
                    };
                    let expected_num_rings = params.num_rings();
                    debug_assert!(
                        n_rings == expected_num_rings,
                        "Receiver's partition ring count does not match the precision",
                    );
                    let expected_num_patches = params.num_patches();
                    debug_assert!(
                        n_patches == expected_num_patches,
                        "Receiver's partition patch count does not match the precision",
                    );
                    params
                },
                _ => {
                    panic!("Unsupported VGMO version: {}", version);
                },
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
                    self.partitioning().write_to_buf(buf);
                },
                _ => {
                    log::error!("Unsupported VGMO[ReceiverParams] version: {version}");
                },
            }
        }
    }

    impl NdfMeasurementParams {
        /// Reads the NDF measurement parameters from the given reader.
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
                    let mut common_info = [0u8; 3];
                    // Read `measurement mode`, `crop to disk`, and `use facet area`
                    reader.read_exact(&mut common_info)?;
                    log::debug!(
                        "Reading NDF measurement params from VGMO file: {}, crop: {}, facet-area: \
                         {}",
                        if common_info[0] == 0 {
                            "By points"
                        } else {
                            "By partition"
                        },
                        common_info[1] != 0,
                        common_info[2] != 0
                    );
                    let mode = match common_info[0] {
                        // By points
                        0 => {
                            let mut buf = [0u8; 36];
                            reader.read_exact(&mut buf)?;
                            let azimuth = StepRangeIncl::<Radians>::read_from_buf(&buf[0..16]);
                            let zenith = StepRangeIncl::<Radians>::read_from_buf(&buf[16..32]);
                            let sample_count = u32::from_le_bytes(buf[32..36].try_into().unwrap());
                            debug_assert_eq!(
                                sample_count as usize,
                                NdfMeasurementParams::expected_samples_count_by_points(
                                    &azimuth, &zenith
                                )
                            );
                            NdfMeasurementMode::ByPoints { azimuth, zenith }
                        },
                        1 => {
                            let (domain, scheme, precision, _n_rings, _n_patches) =
                                SphericalPartition::read_skipping_rings(reader);
                            debug_assert_eq!(
                                domain,
                                SphericalDomain::Upper,
                                "Only upper domain is supported for NDF"
                            );
                            debug_assert_eq!(
                                scheme,
                                PartitionScheme::Beckers,
                                "Only Beckers partition scheme is supported for NDF"
                            );
                            NdfMeasurementMode::ByPartition {
                                precision: precision.theta,
                            }
                        },
                        _ => {
                            panic!("Invalid NDF measurement mode");
                        },
                    };

                    Ok(Self {
                        mode,
                        crop_to_disk: common_info[1] != 0,
                        use_facet_area: common_info[2] != 0,
                    })
                },
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unsupported VGMO[AdfMeasurementParams] version {}", version),
                )),
            }
        }
    }

    impl GafMeasurementParams {
        // TODO: resolve resolution and strict
        /// Reads the MSF measurement parameters from the given reader.
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
                    let azimuth = StepRangeIncl::<Radians>::read_from_buf(&buf[0..16]);
                    let zenith = StepRangeIncl::<Radians>::read_from_buf(&buf[16..32]);
                    let sample_count = u32::from_le_bytes(buf[32..36].try_into().unwrap());
                    debug_assert_eq!(
                        sample_count as usize,
                        GafMeasurementParams::expected_samples_count(&azimuth, &zenith)
                    );
                    Ok(Self {
                        azimuth,
                        zenith,
                        resolution: 512,
                        strict: true,
                    })
                },
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
                    let mut buf = [0u8; 11];
                    reader.read_exact(&mut buf)?;
                    let kind = BsdfKind::from(buf[0]);
                    let incident_medium = Medium::read_from_buf(&buf[1..4]);
                    let transmitted_medium = Medium::read_from_buf(&buf[4..7]);
                    let sim_kind = SimulationKind::try_from(buf[7]).unwrap();
                    let fresnel = buf[8] != 0; // [9] padding bytes
                    let nrays64 = buf[10] == 0xff;

                    let emitter = {
                        let mut buf =
                            vec![0u8; EmitterParams::required_size(version, nrays64).unwrap()]
                                .into_boxed_slice();
                        reader.read_exact(&mut buf)?;
                        EmitterParams::read_from_buf(version, &buf, nrays64)
                    };
                    // TODO: read multiple receivers
                    let receiver = ReceiverParams::read(version, reader);
                    Ok(Self {
                        kind,
                        incident_medium,
                        transmitted_medium,
                        sim_kind,
                        emitter,
                        receivers: vec![receiver],
                        fresnel,
                    })
                },
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
            // TODO: write multiple receivers
            let receiver = &self.receivers[0];
            let nrays64 = self.emitter.num_rays > u32::MAX as u64;
            let buf_size = match version {
                Version {
                    major: 0,
                    minor: 1,
                    patch: 0,
                } => {
                    11 + EmitterParams::required_size(version, nrays64).unwrap()
                        + ReceiverParams::required_size(version, receiver.num_rings()).unwrap()
                },
                _ => {
                    log::error!("Unsupported VGMO version: {}", version.as_string());
                    return Ok(());
                },
            };
            let mut buf = vec![0u8; buf_size].into_boxed_slice();
            buf[0] = self.kind as u8;
            self.incident_medium.write_to_buf(&mut buf[1..4]);
            self.transmitted_medium.write_to_buf(&mut buf[4..7]);
            buf[7] = self.sim_kind.as_u8();
            buf[8] = self.fresnel as u8;
            buf[9] = 0; // padding, reserved for num receivers
            buf[10] = 0; // Type of number for rays,
            if self.emitter.num_rays > u32::MAX as u64 {
                log::debug!("Emitter num rays exceeds u32 max, using u64");
                buf[10] = 0xff;
            }

            let n_written = self.emitter.write_to_buf(version, &mut buf[11..]);
            log::debug!("Written {} bytes for emitter", n_written);
            receiver.write_to_buf(version, &mut buf[n_written + 11..]);
            writer.write_all(&buf)
        }
    }

    pub trait LittleEndianRead: Sized {
        fn read_le_bytes(src: &[u8]) -> Self;
    }

    pub trait LittleEndianWrite: Sized {
        fn write_le_bytes(&self, dst: &mut [u8]);
    }

    macro_rules! impl_endian_read_write {
        ($($t:ty),*) => {
            $(
                impl LittleEndianRead for $t {
                    fn read_le_bytes(src: &[u8]) -> Self {
                        Self::from_le_bytes(src.try_into().expect("Invalid byte slice!"))
                    }
                }

                impl LittleEndianWrite for $t {
                    fn write_le_bytes(&self, dst: &mut [u8]) {
                        dst.copy_from_slice(&self.to_le_bytes())
                    }
                }
            )*
        };
    }

    impl_endian_read_write!(u32, f32, u64, f64);

    /// Writes the given slice to the buffer in little-endian format.
    #[track_caller]
    pub fn write_slice_to_buf<T: LittleEndianWrite>(src: &[T], dst: &mut [u8]) {
        let size: usize = size_of::<T>();
        debug_assert!(
            dst.len() >= size_of_val(src),
            "Write array to buffer: desired size {}, got {}",
            size_of_val(src),
            dst.len()
        );
        for i in 0..src.len() {
            src[i].write_le_bytes(&mut dst[i * size..(i + 1) * size]);
        }
    }

    #[track_caller]
    pub fn write_u64_slice_as_u32_to_buf(src: &[u64], dst: &mut [u8]) {
        let size: usize = size_of::<u32>();
        debug_assert!(
            dst.len() >= src.len() * size,
            "Write array to buffer: desired size {}, got {}",
            src.len() * size,
            dst.len()
        );
        for i in 0..src.len() {
            let val = src[i] as u32;
            dst[i * size..(i + 1) * size].copy_from_slice(&val.to_le_bytes());
        }
    }

    pub fn write_f64_slice_as_f32_to_buf(src: &[f64], dst: &mut [u8]) {
        let size: usize = mem::size_of::<f32>();
        debug_assert!(
            dst.len() >= src.len() * size,
            "Write array to buffer: desired size {}, got {}",
            src.len() * size,
            dst.len()
        );
        for i in 0..src.len() {
            let val = src[i] as f32;
            dst[i * size..(i + 1) * size].copy_from_slice(&val.to_le_bytes());
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
    pub fn read_slice_from_buf<T: LittleEndianRead>(src: &[u8], dst: &mut [T], len: usize) {
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

    pub fn read_u32_slice_as_u64_from_buf(src: &[u8], dst: &mut [u64], len: usize) {
        let size: usize = mem::size_of::<u32>();
        assert_eq!(
            src.len(),
            len * size,
            "Buffer size mismatch, expected {} bytes, got {}",
            len * size,
            src.len()
        );
        for i in 0..len {
            let val = u32::from_le_bytes(src[i * size..(i + 1) * size].try_into().unwrap());
            dst[i] = val as u64;
        }
    }

    pub fn read_f32_slice_as_f64_from_buf(src: &[u8], dst: &mut [f64], len: usize) {
        let size: usize = mem::size_of::<f32>();
        assert_eq!(
            src.len(),
            len * size,
            "Buffer size mismatch, expected {} bytes, got {}",
            len * size,
            src.len()
        );
        for i in 0..len {
            let val = f32::from_le_bytes(src[i * size..(i + 1) * size].try_into().unwrap());
            dst[i] = val as f64;
        }
    }

    impl SingleBsdfMeasurementStats {
        /// The size of the buffer required to read/write the BSDF measurement
        /// stats point.
        pub fn size_in_bytes(n_spectrum: usize, n_bounce: usize, nrays64: bool) -> usize {
            if nrays64 {
                size_of::<u32>()
                    + 2 * size_of::<u64>()
                    + n_spectrum * Self::N_STATS * size_of::<u64>()
                    + n_spectrum * size_of::<f64>()
                    + n_spectrum * n_bounce * 2 * size_of::<f64>()
            } else {
                3 * mem::size_of::<u32>()
                    + Self::N_STATS * n_spectrum * mem::size_of::<u32>()
                    + n_spectrum * mem::size_of::<f32>()
                    + n_spectrum * n_bounce * 2 * mem::size_of::<f32>()
            }
        }

        pub fn size_in_bytes_without_totals(
            n_spectrum: usize,
            n_bounce: usize,
            nrays64: bool,
        ) -> usize {
            if nrays64 {
                n_spectrum * Self::N_STATS * size_of::<u64>()
                    + n_spectrum * size_of::<f64>()
                    + n_spectrum * n_bounce * 2 * size_of::<f64>()
            } else {
                Self::N_STATS * n_spectrum * mem::size_of::<u32>()
                    + n_spectrum * mem::size_of::<f32>()
                    + n_spectrum * n_bounce * 2 * mem::size_of::<f32>()
            }
        }

        /// Writes the BSDF measurement statistics at a single point to the
        /// writer.
        pub fn write<W: Write>(
            &self,
            writer: &mut BufWriter<W>,
            n_spectrum: usize,
            nrays64: bool,
        ) -> Result<(), std::io::Error> {
            let size = Self::size_in_bytes(n_spectrum, self.n_bounce as usize, nrays64);
            let mut buf = vec![0u8; size].into_boxed_slice();
            let mut offset_in_bytes = 0;
            buf[offset_in_bytes..offset_in_bytes + 4].copy_from_slice(&self.n_bounce.to_le_bytes());
            offset_in_bytes += 4;

            if nrays64 {
                buf[offset_in_bytes..offset_in_bytes + 8]
                    .copy_from_slice(&self.n_received.to_le_bytes());
                offset_in_bytes += 8;

                buf[offset_in_bytes..offset_in_bytes + 8]
                    .copy_from_slice(&self.n_missed.to_le_bytes());
                offset_in_bytes += 8;

                // Write n_absorbed, n_reflected, n_captured and n_escaped per wavelength
                write_slice_to_buf::<u64>(
                    &self.n_ray_stats,
                    &mut buf[offset_in_bytes..offset_in_bytes + 8 * n_spectrum * Self::N_STATS],
                );
                offset_in_bytes += 8 * n_spectrum * Self::N_STATS;

                write_slice_to_buf(
                    &self.e_captured,
                    &mut buf[offset_in_bytes..offset_in_bytes + 8 * n_spectrum],
                );
                offset_in_bytes += 8 * n_spectrum;

                write_slice_to_buf(
                    &self.n_ray_per_bounce,
                    &mut buf[offset_in_bytes
                        ..offset_in_bytes + 8 * n_spectrum * self.n_bounce as usize],
                );
                offset_in_bytes += 8 * n_spectrum * self.n_bounce as usize;

                write_slice_to_buf(
                    &self.energy_per_bounce,
                    &mut buf[offset_in_bytes
                        ..offset_in_bytes + 8 * n_spectrum * self.n_bounce as usize],
                );
            } else {
                let n_received = self.n_received as u32;
                buf[offset_in_bytes..offset_in_bytes + 4]
                    .copy_from_slice(&n_received.to_le_bytes());
                offset_in_bytes += 4;

                let n_missed = self.n_missed as u32;
                buf[offset_in_bytes..offset_in_bytes + 4].copy_from_slice(&n_missed.to_le_bytes());
                offset_in_bytes += 4;

                write_u64_slice_as_u32_to_buf(
                    &self.n_ray_stats,
                    &mut buf[offset_in_bytes..offset_in_bytes + 4 * n_spectrum * Self::N_STATS],
                );
                offset_in_bytes += 4 * n_spectrum * Self::N_STATS;

                write_f64_slice_as_f32_to_buf(
                    &self.e_captured,
                    &mut buf[offset_in_bytes..offset_in_bytes + 4 * n_spectrum],
                );

                write_u64_slice_as_u32_to_buf(
                    &self.n_ray_per_bounce,
                    &mut buf[offset_in_bytes
                        ..offset_in_bytes + 4 * n_spectrum * self.n_bounce as usize],
                );
                offset_in_bytes += 4 * n_spectrum * self.n_bounce as usize;

                write_f64_slice_as_f32_to_buf(
                    &self.energy_per_bounce,
                    &mut buf[offset_in_bytes
                        ..offset_in_bytes + 4 * n_spectrum * self.n_bounce as usize],
                );
            }

            writer.write_all(&buf)
        }

        /// Reads the BSDF measurement statistics at a single point from the
        /// buffer.
        pub fn read<R: Read>(
            reader: &mut R,
            n_spectrum: usize,
            nrays64: bool,
        ) -> Result<Self, std::io::Error> {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            let n_bounce = u32::from_le_bytes(buf);

            let (n_received, n_missed) = if nrays64 {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                let n_received = u64::from_le_bytes(buf);
                reader.read_exact(&mut buf)?;
                let n_missed = u64::from_le_bytes(buf);
                (n_received, n_missed)
            } else {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                let n_received = u32::from_le_bytes(buf);
                reader.read_exact(&mut buf)?;
                let n_missed = u32::from_le_bytes(buf);
                (n_received as u64, n_missed as u64)
            };

            let mut buf =
                vec![
                    0u8;
                    Self::size_in_bytes_without_totals(n_spectrum, n_bounce as usize, nrays64)
                ]
                .into_boxed_slice();
            reader.read_exact(&mut buf)?;

            let mut offset = 0;
            let mut n_ray_stats = vec![0u64; n_spectrum * Self::N_STATS].into_boxed_slice();
            let mut e_captured = vec![0f64; n_spectrum].into_boxed_slice();
            let mut n_ray_per_bounce =
                vec![0u64; n_spectrum * n_bounce as usize].into_boxed_slice();
            let mut energy_per_bounce =
                vec![0f64; n_spectrum * n_bounce as usize].into_boxed_slice();

            if nrays64 {
                let n_ray_stats_size = 8 * n_ray_stats.len();
                read_slice_from_buf::<u64>(
                    &buf[offset..n_ray_stats_size],
                    &mut n_ray_stats,
                    n_spectrum * Self::N_STATS,
                );
                offset += n_ray_stats_size;

                let e_captured_size = 8 * e_captured.len();
                read_slice_from_buf::<f64>(
                    &buf[offset..offset + e_captured_size],
                    &mut e_captured,
                    n_spectrum,
                );
                offset += e_captured_size;

                read_slice_from_buf::<u64>(
                    &buf[offset..offset + n_spectrum * n_bounce as usize * 8],
                    &mut n_ray_per_bounce,
                    n_spectrum * n_bounce as usize,
                );
                offset += n_spectrum * n_bounce as usize * 8;

                read_slice_from_buf::<f64>(
                    &buf[offset..offset + n_spectrum * n_bounce as usize * 8],
                    &mut energy_per_bounce,
                    n_spectrum * n_bounce as usize,
                );
            } else {
                let n_ray_stats_size = 4 * n_ray_stats.len();
                read_u32_slice_as_u64_from_buf(
                    &buf[offset..n_ray_stats_size],
                    &mut n_ray_stats,
                    n_spectrum * Self::N_STATS,
                );
                offset += n_ray_stats_size;

                let e_captured_size = 4 * e_captured.len();
                read_f32_slice_as_f64_from_buf(
                    &buf[offset..offset + e_captured_size],
                    &mut e_captured,
                    n_spectrum,
                );
                offset += e_captured_size;

                read_u32_slice_as_u64_from_buf(
                    &buf[offset..offset + n_spectrum * n_bounce as usize * 4],
                    &mut n_ray_per_bounce,
                    n_spectrum * n_bounce as usize,
                );
                offset += n_spectrum * n_bounce as usize * 4;

                read_f32_slice_as_f64_from_buf(
                    &buf[offset..offset + n_spectrum * n_bounce as usize * 4],
                    &mut energy_per_bounce,
                    n_spectrum * n_bounce as usize,
                );
            }

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
        pub fn size_in_bytes(n_bounces: usize, nrays64: bool) -> usize {
            if nrays64 {
                size_of::<u32>() + (n_bounces + 1) * 2 * size_of::<f64>()
            } else {
                size_of::<u32>() + (n_bounces + 1) * 2 * size_of::<f32>()
            }
        }

        /// Reads the data for one single patch.
        pub fn read<R: Read>(
            reader: &mut R,
            nrays64: bool,
        ) -> Result<Option<Self>, std::io::Error> {
            let n_bounce = {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                u32::from_le_bytes(buf)
            };

            if n_bounce == u32::MAX {
                return Ok(None);
            }

            let mut energy_per_bounce = vec![0f64; n_bounce as usize + 1].into_boxed_slice();
            let mut n_ray_per_bounce = vec![0u64; n_bounce as usize + 1].into_boxed_slice();

            if nrays64 {
                let mut buf = vec![0u8; (n_bounce as usize + 1) * 8].into_boxed_slice();

                reader.read_exact(&mut buf)?;
                for (i, val_buf) in buf.chunks(8).enumerate() {
                    n_ray_per_bounce[i] = u64::from_le_bytes(val_buf.try_into().unwrap());
                }

                reader.read_exact(&mut buf)?;
                for (i, val_buf) in buf.chunks(8).enumerate() {
                    energy_per_bounce[i] = f64::from_le_bytes(val_buf.try_into().unwrap());
                }
            } else {
                let mut buf = vec![0u8; (n_bounce as usize + 1) * 4].into_boxed_slice();

                reader.read_exact(&mut buf)?;
                for (i, val_buf) in buf.chunks(4).enumerate() {
                    n_ray_per_bounce[i] = u32::from_le_bytes(val_buf.try_into().unwrap()) as u64;
                }

                reader.read_exact(&mut buf)?;
                for (i, val_buf) in buf.chunks(4).enumerate() {
                    energy_per_bounce[i] = f32::from_le_bytes(val_buf.try_into().unwrap()) as f64;
                }
            }

            Ok(Some(Self {
                n_bounce,
                n_ray_per_bounce,
                energy_per_bounce,
            }))
        }

        /// Writes the data for one single patch.
        pub fn write<W: Write>(
            writer: &mut BufWriter<W>,
            be: &Option<BounceAndEnergy>,
            nrays64: bool,
        ) -> Result<(), std::io::Error> {
            match be {
                None => writer.write_all(&u32::MAX.to_le_bytes()),
                Some(be) => {
                    let size = Self::size_in_bytes(be.n_bounce as usize, nrays64);
                    let mut buf = vec![0u8; size].into_boxed_slice();
                    buf[0..4].copy_from_slice(&be.n_bounce.to_le_bytes());
                    let mut offset = 4;
                    if nrays64 {
                        write_slice_to_buf::<u64>(&be.n_ray_per_bounce, &mut buf[offset..]);
                        offset += 8 * (be.n_bounce as usize + 1);
                        write_slice_to_buf::<f64>(&be.energy_per_bounce, &mut buf[offset..]);
                    } else {
                        write_u64_slice_as_u32_to_buf(&be.n_ray_per_bounce, &mut buf[offset..]);
                        offset += 4 * (be.n_bounce as usize + 1);
                        write_f64_slice_as_f32_to_buf(&be.energy_per_bounce, &mut buf[offset..]);
                    }
                    writer.write_all(&buf)
                },
            }
        }
    }

    impl MeasuredBsdfData {
        pub(crate) fn write_vgonio_brdf<W: Write>(
            writer: &mut W,
            level: MeasuredBrdfLevel,
            brdf: &VgonioBrdf,
        ) -> Result<(), std::io::Error> {
            writer.write_all(&level.as_u32().to_le_bytes())?;
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
                MeasuredBrdfLevel::from(u32::from_le_bytes(buf))
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
                    },
                    Err(_) => {
                        break;
                    },
                }
            }
            Ok(brdfs)
        }

        pub(crate) fn write_raw_measured_data<W: Write>(
            writer: &mut W,
            raw: &RawMeasuredBsdfData,
            nrays64: bool,
        ) -> Result<(), std::io::Error> {
            let mut writer = BufWriter::new(writer);
            // Collected data of the receiver per incident direction per
            // outgoing(patch) direction and per wavelength: `ωi, ωo, λ`.
            for record in raw.records.iter() {
                BounceAndEnergy::write(&mut writer, record, nrays64)?;
            }
            // Writes the statistics of the measurement per incident direction
            for stats in raw.stats.iter() {
                stats.write(&mut writer, raw.spectrum.len(), nrays64)?;
            }
            writer.flush()
        }

        pub(crate) fn read_raw_measured_data<R: Read>(
            reader: &mut R,
            n_wi: usize,
            n_wo: usize,
            n_spectrum: usize,
            n_zenith_in: usize,
            spectrum: &[Nanometres],
            incoming: &[Sph2],
            partition: SphericalPartition,
            nrays64: bool,
        ) -> Result<RawMeasuredBsdfData, std::io::Error> {
            debug_assert_eq!(n_spectrum, spectrum.len(), "Spectrum size mismatch");
            let mut records = vec![None; n_wi * n_wo * n_spectrum];
            for record in records.iter_mut() {
                *record = BounceAndEnergy::read(reader, nrays64)?;
            }

            let mut stats = Box::new_uninit_slice(n_wi);
            for stat in stats.iter_mut() {
                stat.write(SingleBsdfMeasurementStats::read(
                    reader, n_spectrum, nrays64,
                )?);
            }

            Ok(unsafe {
                RawMeasuredBsdfData {
                    n_zenith_in,
                    spectrum: DyArr::from_slice_1d(spectrum),
                    incoming: DyArr::from_slice_1d(incoming),
                    outgoing: partition,
                    records: DyArr::from_vec([n_wi, n_wo, n_spectrum], records),
                    stats: DyArr::from_boxed_slice_1d(stats.assume_init()),
                    #[cfg(feature = "vdbg")]
                    trajectories: Box::new([]),
                    #[cfg(feature = "vdbg")]
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
            // TODO: handle multiple receivers
            match encoding {
                FileEncoding::Ascii => Err(ReadFileErrorKind::UnsupportedEncoding),
                FileEncoding::Binary => {
                    let mut zlib_decoder;
                    let mut gzip_decoder;
                    let mut decoder: Box<&mut dyn Read> = match compression {
                        CompressionScheme::Zlib => {
                            zlib_decoder = flate2::bufread::ZlibDecoder::new(reader);
                            Box::new(&mut zlib_decoder)
                        },
                        CompressionScheme::Gzip => {
                            gzip_decoder = flate2::bufread::GzDecoder::new(reader);
                            Box::new(&mut gzip_decoder)
                        },
                        _ => Box::new(reader),
                    };
                    let spectrum = params.emitter.spectrum.values().collect::<Vec<_>>();
                    let incoming = params.emitter.generate_measurement_points().0;
                    let receiver = &params.receivers[0];
                    let partition = receiver.partitioning();
                    let n_wi = params.n_wi();
                    let n_wo = receiver.num_patches();
                    let n_spectrum = params.n_spectrum();
                    let raw = Self::read_raw_measured_data(
                        &mut decoder,
                        n_wi,
                        n_wo,
                        n_spectrum,
                        params.emitter.measurement_points_zenith_count(),
                        &spectrum,
                        &incoming,
                        partition.clone(),
                        params.emitter.num_rays > u32::MAX as u64,
                    )?;
                    let mut bsdfs =
                        Self::read_measured_bsdf_data(&mut decoder, n_wi, n_wo, n_spectrum)?;

                    let parameterisation = VgonioBrdfParameterisation {
                        n_zenith_i: params.emitter.measurement_points_zenith_count(),
                        incoming: DyArr::from_boxed_slice_1d(
                            params.emitter.generate_measurement_points().0,
                        ),
                        outgoing: partition,
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
                        params: params.clone(),
                        raw,
                        bsdfs: unsafe {
                            mem::transmute::<
                                HashMap<MeasuredBrdfLevel, MaybeUninit<VgonioBrdf>>,
                                HashMap<MeasuredBrdfLevel, VgonioBrdf>,
                            >(bsdfs)
                        },
                    })
                },
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
            let nrays64 = self.params.emitter.num_rays > u32::MAX as u64;
            match compression {
                CompressionScheme::None => {
                    Self::write_raw_measured_data(writer, &self.raw, nrays64)?;
                    Self::write_measured_bsdf_data(writer, self.bsdfs.iter())?;
                },
                CompressionScheme::Zlib => {
                    let mut zlib_encoder =
                        flate2::write::ZlibEncoder::new(vec![], flate2::Compression::default());
                    Self::write_raw_measured_data(&mut zlib_encoder, &self.raw, nrays64)?;
                    Self::write_measured_bsdf_data(&mut zlib_encoder, self.bsdfs.iter())?;
                    writer.write_all(&zlib_encoder.flush_finish()?)?
                },
                CompressionScheme::Gzip => {
                    let mut gzip_encoder =
                        flate2::write::GzEncoder::new(vec![], flate2::Compression::default());
                    Self::write_raw_measured_data(&mut gzip_encoder, &self.raw, nrays64)?;
                    Self::write_measured_bsdf_data(&mut gzip_encoder, self.bsdfs.iter())?;
                    writer.write_all(&gzip_encoder.finish()?)?
                },
                _ => {},
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::measure::{
        bsdf::{
            emitter::EmitterParams,
            receiver::{BounceAndEnergy, ReceiverParams},
            BsdfKind, MeasuredBrdfLevel, MeasuredBsdfData, RawMeasuredBsdfData,
            SingleBsdfMeasurementStats,
        },
        params::{BsdfMeasurementParams, SimulationKind},
    };
    use base::{
        io::{CompressionScheme, FileEncoding},
        math::Sph2,
        medium::Medium,
        partition::{PartitionScheme, SphericalDomain, SphericalPartition},
        range::StepRangeIncl,
        units::{nm, rad, Rads},
        Version,
    };
    use bxdf::brdf::measured::{Origin, VgonioBrdf, VgonioBrdfParameterisation};
    use jabr::array::DyArr;
    use std::{
        collections::HashMap,
        io::{BufReader, BufWriter, Cursor, Write},
        mem::MaybeUninit,
    };

    #[test]
    fn test_bsdf_measurement_params() {
        let params = BsdfMeasurementParams {
            kind: BsdfKind::Brdf,
            incident_medium: Medium::Vacuum,
            transmitted_medium: Medium::Aluminium,
            sim_kind: SimulationKind::WaveOptics,
            emitter: EmitterParams {
                num_rays: 0,
                max_bounces: 0,
                zenith: StepRangeIncl::new(rad!(0.0), rad!(0.0), rad!(0.0)),
                azimuth: StepRangeIncl::new(rad!(0.0), rad!(0.0), rad!(0.0)),
                spectrum: StepRangeIncl::new(nm!(400.0), nm!(700.0), nm!(100.0)),
                num_sectors: 1,
            },
            receivers: vec![ReceiverParams {
                domain: SphericalDomain::Upper,
                precision: Sph2::new(rad!(0.1), rad!(0.1)),
                scheme: PartitionScheme::Beckers,
            }],
            fresnel: false,
        };

        let mut writer = BufWriter::new(vec![]);
        params
            .write_to_vgmo(Version::new(0, 1, 0), &mut writer)
            .unwrap();
        let buf = writer.into_inner().unwrap();
        let mut reader = BufReader::new(Cursor::new(buf));
        let params2 =
            BsdfMeasurementParams::read_from_vgmo(Version::new(0, 1, 0), &mut reader).unwrap();
        assert_eq!(params, params2);
    }

    #[test]
    fn test_bsdf_measurement_stats_point_read_write() {
        let data = SingleBsdfMeasurementStats {
            n_bounce: 3,
            n_received: 123,
            n_missed: 0,
            n_spectrum: 4,
            n_ray_stats: vec![
                1, 2, 3, 4, // n_absorbed
                122, 121, 120, 119, // n_reflected
                100, 120, 110, 115, // n_captured
                22, 1, 10, 4, // n_escaped
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
        assert!(data.is_valid(), "Invalid data");
        let size = SingleBsdfMeasurementStats::size_in_bytes(4, 3, false);
        let mut writer = BufWriter::with_capacity(size, vec![]);
        data.write(&mut writer, 4, false).unwrap();
        let buf = writer.into_inner().unwrap();
        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 = SingleBsdfMeasurementStats::read(&mut reader, 4, false).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_bounce_and_energy_read_write() {
        let data = Some(BounceAndEnergy {
            n_bounce: 11,
            n_ray_per_bounce: vec![33468, 210, 40, 60, 70, 80, 90, 100, 110, 120, 130, 0]
                .into_boxed_slice(),
            energy_per_bounce: vec![
                1349534.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90., 100., 110., 120.,
            ]
            .into_boxed_slice(),
        });
        let mut writer = BufWriter::new(vec![]);
        BounceAndEnergy::write(&mut writer, &data, true).unwrap();
        let buf = writer.into_inner().unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 = BounceAndEnergy::read(&mut reader, true).unwrap();
        assert_eq!(data, data2);

        let data = None;
        let mut writer = BufWriter::new(vec![]);
        BounceAndEnergy::write(&mut writer, &data, true).unwrap();
        let buf = writer.into_inner().unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 = BounceAndEnergy::read(&mut reader, true).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_bsdf_measurement_raw_data() {
        let n_spectrum = 4;
        let n_wi = 4;
        let partition =
            SphericalPartition::new_beckers(SphericalDomain::Upper, Rads::from_degrees(45.0));
        let n_wo = partition.n_patches(); // 8
        let raw = RawMeasuredBsdfData {
            n_zenith_in: 4,
            spectrum: DyArr::from_iterator(
                [n_spectrum as isize],
                vec![nm!(400.0), nm!(500.0), nm!(600.0), nm!(700.0)],
            ),
            incoming: DyArr::splat(Sph2::zero(), [4]),
            outgoing: partition,
            records: DyArr::splat(
                Some(BounceAndEnergy {
                    n_bounce: 3,
                    n_ray_per_bounce: vec![120, 20, 40, 60].into_boxed_slice(),
                    energy_per_bounce: vec![90.0, 20.0, 30.0, 40.0].into_boxed_slice(),
                }),
                [n_wi, n_wo, n_spectrum],
            ),
            stats: DyArr::splat(
                SingleBsdfMeasurementStats {
                    n_bounce: 3,
                    n_received: 1111,
                    n_missed: 0,
                    n_spectrum,
                    n_ray_stats: vec![
                        1, 2, 3, 4, // n_absorbed
                        5, 6, 7, 8, // n_reflected
                        9, 10, 11, 12, // n_captured
                        0, 0, 0, 0, // n_escaped
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
                [4],
            ),

            #[cfg(any(feature = "vdbg", debug_assertions))]
            trajectories: Box::new([]),
            #[cfg(any(feature = "vdbg", debug_assertions))]
            hit_points: Box::new([]),
        };

        let mut writer = BufWriter::new(vec![]);
        MeasuredBsdfData::write_raw_measured_data(&mut writer, &raw).unwrap();
        let buf = writer.into_inner().unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let data2 = MeasuredBsdfData::read_raw_measured_data(
            &mut reader,
            n_wi,
            n_wo,
            n_spectrum,
            raw.spectrum.as_slice(),
            raw.incoming.as_slice(),
            raw.outgoing.clone(),
        )
        .unwrap();
        assert_eq!(raw, data2);
    }

    #[test]
    fn test_bsdf_measured_data() {
        let n_wi = 4;
        let n_wo = 8;
        let n_spectrum = 4;
        let brdf = VgonioBrdf {
            origin: Origin::Simulated,
            incident_medium: Medium::Vacuum,
            transmitted_medium: Medium::Vacuum,
            params: Box::new(VgonioBrdfParameterisation {
                incoming: DyArr::splat(Sph2::zero(), [4]),
                outgoing: SphericalPartition::new_beckers(
                    SphericalDomain::Upper,
                    Rads::from_degrees(45.0),
                ),
            }),
            spectrum: DyArr::from_iterator(
                [4],
                vec![nm!(400.0), nm!(500.0), nm!(600.0), nm!(700.0)],
            ),
            samples: DyArr::splat(1.1, [4, 8, 4]),
        };

        let mut writer = BufWriter::new(vec![]);
        MeasuredBsdfData::write_vgonio_brdf(&mut writer, MeasuredBrdfLevel::from(0u32), &brdf)
            .unwrap();
        let buf = writer.into_inner().unwrap();

        let mut reader = BufReader::new(Cursor::new(buf));
        let mut brdf2 = MaybeUninit::<VgonioBrdf>::uninit();
        MeasuredBsdfData::read_vgonio_brdf(&mut reader, 4, 8, 4, brdf2.as_mut_ptr()).unwrap();
        assert_eq!(brdf.samples, unsafe {
            std::ptr::addr_of!((*brdf2.as_mut_ptr()).samples).read()
        });
    }

    #[test]
    fn test_measured_bsdf_data() {
        let partition =
            SphericalPartition::new_beckers(SphericalDomain::Upper, Rads::from_degrees(45.0));
        let emitter_params = EmitterParams {
            num_rays: 0,
            num_sectors: 1,
            max_bounces: 0,
            zenith: StepRangeIncl {
                start: Rads::from_degrees(0.0),
                stop: Rads::from_degrees(90.0),
                step_size: Rads::from_degrees(90.0),
            },
            azimuth: StepRangeIncl {
                start: Rads::from_degrees(0.0),
                stop: Rads::from_degrees(360.0),
                step_size: Rads::from_degrees(180.0),
            },
            spectrum: StepRangeIncl::new(nm!(400.0), nm!(700.0), nm!(100.0)),
        };
        let incoming = DyArr::from_boxed_slice_1d(emitter_params.generate_measurement_points().0);
        let spectrum = DyArr::from_vec_1d(emitter_params.spectrum.values().collect::<Vec<_>>());
        let n_wi = incoming.len();
        let n_wo = partition.n_patches();
        let n_spectrum = spectrum.len();

        let mut bsdfs = HashMap::new();
        bsdfs.insert(
            MeasuredBrdfLevel::from(0u32),
            VgonioBrdf {
                origin: Origin::Simulated,
                incident_medium: Medium::Vacuum,
                transmitted_medium: Medium::Aluminium,
                params: Box::new(VgonioBrdfParameterisation {
                    n_zenith_i: 4,
                    incoming: incoming.clone(),
                    outgoing: partition.clone(),
                }),
                spectrum: spectrum.clone(),
                samples: DyArr::splat(1.1, [4, 8, 4]),
            },
        );

        let measured = MeasuredBsdfData {
            params: BsdfMeasurementParams {
                kind: BsdfKind::Brdf,
                sim_kind: SimulationKind::WaveOptics,
                incident_medium: Medium::Vacuum,
                transmitted_medium: Medium::Aluminium,
                emitter: emitter_params,
                receivers: vec![ReceiverParams {
                    domain: SphericalDomain::Upper,
                    precision: Sph2 {
                        theta: Rads::from_degrees(45.0),
                        phi: Rads::from_degrees(2.0),
                    },
                    scheme: PartitionScheme::Beckers,
                }],
                fresnel: false,
            },
            raw: RawMeasuredBsdfData {
                spectrum,
                incoming,
                outgoing: partition,
                records: DyArr::splat(
                    Some(BounceAndEnergy {
                        n_bounce: 3,
                        n_ray_per_bounce: vec![120, 20, 40, 60].into_boxed_slice(),
                        energy_per_bounce: vec![90.0, 20.0, 30.0, 40.0].into_boxed_slice(),
                    }),
                    [n_wi, n_wo, n_spectrum],
                ),
                stats: DyArr::splat(
                    SingleBsdfMeasurementStats {
                        n_bounce: 3,
                        n_received: 1111,
                        n_missed: 0,
                        n_spectrum,
                        n_ray_stats: vec![
                            1, 2, 3, 4, // n_absorbed
                            5, 6, 7, 8, // n_reflected
                            9, 10, 11, 12, // n_captured
                            0, 0, 0, 0, // n_escaped
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
                    [4],
                ),

                #[cfg(any(feature = "vdbg", debug_assertions))]
                trajectories: Box::new([]),
                #[cfg(any(feature = "vdbg", debug_assertions))]
                hit_points: Box::new([]),
            },
            bsdfs,
        };

        for compression in [
            CompressionScheme::None,
            CompressionScheme::Zlib,
            CompressionScheme::Gzip,
        ] {
            let mut writer = BufWriter::new(vec![]);
            measured
                .write_to_vgmo(&mut writer, FileEncoding::Binary, compression)
                .unwrap();
            let buf = writer.into_inner().unwrap();

            let mut reader = BufReader::new(Cursor::new(buf));
            let data2 = MeasuredBsdfData::read_from_vgmo(
                &mut reader,
                &measured.params,
                FileEncoding::Binary,
                compression,
            )
            .unwrap();
            assert_eq!(measured.params, data2.params);
            assert_eq!(measured.raw, data2.raw);
            assert_eq!(measured.bsdfs, data2.bsdfs);
        }
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
    cache: &Cache,
    config: &Config,
    output: OutputOptions,
) -> Result<(), VgonioError> {
    println!(
        "    {}>{} Saving {} measurement data...",
        ansi::BRIGHT_YELLOW,
        ansi::RESET,
        data.len()
    );
    let output_dir = config.resolve_output_dir(output.dir.as_deref())?;
    for (i, measurement) in data.iter().enumerate() {
        let filepath = cache.read(|cache| {
            let surf_name = cache
                .get_micro_surface_filepath(measurement.source.micro_surface().unwrap())
                .unwrap()
                .file_stem()
                .unwrap()
                .to_ascii_lowercase();
            output_dir.join(format!(
                "{}_{}_{}",
                measurement.kind().ascii_str(),
                surf_name.to_str().unwrap(),
                base::utils::iso_timestamp_short(
                    measurement.timestamp + chrono::Duration::seconds(i as i64)
                ),
            ))
        });
        println!(
            "      {}-{} Saving to \"{}\"",
            ansi::BRIGHT_CYAN,
            ansi::RESET,
            filepath.display()
        );

        for format in output.formats.iter() {
            match measurement.write_to_file(&filepath, format) {
                Ok(_) => {
                    println!(
                        "      {} Successfully saved to \"{}\", format: {:?}",
                        ansi::CYAN_CHECK,
                        output_dir.display(),
                        format
                    );
                },
                Err(err) => {
                    eprintln!(
                        "        {} Failed to save to \"{}\" with format {:?}: {}",
                        ansi::RED_EXCLAMATION,
                        filepath.display(),
                        format,
                        err,
                    );
                },
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

    match measured.write_to_file(filepath, &format) {
        Ok(_) => {
            println!(
                "      {} Successfully saved as \"{}\"",
                ansi::CYAN_CHECK,
                filepath.display()
            );
        },
        Err(err) => {
            eprintln!(
                "        {} Failed to save as \"{}\": {}",
                ansi::RED_EXCLAMATION,
                filepath.display(),
                err
            );
        },
    }
    Ok(())
}
