//! Measurement data description.

use crate::{
    app::cache::Handle,
    fitting::MeasuredMdfData,
    io::{vgmo::VgmoHeaderExt, OutputFileFormatOptions},
    measure::{
        bsdf::MeasuredBsdfData,
        microfacet::{MeasuredAdfData, MeasuredMsfData, MeasuredSdfData},
        params::{AdfMeasurementMode, MeasurementKind},
    },
    partition::SphericalPartition,
    RangeByStepSizeInclusive,
};
use base::{
    error::VgonioError,
    io::{
        Header, HeaderMeta, ReadFileError, ReadFileErrorKind, WriteFileError, WriteFileErrorKind,
    },
    math::Sph2,
    units::Radians,
    Asset, Version,
};
use chrono::{DateTime, Local};
use std::{
    borrow::Cow,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use surf::MicroSurface;

/// Measurement data source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementDataSource {
    /// Measurement data is loaded from a file.
    Loaded(PathBuf),
    /// Measurement data is generated from a micro-surface.
    Measured(Handle<MicroSurface>),
}

impl MeasurementDataSource {
    /// Returns the path to the measurement data if it is loaded from a file.
    pub fn path(&self) -> Option<&Path> {
        match self {
            MeasurementDataSource::Loaded(p) => Some(p.as_path()),
            MeasurementDataSource::Measured(_) => None,
        }
    }

    /// Returns the micro-surface handle if the measurement data is generated.
    pub fn micro_surface(&self) -> Option<Handle<MicroSurface>> {
        match self {
            MeasurementDataSource::Loaded(_) => None,
            MeasurementDataSource::Measured(ms) => Some(*ms),
        }
    }
}

/// Different kinds of measurement data.
#[derive(Debug, Clone)]
pub enum MeasuredData {
    /// Bidirectional scattering distribution function.
    Bsdf(MeasuredBsdfData),
    /// Microfacet distribution function.
    Adf(MeasuredAdfData),
    /// Shadowing-masking function.
    Msf(MeasuredMsfData),
    /// Microfacet slope distribution function.
    Sdf(MeasuredSdfData),
}

impl MeasuredData {
    /// Returns the kind of the measured data.
    pub fn kind(&self) -> MeasurementKind {
        match self {
            MeasuredData::Bsdf(_) => MeasurementKind::Bsdf,
            MeasuredData::Adf(_) => MeasurementKind::Adf,
            MeasuredData::Msf(_) => MeasurementKind::Msf,
            MeasuredData::Sdf(_) => MeasurementKind::Sdf,
        }
    }

    /// Returns the BSDF data.
    pub fn as_bsdf(&self) -> Option<&MeasuredBsdfData> {
        match self {
            MeasuredData::Bsdf(bsdf) => Some(bsdf),
            _ => None,
        }
    }

    /// Returns the ADF data.
    pub fn as_adf(&self) -> Option<&MeasuredAdfData> {
        match self {
            MeasuredData::Adf(adf) => Some(adf),
            _ => None,
        }
    }

    /// Returns the SDF data.
    pub fn as_sdf(&self) -> Option<&MeasuredSdfData> {
        match self {
            MeasuredData::Sdf(sdf) => Some(sdf),
            _ => None,
        }
    }

    /// Returns the MSF data.
    pub fn as_msf(&self) -> Option<&MeasuredMsfData> {
        match self {
            MeasuredData::Msf(msf) => Some(msf),
            _ => None,
        }
    }

    /// Returns the samples of the measurement data only if it is a
    /// ADF or MSF.
    pub fn adf_or_msf_samples(&self) -> Option<&[f32]> {
        match self {
            MeasuredData::Adf(adf) => Some(&adf.samples),
            MeasuredData::Msf(msf) => Some(&msf.samples),
            _ => None,
        }
    }

    /// Returns the MDF data.
    pub fn as_mdf(&self) -> Option<MeasuredMdfData> {
        match self {
            MeasuredData::Adf(adf) => Some(MeasuredMdfData::Adf(Cow::Borrowed(adf))),
            MeasuredData::Msf(msf) => Some(MeasuredMdfData::Msf(Cow::Borrowed(msf))),
            _ => None,
        }
    }

    /// Returns the azimuthal angle range of the measurement data only if
    /// it is a ADF or MSF measurement.
    pub fn adf_or_msf_azimuth(&self) -> Option<RangeByStepSizeInclusive<Radians>> {
        match self {
            MeasuredData::Adf(adf) => match adf.params.mode {
                AdfMeasurementMode::ByPoints { azimuth, .. } => Some(azimuth),
                AdfMeasurementMode::ByPartition { .. } => None,
            },
            MeasuredData::Msf(msf) => Some(msf.params.azimuth),
            _ => None,
        }
    }

    /// Returns the zenith angle range of the measurement data only if
    /// it is a ADF or MSF measurement.
    pub fn adf_or_msf_zenith(&self) -> Option<RangeByStepSizeInclusive<Radians>> {
        match self {
            MeasuredData::Adf(adf) => match adf.params.mode {
                AdfMeasurementMode::ByPoints { zenith, .. } => Some(zenith),
                AdfMeasurementMode::ByPartition { .. } => None,
            },
            MeasuredData::Msf(msf) => Some(msf.params.zenith),
            _ => None,
        }
    }

    /// Returns the measurement range of the azimuth and zenith angles only if
    /// it is a ADF or MSF measurement.
    pub fn adf_or_msf_angle_ranges(
        &self,
    ) -> Option<(
        RangeByStepSizeInclusive<Radians>,
        RangeByStepSizeInclusive<Radians>,
    )> {
        match self {
            MeasuredData::Adf(adf) => match adf.params.mode {
                AdfMeasurementMode::ByPoints { azimuth, zenith } => Some((azimuth, zenith)),
                // TODO: implement this.
                AdfMeasurementMode::ByPartition { .. } => {
                    eprintln!("Partition mode is not supported yet.");
                    None
                }
            },
            MeasuredData::Msf(msf) => Some((msf.params.azimuth, msf.params.zenith)),
            _ => None,
        }
    }
}

/// Structure for sampling the measurement data.
/// In case we don't have the exact data for the given azimuthal and zenith
/// angles, we do the interpolation.
pub struct MeasuredDataSampler<'a> {
    data: &'a MeasuredData,
    partition: SphericalPartition,
}

impl<'a> MeasuredDataSampler<'a> {
    /// Creates a new sampler for the given measurement data.
    pub fn new(data: &'a MeasuredData) -> Self {
        let partition = match data {
            MeasuredData::Bsdf(bsdf) => bsdf.params.receiver.partitioning(),
            MeasuredData::Adf(adf) => adf.params.receiver.partitioning(),
            MeasuredData::Msf(msf) => msf.params.receiver.partitioning(),
            MeasuredData::Sdf(sdf) => sdf.params.receiver.partitioning(),
        };
        Self { data, partition }
    }

    /// Samples the measurement data for the given incident and outgoing
    /// directions.
    ///
    /// In case the measurement data is a BSDF, the outgoing direction is
    /// required. The BSDF data is going to be sampled depending on the
    /// outgoing direction.
    ///
    /// # Arguments
    ///
    /// * `wi` - Incident direction.
    /// * `wo` - Outgoing direction. Only required for BSDF measurement. For
    ///   ADF, MSF and SDF measurements, this should be `None`.
    ///
    /// # Returns
    ///
    /// The sampled value.
    pub fn sample(&self, wi: Sph2, wo: Option<Sph2>) -> f32 {
        match self.data {
            MeasuredData::Bsdf(bsdf) => {
                let wo = wo.expect("Outgoing direction is required for BSDF measurement.");
                // Find the snapshot for the given incident direction.
                let bsdf = bsdf.snapshots.iter().find(|s| s.wi == wi).expect(
                    "Incident direction is not found in the BSDF snapshots. Currenty, we don't \
                     support the interpolation on incident direction.",
                );
                // Interpolate the BSDF data for the given outgoing direction.
                // Bilinear interpolation is used for the interpolation.
                unimplemented!("Interpolation for BSDF data is not supported yet.")
            }
            _ => unimplemented!("Sampling for the given measurement data is not supported yet."),
        }
    }
}

// TODO: add support for storing data in the memory in a compressed
//       format(maybe LZ4).
/// Structure for storing measurement data in the memory especially
/// when loading from a file.
#[derive(Debug, Clone)]
pub struct MeasurementData {
    /// Internal tag for displaying the measurement data in the GUI.
    pub name: String,
    /// Origin of the measurement data.
    pub source: MeasurementDataSource,
    /// Timestamp of the measurement.
    pub timestamp: DateTime<Local>,
    /// Measurement data.
    pub measured: MeasuredData,
}

impl Asset for MeasurementData {}

impl PartialEq for MeasurementData {
    fn eq(&self, other: &Self) -> bool { self.source == other.source }
}

impl MeasurementData {
    /// Returns the kind of the measurement data.
    pub fn kind(&self) -> MeasurementKind { self.measured.kind() }

    /// Returns the Area Distribution Function data slice for the given
    /// azimuthal angle in radians.
    ///
    /// The returned slice contains two elements, the first one is the
    /// data slice for the given azimuthal angle, the second one is the
    /// data slice for the azimuthal angle that is 180 degrees away from
    /// the given azimuthal angle, if exists.
    ///
    /// Azimuthal angle will be wrapped around to the range [0, 2π).
    ///
    /// 2π will be mapped to 0.
    ///
    /// # Arguments
    ///
    /// * `azimuth_m` - Azimuthal angle of the microfacet normal in radians.
    pub fn ndf_data_slice(&self, azimuth_m: Radians) -> (&[f32], Option<&[f32]>) {
        debug_assert!(self.kind() == MeasurementKind::Adf);
        let (azi, zen) = self.measured.adf_or_msf_angle_ranges().unwrap();
        let azimuth_m = azimuth_m.wrap_to_tau();
        let azimuth_m_idx = azi.index_of(azimuth_m);
        let opposite_azimuth_m = azimuth_m.opposite();
        let opposite_index = if azi.start <= opposite_azimuth_m && opposite_azimuth_m <= azi.stop {
            Some(azi.index_of(opposite_azimuth_m))
        } else {
            None
        };
        let zen_step_count = zen.step_count_wrapped();
        let samples = self.measured.adf_or_msf_samples().unwrap();
        (
            &samples[azimuth_m_idx * zen_step_count..(azimuth_m_idx + 1) * zen_step_count],
            opposite_index
                .map(|index| &samples[index * zen_step_count..(index + 1) * zen_step_count]),
        )
    }

    /// Returns the Masking Shadowing Function data slice for the given
    /// microfacet normal and azimuthal angle of the incident direction.
    ///
    /// The returned slice contains two elements, the first one is the
    /// data slice for the given azimuthal angle, the second one is the
    /// data slice for the azimuthal angle that is 180 degrees away from
    /// the given azimuthal angle, if exists.
    pub fn msf_data_slice(
        &self,
        azimuth_m: Radians,
        zenith_m: Radians,
        azimuth_i: Radians,
    ) -> (&[f32], Option<&[f32]>) {
        debug_assert!(
            self.kind() == MeasurementKind::Msf,
            "measurement data kind should be MicrofacetMaskingShadowing"
        );
        let (self_azimuth, self_zenith) = self.measured.adf_or_msf_angle_ranges().unwrap();
        let azimuth_m = azimuth_m.wrap_to_tau();
        let azimuth_i = azimuth_i.wrap_to_tau();
        let zenith_m = zenith_m.clamp(self_zenith.start, self_zenith.stop);
        let azimuth_m_idx = self_azimuth.index_of(azimuth_m);
        let zenith_m_idx = self_zenith.index_of(zenith_m);
        let azimuth_i_idx = self_azimuth.index_of(azimuth_i);
        let opposite_azimuth_i = azimuth_i.opposite();
        let opposite_azimuth_i_idx = if self_azimuth.start <= opposite_azimuth_i
            && opposite_azimuth_i <= self_azimuth.stop
        {
            Some(self_azimuth.index_of(opposite_azimuth_i))
        } else {
            None
        };
        (
            self.msf_data_slice_inner(azimuth_m_idx, zenith_m_idx, azimuth_i_idx),
            opposite_azimuth_i_idx
                .map(|index| self.msf_data_slice_inner(azimuth_m_idx, zenith_m_idx, index)),
        )
    }

    /// Returns a data slice of the Masking Shadowing Function for the given
    /// indices.
    fn msf_data_slice_inner(
        &self,
        azimuth_m_idx: usize,
        zenith_m_idx: usize,
        azimuth_i_idx: usize,
    ) -> &[f32] {
        let (self_azimuth, self_zenith) = self.measured.adf_or_msf_angle_ranges().unwrap();
        debug_assert!(self.kind() == MeasurementKind::Msf);
        debug_assert!(
            azimuth_m_idx < self_azimuth.step_count_wrapped(),
            "index out of range"
        );
        debug_assert!(
            azimuth_i_idx < self_azimuth.step_count_wrapped(),
            "index out of range"
        );
        debug_assert!(
            zenith_m_idx < self_zenith.step_count_wrapped(),
            "index out of range"
        );
        let zenith_bin_count = self_zenith.step_count_wrapped();
        let azimuth_bin_count = self_azimuth.step_count_wrapped();
        let offset = azimuth_m_idx * zenith_bin_count * azimuth_bin_count * zenith_bin_count
            + zenith_m_idx * azimuth_bin_count * zenith_bin_count
            + azimuth_i_idx * zenith_bin_count;
        &self.measured.adf_or_msf_samples().unwrap()[offset..offset + zenith_bin_count]
    }

    /// Writes the measurement data to a file in VGMO format.
    pub fn write_to_file(
        &self,
        filepath: &Path,
        format: &OutputFileFormatOptions,
    ) -> Result<(), VgonioError> {
        match format {
            OutputFileFormatOptions::Vgmo {
                encoding,
                compression,
            } => {
                let timestamp = {
                    let mut timestamp = [0_u8; 32];
                    timestamp.copy_from_slice(
                        base::utils::iso_timestamp_from_datetime(&self.timestamp).as_bytes(),
                    );
                    timestamp
                };
                let header = Header {
                    meta: HeaderMeta {
                        version: Version::new(0, 1, 0),
                        timestamp,
                        length: 0,
                        sample_size: 4,
                        encoding: *encoding,
                        compression: *compression,
                    },
                    extra: self.measured.as_vgmo_header_ext(),
                };
                let filepath = filepath.with_extension("vgmo");
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&filepath)
                    .map_err(|err| {
                        VgonioError::from_io_error(err, "Failed to open measurement file.")
                    })?;
                let mut writer = BufWriter::new(file);
                crate::io::vgmo::write(&mut writer, header, &self.measured).map_err(|err| {
                    VgonioError::from_write_file_error(
                        WriteFileError {
                            path: filepath.to_owned().into_boxed_path(),
                            kind: err,
                        },
                        "Failed to write VGMO file.",
                    )
                })?;
                #[cfg(feature = "bench")]
                let start = std::time::Instant::now();

                writer.flush().map_err(|err| {
                    VgonioError::from_write_file_error(
                        WriteFileError {
                            path: filepath.to_owned().into_boxed_path(),
                            kind: WriteFileErrorKind::Write(err),
                        },
                        "Failed to flush VGMO file.",
                    )
                })?;
            }
            OutputFileFormatOptions::Exr { resolution } => {
                let filepath = filepath.with_extension("exr");
                match &self.measured {
                    MeasuredData::Bsdf(bsdf) => {
                        bsdf.write_as_exr(&filepath, &self.timestamp, *resolution)?
                    }
                    MeasuredData::Adf(adf) => {
                        adf.write_as_exr(&filepath, &self.timestamp, *resolution)?
                    }
                    MeasuredData::Msf(msf) => {
                        eprintln!("Writing MSF to EXR is not supported yet.");
                    }
                    MeasuredData::Sdf(sdf) => {
                        sdf.write_histogram_as_exr(&filepath, &self.timestamp, *resolution)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Loads the measurement data from a file.
    pub fn read_from_file(filepath: &Path) -> Result<Self, VgonioError> {
        let file = File::open(filepath).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!("Failed to open measurement file: {}", filepath.display()),
            )
        })?;
        let mut reader = BufReader::new(file);
        let header = Header::<VgmoHeaderExt>::read(&mut reader).map_err(|err| {
            VgonioError::from_read_file_error(
                ReadFileError {
                    path: filepath.to_owned().into_boxed_path(),
                    kind: ReadFileErrorKind::Read(err),
                },
                "Failed to read VGMO file.",
            )
        })?;
        log::debug!("Read VGMO file of length: {}", header.meta.length);
        log::debug!("Header: {:?}", header);

        let path = filepath.to_path_buf();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("invalid file stem")
            .to_string();

        let measured = crate::io::vgmo::read(&mut reader, &header).map_err(|err| {
            VgonioError::from_read_file_error(
                ReadFileError {
                    path: filepath.to_owned().into_boxed_path(),
                    kind: err,
                },
                "Failed to read VGMO file.",
            )
        })?;
        let timestamp = DateTime::from(
            DateTime::parse_from_rfc3339(std::str::from_utf8(&header.meta.timestamp).map_err(
                |err| VgonioError::from_utf8_error(err, "Failed to parse timestamp from file."),
            )?)
            .map_err(|err| {
                VgonioError::new(
                    format!(
                        "Failed to parse timestamp from file: {}",
                        filepath.display()
                    ),
                    Some(Box::new(err)),
                )
            })?,
        );

        Ok(MeasurementData {
            name,
            source: MeasurementDataSource::Loaded(path),
            timestamp,
            measured,
        })
    }
}
