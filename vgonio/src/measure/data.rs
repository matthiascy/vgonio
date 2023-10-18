//! Measurement data description.

use crate::{
    app::cache::{Asset, Handle},
    fitting::MeasuredMdfData,
    io::{vgmo::VgmoHeaderExt, OutputFileFormatOptions},
    measure::{
        bsdf::MeasuredBsdfData,
        microfacet::{MeasuredAdfData, MeasuredMsfData, MeasuredSdfData},
        params::MeasurementKind,
    },
    RangeByStepSizeInclusive,
};
use chrono::{DateTime, Local};
use std::{
    borrow::Cow,
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
use vgcore::{
    error::VgonioError,
    io::{
        Header, HeaderMeta, ReadFileError, ReadFileErrorKind, WriteFileError, WriteFileErrorKind,
    },
    units::Radians,
    Version,
};
use vgsurf::MicroSurface;

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
    pub fn bsdf(&self) -> Option<&MeasuredBsdfData> {
        match self {
            MeasuredData::Bsdf(bsdf) => Some(bsdf),
            _ => None,
        }
    }

    /// Returns the ADF data.
    pub fn adf(&self) -> Option<&MeasuredAdfData> {
        match self {
            MeasuredData::Adf(adf) => Some(adf),
            _ => None,
        }
    }

    /// Returns the SDF data.
    pub fn sdf(&self) -> Option<&MeasuredSdfData> {
        match self {
            MeasuredData::Sdf(sdf) => Some(sdf),
            _ => None,
        }
    }

    /// Returns the MSF data.
    pub fn msf(&self) -> Option<&MeasuredMsfData> {
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
    pub fn mdf(&self) -> Option<MeasuredMdfData> {
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
            MeasuredData::Adf(adf) => Some(adf.params.azimuth),
            MeasuredData::Msf(msf) => Some(msf.params.azimuth),
            _ => None,
        }
    }

    /// Returns the zenith angle range of the measurement data only if
    /// it is a ADF or MSF measurement.
    pub fn adf_or_msf_zenith(&self) -> Option<RangeByStepSizeInclusive<Radians>> {
        match self {
            MeasuredData::Adf(adf) => Some(adf.params.zenith),
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
            MeasuredData::Adf(adf) => Some((adf.params.azimuth, adf.params.zenith)),
            MeasuredData::Msf(msf) => Some((msf.params.azimuth, msf.params.zenith)),
            _ => None,
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
        let (az, zn) = self.measured.adf_or_msf_angle_ranges().unwrap();
        let azimuth_m = azimuth_m.wrap_to_tau();
        let azimuth_m_idx = az.index_of(azimuth_m);
        let opposite_azimuth_m = azimuth_m.opposite();
        let opposite_index = if az.start <= opposite_azimuth_m && opposite_azimuth_m <= az.stop {
            Some(az.index_of(opposite_azimuth_m))
        } else {
            None
        };
        let zn_step_count = zn.step_count_wrapped();
        let samples = self.measured.adf_or_msf_samples().unwrap();
        (
            &samples[azimuth_m_idx * zn_step_count..(azimuth_m_idx + 1) * zn_step_count],
            opposite_index
                .map(|index| &samples[index * zn_step_count..(index + 1) * zn_step_count]),
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
                        vgcore::utils::iso_timestamp_from_datetime(&self.timestamp).as_bytes(),
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
                        sdf.write_as_exr(&filepath, &self.timestamp, *resolution)?;
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
