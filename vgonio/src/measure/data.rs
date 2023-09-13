use crate::{
    app::cache::{Asset, Handle},
    fitting::MeasuredMdfData,
    measure::{
        bsdf::MeasuredBsdfData,
        microfacet::{MeasuredAdfData, MeasuredMsfData},
        params::MeasurementKind,
    },
    RangeByStepSizeInclusive,
};
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};
use vgcore::units::Radians;
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
}

impl MeasuredData {
    /// Returns the measurement kind.
    pub fn kind(&self) -> MeasurementKind {
        match self {
            MeasuredData::Adf(_) => MeasurementKind::Adf,
            MeasuredData::Msf(_) => MeasurementKind::Msf,
            MeasuredData::Bsdf(_) => MeasurementKind::Bsdf,
        }
    }

    /// Returns the BSDF data if it is a BSDF.
    pub fn bsdf_data(&self) -> Option<&MeasuredBsdfData> {
        match self {
            MeasuredData::Bsdf(bsdf) => Some(bsdf),
            _ => None,
        }
    }

    /// Returns the MADF data if it is a MADF.
    pub fn adf_data(&self) -> Option<&MeasuredAdfData> {
        match self {
            MeasuredData::Adf(madf) => Some(madf),
            _ => None,
        }
    }

    /// Returns the MMSF data if it is a MMSF.
    pub fn msf_data(&self) -> Option<&MeasuredMsfData> {
        match self {
            MeasuredData::Msf(mmsf) => Some(mmsf),
            _ => None,
        }
    }

    /// Returns the [`MeasuredMdfData`] if it is a ADF or MSF.
    pub fn mdf_data(&self) -> Option<MeasuredMdfData> {
        match self {
            MeasuredData::Bsdf(_) => None,
            MeasuredData::Adf(adf) => Some(MeasuredMdfData::Adf(Cow::Borrowed(adf))),
            MeasuredData::Msf(msf) => Some(MeasuredMdfData::Msf(Cow::Borrowed(msf))),
        }
    }

    /// Returns the zenith range of the measurement data if it is a MADF or
    /// MMSF.
    pub fn adf_or_msf_zenith(&self) -> Option<RangeByStepSizeInclusive<Radians>> {
        match self {
            MeasuredData::Adf(madf) => Some(madf.params.zenith),
            MeasuredData::Msf(mmsf) => Some(mmsf.params.zenith),
            MeasuredData::Bsdf(_) => None,
        }
    }

    /// Returns the azimuth range of the measurement data if it is a MADF or
    /// MMSF.
    pub fn madf_or_mmsf_azimuth(&self) -> Option<RangeByStepSizeInclusive<Radians>> {
        match self {
            MeasuredData::Adf(madf) => Some(madf.params.azimuth),
            MeasuredData::Msf(mmsf) => Some(mmsf.params.azimuth),
            MeasuredData::Bsdf(_) => None,
        }
    }

    // TODO: to be removed
    /// Returns the samples of the measurement data.
    pub fn samples(&self) -> &[f32] {
        match self {
            MeasuredData::Adf(madf) => &madf.samples,
            MeasuredData::Msf(mmsf) => &mmsf.samples,
            MeasuredData::Bsdf(_bsdf) => todo!("implement this"),
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
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        let azimuth_m = azimuth_m.wrap_to_tau();
        let azimuth_m_idx = self_azimuth.index_of(azimuth_m);
        let opposite_azimuth_m = azimuth_m.opposite();
        let opposite_index = if self_azimuth.start <= opposite_azimuth_m
            && opposite_azimuth_m <= self_azimuth.stop
        {
            Some(self_azimuth.index_of(opposite_azimuth_m))
        } else {
            None
        };
        (
            self.ndf_data_slice_inner(azimuth_m_idx),
            opposite_index.map(|index| self.ndf_data_slice_inner(index)),
        )
    }

    /// Returns a data slice of the Area Distribution Function for the given
    /// azimuthal angle index.
    fn ndf_data_slice_inner(&self, azimuth_idx: usize) -> &[f32] {
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        debug_assert!(self.kind() == MeasurementKind::Adf);
        debug_assert!(
            azimuth_idx < self_azimuth.step_count_wrapped(),
            "index out of range"
        );
        let self_zenith = self.measured.adf_or_msf_zenith().unwrap();
        &self.measured.samples()[azimuth_idx * self_zenith.step_count_wrapped()
            ..(azimuth_idx + 1) * self_zenith.step_count_wrapped()]
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
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        let self_zenith = self.measured.adf_or_msf_zenith().unwrap();
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
        let self_azimuth = self.measured.madf_or_mmsf_azimuth().unwrap();
        let self_zenith = self.measured.adf_or_msf_zenith().unwrap();
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
        &self.measured.samples()[offset..offset + zenith_bin_count]
    }
}
