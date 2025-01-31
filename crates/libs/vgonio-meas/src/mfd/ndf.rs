use crate::params::{NdfMeasurementMode, NdfMeasurementParams};
use std::path::Path;
use vgonio_core::{
    error::VgonioError,
    impl_any_measured_trait,
    units::{rad, Radians},
    utils::{
        partition::{DataCarriedOnHemisphereImageWriter, SphericalDomain, SphericalPartition},
        range::StepRangeIncl,
    },
};

/// Structure holding the data for microfacet area distribution measurement.
///
/// TODO: add distribution for the microfacet slope and normal.
///
/// D(m) is the micro-facet area (normal) distribution function, which gives the
/// relative number of facets oriented in any given direction, or, more
/// precisely, the relative total facet surface area per unit solid angle of
/// surface normals pointed in any given direction.
///
/// Microfacet area distribution function (MADF)
/// Microfacet slope distribution function (MSDF)
/// Microfacet normal distribution function (MNDF)
#[derive(Debug, Clone)]
pub struct MeasuredNdfData {
    /// The measurement parameters.
    pub params: NdfMeasurementParams,
    /// The distribution data. The outermost index is the azimuthal angle of the
    /// microfacet normal, and the inner index is the zenith angle of the
    /// microfacet normal.
    pub samples: Box<[f32]>,
}

impl_any_measured_trait!(MeasuredNdfData, Ndf);

impl MeasuredNdfData {
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
    /// * `azim` - Azimuthal angle of the microfacet normal in radians.
    pub fn slice_at(&self, azim: Radians) -> (&[f32], Option<&[f32]>) {
        if self.params.mode.is_by_points() {
            let (azi, zen) = self.measurement_range().unwrap();
            let azim_m = azim.wrap_to_tau();
            let azim_m_idx = azi.index_of(azim_m);
            let opposite_azim_m = azim_m.opposite();
            let opposite_azim_idx = if azi.start <= opposite_azim_m && opposite_azim_m <= azi.stop {
                Some(azi.index_of(opposite_azim_m))
            } else {
                None
            };
            let zen_step_count = zen.step_count_wrapped();
            (
                &self.samples[azim_m_idx * zen_step_count..(azim_m_idx + 1) * zen_step_count],
                opposite_azim_idx.map(|index| {
                    &self.samples[index * zen_step_count..(index + 1) * zen_step_count]
                }),
            )
        } else {
            todo!("Implement slice_at for the partition mode.")
        }
    }

    /// Writes the measured data as an EXR file.
    pub fn write_as_exr(
        &self,
        filepath: &Path,
        timestamp: &chrono::DateTime<chrono::Local>,
        resolution: u32,
    ) -> Result<(), VgonioError> {
        use exr::prelude::*;
        let partition = SphericalPartition::new(
            self.params.mode.partition_scheme_for_data_collection(),
            SphericalDomain::Upper,
            self.params.mode.partition_precision_for_data_collection(),
        );

        // Collect the data following the patches.
        let mut samples_per_patch = vec![0.0; partition.n_patches()];
        match self.params.mode {
            NdfMeasurementMode::ByPoints { zenith, azimuth } => {
                assert!(
                    zenith.step_size > rad!(0.0) && azimuth.step_size > rad!(0.0),
                    "The step size of zenith and azimuth must be greater than 0."
                );
                let n_theta = StepRangeIncl::zero_to_half_pi(zenith.step_size).step_count_wrapped();
                let n_phi = StepRangeIncl::zero_to_tau(azimuth.step_size).step_count_wrapped();
                // NDF samples in ByPoints mode are stored by azimuth first, then by zenith.
                // We need to rearrange the data to match the patch order, which is by zenith
                // first, then by azimuth.
                samples_per_patch
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i_p, v)| {
                        let i_theta = i_p / n_phi;
                        let i_phi = i_p % n_phi;
                        let i_adf = i_phi * n_theta + i_theta;
                        // In case, the number of samples is less than the number of patches.
                        if i_adf < self.samples.len() {
                            *v = self.samples[i_adf];
                        }
                    });
            },
            NdfMeasurementMode::ByPartition { .. } => {
                samples_per_patch.copy_from_slice(&self.samples);
            },
        }

        DataCarriedOnHemisphereImageWriter::new(&partition, resolution).write_as_exr(
            &samples_per_patch,
            filepath,
            timestamp,
            |_| Some(Text::from("NDF")),
            |_| Text::from("NDF"),
        )
    }

    // TODO: review the necessity of this method.
    /// Returns the measurement range of the azimuthal and zenith angles.
    /// The azimuthal angle is in the range [0, 2π] and the zenith angle is in
    /// the range [0, π/2].
    pub fn measurement_range(&self) -> Option<(StepRangeIncl<Radians>, StepRangeIncl<Radians>)> {
        match self.params.mode {
            NdfMeasurementMode::ByPoints { zenith, azimuth } => Some((azimuth, zenith)),
            NdfMeasurementMode::ByPartition { .. } => {
                eprintln!("Measurement range is not available for the partition mode.");
                None
            },
        }
    }

    /// Returns the zenith range of the measurement only if the measurement is
    /// in the ByPoints mode.
    pub fn zenith_range(&self) -> Option<StepRangeIncl<Radians>> {
        match self.params.mode {
            NdfMeasurementMode::ByPoints { zenith, .. } => Some(zenith),
            NdfMeasurementMode::ByPartition { .. } => {
                eprintln!("Zenith range is not available for the partition mode.");
                None
            },
        }
    }
}
