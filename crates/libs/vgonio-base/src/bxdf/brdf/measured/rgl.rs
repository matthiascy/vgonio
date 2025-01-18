use crate::{
    bxdf::{
        brdf::measured::{
            BrdfParam, BrdfParamKind, BrdfSnapshot, BrdfSnapshotIterator, MeasuredBrdf, Origin,
        },
        BrdfProxy, OutgoingDirs, ProxySource,
    },
    impl_any_measured_trait,
    math::{Sph2, Vec3},
    optics::ior::IorRegistry,
    units::{deg, rad, Nanometres},
    utils::{medium::Medium, range::StepRangeIncl},
    AnyMeasured, AnyMeasuredBrdf, BrdfLevel, MeasuredBrdfKind, MeasurementKind,
};
use jabr::array::{DyArr, DynArr};
use std::{borrow::Cow, path::Path};

/// Parametrisation of the BRDF measured in RGL (https://rgl.epfl.ch/pages/lab/material-database) at EPFL by Jonathan Dupuy and Wenzel Jakob.
pub type RglBrdf = MeasuredBrdf<RglBrdfParameterisation, 3>;

impl_any_measured_trait!(@single_level_brdf RglBrdf);

unsafe impl Send for RglBrdf {}
unsafe impl Sync for RglBrdf {}

/// Parameterisation of the RGL BRDF.
///
/// To simplify the implementation, the RGL BRDF is evaluated with the loader
/// `powitacq` instead of reading the raw data (pickled Python files) directly.
///
/// TODO: Read the raw data directly.
/// TODO: use theta_i, phi_i, theta_o, phi_o, lambda
#[derive(Clone, PartialEq, Debug)]
pub struct RglBrdfParameterisation {
    /// Number of zenith angles for the incident directions.
    pub n_zenith_i: usize,
    /// Number of zenith angles for the outgoing directions.
    pub n_zenith_o: usize,
    /// Incident directions.
    /// Directions are stored in spherical coordinates (zenith, azimuth) in a
    /// row-major 1D array. The first dimension is the zenith angle and the
    /// second dimension is the azimuth angle. The zenith angles are in the
    /// range [0, π/2]. The azimuth angles are in the range [0, 2π].
    /// Zenith angles increase the fastest.
    pub incoming: DyArr<Sph2>,
    /// Outgoing directions.
    /// Directions are stored in spherical coordinates (zenith, azimuth) in a
    /// row-major 1D array. The first dimension is the zenith angle and the
    /// second dimension is the azimuth angle. The zenith angles are in the
    /// range [0, π/2]. The azimuth angles are in the range [0, 2π].
    /// Zenith angles increase the fastest.
    pub outgoing: DyArr<Sph2>,
    /// Temporarily store the original data.
    pub original: powitacq::BrdfData,
}

impl BrdfParam for RglBrdfParameterisation {
    fn kind() -> BrdfParamKind { BrdfParamKind::InOutDirs }
}

impl RglBrdfParameterisation {
    pub fn incoming_cartesian(&self) -> DyArr<Vec3> {
        DyArr::from_iterator([-1], self.incoming.iter().map(Sph2::to_cartesian))
    }

    pub fn outgoing_cartesian(&self) -> DyArr<Vec3> {
        DyArr::from_iterator([-1], self.outgoing.iter().map(Sph2::to_cartesian))
    }

    pub fn n_wo(&self) -> usize { self.outgoing.len() }

    pub fn n_wi(&self) -> usize { self.incoming.len() }

    pub fn n_wi_zenith(&self) -> usize { self.n_zenith_i }

    pub fn n_wi_azimuth(&self) -> usize { self.incoming.len() / self.n_zenith_i }

    pub fn n_wo_zenith(&self) -> usize { self.n_zenith_o }

    pub fn n_wo_azimuth(&self) -> usize { self.outgoing.len() / self.n_zenith_o }
}

impl RglBrdf {
    // TODO: check other BRDFs decide if we need to load the BRDFs in the
    // constructor.
    #[cfg(feature = "bxdf_io")]
    pub fn load(path: &Path, transmitted_medium: Medium) -> Self {
        let brdf = powitacq::BrdfData::new(path);
        let spectrum = {
            let spectrum = brdf.wavelengths();
            DyArr::from_iterator([-1], spectrum.iter().map(|&x| Nanometres::new(x)))
        };
        let theta_i =
            StepRangeIncl::new(rad!(0.0), deg!(80.0).to_radians(), deg!(5.0).to_radians());
        let phi_i = StepRangeIncl::new(
            deg!(0.0).to_radians(),
            deg!(360.0).to_radians(),
            deg!(60.0).to_radians(),
        );
        let theta_o =
            StepRangeIncl::new(rad!(0.0), deg!(80.0).to_radians(), deg!(2.0).to_radians());
        let phi_o = StepRangeIncl::new(
            deg!(0.0).to_radians(),
            deg!(360.0).to_radians(),
            deg!(30.0).to_radians(),
        );
        let theta_i_vals = theta_i.values_wrapped().collect::<Box<_>>();
        let phi_i_vals = phi_i.values_wrapped().collect::<Box<_>>();
        let theta_o_vals = theta_o.values_wrapped().collect::<Box<_>>();
        let phi_o_vals = phi_o.values_wrapped().collect::<Box<_>>();

        let incoming = DyArr::from_iterator(
            [-1],
            phi_i_vals.iter().flat_map(|phi_i| {
                theta_i_vals
                    .iter()
                    .map(|theta_i| Sph2::new(*theta_i, *phi_i))
            }),
        );

        let outgoing = DyArr::from_iterator(
            [-1],
            phi_o_vals.iter().flat_map(|phi_o| {
                theta_o_vals
                    .iter()
                    .map(|theta_o| Sph2::new(*theta_o, *phi_o))
            }),
        );

        let n_zenith_i = theta_i_vals.len();
        let n_zenith_o = theta_o_vals.len();

        let parameterisation = RglBrdfParameterisation {
            n_zenith_i,
            n_zenith_o,
            incoming,
            outgoing,
            original: brdf,
        };

        let n_spectrum = spectrum.len();
        let n_wi = parameterisation.n_wi();
        let n_wo = parameterisation.n_wo();

        // Eval the BRDF.
        let samples = {
            let mut samples = vec![0.0f32; n_wi * n_wo * n_spectrum].into_boxed_slice();
            for (idx_wi, wi) in parameterisation.incoming.iter().enumerate() {
                for (idx_wo, wo) in parameterisation.outgoing.iter().enumerate() {
                    let samples_spectrum = parameterisation.original.eval(
                        wi.theta.as_f32(),
                        wi.phi.as_f32(),
                        wo.theta.as_f32(),
                        wo.phi.as_f32(),
                    );
                    let index = idx_wi * n_wo * n_spectrum + idx_wo * n_spectrum;
                    samples[index..index + n_spectrum].copy_from_slice(&samples_spectrum);
                }
            }

            DyArr::from_boxed_slice([n_wi, n_wo, n_spectrum], samples)
        };

        Self {
            kind: MeasuredBrdfKind::Rgl,
            origin: Origin::RealWorld,
            incident_medium: Medium::Vacuum,
            transmitted_medium,
            params: Box::new(parameterisation),
            spectrum,
            samples,
        }
    }

    pub fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Rgl }

    pub fn snapshots(&self) -> BrdfSnapshotIterator<'_, RglBrdfParameterisation, 3> {
        BrdfSnapshotIterator {
            brdf: self,
            n_spectrum: self.spectrum.len(),
            n_incoming: self.params.n_wi(),
            n_outgoing: self.params.n_wo(),
            index: 0,
        }
    }
}

impl<'a> Iterator for BrdfSnapshotIterator<'a, RglBrdfParameterisation, 3> {
    type Item = BrdfSnapshot<'a, f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.n_incoming {
            return None;
        }
        let snapshot_range = self.index * self.n_outgoing * self.n_spectrum
            ..(self.index + 1) * self.n_outgoing * self.n_spectrum;
        let snapshot = BrdfSnapshot {
            wi: self.brdf.params.incoming[self.index],
            n_spectrum: self.n_spectrum,
            samples: &self.brdf.samples.as_slice()[snapshot_range],
        };
        self.index += 1;
        Some(snapshot)
    }
}

impl AnyMeasuredBrdf for RglBrdf {
    crate::any_measured_brdf_trait_common_impl!(RglBrdf, Rgl);

    fn proxy(&self, iors: &IorRegistry) -> BrdfProxy {
        let iors_i = iors
            .ior_of_spectrum(self.incident_medium, self.spectrum.as_slice())
            .unwrap()
            .into_vec();
        let iors_t = iors
            .ior_of_spectrum(self.transmitted_medium, self.spectrum.as_slice())
            .unwrap()
            .into_vec();
        let n_theta_i = self.params.n_wi_zenith();
        let n_phi_i = self.params.n_wi_azimuth();
        let n_theta_o = self.params.n_wo_zenith();
        let n_phi_o = self.params.n_wo_azimuth();
        let n_spectrum = self.spectrum.len();
        let i_thetas = DyArr::from_iterator(
            [-1],
            self.params
                .incoming
                .iter()
                .take(n_theta_i)
                .map(|sph| sph.theta.as_f32()),
        );
        let i_phis = DyArr::from_iterator(
            [-1],
            self.params
                .incoming
                .iter()
                .step_by(n_theta_i)
                .map(|sph| sph.phi.as_f32()),
        );
        let o_thetas = DyArr::from_iterator(
            [-1],
            self.params
                .outgoing
                .iter()
                .take(n_theta_o)
                .map(|sph| sph.theta.as_f32()),
        );
        let o_phis = DyArr::from_iterator(
            [-1],
            self.params
                .outgoing
                .iter()
                .step_by(n_theta_o)
                .map(|sph| sph.phi.as_f32()),
        );
        // Rearrange the samples to match the new dimensions.
        // The original dimensions are [n_phi_i, n_theta_i, n_phi_o, n_theta_o,
        // n_spectrum] The target dimensions are [n_theta_i, n_phi_i, n_theta_o,
        // n_phi_o, n_spectrum]
        let mut resampled = DynArr::zeros(&[n_theta_i, n_phi_i, n_theta_o, n_phi_o, n_spectrum]);
        let &[stride_theta_i, stride_phi_i, stride_theta_o, stride_phi_o, ..] = resampled.strides()
        else {
            panic!("Invalid strides!");
        };
        let org_strides = self.samples.strides();
        for i in 0..n_theta_i {
            for j in 0..n_phi_i {
                for k in 0..n_theta_o {
                    for l in 0..n_phi_o {
                        let org_wi_idx = i + j * n_theta_i;
                        let org_wo_idx = k + l * n_theta_o;
                        let org_offset = org_wi_idx * org_strides[0] + org_wo_idx * org_strides[1];
                        let new_offset = i * stride_theta_i
                            + j * stride_phi_i
                            + k * stride_theta_o
                            + l * stride_phi_o;
                        resampled.as_mut_slice()[new_offset..new_offset + n_spectrum]
                            .copy_from_slice(
                                &self.samples.as_slice()[org_offset..org_offset + n_spectrum],
                            );
                    }
                }
            }
        }

        let o_dirs = OutgoingDirs::new_grid(Cow::Owned(o_thetas), Cow::Owned(o_phis));
        BrdfProxy {
            has_nan: false,
            brdf: self,
            source: ProxySource::Measured,
            i_thetas: Cow::Owned(i_thetas),
            i_phis: Cow::Owned(i_phis),
            o_dirs,
            resampled: Cow::Owned(resampled),
            iors_i: Cow::Owned(iors_i),
            iors_t: Cow::Owned(iors_t),
        }
    }
}
