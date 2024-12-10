use crate::{
    brdf::{
        measured::{
            BrdfParameterisation, BrdfSnapshot, BrdfSnapshotIterator, MeasuredBrdf, Origin,
            ParametrisationKind, VgonioBrdfParameterisation,
        },
        Bxdf,
    },
    Scattering,
};
use base::{
    impl_measured_data_trait, math,
    math::{Sph2, Vec3},
    medium::Medium,
    optics::ior::RefractiveIndexRegistry,
    range::RangeByStepSizeInclusive,
    units::{deg, rad, Nanometres, Radians},
    ErrorMetric, MeasuredBrdfKind, MeasuredData, MeasurementKind, ResidualErrorMetric,
};
use jabr::array::DyArr;
use std::path::Path;

#[cfg(feature = "fitting")]
use crate::brdf::measured::AnalyticalFit;

/// Parametrisation of the BRDF measured in RGL (https://rgl.epfl.ch/pages/lab/material-database) at EPFL by Jonathan Dupuy and Wenzel Jakob.
pub type RglBrdf = MeasuredBrdf<RglBrdfParameterisation, 3>;

impl_measured_data_trait!(RglBrdf, Bsdf, Some(MeasuredBrdfKind::Rgl));

unsafe impl Send for RglBrdf {}
unsafe impl Sync for RglBrdf {}

/// Parameterisation of the RGL BRDF.
///
/// To simplify the implementation, the RGL BRDF is evaluated with the loader
/// `powitacq` instead of reading the raw data (pickled Python files) directly.
///
/// TODO: Read the raw data directly.
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

impl BrdfParameterisation for RglBrdfParameterisation {
    fn kind() -> ParametrisationKind { ParametrisationKind::IncidentDirection }
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
    pub fn load(path: &Path, transmitted_medium: Medium) -> Self {
        let brdf = powitacq::BrdfData::new(path);
        let spectrum = {
            let mut spectrum = brdf.wavelengths();
            DyArr::from_iterator([-1], spectrum.iter().map(|&x| Nanometres::new(x)))
        };
        let theta_i = RangeByStepSizeInclusive::new(
            rad!(0.0),
            deg!(80.0).to_radians(),
            deg!(5.0).to_radians(),
        );
        let phi_i = RangeByStepSizeInclusive::new(
            deg!(0.0).to_radians(),
            deg!(360.0).to_radians(),
            deg!(60.0).to_radians(),
        );
        let theta_o = RangeByStepSizeInclusive::new(
            rad!(0.0),
            deg!(80.0).to_radians(),
            deg!(2.0).to_radians(),
        );
        let phi_o = RangeByStepSizeInclusive::new(
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

#[cfg(feature = "fitting")]
impl AnalyticalFit for RglBrdf {
    type Params = RglBrdfParameterisation;

    impl_analytical_fit_trait!(self);

    fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Rgl }

    fn new_analytical(
        medium_i: Medium,
        medium_t: Medium,
        spectrum: &[Nanometres],
        params: &Self::Params,
        model: &dyn Bxdf<Params = [f64; 2]>,
        iors: &RefractiveIndexRegistry,
    ) -> Self
    where
        Self: Sized,
    {
        let iors_t = iors.ior_of_spectrum(medium_i, spectrum).unwrap();
        let iors_i = iors.ior_of_spectrum(medium_t, spectrum).unwrap();
        let n_spectrum = spectrum.len();
        let n_wi = params.n_wi();
        let n_wo = params.n_wo();
        let mut samples = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        for (idx_wi, wi) in params.incoming.iter().enumerate() {
            let wi_cart = wi.to_cartesian();
            for (idx_wo, wo) in params.outgoing.iter().enumerate() {
                let wo_cart = wo.to_cartesian();
                let spectral_samples = Scattering::eval_reflectance_spectrum(
                    model, &wi_cart, &wo_cart, &iors_i, &iors_t,
                )
                .iter()
                .map(|&x| x as f32)
                .collect::<Box<_>>();
                let offset = idx_wi * n_wo * n_spectrum + idx_wo * n_spectrum;
                samples.as_mut_slice()[offset..offset + n_spectrum]
                    .copy_from_slice(&spectral_samples);
            }
        }

        Self {
            origin: Origin::RealWorld,
            incident_medium: medium_i,
            transmitted_medium: medium_t,
            params: Box::new(params.clone()),
            spectrum: DyArr::from_iterator([-1], spectrum.iter().copied()),
            samples,
        }
    }

    fn distance(&self, other: &Self, metric: ErrorMetric, rmetric: ResidualErrorMetric) -> f64
    where
        Self: Sized,
    {
        assert_eq!(
            self.spectrum.len(),
            other.spectrum.len(),
            "Spectra must have the same length"
        );
        if self.params() != other.params() {
            panic!("Parameterisations must be the same");
        }
        let factor = match metric {
            ErrorMetric::Mse => math::rcp_f64(self.samples().len() as f64),
            ErrorMetric::Nllsq => 0.5,
        };
        match rmetric {
            ResidualErrorMetric::Identity => {
                self.samples
                    .iter()
                    .zip(other.samples.iter())
                    .fold(0.0, |acc, (x, y)| {
                        let diff = *x as f64 - *y as f64;
                        acc + math::sqr(diff) * factor
                    })
            },
            ResidualErrorMetric::JLow => {
                self.snapshots()
                    .zip(other.snapshots())
                    .fold(0.0f64, |acc, (xs, ys)| {
                        let cos_theta_i = xs.wi.theta.cos() as f64;
                        acc + xs.samples.iter().zip(ys.samples.iter()).fold(
                            0.0f64,
                            |acc, (a, b)| {
                                let diff = (*a as f64 * cos_theta_i + 1.0).ln()
                                    - (*b as f64 * cos_theta_i + 1.0).ln();
                                acc + math::sqr(diff) * factor
                            },
                        )
                    })
            },
        }
    }

    fn filtered_distance(
        &self,
        other: &Self,
        metric: ErrorMetric,
        rmetric: ResidualErrorMetric,
        limit: Radians,
    ) -> f64
    where
        Self: Sized,
    {
        assert_eq!(self.spectrum(), other.spectrum(), "Spectra must be equal!");
        if self.params != other.params {
            panic!("Parameterization must be the same!");
        }
        if (limit.as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6 {
            return self.distance(other, metric, rmetric);
        }
        let n_wo = self.params.n_wo();
        let n_wi_filtered = self
            .params
            .incoming
            .iter()
            .position(|sph| sph.theta >= limit)
            .unwrap_or(self.params.n_wi());
        let n_wo_filtered = self
            .params
            .outgoing
            .iter()
            .position(|sph| sph.theta >= limit)
            .unwrap_or(self.params.n_wo());
        let n_spectrum = self.spectrum.len();
        let n_samples = n_wi_filtered * n_wo_filtered * n_spectrum;

        let factor = match metric {
            ErrorMetric::Mse => math::rcp_f64(n_samples as f64),
            ErrorMetric::Nllsq => 0.5,
        };

        let mut dist = 0.0;
        for i in 0..n_wi_filtered {
            for j in 0..n_wo_filtered {
                for k in 0..n_spectrum {
                    let offset = i * n_wo * n_spectrum + j * n_spectrum + k;
                    let diff = self.samples[offset] as f64 - other.samples[offset] as f64;
                    dist += math::sqr(diff) * factor;
                }
            }
        }

        dist
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
