use crate::{
    fitting::{FittingProblem, FittingReport, ResidualErrorMetric},
    measure::bsdf::MeasuredBrdfLevel,
};
use base::{
    math::Vec3,
    optics::ior::{Ior, RefractiveIndexRegistry},
    range::RangeByStepSizeInclusive,
    units::Radians,
    ErrorMetric, Isotropy,
};
use bxdf::{
    brdf::{
        analytical::microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf},
        measured::{
            AnalyticalFit, BrdfParameterisation, ClausenBrdf, ClausenBrdfParameterisation,
            MeasuredBrdfKind, VgonioBrdf, VgonioBrdfParameterisation, Yan2018Brdf,
            Yan2018BrdfParameterisation,
        },
        Bxdf,
    },
    distro::MicrofacetDistroKind,
    Scattering,
};
use jabr::array::DyArr;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, U1, U2};
use std::{any::Any, convert::identity, fmt::Display};

/// The fitting problem for the microfacet based BSDF model.
///
/// The fitting procedure is based on the Levenberg-Marquardt algorithm.
pub struct MicrofacetBrdfFittingProblem<'a, P: BrdfParameterisation> {
    /// The measured BSDF data.
    pub measured: &'a (dyn AnalyticalFit<Params = P> + Sync),
    /// The target BSDF model.
    pub target: MicrofacetDistroKind,
    /// The level of the measured BRDF data to be fitted.
    pub level: MeasuredBrdfLevel,
    /// The refractive indices of the incident medium at the measured
    /// wavelengths.
    pub iors_i: Box<[Ior]>,
    /// The refractive indices of the transmitted medium at the measured
    /// wavelengths.
    pub iors_t: Box<[Ior]>,
    /// The initial guess for the roughness parameter.
    pub initial_guess: RangeByStepSizeInclusive<f64>,
    /// The polar angle limit for the data fitting.
    pub theta_limit: Radians,
}

impl<'a, P: BrdfParameterisation> Display for MicrofacetBrdfFittingProblem<'a, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BSDF Fitting Problem")
            .field("target", &self.target)
            .finish()
    }
}

impl<'a, P: BrdfParameterisation> MicrofacetBrdfFittingProblem<'a, P> {
    /// Creates a new BSDF fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured BSDF data.
    /// * `target` - The target BSDF model.
    pub fn new(
        measured: &'a (dyn AnalyticalFit<Params = P> + Sync),
        target: MicrofacetDistroKind,
        initial: RangeByStepSizeInclusive<f64>,
        level: MeasuredBrdfLevel, // temporarily only one level is supported
        iors: &'a RefractiveIndexRegistry,
        theta_limit: Radians,
    ) -> Self {
        let spectrum = measured.spectrum();
        let iors_i = iors
            .ior_of_spectrum(measured.incident_medium(), &spectrum)
            .unwrap_or_else(|| {
                panic!(
                    "missing refractive indices for {:?}",
                    measured.incident_medium()
                )
            });
        let iors_t = iors
            .ior_of_spectrum(measured.transmitted_medium(), &spectrum)
            .unwrap_or_else(|| {
                panic!(
                    "missing refractive indices for {:?}",
                    measured.transmitted_medium()
                )
            });
        Self {
            measured,
            target,
            level,
            iors_i,
            iors_t,
            initial_guess: initial,
            theta_limit,
        }
    }
}

/// Initialises the microfacet based BSDF models with the given range of
/// roughness parameters as the initial guess.
fn initialise_microfacet_bsdf_models(
    min: f64,
    max: f64,
    num: u32,
    target: MicrofacetDistroKind,
) -> Vec<Box<dyn Bxdf<Params = [f64; 2]>>> {
    let step = (max - min) / num as f64;
    match target {
        MicrofacetDistroKind::TrowbridgeReitz => (0..=num)
            .map(|i| {
                let alpha = min + step * i as f64;
                Box::new(TrowbridgeReitzBrdf::new(alpha, alpha)) as _
            })
            .collect(),
        MicrofacetDistroKind::Beckmann => (0..=num)
            .map(|i| {
                let alpha = min + step * i as f64;
                Box::new(BeckmannBrdf::new(alpha, alpha)) as _
            })
            .collect(),
    }
}

macro_rules! switch_isotropy {
    ($brdf_ty:ident<$brdf_params_ty:ident> => $isotropy:ident, $self:ident, $brdf:ident, $model:ident, $solver:ident, $rmetric:ident) => {
        match $isotropy {
            Isotropy::Isotropic => {
                let problem = BrdfFittingProblemProxy::<
                    $brdf_params_ty,
                    $brdf_ty,
                    { Isotropy::Isotropic },
                >::new(
                    $brdf,
                    $model,
                    &$self.iors_i,
                    &$self.iors_t,
                    $self.theta_limit,
                    $rmetric,
                );
                let (result, report) = $solver.minimize(problem);
                (result.model, report)
            },
            Isotropy::Anisotropic => {
                let problem = BrdfFittingProblemProxy::<
                    $brdf_params_ty,
                    $brdf_ty,
                    { Isotropy::Anisotropic },
                >::new(
                    $brdf,
                    $model,
                    &$self.iors_i,
                    &$self.iors_t,
                    $self.theta_limit,
                    $rmetric,
                );
                let (result, report) = $solver.minimize(problem);
                (result.model, report)
            },
        }
    };
}

// Actual implementation of the fitting problem.
impl<'a, P: BrdfParameterisation> FittingProblem for MicrofacetBrdfFittingProblem<'a, P> {
    type Model = Box<dyn Bxdf<Params = [f64; 2]>>;

    fn lsq_lm_fit(
        self,
        isotropy: Isotropy,
        rmetric: ResidualErrorMetric,
    ) -> FittingReport<Self::Model> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let solver = LevenbergMarquardt::new();
        let mut results = {
            initialise_microfacet_bsdf_models(
                self.initial_guess.start,
                self.initial_guess.stop,
                self.initial_guess.step_count() as u32 - 1,
                self.target,
            )
            .into_par_iter()
            .filter_map(|model| {
                let init_guess = model.params();
                let kind = model.family();
                let (model, report) = match self.measured.kind() {
                    MeasuredBrdfKind::Clausen => {
                        let brdf = self.measured.as_any().downcast_ref::<ClausenBrdf>().unwrap();
                        let problem = BrdfFittingProblemProxy::<
                            ClausenBrdfParameterisation,
                            ClausenBrdf,
                            { Isotropy::Isotropic },
                        >::new(brdf, model, &self.iors_i, &self.iors_t, self.theta_limit, rmetric);
                        let (result, report) = solver.minimize(problem);
                        (result.model, report)
                    },
                    MeasuredBrdfKind::Vgonio => {
                        let brdf = self.measured.as_any().downcast_ref::<VgonioBrdf>().unwrap();
                        switch_isotropy!(VgonioBrdf<VgonioBrdfParameterisation> => isotropy, self, brdf, model, solver, rmetric)
                    },
                    MeasuredBrdfKind::Yan2018 => {
                        let brdf = self.measured.as_any().downcast_ref::<Yan2018Brdf>().unwrap();
                        switch_isotropy!(Yan2018Brdf<Yan2018BrdfParameterisation> => isotropy, self, brdf, model, solver, rmetric)
                    },
                    _ => {
                        log::warn!("Unsupported BRDF kind: {:?}", self.measured.kind());
                        return None;
                    },
                };
                log::debug!(
                    "Fitting {isotropy} BRDF model: {kind:?} with αx = {} αy = {}\n    - fitted αx = {} αy = {}\n    - report: \
                     {report:?}",
                    init_guess[0],
                    init_guess[1],
                    model.params()[0],
                    model.params()[1],
                );
                match report.termination {
                    TerminationReason::Converged { .. } | TerminationReason::LostPatience => Some((model, report)),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
        };
        results.shrink_to_fit();
        FittingReport::new(results)
    }
}

struct BrdfFittingProblemProxy<'a, P, M, const I: Isotropy>
where
    P: BrdfParameterisation,
    M: AnalyticalFit<Params = P>,
{
    /// The measured BSDF data.
    measured: &'a M,
    /// The target BSDF model.
    model: Box<dyn Bxdf<Params = [f64; 2]>>,
    /// The refractive indices of the incident medium at the measured
    /// wavelengths.
    iors_i: &'a [Ior],
    /// The refractive indices of the transmitted medium at the measured
    /// wavelengths.
    iors_t: &'a [Ior],
    /// Polar angle limits for the data fitting.
    /// This is used to filter out the aberrant data points close to the grazing
    /// angles.
    theta_limit: Radians,
    /// Error metric for the residuals.
    error_metric: ResidualErrorMetric,
    /// Cached data for the fitting problem.
    cache: Box<dyn Any>,
}

/// The cache
pub struct CartesianDirectionsCache {
    /// Incident directions in cartesian coordinates.
    wis: DyArr<Vec3>,
    /// Outgoing directions in cartesian coordinates.
    wos: DyArr<Vec3>,
}

/// Common implementation for the fitting problem proxy using the
/// CartesianDirectionsCache.
macro_rules! impl_fitting_proxy_using_cartesian_cache {
    ($brdf:ident, $brdf_params:ident, $self:ident) => {
        impl<'a, const I: Isotropy> BrdfFittingProblemProxy<'a, $brdf_params, $brdf, I> {
            pub fn new(
                measured: &'a $brdf,
                model: Box<dyn Bxdf<Params = [f64; 2]>>,
                iors_i: &'a [Ior],
                iors_t: &'a [Ior],
                theta_limit: Radians,
                error_metric: ResidualErrorMetric,
            ) -> Self {
                let cache = CartesianDirectionsCache {
                    wis: measured.params().incoming_cartesian(),
                    wos: measured.params().outgoing_cartesian(),
                };
                Self {
                    measured,
                    model,
                    iors_i,
                    iors_t,
                    theta_limit,
                    error_metric,
                    cache: Box::new(cache),
                }
            }

            /// Evaluates the residuals of the fitting problem.
            pub fn eval_residuals(&$self) -> Box<[f64]> {
                log::debug!("evaluating residuals");
                // The number of residuals is n_wi * n_wo * n_spectrum.
                let n_wo = $self.measured.params.n_wo();
                let n_wi = $self.measured.params.n_wi();
                let n_spectrum = $self.measured.spectrum.len();
                let cache = $self
                .cache
                .downcast_ref::<CartesianDirectionsCache>()
                .unwrap();
                // Row-major [snapshot, patch, wavelength]
                let mut rs = Box::new_uninit_slice(n_wi * n_wo * n_spectrum);
                for (i, snapshot) in $self.measured.snapshots().enumerate() {
                    let cos_theta_i = snapshot.wi.theta.cos() as f64;
                    let wi = snapshot.wi.to_cartesian();
                        cache
                        .wos
                        .iter()
                        .zip(snapshot.samples.chunks(n_spectrum))
                        .enumerate()
                        .for_each(|(j, (wo, samples))| {
                            let modelled_values = Scattering::eval_reflectance_spectrum(
                                $self.model.as_ref(),
                                &wi,
                                &wo,
                                $self.iors_i,
                                $self.iors_t,
                            );
                            match $self.error_metric {
                                ResidualErrorMetric::Identity => {
                                    samples
                                        .iter()
                                        .zip(modelled_values.iter())
                                        .enumerate()
                                        .for_each(|(k, (measured, modelled))| {
                                            rs[i * n_wo * n_spectrum + j * n_spectrum + k]
                                                .write(modelled - *measured as f64);
                                        });
                                },
                                ResidualErrorMetric::JLow => {
                                    samples
                                        .iter()
                                        .zip(modelled_values.iter())
                                        .enumerate()
                                        .for_each(|(k, (measured, modelled))| {
                                            let me = (*measured as f64 * cos_theta_i + 1.0).ln();
                                            let md = (*modelled * cos_theta_i + 1.0).ln();
                                            rs[i * n_wo * n_spectrum + j * n_spectrum + k]
                                                .write(md - me);
                                        });
                                },
                            }
                    });
                }
                unsafe { rs.assume_init() }
            }

            /// Evaluates the residuals of the fitting problem with filtering.
            pub fn eval_residuals_filtered(&$self) -> Box<[f64]> {
                log::debug!("evaluating residuals with filtering");
                // The number of residuals is n_wi * n_wo * n_spectrum.
                let n_wo = $self.measured.params.n_wo();
                let n_wi = $self.measured.params.n_wi();
                let n_spectrum = $self.measured.spectrum.len();
                let cache = $self
                    .cache
                    .downcast_ref::<CartesianDirectionsCache>()
                    .unwrap();
                let n_wi_filtered = cache
                    .wis
                    .iter()
                    .position(|wi| base::math::theta(wi) >= $self.theta_limit)
                    .unwrap();
                let n_wo_filtered = cache
                    .wos
                    .iter()
                    .position(|wo| base::math::theta(wo) >= $self.theta_limit)
                    .unwrap();
                // Row-major [snapshot, patch, wavelength] = [wi, wo, wavelength]
                let mut rs = Box::new_uninit_slice(n_wi_filtered * n_wo_filtered * n_spectrum);
                // Incident directions.
                for (i, snapshot) in $self.measured.snapshots().enumerate() {
                    if i >= n_wi_filtered {
                        break;
                    }
                    let wi = snapshot.wi.to_cartesian();
                    // Outgoing directions.
                    for (j, (wo, samples)) in cache
                        .wos
                        .iter()
                        .zip(snapshot.samples.chunks(n_spectrum))
                        .enumerate()
                    {
                        if j >= n_wo_filtered {
                            break;
                        }
                        let modelled_values = Scattering::eval_reflectance_spectrum(
                            $self.model.as_ref(),
                            &wi,
                            &wo,
                            $self.iors_i,
                            $self.iors_t,
                        );
                        // Wavelengths.
                        for (k, (measured, modelled)) in samples.iter().zip(modelled_values.iter()).enumerate()
                        {
                            rs[i * n_wo_filtered * n_spectrum + j * n_spectrum + k]
                            .write(modelled - *measured as f64);
                        }
                    }
                }
                unsafe { rs.assume_init() }
            }

            fn jacobian_iso(&$self) -> Box<[f64]> {
                let n_spectrum = $self.measured.spectrum.len();
                let n_wi = $self.measured.params.n_wi();
                let n_wo = $self.measured.params.n_wo();
                let cache = $self
                    .cache
                   .downcast_ref::<CartesianDirectionsCache>()
                    .unwrap();
                let derivative_per_wavelength = $self
                    .iors_i
                    .iter()
                    .zip($self.iors_t.iter())
                    .map(|(ior_i, ior_t)| {
                        $self.model
                           .pds_iso(cache.wis.as_ref(), cache.wos.as_ref(), ior_i, ior_t)
                    })
                    .collect::<Vec<_>>();
                // Re-arrange the shape of the derivatives to match the residuals:
                // [snapshot, patch, wavelength]
                let mut jacobian = vec![0.0; n_wi * n_wo * n_spectrum].into_boxed_slice();
                for i in 0..n_wi {
                    for j in 0..n_wo {
                        for k in 0..n_spectrum {
                            let offset = i * n_wo * n_spectrum + j * n_spectrum + k;
                            jacobian[offset] = derivative_per_wavelength[k][i * n_wo + j];
                        }
                    }
                }
                jacobian
            }

            fn jacobian_iso_filtered(&$self) -> Box<[f64]> {
                let n_spectrum = $self.measured.spectrum.len();
                let n_wi = $self.measured.params.n_wi();
                let n_wo = $self.measured.params.n_wo();
                let cache = $self
                    .cache
                    .downcast_ref::<CartesianDirectionsCache>()
                    .unwrap();
                let n_wi_filtered = cache
                    .wis
                    .iter()
                    .position(|wi| base::math::theta(wi) >= $self.theta_limit)
                    .unwrap();
                let n_wo_filtered = cache
                    .wos
                    .iter()
                    .position(|wo| base::math::theta(wo) >= $self.theta_limit)
                    .unwrap();
                let derivative_per_wavelength = $self
                    .iors_i
                    .iter()
                    .zip($self.iors_t.iter())
                    .map(|(ior_i, ior_t)| {
                        $self.model
                            .pds_iso(cache.wis.as_ref(), cache.wos.as_ref(), ior_i, ior_t)
                    })
                    .collect::<Vec<_>>();
                // Re-arrange the shape of the derivatives to match the residuals:
                // [wi, wo, wavelength]
                let mut jacobian = vec![0.0; n_wi_filtered * n_wo_filtered * n_spectrum].into_boxed_slice();
                for i in 0..n_wi_filtered {
                    for j in 0..n_wo_filtered {
                        for k in 0..n_spectrum {
                            let offset = i * n_wo_filtered * n_spectrum + j * n_spectrum + k;
                            jacobian[offset] = derivative_per_wavelength[k][i * n_wo + j];
                        }
                    }
                }
                jacobian
            }

            fn jacobian_aniso(&$self) -> Box<[f64]> {
                let n_wi = $self.measured.params.n_wi();
                let n_wo = $self.measured.params.n_wo();
                let cache = $self
                    .cache
                    .downcast_ref::<CartesianDirectionsCache>()
                    .unwrap();
                let n_wi_filtered = cache
                    .wis
                    .iter()
                    .position(|wi| base::math::theta(wi) >= $self.theta_limit)
                    .unwrap();
                let n_wo_filtered = cache
                    .wos
                    .iter()
                    .position(|wo| base::math::theta(wo) >= $self.theta_limit)
                    .unwrap();
                let derivative_per_wavelength = $self
                    .iors_i
                    .iter()
                    .zip($self.iors_t.iter())
                    .map(|(ior_i, ior_t)| {
                        $self.model.pds(cache.wis.as_ref(), cache.wos.as_ref(), ior_i, ior_t)
                    })
                    .collect::<Vec<_>>();
                let n_spectrum = $self.measured.spectrum.len();
                // Re-arrange the shape of the derivatives to match the residuals:
                // [wi, wo, wavelength, alpha]
                //  n_wi, n_wo, n_wavelength, 2
                let mut jacobian =
                vec![0.0; n_wi_filtered * n_wo_filtered * n_spectrum * 2].into_boxed_slice();
                for i in 0..n_wi_filtered {
                    for j in 0..n_wo_filtered {
                        for k in 0..n_spectrum {
                            let offset = i * n_wo_filtered * n_spectrum * 2 + j * n_spectrum * 2 + k * 2;
                            jacobian[offset] = derivative_per_wavelength[k][i * n_wo * 2 + j * 2];
                            jacobian[offset + 1] = derivative_per_wavelength[k][i * n_wo * 2 + j * 2 + 1];
                        }
                    }
                }
                jacobian
            }

            fn jacobian_aniso_filtered(&$self) -> Box<[f64]> {
                let n_wi = $self.measured.params.n_wi();
                let n_wo = $self.measured.params.n_wo();
                let cache = $self
                    .cache
                    .downcast_ref::<CartesianDirectionsCache>()
                    .unwrap();
                let derivative_per_wavelength = $self
                    .iors_i
                    .iter()
                    .zip($self.iors_t.iter())
                    .map(|(ior_i, ior_t)| {
                    $self.model
                            .pds(cache.wis.as_ref(), cache.wos.as_ref(), ior_i, ior_t)
                    })
                    .collect::<Vec<_>>();
                let n_spectrum = $self.measured.spectrum.len();
                // Re-arrange the shape of the derivatives to match the residuals:
                // [wi, wo, wavelength, alpha]
                //  n_wi, n_wo, n_wavelength, 2
                let mut jacobian = vec![0.0; n_wi * n_wo * n_spectrum * 2].into_boxed_slice();
                for i in 0..n_wi {
                    for j in 0..n_wo {
                        for k in 0..n_spectrum {
                            let offset = i * n_wo * n_spectrum * 2 + j * n_spectrum * 2 + k * 2;
                            jacobian[offset] = derivative_per_wavelength[k][i * n_wo * 2 + j * 2];
                            jacobian[offset + 1] = derivative_per_wavelength[k][i * n_wo * 2 + j * 2 + 1];
                        }
                    }
                }
                jacobian
            }
        }
    };
}

impl_fitting_proxy_using_cartesian_cache!(VgonioBrdf, VgonioBrdfParameterisation, self);
impl_fitting_proxy_using_cartesian_cache!(Yan2018Brdf, Yan2018BrdfParameterisation, self);

// Macro to generate the residuals method for the fitting problem proxy to avoid
// code duplication.
macro_rules! residuals {
    ($self:ident) => {
        if ($self.theta_limit.as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6 {
            Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
                &$self.eval_residuals(),
            ))
        } else {
            Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
                &$self.eval_residuals_filtered(),
            ))
        }
    };
}

macro_rules! impl_lsq_using_cartesian_cache {
    (iso $brdf:ty, $brdf_params:ty, $self:ident) => {
        impl<'a> LeastSquaresProblem<f64, Dyn, U1> for BrdfFittingProblemProxy<'a, $brdf_params, $brdf, { Isotropy::Isotropic }>
        {
            type ResidualStorage = VecStorage<f64, Dyn, U1>;
            type JacobianStorage = Owned<f64, Dyn, U1>;
            type ParameterStorage = Owned<f64, U1, U1>;

            fn set_params(&mut $self, params: &Vector<f64, U1, Self::ParameterStorage>) {
                $self.model.set_params(&[params[0], params[0]]);
            }

            fn params(&$self) -> Vector<f64, U1, Self::ParameterStorage> {
                Vector::<f64, U1, Self::ParameterStorage>::new($self.model.params()[0])
            }

            fn residuals(&$self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
                residuals!($self)
            }

            fn jacobian(&$self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
                if ($self.theta_limit.as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6 {
                    Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
                        &$self.jacobian_iso(),
                    ))
                } else {
                    Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
                        &$self.jacobian_iso_filtered(),
                    ))
                }
            }
        }
    };

    (aniso $brdf:ty, $brdf_params:ty, $self:ident) => {
        impl<'a> LeastSquaresProblem<f64, Dyn, U2> for BrdfFittingProblemProxy<'a, $brdf_params, $brdf, { Isotropy::Anisotropic }>
        {
            type ResidualStorage = VecStorage<f64, Dyn, U1>;
            type JacobianStorage = Owned<f64, Dyn, U2>;
            type ParameterStorage = Owned<f64, U2, U1>;

            fn set_params(&mut $self, params: &Vector<f64, U2, Self::ParameterStorage>) {
                $self.model.set_params(&[params[0], params[1]]);
            }

            fn params(&$self) -> Vector<f64, U2, Self::ParameterStorage> {
                Vector::<f64, U2, Self::ParameterStorage>::from($self.model.params())
            }

            fn residuals(&$self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> { residuals!($self) }

            fn jacobian(&$self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
                if ($self.theta_limit.as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6 {
                    Some(OMatrix::<f64, Dyn, U2>::from_row_slice(
                        &$self.jacobian_aniso(),
                    ))
                } else {
                    Some(OMatrix::<f64, Dyn, U2>::from_row_slice(
                        &$self.jacobian_aniso_filtered(),
                    ))
                }
            }
        }
    };
}

impl_lsq_using_cartesian_cache!(iso VgonioBrdf, VgonioBrdfParameterisation, self);
impl_lsq_using_cartesian_cache!(aniso VgonioBrdf, VgonioBrdfParameterisation, self);
impl_lsq_using_cartesian_cache!(iso Yan2018Brdf, Yan2018BrdfParameterisation, self);
impl_lsq_using_cartesian_cache!(aniso Yan2018Brdf, Yan2018BrdfParameterisation, self);

impl<'a, const I: Isotropy>
    BrdfFittingProblemProxy<'a, ClausenBrdfParameterisation, ClausenBrdf, I>
{
    pub fn new(
        measured: &'a ClausenBrdf,
        model: Box<dyn Bxdf<Params = [f64; 2]>>,
        iors_i: &'a [Ior],
        iors_t: &'a [Ior],
        theta_limit: Radians,
        error_metric: ResidualErrorMetric,
    ) -> Self {
        Self {
            measured,
            model,
            iors_i,
            iors_t,
            theta_limit,
            error_metric,
            cache: Box::new(()),
        }
    }

    /// Evaluates the residuals of the fitting problem.
    pub fn eval_residuals(&self) -> Box<[f64]> {
        let n_spectrum = self.measured.n_spectrum();
        let n_wo = self.measured.n_wo();
        self.measured
            .params
            .wi_wos_iter()
            .filter_map(|(i, (wi, wos))| {
                if wi.theta >= self.theta_limit {
                    return None;
                }
                let cos_theta_i = wi.theta.cos() as f64;
                let wi = wi.to_cartesian();
                Some(
                    wos.iter()
                        .enumerate()
                        .filter(|(_, wo)| wo.theta < self.theta_limit)
                        .flat_map(move |(j, wo)| {
                            let wo = wo.to_cartesian();
                            // Reuse the memory of the modelled samples to store the residuals.
                            let mut modelled = Scattering::eval_reflectance_spectrum(
                                self.model.as_ref(),
                                &wi,
                                &wo,
                                &self.iors_i,
                                &self.iors_t,
                            )
                            .into_vec();
                            let offset = i * n_wo * n_spectrum + j * n_spectrum;
                            let measured =
                                &self.measured.samples.as_slice()[offset..offset + n_spectrum];

                            if self.error_metric == ResidualErrorMetric::Identity {
                                modelled.iter_mut().zip(measured.iter()).for_each(
                                    |(modelled, measured)| {
                                        *modelled = *measured as f64 - *modelled;
                                    },
                                );
                            } else if self.error_metric == ResidualErrorMetric::JLow {
                                modelled.iter_mut().zip(measured.iter()).for_each(
                                    |(modelled, measured)| {
                                        let me = (*measured as f64 * cos_theta_i + 1.0).ln();
                                        let md = (*modelled * cos_theta_i + 1.0).ln();
                                        *modelled = md - me;
                                    },
                                )
                            }
                            modelled.into_iter()
                        }),
                )
            })
            .flatten()
            .collect::<Box<_>>()
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U1>
    for BrdfFittingProblemProxy<
        'a,
        ClausenBrdfParameterisation,
        ClausenBrdf,
        { Isotropy::Isotropic },
    >
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, params: &Vector<f64, U1, Self::ParameterStorage>) {
        self.model.set_params(&[params[0], params[0]]);
    }

    fn params(&self) -> Vector<f64, U1, Self::ParameterStorage> {
        Vector::<f64, U1, Self::ParameterStorage>::new(self.model.params()[0])
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
            &self.eval_residuals(),
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        let pds = self
            .measured
            .params
            .all_wi_wo_iter()
            .filter_map(|(wi, wo)| {
                if wi.theta >= self.theta_limit || wo.theta >= self.theta_limit {
                    return None;
                }
                let wi = wi.to_cartesian();
                let wo = wo.to_cartesian();
                Some(
                    self.iors_i
                        .iter()
                        .zip(self.iors_t.iter())
                        .map(move |(ior_i, ior_t)| self.model.pd_iso(&wi, &wo, ior_i, ior_t)),
                )
            })
            .flatten()
            .collect::<Vec<_>>();
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(&pds))
    }
}
