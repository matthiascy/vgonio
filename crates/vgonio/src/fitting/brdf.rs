use crate::{
    app::cache::RawCache,
    fitting::{FittingProblem, FittingReport},
    measure::bsdf::MeasuredBrdfLevel,
};
use base::{
    math::Vec3,
    optics::ior::{Ior, RefractiveIndexRegistry},
    range::RangeByStepSizeInclusive,
    Isotropy,
};
use bxdf::{
    brdf::{
        analytical::microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf},
        measured::{
            AnalyticalFit, BrdfParameterisation, ClausenBrdf, ClausenBrdfParameterisation,
            MeasuredBrdfKind, VgonioBrdf, VgonioBrdfParameterisation,
        },
        Bxdf,
    },
    distro::MicrofacetDistroKind,
    Scattering,
};
use jabr::array::DyArr;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, U1, U2};
use std::{any::Any, fmt::Display};

/// The fitting problem for the microfacet based BSDF model.
///
/// The fitting procedure is based on the Levenberg-Marquardt algorithm.
pub struct MicrofacetBrdfFittingProblem<'a, P: BrdfParameterisation, M: AnalyticalFit<Params = P>> {
    /// The measured BSDF data.
    pub measured: &'a M,
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
    /// The refractive index registry.
    // Used to retrieve the refractive indices of the incident and transmitted
    // mediums while calculating the modelled BSDF maximum values.
    pub iors: &'a RefractiveIndexRegistry,
    /// The initial guess for the roughness parameter.
    pub initial_guess: RangeByStepSizeInclusive<f64>,
}

impl<'a, P: BrdfParameterisation, M: AnalyticalFit<Params = P>> Display
    for MicrofacetBrdfFittingProblem<'a, P, M>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BSDF Fitting Problem")
            .field("target", &self.target)
            .finish()
    }
}

impl<'a, P: BrdfParameterisation, M: AnalyticalFit<Params = P>>
    MicrofacetBrdfFittingProblem<'a, P, M>
{
    /// Creates a new BSDF fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured BSDF data.
    /// * `target` - The target BSDF model.
    pub fn new(
        measured: &'a M,
        target: MicrofacetDistroKind,
        initial: RangeByStepSizeInclusive<f64>,
        level: MeasuredBrdfLevel, // temporarily only one level is supported
        cache: &'a RawCache,
    ) -> Self {
        let spectrum = measured.spectrum();
        let iors_i = cache
            .iors
            .ior_of_spectrum(measured.incident_medium(), &spectrum)
            .unwrap_or_else(|| {
                panic!(
                    "missing refractive indices for {:?}",
                    measured.incident_medium()
                )
            });
        let iors_t = cache
            .iors
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
            iors: &cache.iors,
            initial_guess: initial,
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

// Actual implementation of the fitting problem.
impl<'a, P: BrdfParameterisation, M: AnalyticalFit<Params = P>> FittingProblem
    for MicrofacetBrdfFittingProblem<'a, P, M>
{
    type Model = Box<dyn Bxdf<Params = [f64; 2]>>;

    fn lsq_lm_fit(self, isotropy: Isotropy) -> FittingReport<Self::Model> {
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
                let (model, report) = match isotropy {
                    Isotropy::Isotropic => {
                        let problem = BrdfFittingProblemProxy::<P, M, { Isotropy::Isotropic }>::new(
                            self.measured,
                            model,
                            self.iors,
                            &self.iors_i,
                            &self.iors_t,
                        );
                        let (result, report) = solver.minimize(problem);
                        (result.model, report)
                    }
                    Isotropy::Anisotropic => {
                        let problem =
                            BrdfFittingProblemProxy::<P, M, { Isotropy::Anisotropic }>::new(
                                self.measured,
                                model,
                                self.iors,
                                &self.iors_i,
                                &self.iors_t,
                            );
                        let (result, report) = solver.minimize(problem);
                        (result.model, report)
                    }
                };
                log::debug!(
                    "Fitting {} BRDF model: {:?} with αx = {} αy = {}\n    - fitted αx = {} αy = \
                     {}\n    - report: {:?}",
                    isotropy,
                    kind,
                    init_guess[0],
                    init_guess[1],
                    model.params()[0],
                    model.params()[1],
                    report
                );
                match report.termination {
                    TerminationReason::Converged { .. } | TerminationReason::LostPatience => {
                        Some((model, report))
                    }
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
    /// The refractive index registry.
    iors: &'a RefractiveIndexRegistry,
    /// The refractive indices of the incident medium at the measured
    /// wavelengths.
    iors_i: &'a [Ior],
    /// The refractive indices of the transmitted medium at the measured
    /// wavelengths.
    iors_t: &'a [Ior],
    /// Cached data for the fitting problem.
    cache: Box<dyn Any>,
}

pub struct VgonioBrdfFittingCache {
    /// Incident directions in cartesian coordinates.
    wis: DyArr<Vec3>,
    /// Outgoing directions in cartesian coordinates.
    wos: DyArr<Vec3>,
}

impl<'a, P: BrdfParameterisation, M: AnalyticalFit<Params = P>, const I: Isotropy>
    BrdfFittingProblemProxy<'a, P, M, I>
{
    pub fn new(
        measured: &'a M,
        model: Box<dyn Bxdf<Params = [f64; 2]>>,
        iors: &'a RefractiveIndexRegistry,
        iors_i: &'a [Ior],
        iors_t: &'a [Ior],
    ) -> Self {
        let cache: Box<dyn Any> = match measured.kind() {
            MeasuredBrdfKind::Vgonio => {
                let cache = VgonioBrdfFittingCache {
                    wis: measured.params().incoming_cartesian(),
                    wos: measured.params().outgoing_cartesian(),
                };
                Box::new(cache) as _
            }
            _ => Box::new(()) as _,
        };
        Self {
            measured,
            model,
            iors,
            iors_i,
            iors_t,
            cache,
        }
    }
}

impl<'a, const I: Isotropy> BrdfFittingProblemProxy<'a, VgonioBrdfParameterisation, VgonioBrdf, I> {
    /// Evaluates the residuals of the fitting problem.
    pub fn eval_residuals(&self) -> Box<[f64]> {
        // The number of residuals is n_wi * n_wo * n_spectrum.
        let n_wo = self.measured.params.n_wo();
        let n_wi = self.measured.params.n_wi();
        let n_spectrum = self.measured.spectrum.len();
        let mut cache = self.cache.downcast_ref::<VgonioBrdfFittingCache>().unwrap();
        // Row-major [snapshot, patch, wavelength]
        let mut rs = Box::new_uninit_slice(n_wi * n_wo * n_spectrum);
        for (i, snapshot) in self.measured.snapshots().enumerate() {
            let wi = snapshot.wi.to_cartesian();
            cache
                .wos
                .iter()
                .zip(snapshot.samples.chunks(n_spectrum))
                .enumerate()
                .for_each(|(j, (wo, samples))| {
                    let modelled_values = Scattering::eval_reflectance_spectrum(
                        self.model.as_ref(),
                        &wi,
                        &wo,
                        self.iors_i,
                        self.iors_t,
                    );
                    samples
                        .iter()
                        .zip(modelled_values.iter())
                        .enumerate()
                        .for_each(|(k, (measured, modelled))| {
                            rs[i * n_wo * n_spectrum + j * n_spectrum + k]
                                .write(modelled - *measured as f64);
                        });
                });
        }
        unsafe { rs.assume_init() }
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U1>
    for BrdfFittingProblemProxy<'a, VgonioBrdfParameterisation, VgonioBrdf, { Isotropy::Isotropic }>
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

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
            &self.eval_residuals(),
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        let n_spectrum = self.measured.spectrum.len();
        let n_wi = self.measured.params.n_wi();
        let n_wo = self.measured.params.n_wo();
        let cache = self.cache.downcast_ref::<VgonioBrdfFittingCache>().unwrap();
        let derivative_per_wavelength = self
            .iors_i
            .iter()
            .zip(self.iors_t.iter())
            .map(|(ior_i, ior_t)| {
                self.model
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
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(&jacobian))
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U2>
    for BrdfFittingProblemProxy<
        'a,
        VgonioBrdfParameterisation,
        VgonioBrdf,
        { Isotropy::Anisotropic },
    >
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, params: &Vector<f64, U2, Self::ParameterStorage>) {
        self.model.set_params(&[params[0], params[1]]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::from(self.model.params())
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
            &self.eval_residuals(),
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        let n_wi = self.measured.params.n_wi();
        let n_wo = self.measured.params.n_wo();
        let cache = self.cache.downcast_ref::<VgonioBrdfFittingCache>().unwrap();
        let derivative_per_wavelength = self
            .iors_i
            .iter()
            .zip(self.iors_t.iter())
            .map(|(ior_i, ior_t)| {
                self.model
                    .pds(cache.wis.as_ref(), cache.wos.as_ref(), ior_i, ior_t)
            })
            .collect::<Vec<_>>();
        let n_spectrum = self.measured.spectrum.len();
        // Re-arrange the shape of the derivatives to match the residuals:
        // [snapshot, patch, wavelength, alpha]
        //  n_snapshot, n_patch, n_wavelength, 2
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
        Some(OMatrix::<f64, Dyn, U2>::from_row_slice(&jacobian))
    }
}

impl<'a, const I: Isotropy>
    BrdfFittingProblemProxy<'a, ClausenBrdfParameterisation, ClausenBrdf, I>
{
    pub fn new(
        measured: &'a ClausenBrdf,
        model: Box<dyn Bxdf<Params = [f64; 2]>>,
        iors: &'a RefractiveIndexRegistry,
        iors_i: &'a [Ior],
        iors_t: &'a [Ior],
    ) -> Self {
        Self {
            measured,
            model,
            iors,
            iors_i,
            iors_t,
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
            .flat_map(|(i, (wi, wos))| {
                let wi = wi.to_cartesian();
                wos.iter().enumerate().flat_map(move |(j, wo)| {
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
                    let measured = &self.measured.samples.as_slice()[offset..offset + n_spectrum];
                    modelled
                        .iter_mut()
                        .zip(measured.iter())
                        .for_each(|(modelled, measured)| {
                            *modelled = *measured as f64 - *modelled;
                        });
                    modelled.into_iter()
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
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
            .flat_map(|(wi_sph, wo_sph)| {
                let wi = wi_sph.to_cartesian();
                let wo = wo_sph.to_cartesian();
                self.iors_i
                    .iter()
                    .zip(self.iors_t)
                    .map(move |(ior_i, ior_t)| self.model.pd_iso(&wi, &wo, ior_i, ior_t))
            })
            .collect::<Vec<_>>();
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(&pds))
    }
}
