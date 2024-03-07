use crate::{
    app::cache::{Cache, RawCache},
    fitting::{FittingProblem, FittingReport},
    measure::bsdf::MeasuredBsdfData,
    partition::SphericalPartition,
};
use base::{
    math::{sph_to_cart, Vec3, Vec3A},
    optics::{fresnel, ior::RefractiveIndex},
    Isotropy,
};
use bxdf::{
    brdf::microfacet::{BeckmannBrdfModel, TrowbridgeReitzBrdfModel},
    MicrofacetBasedBrdfFittingModel, MicrofacetBasedBrdfModel, MicrofacetBasedBrdfModelKind,
};
use jabr::optics::reflect;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, U1, U2};
use std::{
    borrow::Cow,
    fmt::{Debug, Display},
};

/// The fitting problem for the microfacet based BSDF model.
///
/// The fitting procedure ir based on the Levenberg-Marquardt algorithm.
pub struct MicrofacetBrdfFittingProblem<'a> {
    /// The measured BSDF data.
    pub measured: Cow<'a, MeasuredBsdfData>,
    /// The target BSDF model.
    pub target: MicrofacetBasedBrdfModelKind,
    /// The refractive indices of the incident medium at the measured
    /// wavelengths.
    pub iors_i: Box<[RefractiveIndex]>,
    /// The refractive indices of the transmitted medium at the measured
    /// wavelengths.
    pub iors_t: Box<[RefractiveIndex]>,
}

impl<'a> Display for MicrofacetBrdfFittingProblem<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BSDF Fitting Problem")
            .field("target", &self.target)
            .finish()
    }
}

impl<'a> MicrofacetBrdfFittingProblem<'a> {
    /// Creates a new BSDF fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured BSDF data.
    /// * `target` - The target BSDF model.
    pub fn new(
        measured: &'a MeasuredBsdfData,
        target: MicrofacetBasedBrdfModelKind,
        cache: &RawCache,
    ) -> Self {
        let wavelengths = measured
            .params
            .emitter
            .spectrum
            .values()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let iors_i = cache
            .iors
            .ior_of_spectrum(measured.params.incident_medium, &wavelengths)
            .unwrap_or_else(|| {
                panic!(
                    "missing refractive indices for {:?}",
                    measured.params.incident_medium
                )
            });
        let iors_t = cache
            .iors
            .ior_of_spectrum(measured.params.transmitted_medium, &wavelengths)
            .unwrap_or_else(|| {
                panic!(
                    "missing refractive indices for {:?}",
                    measured.params.transmitted_medium
                )
            });
        Self {
            measured: Cow::Borrowed(measured),
            target,
            iors_i,
            iors_t,
        }
    }
}

/// Initialises the microfacet based BSDF models with the given range of
/// roughness parameters as the initial guess.
fn initialise_microfacet_bsdf_models(
    min: f64,
    max: f64,
    num: u32,
    target: MicrofacetBasedBrdfModelKind,
) -> Vec<Box<dyn MicrofacetBasedBrdfFittingModel>> {
    let step = (max - min) / (num - 1) as f64;
    match target {
        MicrofacetBasedBrdfModelKind::TrowbridgeReitz => (0..num)
            .map(|i| {
                let alpha = min + step * i as f64;
                Box::new(TrowbridgeReitzBrdfModel::new(alpha, alpha)) as _
            })
            .collect(),
        MicrofacetBasedBrdfModelKind::Beckmann => (0..num)
            .map(|i| {
                let alpha = min + step * i as f64;
                Box::new(BeckmannBrdfModel::new(alpha, alpha)) as _
            })
            .collect(),
    }
}

// Actual implementation of the fitting problem.
impl<'a> FittingProblem for MicrofacetBrdfFittingProblem<'a> {
    type Model = Box<dyn MicrofacetBasedBrdfModel>;

    fn lsq_lm_fit(self) -> FittingReport<Self::Model> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let solver = LevenbergMarquardt::new();
        let mut results = {
            initialise_microfacet_bsdf_models(0.2, 0.3, 100, self.target)
                // .into_par_iter()
                .into_iter()
                .filter_map(|model| {
                    log::debug!(
                        "Fitting Isotropic BRDF model: {:?} with αx = {} αy = {}",
                        model.kind(),
                        model.alpha_x(),
                        model.alpha_y()
                    );
                    let problem = MicrofacetBrdfFittingProblemProxy::<{ Isotropy::Isotropic }>::new(
                        &self.measured,
                        model,
                        &self.iors_i,
                        &self.iors_t,
                    );
                    let (result, report) = solver.minimize(problem);
                    log::debug!(
                        "Fitted αx = {} αy = {}, report: {:?}",
                        result.model.alpha_x(),
                        result.model.alpha_y(),
                        report
                    );
                    match report.termination {
                        TerminationReason::Converged { .. } => Some((result.model as _, report)),
                        _ => None,
                    }
                })
                .collect::<Vec<_>>()
        };
        results.shrink_to_fit();
        FittingReport::new(results, |m| m.alpha_x() > 0.0 && m.alpha_y() > 0.0)
    }
}

struct MicrofacetBrdfFittingProblemProxy<'a, const I: Isotropy> {
    /// The measured BSDF data.
    measured: &'a MeasuredBsdfData,
    /// Maximum value of the measured samples for each snapshot. Only the first
    /// spectral sample is considered.
    max_measured: Box<[f64]>,
    /// The target BSDF model.
    model: Box<dyn MicrofacetBasedBrdfFittingModel>,
    /// The actual partition (not the params) of the measured BSDF data.
    partition: SphericalPartition,
    /// The refractive indices of the incident medium at the measured
    /// wavelengths.
    iors_i: &'a [RefractiveIndex],
    /// The refractive indices of the transmitted medium at the measured
    /// wavelengths.
    iors_t: &'a [RefractiveIndex],
    /// Incident directions
    wis: Box<[Vec3]>,
    /// Outgoing directions
    wos: Box<[Vec3]>,
}

fn eval_residuals<const I: Isotropy>(problem: &MicrofacetBrdfFittingProblemProxy<I>) -> Box<[f64]> {
    // The number of residuals is the number of patches times the number of
    // snapshots.
    let n_patches = problem.partition.num_patches();
    let residuals_count = n_patches * problem.measured.snapshots.len();
    let mut rs = Box::new_uninit_slice(residuals_count);
    problem
        .measured
        .snapshots
        .iter()
        .enumerate()
        .for_each(|(i, snapshot)| {
            let wi = {
                let sph = snapshot.wi;
                sph_to_cart(sph.theta, sph.phi)
            };
            let max_measured = problem.max_measured[i];
            // Use precise max value for the modelled samples
            let wo: Vec3 = fresnel::reflect(Vec3A::from(-wi), Vec3A::Z).into();
            let max_modelled = problem
                .model
                .eval(wi, wo, &problem.iors_i[0], &problem.iors_t[0]);
            problem
                .partition
                .patches
                .iter()
                .enumerate()
                .for_each(|(j, patch)| {
                    let wo = {
                        let c = patch.center();
                        sph_to_cart(c.theta, c.phi)
                    };
                    // Only the first wavelength is used. TODO: Use all
                    let measured = snapshot.samples[j][0] as f64 / max_measured;
                    let modelled =
                        problem
                            .model
                            .eval(wi, wo, &problem.iors_i[0], &problem.iors_t[0])
                            / max_modelled;
                    rs[i * n_patches + j].write(modelled - measured);
                });
        });
    unsafe { rs.assume_init() }
}

impl<'a, const I: Isotropy> MicrofacetBrdfFittingProblemProxy<'a, I> {
    pub fn new(
        measured: &'a MeasuredBsdfData,
        model: Box<dyn MicrofacetBasedBrdfFittingModel>,
        iors_i: &'a [RefractiveIndex],
        iors_t: &'a [RefractiveIndex],
    ) -> Self {
        let partition = measured.params.receiver.partitioning();
        let max_measured = measured
            .snapshots
            .iter()
            .map(|snapshot| {
                snapshot
                    .samples
                    .iter()
                    .fold(0.0, |m: f64, s| m.max(s[0] as f64))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let wis = measured
            .snapshots
            .iter()
            .map(|s| s.wi.to_cartesian())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let wos = partition
            .patches
            .iter()
            .map(|p| p.center().to_cartesian())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            measured,
            max_measured,
            model,
            partition,
            iors_i,
            iors_t,
            wis,
            wos,
        }
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U1>
    for MicrofacetBrdfFittingProblemProxy<'a, { Isotropy::Isotropic }>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, params: &Vector<f64, U1, Self::ParameterStorage>) {
        self.model.set_alpha_x(params[0]);
        self.model.set_alpha_y(params[0]);
    }

    fn params(&self) -> Vector<f64, U1, Self::ParameterStorage> {
        Vector::<f64, U1, Self::ParameterStorage>::new(self.model.alpha_x())
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(&eval_residuals(
            self,
        )))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        // Temporary implementation: only first wavelength is used.
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
            &self.model.partial_derivatives_isotropic(
                &self.wis,
                &self.wos,
                &self.iors_i[0],
                &self.iors_t[0],
            ),
        ))
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U2>
    for MicrofacetBrdfFittingProblemProxy<'a, { Isotropy::Anisotropic }>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    impl_least_squares_problem_common_methods!(self, Vector<f64, U2, Self::ParameterStorage>);

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(&eval_residuals(
            self,
        )))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        // Temporary implementation: only first wavelength is used.
        Some(OMatrix::<f64, Dyn, U2>::from_row_slice(
            &self
                .model
                .partial_derivatives(&self.wis, &self.wos, &self.iors_i[0], &self.iors_t[0]),
        ))
    }
}
