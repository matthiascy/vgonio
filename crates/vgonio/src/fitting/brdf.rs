use crate::{
    app::cache::{Cache, RawCache, RefractiveIndexRegistry},
    fitting::{err::generate_analytical_brdf, FittingProblem, FittingReport},
    measure::bsdf::MeasuredBsdfData,
    partition::SphericalPartition,
    RangeByStepCountInclusive, RangeByStepSizeInclusive,
};
use base::{
    math::{sph_to_cart, Vec3, Vec3A},
    optics::{fresnel, ior::RefractiveIndexRecord},
    Isotropy,
};
use bxdf::{
    brdf::microfacet::{BeckmannBrdfModel, TrowbridgeReitzBrdfModel},
    MicrofacetBasedBrdfFittingModel, MicrofacetBasedBrdfModel, MicrofacetBrdfKind,
};
use jabr::optics::reflect;
use levenberg_marquardt::{
    LeastSquaresProblem, LevenbergMarquardt, MinimizationReport, TerminationReason,
};
use nalgebra::{Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, U1, U2};
use rayon::{iter::IndexedParallelIterator, slice::ParallelSlice};
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
    pub target: MicrofacetBrdfKind,
    /// The refractive indices of the incident medium at the measured
    /// wavelengths.
    pub iors_i: Box<[RefractiveIndexRecord]>,
    /// The refractive indices of the transmitted medium at the measured
    /// wavelengths.
    pub iors_t: Box<[RefractiveIndexRecord]>,
    /// The refractive index registry.
    // Used to retrieve the refractive indices of the incident and transmitted
    // mediums while calculating the modelled BSDF maxiumum values.
    pub iors: &'a RefractiveIndexRegistry,
    /// The initial guess for the roughness parameter.
    pub initial_guess: RangeByStepSizeInclusive<f64>,
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
        target: MicrofacetBrdfKind,
        initial: RangeByStepSizeInclusive<f64>,
        cache: &'a RawCache,
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
    target: MicrofacetBrdfKind,
) -> Vec<Box<dyn MicrofacetBasedBrdfFittingModel>> {
    let step = (max - min) / num as f64;
    match target {
        MicrofacetBrdfKind::TrowbridgeReitz => (0..=num)
            .map(|i| {
                let alpha = min + step * i as f64;
                Box::new(TrowbridgeReitzBrdfModel::new(alpha, alpha)) as _
            })
            .collect(),
        MicrofacetBrdfKind::Beckmann => (0..=num)
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
            initialise_microfacet_bsdf_models(
                self.initial_guess.start,
                self.initial_guess.stop,
                self.initial_guess.step_count() as u32 - 1,
                self.target,
            )
            .into_par_iter()
            .filter_map(|model| {
                let init_guess = (model.alpha_x(), model.alpha_y());
                log::debug!(
                    "Fitting Isotropic BRDF model: {:?} with αx = {} αy = {}",
                    model.kind(),
                    init_guess.0,
                    init_guess.1
                );
                let kind = model.kind();
                let problem = MicrofacetBrdfFittingProblemProxy::<{ Isotropy::Isotropic }>::new(
                    &self.measured,
                    model,
                    self.iors,
                    &self.iors_i,
                    &self.iors_t,
                );
                let (result, report) = solver.minimize(problem);
                log::debug!(
                    "Fitting Isotropic BRDF model: {:?} with αx = {} αy = {}\n    - fitted αx = \
                     {} αy = {}\n    - report: {:?}",
                    kind,
                    init_guess.0,
                    init_guess.1,
                    result.model.alpha_x(),
                    result.model.alpha_y(),
                    report
                );
                match report.termination {
                    TerminationReason::Converged { .. } | TerminationReason::LostPatience => {
                        Some((result.model as _, report))
                    }
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
        };
        results.shrink_to_fit();
        FittingReport::new(results, |m: &Box<dyn MicrofacetBasedBrdfModel>| {
            m.alpha_x() > 0.0 && m.alpha_y() > 0.0
        })
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
    /// Per snapshot maximum value of the modelled samples with the
    /// corresponding roughness. (alpha, max_modelled_per_snapshot)
    max_modelled: Vec<(f64, Box<[f64]>)>, /* Memory inefficient, but it's a temporary solution.
                                           * TODO: sharing the max_modelled between threads
                                           * for one fitting problem there is no need to
                                           * store the maximum modelled values for each
                                           * roughness. */
    /// The actual partition (not the params) of the measured BSDF data.
    partition: SphericalPartition,
    /// The refractive index registry.
    iors: &'a RefractiveIndexRegistry,
    /// The refractive indices of the incident medium at the measured
    /// wavelengths.
    iors_i: &'a [RefractiveIndexRecord],
    /// The refractive indices of the transmitted medium at the measured
    /// wavelengths.
    iors_t: &'a [RefractiveIndexRecord],
    /// Incident directions
    wis: Box<[Vec3]>,
    /// Outgoing directions
    wos: Box<[Vec3]>,
}

impl<'a> MicrofacetBrdfFittingProblemProxy<'a, { Isotropy::Isotropic }> {
    pub fn new(
        measured: &'a MeasuredBsdfData,
        model: Box<dyn MicrofacetBasedBrdfFittingModel>,
        iors: &'a RefractiveIndexRegistry,
        iors_i: &'a [RefractiveIndexRecord],
        iors_t: &'a [RefractiveIndexRecord],
    ) -> Self {
        let partition = measured.params.receiver.partitioning();
        let max_measured = measured
            .snapshots
            .iter()
            .map(|snapshot| {
                snapshot
                    .samples
                    .iter()
                    .fold(0.0f64, |m, s| m.max(s[0] as f64))
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
        let mut problem = Self {
            measured,
            max_measured,
            model,
            max_modelled: vec![],
            partition,
            iors,
            iors_i,
            iors_t,
            wis,
            wos,
        };
        // Set the initial parameters of the model to trigger the calculation of
        // the maximum modelled values.
        let ax = problem.model.alpha_x();
        problem.set_params(&Vector::<f64, U1, Owned<f64, U1, U1>>::new(ax));
        problem
    }
}

fn eval_residuals<const I: Isotropy>(problem: &MicrofacetBrdfFittingProblemProxy<I>) -> Box<[f64]> {
    // The number of residuals is the number of patches times the number of
    // snapshots.
    let n_patches = problem.partition.num_patches();
    let residuals_count = n_patches * problem.measured.snapshots.len();
    let mut rs = Box::new_uninit_slice(residuals_count);
    let max_modelled = problem
        .max_modelled
        .iter()
        .find(|(alpha, _)| *alpha == problem.model.alpha_x())
        .unwrap()
        .1
        .as_ref();
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
            let max_modelled = max_modelled[i];
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

impl<'a> LeastSquaresProblem<f64, Dyn, U1>
    for MicrofacetBrdfFittingProblemProxy<'a, { Isotropy::Isotropic }>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, params: &Vector<f64, U1, Self::ParameterStorage>) {
        self.model.set_alpha_x(params[0]);
        self.model.set_alpha_y(params[0]);
        if self
            .max_modelled
            .iter()
            .find(|(alpha, _)| *alpha == self.model.alpha_x())
            .is_none()
        {
            let (_, maxes) = generate_analytical_brdf(
                &self.measured.params,
                self.model.as_ref(),
                self.iors,
                true,
            );
            // currently only the first wavelength is used
            let first_wavelength_maxes = maxes
                .iter()
                .map(|s| s[0] as f64)
                .collect::<Vec<_>>()
                .into_boxed_slice();
            self.max_modelled
                .push((self.model.alpha_x(), first_wavelength_maxes));
        }
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
