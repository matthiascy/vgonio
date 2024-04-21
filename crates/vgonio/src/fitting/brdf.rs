pub mod sampled;

use crate::{
    app::cache::{RawCache, RefractiveIndexRegistry},
    fitting::{err::generate_analytical_brdf, FittingProblem, FittingReport},
    measure::bsdf::MeasuredBsdfData,
};
use base::{
    math::Vec3, optics::ior::Ior, partition::SphericalPartition, range::RangeByStepSizeInclusive,
    Isotropy,
};
use bxdf::{
    brdf::{
        microfacet::{BeckmannBrdf, MicrofacetBrdf, TrowbridgeReitzBrdf, TrowbridgeReitzBrdfModel},
        Bxdf,
    },
    distro::{MicrofacetDistroKind, TrowbridgeReitzDistribution},
    Scattering,
};
use jabr::optics::reflect;
use levenberg_marquardt::{
    LeastSquaresProblem, LevenbergMarquardt, MinimizationReport, TerminationReason,
};
use log::log;
use nalgebra::{Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, U1, U2};
use rayon::{
    iter::{IndexedParallelIterator, ParallelBridge},
    slice::ParallelSlice,
};
use std::{borrow::Cow, fmt::Display};

/// The fitting problem for the microfacet based BSDF model.
///
/// The fitting procedure is based on the Levenberg-Marquardt algorithm.
pub struct MicrofacetBrdfFittingProblem<'a> {
    /// The measured BSDF data.
    pub measured: Cow<'a, MeasuredBsdfData>,
    /// The target BSDF model.
    pub target: MicrofacetDistroKind,
    /// Whether to normalise the measured data.
    pub normalise: bool,
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
        target: MicrofacetDistroKind,
        initial: RangeByStepSizeInclusive<f64>,
        normalise: bool,
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
            normalise,
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
impl<'a> FittingProblem for MicrofacetBrdfFittingProblem<'a> {
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
                        let problem =
                            MicrofacetBrdfFittingProblemProxy::<{ Isotropy::Isotropic }>::new(
                                &self.measured,
                                model,
                                self.normalise,
                                self.iors,
                                &self.iors_i,
                                &self.iors_t,
                            );
                        let (result, report) = solver.minimize(problem);
                        (result.model, report)
                    }
                    Isotropy::Anisotropic => {
                        let problem =
                            MicrofacetBrdfFittingProblemProxy::<{ Isotropy::Anisotropic }>::new(
                                &self.measured,
                                model,
                                self.normalise,
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
        FittingReport::new(results, |m: &Box<dyn Bxdf<Params = [f64; 2]>>| {
            m.params()[0] > 0.0 && m.params()[1] > 0.0
        })
    }
}

struct MicrofacetBrdfFittingProblemProxy<'a, const I: Isotropy> {
    /// The measured BSDF data.
    measured: &'a MeasuredBsdfData,
    /// Whether to normalise the measured data.
    normalise: bool,
    /// Maximum value of the measured samples for each snapshot (incident
    /// direction) and wavelength. The first layer is the snapshot index, and
    /// the second layer is the wavelength index.
    max_measured: Option<Box<[f64]>>,
    /// The target BSDF model.
    model: Box<dyn Bxdf<Params = [f64; 2]>>,
    /// Per snapshot maximum value of the modelled samples with the
    /// corresponding roughness. (alpha, max_modelled_per_snapshot)
    max_modelled: Option<Vec<((f64, f64), Box<[f64]>)>>, /* Memory inefficient, but it's a
                                                          * temporary
                                                          * solution.
                                                          * TODO: sharing the max_modelled
                                                          * between
                                                          * threads
                                                          * for one fitting problem there
                                                          * is no
                                                          * need to
                                                          * store the maximum modelled
                                                          * values
                                                          * for each
                                                          * roughness. */
    n_wavelengths: usize,
    /// The actual partition (not the params) of the measured BSDF data.
    partition: SphericalPartition,
    /// The refractive index registry.
    iors: &'a RefractiveIndexRegistry,
    /// The refractive indices of the incident medium at the measured
    /// wavelengths.
    iors_i: &'a [Ior],
    /// The refractive indices of the transmitted medium at the measured
    /// wavelengths.
    iors_t: &'a [Ior],
    /// Incident directions
    wis: Box<[Vec3]>,
    /// Outgoing directions
    wos: Box<[Vec3]>,
}

fn new_microfacet_brdf_fitting_problem_proxy_common(
    measured: &MeasuredBsdfData,
    n_wavelengths: usize,
    normalise: bool,
) -> (
    Option<Box<[f64]>>,
    Box<[Vec3]>,
    Box<[Vec3]>,
    SphericalPartition,
) {
    let partition = measured.params.receiver.partitioning();
    use rayon::iter::ParallelIterator;
    let max_measured = normalise.then(|| {
        let mut max = vec![-1.0; measured.snapshots.len() * n_wavelengths].into_boxed_slice();
        measured
            .snapshots
            .iter()
            .zip(max.chunks_mut(n_wavelengths))
            .par_bridge()
            .for_each(|(snapshot, max)| {
                for spectral_samples in snapshot.samples.iter() {
                    for (j, s) in spectral_samples.iter().enumerate() {
                        max[j] = f64::max(max[j], *s as f64);
                    }
                }
            });
        max
    });

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
    (max_measured, wis, wos, partition)
}

impl<'a> MicrofacetBrdfFittingProblemProxy<'a, { Isotropy::Isotropic }> {
    pub fn new(
        measured: &'a MeasuredBsdfData,
        model: Box<dyn Bxdf<Params = [f64; 2]>>,
        normalise: bool,
        iors: &'a RefractiveIndexRegistry,
        iors_i: &'a [Ior],
        iors_t: &'a [Ior],
    ) -> Self {
        let (max_measured, wis, wos, partition) =
            new_microfacet_brdf_fitting_problem_proxy_common(measured, iors_i.len(), normalise);
        let max_modelled = normalise.then_some(vec![]);
        // Set the initial parameters of the model to trigger the calculation of
        // the maximum modelled values.
        let mut problem = Self {
            measured,
            normalise,
            max_measured,
            model,
            max_modelled,
            n_wavelengths: iors_i.len(),
            partition,
            iors,
            iors_i,
            iors_t,
            wis,
            wos,
        };
        let [ax, _] = problem.model.params();
        problem.set_params(&Vector::<f64, U1, Owned<f64, U1, U1>>::new(ax));
        problem
    }
}

impl<'a> MicrofacetBrdfFittingProblemProxy<'a, { Isotropy::Anisotropic }> {
    pub fn new(
        measured: &'a MeasuredBsdfData,
        model: Box<dyn Bxdf<Params = [f64; 2]>>,
        normalise: bool,
        iors: &'a RefractiveIndexRegistry,
        iors_i: &'a [Ior],
        iors_t: &'a [Ior],
    ) -> Self {
        let (max_measured, wis, wos, partition) =
            new_microfacet_brdf_fitting_problem_proxy_common(measured, iors_i.len(), normalise);
        let max_modelled = normalise.then_some(vec![]);
        let mut problem = Self {
            measured,
            normalise,
            max_measured,
            model,
            max_modelled,
            n_wavelengths: iors_i.len(),
            partition,
            iors,
            iors_i,
            iors_t,
            wis,
            wos,
        };
        let [ax, ay] = problem.model.params();
        problem.set_params(&Vector::<f64, U2, Owned<f64, U2, U1>>::new(ax, ay));
        problem
    }
}

fn eval_residuals<const I: Isotropy>(problem: &MicrofacetBrdfFittingProblemProxy<I>) -> Box<[f64]> {
    // The number of residuals is the number of patches times the number of
    // snapshots.
    let n_patches = problem.partition.num_patches();
    let residuals_count = problem.measured.snapshots.len() * n_patches * problem.n_wavelengths;
    let mut rs = Box::new_uninit_slice(residuals_count); // Row-major [snapshot, patch, wavelength]
    let max_modelled_values = problem.max_modelled.as_ref().map(|maxes| {
        maxes
            .iter()
            .find(|((ax, _), _)| *ax == problem.model.params()[0])
            .unwrap()
            .1
            .as_ref()
    });
    let max_measured_values = problem.max_measured.as_ref().map(|m| m.as_ref());
    let n_wavelengths = problem.n_wavelengths;
    problem
        .measured
        .snapshots
        .iter()
        .enumerate()
        .for_each(|(i, snapshot)| {
            let wi = snapshot.wi.to_cartesian();
            let max_values_range = i * n_wavelengths..(i + 1) * n_wavelengths;
            let max_measured = max_measured_values.map(|m| &m[max_values_range.clone()]);
            let max_modelled = max_modelled_values.map(|m| &m[max_values_range.clone()]);
            problem
                .partition
                .patches
                .iter()
                .enumerate()
                .for_each(|(j, patch)| {
                    let wo = patch.center().to_cartesian();
                    let modelled_values = Scattering::eval_reflectance_spectrum(
                        problem.model.as_ref(),
                        &wi,
                        &wo,
                        &problem.iors_i,
                        &problem.iors_t,
                    );
                    let measured_values = &snapshot.samples[j];
                    match (max_measured, max_modelled) {
                        (Some(max_measured), Some(max_modelled)) => {
                            for k in 0..n_wavelengths {
                                let max_measured_value = if max_measured[k] == 0.0 {
                                    1.0
                                } else {
                                    max_measured[k]
                                };
                                let max_modelled_value = if max_modelled[k] == 0.0 {
                                    1.0
                                } else {
                                    max_modelled[k]
                                };
                                let measured = measured_values[k] as f64 / max_measured_value;
                                let modelled = modelled_values[k] / max_modelled_value;
                                rs[i * n_patches * n_wavelengths + j * n_wavelengths + k]
                                    .write(modelled - measured);
                            }
                        }
                        (None, None) => {
                            for k in 0..n_wavelengths {
                                let measured = measured_values[k] as f64;
                                let modelled = modelled_values[k];
                                rs[i * n_patches * n_wavelengths + j * n_wavelengths + k]
                                    .write(modelled - measured);
                            }
                        }
                        _ => {
                            unreachable!()
                        }
                    }
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
        self.model.set_params(&[params[0], params[0]]);
        let alpha = self.model.params()[0];
        if self.normalise {
            if self
                .max_modelled
                .as_mut()
                .unwrap()
                .iter_mut()
                .find(|((ax, _), _)| *ax == alpha)
                .is_none()
            {
                // TODO: we are repeating the calculation of the brdf, maybe we can
                // reuse not only the maximum values but also the brdf itself.
                let maxes = generate_analytical_brdf(
                    &self.measured.params,
                    self.model.as_ref(),
                    self.iors,
                    true,
                )
                .max_values
                .iter()
                .map(|m| *m as f64)
                .collect::<Vec<_>>()
                .into_boxed_slice();
                self.max_modelled
                    .as_mut()
                    .unwrap()
                    .push(((alpha, alpha), maxes));
            }
        }
    }

    fn params(&self) -> Vector<f64, U1, Self::ParameterStorage> {
        Vector::<f64, U1, Self::ParameterStorage>::new(self.model.params()[0])
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(&eval_residuals(
            self,
        )))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        let derivative_per_wavelength = self
            .iors_i
            .iter()
            .zip(self.iors_t.iter())
            .map(|(ior_i, ior_t)| self.model.pds_iso(&self.wis, &self.wos, ior_i, ior_t))
            .collect::<Vec<_>>();
        // Re-arrange the shape of the derivatives to match the residuals:
        // [snapshot, patch, wavelength]
        let mut jacobian =
            vec![
                0.0;
                self.measured.snapshots.len() * self.partition.num_patches() * self.n_wavelengths
            ]
            .into_boxed_slice();
        for (i, snapshot) in self.measured.snapshots.iter().enumerate() {
            for (j, patch) in self.partition.patches.iter().enumerate() {
                for k in 0..self.n_wavelengths {
                    let offset = i * self.partition.num_patches() * self.n_wavelengths
                        + j * self.n_wavelengths
                        + k;
                    jacobian[offset] =
                        derivative_per_wavelength[k][i * self.partition.num_patches() + j];
                }
            }
        }
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(&jacobian))
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U2>
    for MicrofacetBrdfFittingProblemProxy<'a, { Isotropy::Anisotropic }>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, params: &Vector<f64, U2, Self::ParameterStorage>) {
        self.model.set_params(&[params[0], params[1]]);
        let [alpha_x, alpha_y] = self.model.params();
        if self.normalise {
            if self
                .max_modelled
                .as_mut()
                .unwrap()
                .iter_mut()
                .find(|((ax, ay), _)| *ax == alpha_x && *ay == alpha_y)
                .is_none()
            {
                let maxes = generate_analytical_brdf(
                    &self.measured.params,
                    self.model.as_ref(),
                    self.iors,
                    true,
                )
                .max_values
                .iter()
                .map(|m| *m as f64)
                .collect::<Vec<_>>()
                .into_boxed_slice();
                self.max_modelled
                    .as_mut()
                    .unwrap()
                    .push(((alpha_x, alpha_y), maxes));
            }
        }
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::from(self.model.params())
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(&eval_residuals(
            self,
        )))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        let derivative_per_wavelength = self
            .iors_i
            .iter()
            .zip(self.iors_t.iter())
            .map(|(ior_i, ior_t)| self.model.pds(&self.wis, &self.wos, ior_i, ior_t))
            .collect::<Vec<_>>();
        // Re-arrange the shape of the derivatives to match the residuals:
        // [snapshot, patch, wavelength, alpha]
        //  n_snapshot, n_patch, n_wavelength, 2
        let mut jacobian = vec![
            0.0;
            self.measured.snapshots.len()
                * self.partition.num_patches()
                * self.n_wavelengths
                * 2
        ]
        .into_boxed_slice();
        for (i, snapshot) in self.measured.snapshots.iter().enumerate() {
            for (j, patch) in self.partition.patches.iter().enumerate() {
                for k in 0..self.n_wavelengths {
                    let offset = i * self.partition.num_patches() * self.n_wavelengths * 2
                        + j * self.n_wavelengths * 2
                        + k * 2;
                    jacobian[offset] =
                        derivative_per_wavelength[k][i * self.partition.num_patches() * 2 + j * 2];
                    jacobian[offset + 1] = derivative_per_wavelength[k]
                        [i * self.partition.num_patches() * 2 + j * 2 + 1];
                }
            }
        }
        Some(OMatrix::<f64, Dyn, U2>::from_row_slice(&jacobian))
    }
}
