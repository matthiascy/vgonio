use crate::{
    app::cache::{RawCache, RefractiveIndexRegistry},
    fitting::{
        brdf::initialise_microfacet_bsdf_models, err::generate_analytical_brdf_from_sampled_brdf,
        FittingProblem, FittingReport,
    },
    measure::data::SampledBrdf,
};
use base::{medium::Medium, optics::ior::Ior, range::RangeByStepSizeInclusive, Isotropy};
use bxdf::{brdf::Bxdf, distro::MicrofacetDistroKind, Scattering};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::{Dyn, Matrix, OMatrix, Owned, VecStorage, Vector, U1, U2};
use std::fmt::Display;

pub struct SampledBrdfFittingProblem<'a> {
    pub measured: &'a SampledBrdf,
    pub target: MicrofacetDistroKind,
    pub iors_i: Box<[Ior]>,
    pub iors_t: Box<[Ior]>,
    pub iors: &'a RefractiveIndexRegistry,
    pub initial_guess: RangeByStepSizeInclusive<f64>,
}

impl<'a> Display for SampledBrdfFittingProblem<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BSDF Fitting Problem (sampled brdf)")
            .field("target", &self.target)
            .finish()
    }
}

impl<'a> SampledBrdfFittingProblem<'a> {
    pub fn new(
        measured: &'a SampledBrdf,
        target: MicrofacetDistroKind,
        initial: RangeByStepSizeInclusive<f64>,
        cache: &'a RawCache,
    ) -> Self {
        let iors_i = cache
            .iors
            .ior_of_spectrum(Medium::Air, &measured.spectrum)
            .unwrap();
        let iors_t = cache
            .iors
            .ior_of_spectrum(Medium::Aluminium, &measured.spectrum)
            .unwrap();
        Self {
            measured,
            target,
            iors_i,
            iors_t,
            iors: &cache.iors,
            initial_guess: initial,
        }
    }
}

impl<'a> FittingProblem for SampledBrdfFittingProblem<'a> {
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
                            SampledBrdfFittingProblemProxy::<{ Isotropy::Isotropic }>::new(
                                &self.measured,
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
                            SampledBrdfFittingProblemProxy::<{ Isotropy::Anisotropic }>::new(
                                &self.measured,
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
        FittingReport::new(results, |m: &Box<dyn Bxdf<Params = [f64; 2]>>| {
            m.params()[0] > 0.0 && m.params()[1] > 0.0
        })
    }
}

struct SampledBrdfFittingProblemProxy<'a, const I: Isotropy> {
    measured: &'a SampledBrdf,
    model: Box<dyn Bxdf<Params = [f64; 2]>>,
    iors: &'a RefractiveIndexRegistry,
    iors_i: &'a [Ior],
    iors_t: &'a [Ior],
}

impl<'a> SampledBrdfFittingProblemProxy<'a, { Isotropy::Isotropic }> {
    pub fn new(
        measured: &'a SampledBrdf,
        model: Box<dyn Bxdf<Params = [f64; 2]>>,
        iors: &'a RefractiveIndexRegistry,
        iors_i: &'a [Ior],
        iors_t: &'a [Ior],
    ) -> Self {
        let mut problem = Self {
            measured,
            model,
            iors,
            iors_i,
            iors_t,
        };
        // Set the initial parameters of the model to trigger the calculation of
        // the maximum modelled values.
        let [ax, _] = problem.model.params();
        problem.set_params(&Vector::<f64, U1, Owned<f64, U1, U1>>::new(ax));
        problem
    }
}

impl<'a> SampledBrdfFittingProblemProxy<'a, { Isotropy::Anisotropic }> {
    pub fn new(
        measured: &'a SampledBrdf,
        model: Box<dyn Bxdf<Params = [f64; 2]>>,
        iors: &'a RefractiveIndexRegistry,
        iors_i: &'a [Ior],
        iors_t: &'a [Ior],
    ) -> Self {
        let mut problem = Self {
            measured,
            model,
            iors,
            iors_i,
            iors_t,
        };
        let [ax, ay] = problem.model.params();
        problem.set_params(&Vector::<f64, U2, Owned<f64, U2, U1>>::new(ax, ay));
        problem
    }
}

fn eval_sampled_brdf_residuals<const I: Isotropy>(
    problem: &SampledBrdfFittingProblemProxy<I>,
) -> Box<[f64]> {
    let n_lambda = problem.measured.n_lambda();
    let n_wo = problem.measured.n_wo();
    problem
        .measured
        .wi_wo_pairs
        .iter()
        .enumerate()
        .flat_map(|(i, (wi, wos))| {
            let wi = wi.to_cartesian();
            wos.iter().enumerate().flat_map(move |(j, wo)| {
                let wo = wo.to_cartesian();
                // Reuse the memory of the modelled samples to store the residuals.
                let mut modelled = Scattering::eval_reflectance_spectrum(
                    problem.model.as_ref(),
                    &wi,
                    &wo,
                    &problem.iors_i,
                    &problem.iors_t,
                )
                .into_vec();
                let offset = i * n_wo * n_lambda + j * n_lambda;
                let measured = &problem.measured.samples[offset..offset + n_lambda];
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

impl<'a> LeastSquaresProblem<f64, Dyn, U1>
    for SampledBrdfFittingProblemProxy<'a, { Isotropy::Isotropic }>
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
            &eval_sampled_brdf_residuals(self),
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        let pds = self
            .measured
            .wios()
            .iter()
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

impl<'a> LeastSquaresProblem<f64, Dyn, U2>
    for SampledBrdfFittingProblemProxy<'a, { Isotropy::Anisotropic }>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U2>;
    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, params: &Vector<f64, U2, Self::ParameterStorage>) {
        self.model.set_params(&[params[0], params[1]]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::from_row_slice(&self.model.params())
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        Some(OMatrix::<f64, Dyn, U1>::from_row_slice(
            &eval_sampled_brdf_residuals(self),
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        let pds = self
            .measured
            .wios()
            .iter()
            .flat_map(|(wi_sph, wo_sph)| {
                let wi = wi_sph.to_cartesian();
                let wo = wo_sph.to_cartesian();
                self.iors_i
                    .iter()
                    .zip(self.iors_t)
                    .flat_map(move |(ior_i, ior_t)| self.model.pd(&wi, &wo, ior_i, ior_t))
            })
            .collect::<Vec<_>>();
        Some(OMatrix::<f64, Dyn, U2>::from_row_slice(&pds))
    }
}
