use crate::{fitting::FittingReport, measure::microfacet::MeasuredMmsfData};
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, MinimizationReport};
use nalgebra::{Dyn, Matrix, Owned, VecStorage, Vector, U1};
use vgcore::math::{Handedness, SphericalCoord, Vec3};

use super::{
    FittingProblem, MicrofacetMaskingShadowingModel, MicrofacetModelFamily, ReflectionModelFamily,
};

pub struct MmsfFittingProblem<'a> {
    inner: InnerMmsfFittingProblem<'a>,
}

impl<'a> MmsfFittingProblem<'a> {
    pub fn new<M: MicrofacetMaskingShadowingModel + 'static>(
        measured: &'a MeasuredMmsfData,
        model: M,
        normal: Vec3,
    ) -> Self {
        Self {
            inner: InnerMmsfFittingProblem {
                measured,
                normal,
                model: Box::new(model),
            },
        }
    }
}

struct InnerMmsfFittingProblem<'a> {
    measured: &'a MeasuredMmsfData,
    normal: Vec3,
    model: Box<dyn MicrofacetMaskingShadowingModel>,
}

impl<'a> LeastSquaresProblem<f64, Dyn, U1> for InnerMmsfFittingProblem<'a> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U1>;
    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector<f64, U1, Self::ParameterStorage>) {
        self.model.set_param(x[0]);
    }

    fn params(&self) -> Vector<f64, U1, Self::ParameterStorage> {
        Vector::<f64, U1, Self::ParameterStorage>::new(self.model.param())
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        let phi_step_count = self.measured.params.azimuth.step_count_wrapped();
        let theta_step_count = self.measured.params.zenith.step_count_wrapped();
        // Only the first phi_step_count * theta_step_count samples are used
        let residuals = self
            .measured
            .samples
            .iter()
            .take(phi_step_count * theta_step_count)
            .enumerate()
            .map(|(idx, meas)| {
                let phi_v_idx = idx / theta_step_count;
                let theta_v_idx = idx % theta_step_count;
                let theta_v = self.measured.params.zenith.step(theta_v_idx);
                let cos_theta_v = theta_v.cos() as f64;
                self.model.eval_with_cos_theta_v(cos_theta_v) - *meas as f64
            })
            .collect();
        Some(Matrix::<f64, Dyn, U1, Self::ResidualStorage>::from_vec(
            residuals,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        let alpha = self.params().x;
        let alpha2 = alpha * alpha;
        let phi_step_count = self.measured.params.azimuth.step_count_wrapped();
        let theta_step_count = self.measured.params.zenith.step_count_wrapped();
        let derivatives = match self.model.family() {
            ReflectionModelFamily::Microfacet(m) => match m {
                MicrofacetModelFamily::TrowbridgeReitz => self
                    .measured
                    .samples
                    .iter()
                    .take(phi_step_count * theta_step_count)
                    .enumerate()
                    .map(|(idx, meas)| {
                        let phi_v_idx = idx / theta_step_count;
                        let theta_v_idx = idx % theta_step_count;
                        let theta_v = self.measured.params.zenith.step(theta_v_idx);
                        let phi_v = self.measured.params.azimuth.step(phi_v_idx);
                        let v = SphericalCoord::new(1.0, theta_v, phi_v)
                            .to_cartesian(Handedness::RightHandedYUp);
                        let cos_theta_v = v.dot(self.normal) as f64;
                        let cos_theta_v2 = cos_theta_v * cos_theta_v;
                        let tan_theta_v2 = if cos_theta_v2 == 0.0 {
                            f64::INFINITY
                        } else {
                            1.0 / cos_theta_v2 - cos_theta_v2
                        };
                        let a = (alpha2 * tan_theta_v2 + 1.0).sqrt();
                        let numerator = 2.0 * alpha * tan_theta_v2;
                        let denominator = a * a * a + 2.0 * a * a + a;
                        numerator / denominator
                    })
                    .collect(),
                MicrofacetModelFamily::BeckmannSpizzichino => self
                    .measured
                    .samples
                    .iter()
                    .take(phi_step_count * theta_step_count)
                    .enumerate()
                    .map(|(idx, meas)| {
                        let phi_v_idx = idx / theta_step_count;
                        let theta_v_idx = idx % theta_step_count;
                        let theta_v = self.measured.params.zenith.step(theta_v_idx);
                        let phi_v = self.measured.params.azimuth.step(phi_v_idx);
                        let v = SphericalCoord::new(1.0, theta_v, phi_v)
                            .to_cartesian(Handedness::RightHandedYUp);
                        let cos_theta_v = v.dot(self.normal) as f64;
                        let cos_theta_v2 = cos_theta_v * cos_theta_v;
                        let tan_theta_v = if cos_theta_v2 == 0.0 {
                            f64::INFINITY
                        } else {
                            (1.0 - cos_theta_v2).sqrt() / cos_theta_v
                        };

                        let cot_theta_v = 1.0 / tan_theta_v;
                        let a = cot_theta_v / alpha;
                        let e = (-a * a).exp();
                        let erf = libm::erf(a);
                        let sqrt_pi = std::f64::consts::PI.sqrt();
                        let b = 1.0 + erf + e * alpha * tan_theta_v / sqrt_pi;
                        let numerator = 2.0 * e * tan_theta_v;
                        let denominator = sqrt_pi * b * b;
                        numerator / denominator
                    })
                    .collect(),
            },
        };
        Some(Matrix::<f64, Dyn, U1, Self::JacobianStorage>::from_vec(
            derivatives,
        ))
    }
}

impl<'a> FittingProblem for MmsfFittingProblem<'a> {
    type Model = Box<dyn MicrofacetMaskingShadowingModel>;

    fn lsq_lm_fit(self) -> FittingReport<Self::Model> {
        let solver = LevenbergMarquardt::new();
        let (result, report) = solver.minimize(self.inner);
        FittingReport {
            best: 0,
            reports: vec![(result.model, report)],
        }
    }
}
