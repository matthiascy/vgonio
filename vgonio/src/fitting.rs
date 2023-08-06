use crate::measure::microfacet::MeasuredMadfData;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{Dyn, Matrix, Owned, VecStorage, Vector1, Vector3, U1};
use vgcore::math::{Cartesian, Handedness, SphericalCoord, Vec3};

/// Fitting procedure trying to find the width parameter for the microfacet area
/// (normal) distribution function (NDF) of the Trowbridge-Reitz (GGX) model.
///
/// Use Levenberg-Marquardt to fit this parameter to measured data.
/// The problem has 1 parameter and 1 residual.
struct MadfFittingTrowbridgeReitz<'a> {
    /// Measured data to fit to.
    measured: &'a MeasuredMadfData,
    /// The normal vector of the microfacet.
    normal: Vec3,
    /// Width parameter of the NDF.
    width: f32,
}

impl<'a> MadfFittingTrowbridgeReitz<'a> {
    /// Creates a new fitting problem.
    ///
    /// # Arguments
    ///
    /// * `measured` - The measured data to fit to.
    /// * `n` - The normal vector of the microfacet.
    pub fn new(measured: &'a MeasuredMadfData, n: Vec3) -> Self {
        Self {
            measured,
            normal: n,
            width: 0.1,
        }
    }

    /// Evaluates the function at the given point with the current parameters.
    fn eval(&self, m: Vec3) -> f32 {
        let alpha2 = self.width * self.width;
        let cos_theta_m = m.dot(self.normal);
        let cos_theta_m2 = cos_theta_m * cos_theta_m;
        let tan_theta_m2 = 1.0 / cos_theta_m2 - cos_theta_m2;
        let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
        alpha2 / (std::f32::consts::PI * cos_theta_m4 * (alpha2 + tan_theta_m2).powi(2))
    }
}

impl<'a> LeastSquaresProblem<f32, Dyn, U1> for MadfFittingTrowbridgeReitz<'a> {
    type ResidualStorage = VecStorage<f32, Dyn, U1>;
    type JacobianStorage = Owned<f32, Dyn, U1>;
    type ParameterStorage = Owned<f32, U1, U1>;

    fn set_params(&mut self, x: &Vector1<f32>) { self.width = x[0]; }

    fn params(&self) -> Vector1<f32> { Vector1::new(self.width) }

    fn residuals(&self) -> Option<Matrix<f32, Dyn, U1, Self::ResidualStorage>> {
        let theta_step_count = self.measured.params.zenith.step_count_wrapped();
        let residuals = self
            .measured
            .samples
            .iter()
            .enumerate()
            .map(|(idx, meas)| {
                let phi_idx = idx / theta_step_count;
                let theta_idx = idx % theta_step_count;
                let theta = self.measured.params.zenith.step(theta_idx);
                let phi = self.measured.params.azimuth.step(phi_idx);
                let m =
                    SphericalCoord::new(1.0, theta, phi).to_cartesian(Handedness::RightHandedYUp);
                self.eval(m)
            })
            .collect();
        Some(Matrix::<f32, Dyn, U1, Self::ResidualStorage>::from_vec(
            residuals,
        ))
    }

    fn jacobian(&self) -> Option<Matrix<f32, Dyn, U1, Self::JacobianStorage>> {
        let alpha = self.width;
        let alpha2 = alpha * alpha;
        let derivatives = self
            .measured
            .samples
            .iter()
            .enumerate()
            .map(|(idx, meas)| {
                let phi_idx = idx / self.measured.params.zenith.step_count_wrapped();
                let theta_idx = idx % self.measured.params.zenith.step_count_wrapped();
                let theta = self.measured.params.zenith.step(theta_idx);
                let phi = self.measured.params.azimuth.step(phi_idx);
                let m =
                    SphericalCoord::new(1.0, theta, phi).to_cartesian(Handedness::RightHandedYUp);

                let cos_theta_m = m.dot(self.normal);
                let cos_theta_m2 = cos_theta_m * cos_theta_m;
                let tan_theta_m2 = 1.0 / cos_theta_m2 - cos_theta_m2;
                let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;

                let upper = 2.0 * alpha * (tan_theta_m2 - alpha2);
                let lower = std::f32::consts::PI * cos_theta_m4 * (alpha2 + tan_theta_m2).powi(3);
                upper / lower
            })
            .collect();
        Some(Matrix::<f32, Dyn, U1, Self::JacobianStorage>::from_vec(
            derivatives,
        ))
    }
}
