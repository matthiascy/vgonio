use crate::{
    brdf::Bxdf,
    distro::MicrofacetDistroKind,
    fitting::{
        brdf::{AnalyticalFit2, BrdfFittingProxy},
        FittingReport,
    },
};
use base::{
    optics::ior::{Ior, IorRegistry},
    units::Radians,
    ErrorMetric, Isotropy, Weighting,
};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{Dyn, Matrix, Owned, VecStorage, Vector, U1, U2};

pub fn brdf_fitting_nllsq_fit<Brdf: AnalyticalFit2>(
    measured: &Brdf,
    distro: MicrofacetDistroKind,
    metric: ErrorMetric,
    max_theta_i: Radians,
    max_theta_o: Radians,
    weighting: Weighting,
    iors: &IorRegistry,
) -> FittingReport<Box<dyn Bxdf<Params = [f64; 2]>>> {
    let _ = max_theta_o;
    todo!()
}
/// A proxy for the BRDF fitting problem using the NLLSQ algorithm.
pub struct NllsqBrdfFittingProxy<'a, Brdf, const I: Isotropy>
where
    Brdf: AnalyticalFit2,
{
    /// The proxy for the BRDF data.
    pub proxy: BrdfFittingProxy<'a, Brdf>,
    /// Cached IORs for the incident medium.
    iors_i: &'a [Ior],
    /// Cached IORs for the transmitted medium.
    iors_t: &'a [Ior],
    /// The target model being fitted to the measured data.
    model: Box<dyn Bxdf<Params = [f64; 2]>>,
}

impl<'a, Brdf> LeastSquaresProblem<f64, Dyn, U1>
    for NllsqBrdfFittingProxy<'a, Brdf, { Isotropy::Isotropic }>
where
    Brdf: AnalyticalFit2,
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;

    type JacobianStorage = Owned<f64, Dyn, U1>;

    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector<f64, U1, Self::ParameterStorage>) { todo!() }

    fn params(&self) -> Vector<f64, U1, Self::ParameterStorage> { todo!() }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> { todo!() }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> { todo!() }
}

impl<'a, Brdf> LeastSquaresProblem<f64, Dyn, U2>
    for NllsqBrdfFittingProxy<'a, Brdf, { Isotropy::Anisotropic }>
where
    Brdf: AnalyticalFit2,
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;

    type JacobianStorage = Owned<f64, Dyn, U2>;

    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, x: &Vector<f64, U2, Self::ParameterStorage>) { todo!() }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> { todo!() }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> { todo!() }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> { todo!() }
}
