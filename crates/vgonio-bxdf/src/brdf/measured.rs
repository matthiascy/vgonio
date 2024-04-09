use jabr::array::DyArr;

pub struct MeasuredBrdf {
    /// Sampled BRDF data with 4 dimensions: theta_i, phi_i, theta_o, phi_o.
    samples: DyArr<f64, 4>,
}
