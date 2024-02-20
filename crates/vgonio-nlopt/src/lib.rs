/// A representation of a non-linear function.
pub trait NonLinearFn {
    /// Number of parameters in the function.
    fn num_params(&self) -> usize;
    /// Parameters of the function.
    fn params(&self) -> Box<[f64]>;
    /// Set parameters of the function.
    fn set_params(&mut self, params: &[f64]);
    /// Evaluate the function at the given point.
    fn eval(&self, x: &[f64]) -> f64;
    /// Evaluate the derivatives of the function with respect to its parameters
    /// at the given point.
    fn eval_params_derivatives(&self, x: &[f64]) -> Box<[f64]>;
}
