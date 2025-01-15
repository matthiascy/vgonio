//! Fitting of microfacet distribution functions and bidirectional
//! scattering distribution functions.

/// Common method when implementing [`LeastSquaresProblem`] for a fitting
macro_rules! impl_least_squares_problem_common_methods {
    (@aniso => $self:ident, $params_ty:ty) => {
        fn set_params(&mut $self, params: &$params_ty) {
            $self.model.set_params(params.as_ref());
        }

        fn params(&$self) -> $params_ty {
            let [x, y] = $self.model.params();
            <$params_ty>::new(x, y)
        }
    };
    (@iso2 => $self:ident, $params_ty:ty) => {
        fn set_params(&mut $self, params: &$params_ty) {
            $self.model.set_params(&[params[0], params[0]]);
        }

        fn params(&$self) -> $params_ty {
            let [x, _] = $self.model.params();
            <$params_ty>::new(x)
        }
    }
}

mod mfd;

use base::bxdf::fitting::{FittedModel, FittingProblemKind};
pub use mfd::*;

use base::Symmetry;
use std::fmt::Debug;

/// A collection of fitted models without repetition.
#[derive(Debug, Clone, Default)]
pub struct FittedModels(Vec<FittedModel>);

impl FittedModels {
    /// Checks if the collection already contains a model with the same kind and
    /// symmetry.
    pub fn contains(
        &self,
        kind: &FittingProblemKind,
        scale: Option<f32>,
        symmetry: Symmetry,
    ) -> bool {
        self.0
            .iter()
            .any(|f| f.kind() == *kind && f.scale() == scale && f.symmetry() == symmetry)
    }

    /// Push a new model to the collection.
    pub fn push(&mut self, model: FittedModel) { self.0.push(model); }
}

impl AsRef<[FittedModel]> for FittedModels {
    fn as_ref(&self) -> &[FittedModel] { self.0.as_ref() }
}
