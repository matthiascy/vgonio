//! App-side fitting helpers (GUI). The fitting *capability* lives in the
//! `vgn_fitting` crate; this module holds only GUI-facing accumulation types.

use vgn_bxdf::fitting::{FittedModel, FittingProblemKind};
use vgn_core::Symmetry;

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
