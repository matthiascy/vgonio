use crate::acq::desc::MeasurementDesc;
use crate::acq::ior::RefractiveIndexDatabase;
use crate::htfld::Heightfield;

#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum BxdfKind {
    InPlane,
}

/// Measurement of the in-plane BRDF (incident angle and outgoing angle are on
/// the same plane).
pub fn measure_in_plane_brdf(
    _desc: &MeasurementDesc,
    _ior_db: &RefractiveIndexDatabase,
    _surfaces: &[Heightfield],
) -> Vec<f32> {
    // TODO: Implement
    vec![]
}
