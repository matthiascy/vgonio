use crate::acq::Medium;
use crate::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum BxdfKind {
    InPlane,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AngleUnit {
    #[serde(rename = "deg")]
    Degrees,

    #[serde(rename = "rad")]
    Radians,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LengthUnit {
    #[serde(rename = "m")]
    Meters,

    #[serde(rename = "mm")]
    Millimetres,

    #[serde(rename = "um")]
    Micrometres,

    #[serde(rename = "nm")]
    Nanometres,

    #[serde(rename = "pm")]
    Picometres,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(into = "[f32; 3]", from = "[f32; 3]")]
pub struct Spectrum {
    /// Initial wavelength of the spectrum.
    pub start: f32,

    /// Final wavelength of the spectrum.
    pub stop: f32,

    /// Increment between wavelength samples.
    pub step: f32,
}

impl From<[f32; 3]> for Spectrum {
    fn from(vals: [f32; 3]) -> Self {
        Self {
            start: vals[0],
            stop: vals[1],
            step: vals[2],
        }
    }
}

impl From<Spectrum> for [f32; 3] {
    fn from(spec: Spectrum) -> Self {
        [spec.start, spec.stop, spec.step]
    }
}

/// Description of the BRDF collector.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Collector {
    pub radius: f32,
    pub shape: CollectorShape,
    pub partition: CollectorPartition,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectorShape {
    /// Only capture the upper part of the sphere.
    UpperHemisphere,

    /// Only capture the lower part of the sphere.
    LowerHemisphere,

    /// Capture the whole sphere.
    WholeSphere,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectorPartition {
    /// The collector is partitioned into a number of regions with the same
    /// angular interval.
    EqualAngle(f32),

    /// The collector is partitioned into a number of regions with the same
    /// area (solid angle).
    EqualArea(f32),

    /// The collector is partitioned into a number of regions with the same
    /// projected area (projected solid angle).
    EqualProjectedArea(f32),
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum Position {
    /// The position defined by spherical coordinates: radius (r), zenith (θ)
    /// and azimuth (φ).
    Spherical(f32, f32, f32),

    /// The position defined by cartesian coordinates: x, y and z.
    Cartesian(f32, f32, f32),
}

/// Description of the light source.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LightSource {
    /// The light source's position in either spherical coordinates or cartesian
    /// coordinates.
    pub position: Position,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum MeasurementKind {
    Bxdf { kind: BxdfKind },
    Ndf,
}

#[derive(Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeasurementDesc {
    pub angle_unit: AngleUnit,
    pub length_unit: LengthUnit,
    pub measurement_kind: MeasurementKind,
    pub rays_count: u32,
    pub max_bounces: u32,
    pub incident_medium: Medium,
    pub transmitted_medium: Medium,
    pub surfaces: Vec<PathBuf>,
    pub spectrum: Spectrum,
    pub collector: Collector,
    pub light_source: LightSource,
}

impl MeasurementDesc {
    pub fn load_from_file(filepath: &Path) -> Result<MeasurementDesc, Error> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        serde_yaml::from_reader(reader).map_err(Error::from)
    }
}

#[test]
fn scene_desc_serialization() {
    use std::io::Write;
    let desc_0 = MeasurementDesc {
        angle_unit: AngleUnit::Degrees,
        length_unit: LengthUnit::Meters,
        measurement_kind: MeasurementKind::Bxdf {
            kind: BxdfKind::InPlane,
        },
        rays_count: 100,
        max_bounces: 10,
        incident_medium: Medium::Air,
        transmitted_medium: Medium::Air,
        surfaces: vec![PathBuf::from("/tmp/scene.obj")],
        spectrum: Spectrum {
            start: 380.0,
            stop: 780.0,
            step: 10.0,
        },
        collector: Collector {
            radius: 0.1,
            shape: CollectorShape::WholeSphere,
            partition: CollectorPartition::EqualArea(0.1),
        },
        light_source: LightSource {
            position: Position::Spherical(1.0, 0.0, 0.0),
        },
    };
    let desc_1 = desc_0.clone();

    let serialized_0 = serde_yaml::to_string(&desc_0).unwrap();
    let serialized_1 = serde_yaml::to_string(&desc_1).unwrap();

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open("./scene_desc.yml")
        .unwrap();

    file.write_all(serialized_0.as_bytes()).unwrap();
    file.write_all(serialized_0.as_bytes()).unwrap();

    let deserialized_0: MeasurementDesc = serde_yaml::from_str(&serialized_0).unwrap();
    let deserialized_1: MeasurementDesc = serde_yaml::from_str(&serialized_1).unwrap();

    assert_eq!(desc_0, deserialized_0);
    assert_eq!(desc_1, deserialized_1);
}
