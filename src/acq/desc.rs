use crate::acq::bxdf::BxdfKind;
use crate::acq::collector::CollectorDesc;
use crate::acq::Medium;
use crate::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

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
pub struct EmitterDesc {
    /// The light source's position in either spherical coordinates or cartesian
    /// coordinates.
    pub position: Position,
    pub spectrum: Spectrum,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum MeasurementKind {
    Bxdf { kind: BxdfKind },
    Ndf,
}

/// Description of the measurement.
///
/// Note: angle in the description file is always in degrees.
#[derive(Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeasurementDesc {
    pub length_unit: LengthUnit,
    pub measurement_kind: MeasurementKind,
    pub rays_count: u32,
    pub max_bounces: u32,
    pub incident_medium: Medium,
    pub transmitted_medium: Medium,
    pub surfaces: Vec<PathBuf>,
    pub emitter: EmitterDesc,
    pub collector: CollectorDesc,
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
        length_unit: LengthUnit::Meters,
        measurement_kind: MeasurementKind::Bxdf {
            kind: BxdfKind::InPlane,
        },
        rays_count: 100,
        max_bounces: 10,
        incident_medium: Medium::Air,
        transmitted_medium: Medium::Air,
        surfaces: vec![PathBuf::from("/tmp/scene.obj")],
        collector: CollectorDesc {
            radius: 0.1,
            shape: CollectorShape::WholeSphere,
            partition: CollectorPartition::EqualArea(0.1),
        },
        emitter: EmitterDesc {
            position: Position::Spherical(1.0, 0.0, 0.0),
            spectrum: Spectrum {
                start: 380.0,
                stop: 780.0,
                step: 10.0,
            },
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
