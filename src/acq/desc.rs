use crate::acq::bxdf::BxdfKind;
use crate::acq::util::{SphericalPartition, SphericalShape};
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
#[serde(into = "[T; 3]", from = "[T; 3]")]
pub struct Range<T: Copy> {
    /// Initial wavelength of the spectrum.
    pub start: T,

    /// Final wavelength of the spectrum.
    pub stop: T,

    /// Increment between wavelength samples.
    pub step: T,
}

impl<T: Copy> From<[T; 3]> for Range<T> {
    fn from(vals: [T; 3]) -> Self {
        Self {
            start: vals[0],
            stop: vals[1],
            step: vals[2],
        }
    }
}

impl<T: Copy> From<Range<T>> for [T; 3] {
    fn from(range: Range<T>) -> Self {
        [range.start, range.stop, range.step]
    }
}

impl<T: Copy> From<(T, T, T)> for Range<T> {
    fn from(vals: (T, T, T)) -> Self {
        Self {
            start: vals.0,
            stop: vals.1,
            step: vals.2,
        }
    }
}

impl<T: Copy> From<Range<T>> for (T, T, T) {
    fn from(range: Range<T>) -> Self {
        (range.start, range.stop, range.step)
    }
}

impl<T: Default + Copy> Default for Range<T> {
    fn default() -> Self {
        Self {
            start: T::default(),
            stop: T::default(),
            step: T::default(),
        }
    }
}

impl Range<f32> {
    pub fn samples_count(&self) -> usize {
        ((self.stop - self.start) / self.step).floor() as usize
    }
}


#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum RadiusDesc {
    /// Radius is deduced from the dimension of the surface.
    Auto,

    /// Radius is given explicitly.
    Fixed(f32),
}

/// Description of the light source.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EmitterDesc {
    /// Number of emitted rays.
    pub num_rays: u32,

    /// Max bounces allowed for the emitted rays.
    pub max_bounces: u32,

    /// Radius (r) specifying the spherical coordinates of the light source.
    pub radius: RadiusDesc,

    /// Partition of the emitter sphere.
    pub partition: SphericalPartition,

    /// Light source's spectrum.
    pub spectrum: Range<f32>,
}

/// Description of the BRDF collector.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CollectorDesc {
    /// Radius of the underlying shape of the collector.
    pub radius: RadiusDesc,

    /// Exact spherical shape of the collector.
    pub shape: SphericalShape,

    /// Partition of the collector patches.
    pub partition: SphericalPartition,
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
    /// Length unit of the measurement.
    pub length_unit: LengthUnit,

    /// The measurement kind.
    pub measurement_kind: MeasurementKind,

    /// Incident medium of the measurement.
    pub incident_medium: Medium,

    /// Transmitted medium of the measurement (medium of the surface).
    pub transmitted_medium: Medium,

    /// Surfaces to be measured. surface's path can be prefixed with either
    /// `user://` or `local://` to indicate the user-defined data file path or
    /// system-defined data file path.
    pub surfaces: Vec<PathBuf>,

    /// Description of the emitter.
    pub emitter: EmitterDesc,

    /// Description of the collector.
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
        incident_medium: Medium::Air,
        transmitted_medium: Medium::Air,
        surfaces: vec![PathBuf::from("/tmp/scene.obj")],
        collector: CollectorDesc {
            radius: 0.1,
            shape: SphericalShape::WholeSphere,
            partition: SphericalPartition::EqualArea {
                theta: (0.0, 90.0, 45),
                phi: Range {
                    start: 0.0,
                    stop: 360.0,
                    step: 0.0,
                },
            },
        },
        emitter: EmitterDesc {
            num_rays: 0,
            max_bounces: 0,
            radius: 0.0,
            spectrum: Range {
                start: 380.0,
                stop: 780.0,
                step: 10.0,
            },
            partition: SphericalPartition::EqualArea {
                theta: (0.0, 90.0, 45),
                phi: Range {
                    start: 0.0,
                    stop: 360.0,
                    step: 0.0,
                },
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
