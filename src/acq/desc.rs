use crate::{
    acq::{
        bsdf::BsdfKind,
        util::{SphericalPartition, SphericalShape},
        Length, LengthUnit, Medium, Metre, Metres, RayTracingMethod,
    },
    Error,
};
use std::{
    fs::File,
    io::Read,
    ops::Sub,
    path::{Path, PathBuf},
};

/// Helper struct used to specify the range of all kind of measurement (mostly angle ranges).
#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize)]
#[serde(into = "[T; 3]", from = "[T; 3]")]
pub struct Range<T: Copy + Clone> {
    /// Initial wavelength of the spectrum.
    pub start: T,

    /// Final wavelength of the spectrum.
    pub stop: T,

    /// Increment between wavelength samples.
    pub step: T,
}

impl<T: Copy + Clone> PartialEq for Range<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.stop == other.stop && self.step == other.step
    }
}

impl<T: Copy + Clone> Eq for Range<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> Range<T> {
    /// Maps a function over the start and stop of the range.
    pub fn map(&self, f: impl Fn(T) -> T) -> Range<T> {
        Range {
            start: f(self.start),
            stop: f(self.stop),
            step: self.step,
        }
    }

    /// Returns the span of the range.
    pub fn span(&self) -> T
    where
        T: Sub<Output = T>,
    {
        self.stop - self.start
    }
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
    fn from(range: Range<T>) -> Self { [range.start, range.stop, range.step] }
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
    fn from(range: Range<T>) -> Self { (range.start, range.stop, range.step) }
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
    /// Returns the sample count of the range.
    pub fn samples_count(&self) -> usize { ((self.stop - self.start) / self.step).floor() as usize }
}

impl<A: LengthUnit> Range<Length<A>> {
    /// Returns the sample count of the range.
    pub fn samples_count(&self) -> usize { ((self.stop.value - self.start.value) / self.step.value).floor() as usize }
}

/// Describes the radius of measurement.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum RadiusDesc {
    /// Radius is deduced from the dimension of the surface.
    Auto,

    /// Radius is given explicitly.
    Fixed(f32),
}

impl RadiusDesc {
    /// Whether the radius is deduced from the dimension of the surface.
    pub fn is_auto(&self) -> bool {
        match self {
            RadiusDesc::Auto => true,
            RadiusDesc::Fixed(_) => false,
        }
    }
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

    /// Angle (theta) specifying the spherical coordinates of the light source
    /// (**in degrees**).
    pub zenith: Range<f32>,

    /// Angle (phi) specifying the spherical coordinates of the light source
    /// (**in degrees**).
    pub azimuth: Range<f32>,

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

/// Supported type of measurement.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum MeasurementKind {
    /// Bidirectional Scattering Distribution Function
    Bsdf(BsdfKind),

    /// Normal Distribution Function
    Ndf,
}

/// Description of the measurement.
///
/// Note: angle in the description file is always in degrees.
#[derive(Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeasurementDesc {
    /// The measurement kind.
    pub measurement_kind: MeasurementKind,

    /// Ray tracing method used for the measurement.
    pub tracing_method: RayTracingMethod,

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
    /// Load measurement descriptions from a file. A file may contain multiple
    /// descriptions.
    pub fn load_from_file(filepath: &Path) -> Result<Vec<MeasurementDesc>, Error> {
        use serde::Deserialize;

        let mut file = File::open(filepath)?;
        let content = {
            let mut str_ = String::new();
            file.read_to_string(&mut str_)?;
            str_
        };
        let measurements = serde_yaml::Deserializer::from_str(&content)
            .into_iter()
            .map(|doc| MeasurementDesc::deserialize(doc).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(measurements)
    }
}

impl Default for MeasurementDesc {
    fn default() -> Self {
        Self {
            measurement_kind: MeasurementKind::Ndf,
            tracing_method: RayTracingMethod::Standard,
            incident_medium: Medium::Air,
            transmitted_medium: Medium::Air,
            surfaces: vec![],
            emitter: EmitterDesc {
                num_rays: 1000,
                max_bounces: 10,
                radius: RadiusDesc::Auto,
                zenith: Range::<f32> {
                    start: 0.0,
                    stop: 90.0,
                    step: 5.0,
                },
                azimuth: Range::<f32> {
                    start: 0.0,
                    stop: 360.0,
                    step: 120.0,
                },
                spectrum: Default::default(),
            },
            collector: CollectorDesc {
                radius: RadiusDesc::Auto,
                shape: SphericalShape::UpperHemisphere,
                partition: SphericalPartition::EqualArea {
                    zenith: (0.0, 0.0, 0),
                    azimuth: Range::<f32> {
                        start: 0.0,
                        stop: 0.0,
                        step: 0.0,
                    },
                },
            },
        }
    }
}

#[test]
fn scene_desc_serialization() {
    use std::io::{Cursor, Write};

    let desc = MeasurementDesc {
        measurement_kind: MeasurementKind::Bsdf(BsdfKind::InPlaneBrdf),
        tracing_method: RayTracingMethod::Standard,
        incident_medium: Medium::Air,
        transmitted_medium: Medium::Air,
        surfaces: vec![PathBuf::from("/tmp/scene.obj")],
        collector: CollectorDesc {
            radius: RadiusDesc::Auto,
            shape: SphericalShape::WholeSphere,
            partition: SphericalPartition::EqualArea {
                zenith: (0.0, 90.0, 45),
                azimuth: Range {
                    start: 0.0,
                    stop: 360.0,
                    step: 0.0,
                },
            },
        },
        emitter: EmitterDesc {
            num_rays: 0,
            max_bounces: 0,
            radius: RadiusDesc::Auto,
            spectrum: Range {
                start: 380.0,
                stop: 780.0,
                step: 10.0,
            },
            zenith: Range {
                start: 0.0,
                stop: 90.0,
                step: 0.0,
            },
            azimuth: Range {
                start: 0.0,
                stop: 360.0,
                step: 0.0,
            },
        },
    };

    let serialized = serde_yaml::to_string(&desc).unwrap();

    let mut file = Cursor::new(vec![0u8; 128]);
    file.write_all(serialized.as_bytes()).unwrap();

    file.set_position(0);
    let deserialized_0: MeasurementDesc = serde_yaml::from_reader(file).unwrap();
    let deserialized_1: MeasurementDesc = serde_yaml::from_str(&serialized).unwrap();

    println!("{}", serialized);

    assert_eq!(desc, deserialized_0);
    assert_eq!(desc, deserialized_1);
    assert_eq!(deserialized_0, deserialized_1);
}
