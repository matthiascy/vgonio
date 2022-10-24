use crate::{
    acq::{
        bsdf::BsdfKind,
        util::{SphericalPartition, SphericalDomain},
        Length, LengthUnit, Medium, UMetre, Metres, RayTracingMethod,
    },
    Error,
};
use std::{
    fs::File,
    io::Read,
    ops::Sub,
    path::{Path, PathBuf},
};
use crate::acq::collector::CollectorScheme;
use crate::acq::{degrees, Patch};

/// Helper struct used to specify the range of all kind of measurement (mostly angle ranges).
#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize)]
#[serde(into = "[T; 3]", from = "[T; 3]")]
pub struct Range<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,

    /// Final value of the range.
    pub stop: T,

    /// Increment between two consecutive values of the range.
    pub step: T,
}

/// Helper struct used to specify the range without knowing the step size but the number of samples.
#[derive(Debug, Copy, Clone, serde::Serialize, serde::Deserialize)]
#[serde(into = "(T, T, usize)", from = "(T, T, usize)")]
pub struct Range2<T: Copy + Clone> {
    /// Initial value of the range.
    pub start: T,

    /// Final value of the range.
    pub stop: T,

    /// Number of samples.
    pub count: usize
}

impl<T: Copy + Clone> PartialEq for Range<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.stop == other.stop && self.step == other.step
    }
}

impl<T: Copy + Clone> PartialEq for Range2<T>
    where T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.stop == other.stop && self.count == other.count
    }
}

impl<T: Copy + Clone> Eq for Range<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> Eq for Range2<T> where T: PartialEq + Eq {}

impl<T: Copy + Clone> Range<T> {
    /// Maps a function over the start and stop of the range.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> Range<U> {
        Range {
            start: f(self.start),
            stop: f(self.stop),
            step: f(self.step),
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

impl<T: Copy + Clone> Range2<T> {
    /// Maps a function over the start and stop of the range.
    pub fn map<U: Copy>(&self, f: impl Fn(T) -> U) -> Range2<U> {
        Range2 {
            start: f(self.start),
            stop: f(self.stop),
            count: self.count,
        }
    }

    /// Returns the span of the range.
    pub fn span(&self) -> T
        where T: Sub<Output = T>,
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

impl<T: Copy> From<(T, T, usize)> for Range2<T> {
    fn from(vals: (T, T, usize)) -> Self {
        Self {
            start: vals.0,
            stop: vals.1,
            count: vals.2,
        }
    }
}

impl<T: Copy> From<Range<T>> for (T, T, T) {
    fn from(range: Range<T>) -> Self { (range.start, range.stop, range.step) }
}

impl<T: Copy> From<Range2<T>> for (T, T, usize) {
    fn from(range: Range2<T>) -> Self { (range.start, range.stop, range.count) }
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

impl<T: Default + Copy> Default for Range2<T> {
    fn default() -> Self {
        Self {
            start: T::default(),
            stop: T::default(),
            count: 0,
        }
    }
}

impl Range<f32> {
    /// Returns the sample count of the range.
    pub fn samples_count(&self) -> usize { ((self.stop - self.start) / self.step).floor() as usize }
}

impl Range2<f32> {
    pub fn step_size(&self) -> f32 {
        (self.stop - self.start) / (self.count as f32)
    }
}

impl<A: LengthUnit> Range<Length<A>> {
    /// Returns the sample count of the range.
    pub fn samples_count(&self) -> usize {
        ((self.stop.value - self.start.value) / self.step.value).floor() as usize
    }
}

impl<A: LengthUnit> Range2<Length<A>> {
    pub fn step_size(&self) -> Length<A> {
        Length::new((self.stop.value - self.start.value) / (self.count as f32))
    }
}

/// Describes the radius of measurement.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")] // TODO: use case_insensitive in the future
pub enum Radius {
    /// Radius is dynamically deduced from the dimension of the surface.
    Dynamic,

    /// Radius is given explicitly.
    Fixed(Metres),
}

impl Radius {
    /// Whether the radius is dynamically deduced from the dimension of the surface.
    pub fn is_dynamic(&self) -> bool {
        match self {
            Radius::Dynamic => true,
            Radius::Fixed(_) => false,
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
    pub radius: Radius,

    /// Angle (theta) specifying the position in spherical coordinates of the
    /// light source (**in degrees**).
    pub zenith: Range<f32>,

    /// Angle (phi, zenith) specifying the position in spherical coordinates of the
    /// light source (**in degrees**).
    pub azimuth: Range<f32>,

    /// Light source's spectrum.
    pub spectrum: Range<f32>,
}

/// Description of the BRDF collector.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CollectorDesc {
    /// Radius of the underlying shape of the collector.
    pub radius: Radius,

    /// Capture scheme of the collector.
    pub scheme: CollectorScheme,
}

impl CollectorDesc {
    /// Generates the patches of the collector in case
    /// of a spherical partition.
    pub fn generate_patches(&self) -> Option<Vec<Patch>> {
        match self.scheme {
            CollectorScheme::Partitioned { domain, partition } => {
                Some(partition.generate_patches(domain))
            }
            CollectorScheme::Individual { .. } => { None }
        }
    }
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
                radius: Radius::Dynamic,
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
                radius: Radius::Dynamic,
                scheme: CollectorScheme::Partitioned {
                    domain: SphericalDomain::UpperHemisphere,
                    partition: SphericalPartition::EqualArea {
                        zenith: Range2 {
                            start: degrees!(0.0),
                            stop: degrees!(90.0),
                            count: 0,
                        },
                        azimuth: Range {
                            start: degrees!(0.0),
                            stop: degrees!(0.0),
                            step: degrees!(0.0),
                        },
                    },
                }
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
            radius: Radius::Dynamic,
            shape: SphericalDomain::WholeSphere,
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
            radius: Radius::Dynamic,
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
