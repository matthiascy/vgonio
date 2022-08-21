use crate::acq::{
    collector::Patch,
    desc::{EmitterDesc, RadiusDesc, Range},
    util::SphericalCoord,
    Ray,
};

/// Light emitter of the virtual gonio-photometer. The shape of the emitter
/// patch is spherical cap. The patch will be rotated during the measurement.
///
/// Note: need to update the radius for each surface before the measurement to
/// make sure that the surface is covered by the patch.
pub struct Emitter {
    /// Number of rays emitted by the emitter.
    pub num_rays: u32,

    /// Distance of the emitter from the origin.
    pub radius: Option<f32>,

    /// Inclination angle range of the emitter (in radians).
    pub zenith: Range<f32>,

    /// Azimuthal angle range of the emitter (in radians).
    pub azimuth: Range<f32>,

    /// The patch from which the rays are emitted.
    pub patch: Patch,

    /// Samples generated for the patch.
    pub samples: Vec<glam::Vec3>,
}

impl From<EmitterDesc> for Emitter {
    fn from(desc: EmitterDesc) -> Self {
        let theta = desc.zenith.map(f32::to_radians);
        let phi = desc.azimuth.map(f32::to_radians);
        let samples = uniform_sampling_on_unit_sphere(
            desc.num_rays,
            theta.start,
            theta.stop,
            phi.start,
            phi.stop,
        );
        let radius = match desc.radius {
            RadiusDesc::Auto => None,
            RadiusDesc::Fixed(val) => Some(val),
        };
        Self {
            num_rays: desc.num_rays,
            radius,
            zenith: theta,
            azimuth: phi,
            patch: Patch::new(
                (-theta.step / 2.0, theta.step / 2.0),
                (-phi.step / 2.0, phi.step / 2.0),
            ),
            samples,
        }
    }
}

impl Emitter {
    pub fn set_radius(&mut self, radius: f32) { self.radius = Some(radius); }

    /// All possible positions of the emitter.
    pub fn positions(&self) -> Vec<SphericalCoord> {
        let n_zenith = ((self.zenith.stop - self.zenith.start) / self.zenith.step).ceil() as usize;
        let n_azimuth =
            ((self.azimuth.stop - self.azimuth.start) / self.azimuth.step).ceil() as usize;

        (0..n_zenith)
            .into_iter()
            .flat_map(|i_theta| {
                (0..n_azimuth).into_iter().map(move |i_phi| SphericalCoord {
                    zenith: self.zenith.start + i_theta as f32 * self.zenith.step,
                    azimuth: self.azimuth.start + i_phi as f32 * self.azimuth.step,
                })
            })
            .collect()
    }

    /// Emits rays from the patch located at `pos`.
    pub fn emit_rays(&self, pos: SphericalCoord) -> Vec<Ray> {
        let r = self.radius.expect("radius not set");
        let mat = glam::Mat3::from_axis_angle(glam::Vec3::Y, pos.zenith)
            * glam::Mat3::from_axis_angle(glam::Vec3::Z, pos.azimuth);
        let dir = -pos.into_cartesian();

        self.samples
            .iter()
            .map(|sample| {
                let origin = mat * *sample * r;
                Ray { o: origin, d: dir }
            })
            .collect()
    }
}

/// Generates uniformly distributed samples on the unit sphere.
/// Right-handed Z-up coordinate system is used.
///
/// x = cos phi * sin theta
/// y = sin phi * sin theta
/// z = cos theta
pub fn uniform_sampling_on_unit_sphere(
    num_samples: u32,
    theta_start: f32,
    theta_stop: f32,
    phi_start: f32,
    phi_stop: f32,
) -> Vec<glam::Vec3> {
    use rand_distr::Distribution;
    use std::f32::consts::PI;

    let mut rng = rand::thread_rng();
    let uniform = rand_distr::Uniform::new(0.0, 1.0);
    let mut samples = vec![];

    let mut i = 0;
    while i < num_samples {
        let phi = uniform.sample(&mut rng) * PI * 2.0;
        let theta = (1.0 - 2.0 * uniform.sample(&mut rng)).acos();

        if (theta_start..theta_stop).contains(&theta) && (phi_start..phi_stop).contains(&phi) {
            samples.push(glam::Vec3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ));
            i += 1;
        }
    }

    samples
}
