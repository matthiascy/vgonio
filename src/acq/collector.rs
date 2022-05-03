use crate::acq::desc::Range;

/// The virtual goniophotometer's detectors represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
/// The detectors are positioned on the center of each patch; the patches
/// are partitioned using 1.0 as radius.
#[derive(Clone, Debug)]
pub struct Collector {
    pub radius: f32,
    pub shape: Shape,
    pub partition: Partition,
    pub patches: Vec<Patch>,
}

/// Represents a patch on the spherical [`Collector`].
#[derive(Copy, Clone, Debug)]
pub struct Patch {
    /// Polar angle of the center of the patch in radians.
    pub zenith: f32,

    /// Azimuthal angle of the center of the patch in radians.
    pub azimuth: f32,
}

/// Description of the BRDF collector.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CollectorDesc {
    /// Radius of the underlying shape of the collector.
    pub radius: f32,

    /// Exact spherical shape of the collector.
    pub shape: Shape,

    /// Partition of the collector patches.
    pub partition: Partition,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    /// Only capture the upper part of the sphere.
    UpperHemisphere,

    /// Only capture the lower part of the sphere.
    LowerHemisphere,

    /// Capture the whole sphere.
    WholeSphere,
}

/// Partition of the collector spherical shape, each patch served as a detector.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Partition {
    /// The collector is partitioned into a number of regions with the same
    /// angular interval. Angles are expressed in *degrees*.
    EqualAngle {
        /// Range of interest of the polar angle θ (start, stop, step).
        theta: Range<f32>,

        /// Range of interest of the azimuthal angle φ (start, stop, step).
        phi: Range<f32>,
    },

    /// The collector is partitioned into a number of regions with the same
    /// area (solid angle), azimuthal angle φ is divided into equal intervals;
    /// polar angle θ is divided non uniformly to guarantee equal area patches.
    /// Angles are expressed in *degrees*
    EqualArea {
        /// Range of interest of the polar angle θ (start, stop, count).
        theta: (f32, f32, u32),

        /// Range of interest interval of the azimuthal angle φ (start, stop,
        /// step).
        phi: Range<f32>,
    },

    /// The collector is partitioned into a number of regions with the same
    /// projected area (projected solid angle).
    EqualProjectedArea {
        /// Range of interest of the polar angle θ (start, stop, count).
        theta: (f32, f32, u32),

        /// Range of interest interval of the azimuthal angle φ (start, stop,
        /// step) in degrees.
        phi: Range<f32>,
    },
}

impl Partition {
    pub fn kind_str(&self) -> &'static str {
        match self {
            Partition::EqualAngle { .. } => "equal angular interval",
            Partition::EqualArea { .. } => "equal area (solid angle)",
            Partition::EqualProjectedArea { .. } => "equal projected area (projected solid angle)",
        }
    }

    pub fn theta_range_str(&self) -> String {
        match self {
            Partition::EqualAngle { theta, .. } => format!(
                "{}° - {}°, step size {}°",
                theta.start, theta.stop, theta.step
            ),
            Partition::EqualArea { theta, .. } | Partition::EqualProjectedArea { theta, .. } => {
                format!("{}° - {}°, samples count {}", theta.0, theta.1, theta.2)
            }
        }
    }

    pub fn phi_range_str(&self) -> String {
        match self {
            Partition::EqualAngle { phi, .. }
            | Partition::EqualArea { phi, .. }
            | Partition::EqualProjectedArea { phi, .. } => {
                format!("{}° - {}°, step size {}°", phi.start, phi.stop, phi.step)
            }
        }
    }
}

impl From<CollectorDesc> for Collector {
    fn from(desc: CollectorDesc) -> Self {
        // todo: differentiate collector shape
        let patches = match desc.partition {
            Partition::EqualAngle {
                theta:
                    Range {
                        start: theta_start,
                        stop: theta_stop,
                        step: theta_step,
                    },
                phi:
                    Range {
                        start: phi_start,
                        stop: phi_stop,
                        step: phi_step,
                    },
            } => {
                let theta_start = theta_start.to_radians();
                let theta_stop = theta_stop.to_radians();
                let theta_step = theta_step.to_radians();
                let phi_start = phi_start.to_radians();
                let phi_stop = phi_stop.to_radians();
                let phi_step = phi_step.to_radians();

                let n_theta = ((theta_stop - theta_start) / theta_step).ceil() as usize;
                let n_phi = ((phi_stop - phi_start) / phi_step).ceil() as usize;

                let mut patches = Vec::with_capacity(n_theta * n_phi);
                for theta in 0..n_theta {
                    for phi in 0..n_phi {
                        patches.push(Patch {
                            zenith: (theta as f32 + 0.5) * theta_step,
                            azimuth: (phi as f32 + 0.5) * phi_step,
                        });
                    }
                }
                patches
            }
            Partition::EqualArea {
                theta: (theta_start, theta_stop, count),
                phi:
                    Range {
                        start: phi_start,
                        stop: phi_stop,
                        step: phi_step,
                    },
            } => {
                // Uniformly divide the zenith angle into count patches. Suppose r == 1
                // Spherical cap area = 2πrh, where r is the radius of the sphere on which
                // resides the cap, and h is the height from the top of the cap
                // to the bottom of the cap.
                let theta_start = theta_start.to_radians();
                let theta_stop = theta_stop.to_radians();
                let phi_start = phi_start.to_radians();
                let phi_stop = phi_stop.to_radians();
                let phi_step = phi_step.to_radians();

                let h_start = 1.0 - theta_start.cos();
                let h_stop = 1.0 - theta_stop.cos();
                let h_step = (h_stop - h_start) / count as f32;

                let n_theta = count as usize;
                let n_phi = ((phi_stop - phi_start) / phi_step).ceil() as usize;

                let mut patches = Vec::with_capacity(n_theta * n_phi);
                for theta in 0..n_theta {
                    for phi in 0..n_phi {
                        patches.push(Patch {
                            zenith: (1.0 - (h_step * (theta as f32 + 0.5) + h_start)).acos(),  // use the median angle (+ 0.5) as zenith
                            azimuth: (phi_start + phi as f32 * phi_step) as f32,
                        });
                    }
                }
                patches
            }
            Partition::EqualProjectedArea {
                theta: (start, stop, count),
                phi:
                    Range {
                        start: phi_start,
                        stop: phi_stop,
                        step: phi_step,
                    },
            } => {
                // Non-uniformly divide the radius of the disk after the projection.
                // Disk area is linearly proportional to squared radius.
                let theta_start = start.to_radians();
                let theta_stop = stop.to_radians();
                let phi_start = phi_start.to_radians();
                let phi_stop = phi_stop.to_radians();
                let phi_step = phi_step.to_radians();
                // Calculate radius range.
                let r_start = theta_start.sin();
                let r_stop = theta_stop.sin();
                let r_start_sqr = r_start * r_start;
                let r_stop_sqr = r_stop * r_stop;
                let factor = 1.0 / count as f32;
                let n_theta = count as usize;
                let n_phi = ((phi_stop - phi_start) / phi_step).ceil() as usize;

                let mut patches = Vec::with_capacity(n_theta * n_phi);
                for i in 0..n_theta {
                    // Linearly interpolate squared radius range.
                    // Projected area is proportional to squared radius.
                    //                 1st           2nd           3rd
                    // O- - - - | - - - I - - - | - - - I - - - | - - - I - - -|
                    //     r_start_sqr                               r_stop_sqr
                    let r_sqr =
                        r_start_sqr + (r_stop_sqr - r_start_sqr) * factor * (i as f32 + 0.5);
                    let r = r_sqr.sqrt();
                    let theta = r.asin();
                    for phi in 0..((phi_stop - phi_start) / phi_step).ceil() as usize {
                        patches.push(Patch {
                            zenith: theta,
                            azimuth: (phi_start + phi as f32 * phi_step) as f32,
                        });
                    }
                }
                patches
            }
        };

        Self {
            radius: desc.radius,
            shape: desc.shape,
            partition: desc.partition,
            patches,
        }
    }
}
