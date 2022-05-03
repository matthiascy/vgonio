/// The virtual goniophotometer's detectors represented by the patches
/// of a sphere (or an hemisphere) positioned around the specimen.
/// The detectors are positioned on the center of each patch; the patches
/// are partitioned using 1.0 as radius.
pub struct Collector {
    pub radius: f32,
    pub shape: Shape,
    pub partition: Partition,
    pub detectors: Vec<Detector>,
}

/// Represents a patch on the spherical [`Collector`].
pub struct Detector {
    /// Polar angle of the center of the patch.
    pub zenith: f32,

    /// Azimuthal angle of the center of the patch.
    pub azimuth: f32,

    /// Measured value of the patch.
    pub measured: f32,
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
        theta: (f32, f32, f32),

        /// Range of interest of the azimuthal angle φ (start, stop, step).
        phi: (f32, f32, f32),
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
        phi: (f32, f32, f32),
    },

    /// The collector is partitioned into a number of regions with the same
    /// projected area (projected solid angle).
    EqualProjectedArea {
        /// Range of interest of the polar angle θ (start, stop, count).
        theta: (f32, f32, u32),

        /// Range of interest interval of the azimuthal angle φ (start, stop,
        /// step) in degrees.
        phi: (f32, f32, f32),
    },
}

impl From<CollectorDesc> for Collector {
    fn from(desc: CollectorDesc) -> Self {
        // todo: differentiate collector shape
        let detectors = match desc.partition {
            Partition::EqualAngle {
                theta: (theta_start, theta_stop, theta_step),
                phi: (phi_start, phi_stop, phi_step),
            } => {
                let theta_start = theta_start.to_radians();
                let theta_stop = theta_stop.to_radians();
                let theta_step = theta_step.to_radians();
                let phi_start = phi_start.to_radians();
                let phi_stop = phi_stop.to_radians();
                let phi_step = phi_step.to_radians();

                let n_theta = ((theta_stop - theta_start) / theta_step).ceil() as usize;
                let n_phi = ((phi_stop - phi_start) / phi_step).ceil() as usize;

                let mut detectors = Vec::with_capacity(n_theta * n_phi);
                for theta in 0..n_theta {
                    for phi in 0..n_phi {
                        detectors.push(Detector {
                            zenith: (theta as f32 + 0.5) * theta_step,
                            azimuth: (phi as f32 + 0.5) * phi_step,
                            measured: 0.0,
                        });
                    }
                }
                detectors
            }
            Partition::EqualArea {
                theta: (start, end, count),
                phi: (phi_start, phi_stop, phi_step),
            } => {
                // Uniformly divide the zenith angle into count patches. Suppose r == 1
                // Spherical cap area = 2πrh, where r is the radius of the sphere on which
                // resides the cap, and h is the height from the top of the cap
                // to the bottom of the cap.
                let h_step = 1.0 / count as f32;
                let phi_start = phi_start.to_radians();
                let phi_stop = phi_stop.to_radians();
                let phi_step = phi_step.to_radians();

                let n_theta = count as usize;
                let n_phi = ((phi_stop - phi_start) / phi_step).ceil() as usize;

                let mut detectors = Vec::with_capacity(n_theta * n_phi);
                for theta in 0..n_theta {
                    for phi in 0..n_phi {
                        detectors.push(Detector {
                            zenith: (1.0 - h_step * (theta as f32 + 0.5)).acos(),  // use the median angle (+ 0.5) as zenith
                            azimuth: (phi_start + phi as f32 * phi_step) as f32,
                            measured: 0.0,
                        });
                    }
                }
                detectors
            }
            Partition::EqualProjectedArea{
                theta: (start, stop, count),
                phi: (phi_start, phi_stop, phi_step) } => {
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

                let mut detectors = Vec::with_capacity(n_theta * n_phi);
                for i in 0..n_theta {
                    // Linearly interpolate squared radius range.
                    // Projected area is proportional to squared radius.
                    //                 1st           2nd           3rd
                    // O- - - - | - - - I - - - | - - - I - - - | - - - I - - -|
                    //     r_start_sqr                               r_stop_sqr
                    let r_sqr = r_start_sqr + (r_stop_sqr - r_start_sqr) * factor * (i as f32 + 0.5);
                    let r = r_sqr.sqrt();
                    let theta = r.asin();
                    for phi in 0..((phi_stop - phi_start) / phi_step).ceil() as usize {
                        detectors.push(Detector {
                            zenith: theta,
                            azimuth: (phi_start + phi as f32 * phi_step) as f32,
                            measured: 0.0,
                        });
                    }
                }
                detectors
            }
        };

        Self {
            radius: desc.radius,
            shape: desc.shape,
            partition: desc.partition,
            detectors,
        }
    }
}
