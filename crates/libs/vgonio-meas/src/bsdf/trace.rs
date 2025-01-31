/// Enumeration of the different ways to trace rays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RtcMethod {
    /// Ray tracing using Intel's Embree library.
    #[cfg(feature = "embree")]
    Embree,
    /// Ray tracing using Nvidia's OptiX library.
    #[cfg(feature = "optix")]
    Optix,
    /// Customised grid ray tracing method.
    Grid,
}

#[cfg(feature = "vdbg")]
/// Energy after a ray is reflected by the micro-surface.
///
/// Used during the data collection process.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Energy {
    /// The ray of a certain wavelength is absorbed by the micro-surface.
    Absorbed,
    /// The ray of a specific wavelength is reflected by the micro-surface.
    Reflected(f32),
}

/// Bounce and energy of a patch for each bounce.
///
/// The index of the array corresponds to the bounce number starting from 1.
/// At index 0, the data is the sum of all the bounces.
#[derive(Debug, Clone, Default)]
pub struct BounceAndEnergy {
    /// Maximum number of bounces of rays hitting the patch.
    pub n_bounce: u32,
    /// Number of rays hitting the patch for each bounce.
    pub n_ray_per_bounce: Box<[u64]>,
    /// Total energy of rays hitting the patch for each bounce.
    pub energy_per_bounce: Box<[f64]>,
}

impl BounceAndEnergy {
    /// Creates a new bounce and energy.
    pub fn empty(bounces: usize) -> Self {
        Self {
            n_bounce: bounces as u32,
            n_ray_per_bounce: vec![0; bounces + 1].into_boxed_slice(),
            energy_per_bounce: vec![0.0; bounces + 1].into_boxed_slice(),
        }
    }

    /// Reallocates the memory in case the number of bounces is greater
    /// than the current number of bounces. The function creates a new
    /// array of the number of rays and energy per bounce and then
    /// copies the data from the old arrays to the new arrays.
    pub fn reallocate(&mut self, bounces: usize) {
        if bounces as u32 <= self.n_bounce {
            return;
        }
        let mut n_ray_per_bounce = vec![0; bounces + 1];
        let mut energy_per_bounce = vec![0.0; bounces + 1];
        n_ray_per_bounce[..=self.n_bounce as usize].copy_from_slice(&self.n_ray_per_bounce);
        energy_per_bounce[..=self.n_bounce as usize].copy_from_slice(&self.energy_per_bounce);
        self.n_ray_per_bounce = n_ray_per_bounce.into_boxed_slice();
        self.energy_per_bounce = energy_per_bounce.into_boxed_slice();
        self.n_bounce = bounces as u32;
    }

    /// Returns the total number of rays.
    pub fn total_rays(&self) -> u64 { self.n_ray_per_bounce[0] }

    /// Returns the total energy of rays.
    pub fn total_energy(&self) -> f64 { self.energy_per_bounce[0] }
}

impl PartialEq for BounceAndEnergy {
    fn eq(&self, other: &Self) -> bool {
        self.n_bounce == other.n_bounce
            && self.n_ray_per_bounce == other.n_ray_per_bounce
            && self.energy_per_bounce == other.energy_per_bounce
    }
}
