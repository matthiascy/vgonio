//! BRDF data from the MERL database.
#[cfg(feature = "bxdf_io")]
use crate::error::VgonioError;
use crate::{
    bxdf::{
        brdf::{
            io2hd_sph,
            measured::{BrdfParam, BrdfParamKind, MeasuredBrdf, Origin},
        },
        BrdfProxy, OutgoingDirs, ProxySource,
    },
    impl_any_measured_trait,
    math::Sph2,
    optics::ior::IorRegistry,
    units::{nm, Nanometres, Radians},
    utils::medium::Medium,
    AnyMeasured, AnyMeasuredBrdf, BrdfLevel, MeasuredBrdfKind, MeasurementKind,
};
use jabr::array::{s, DArr, DyArr, DynArr};
use std::borrow::Cow;
#[cfg(feature = "bxdf_io")]
use std::path::Path;

/// Parameterisation for a measured BRDF from the MERL database: <http://www.merl.com/brdf/>
#[derive(Debug, Clone, PartialEq)]
pub struct MerlBrdfParam {
    /// The zenith angles difference between the incident and outgoing
    /// directions.
    zenith_d: DArr<Radians, s![90]>,
    /// The azimuthal angles difference between the incident and outgoing
    /// directions.
    azimuth_d: DArr<Radians, s![180]>,
    /// Zenith angles of the half-vector.
    zenith_h: DArr<Radians, s![90]>,
}

impl MerlBrdfParam {
    /// The number of zenith angles for the half-vector.
    pub const RES_THETA_H: u32 = 90;
    /// The number of zenith ang90les for the difference vector.
    pub const RES_THETA_D: u32 = 90;
    /// The number of azimuthal angles for the difference vector.
    /// The actual number of samp90les is twice this value, as the BRDF is
    /// isotropic and the reciprocity principle applies.
    pub const RES_PHI_D: u32 = 180;
    /// The total number of sample90s.
    pub const RES_TOTAL: u32 = Self::RES_THETA_H * Self::RES_THETA_D * Self::RES_PHI_D;
    /// The scale factor for the Red channel.
    pub const R_SCALE: f64 = 1.0 / 1500.0;
    /// The scale factor for the Green channel.
    pub const G_SCALE: f64 = 1.15 / 1500.0;
    /// The scale factor for the Blue channel.
    pub const B_SCALE: f64 = 1.66 / 1500.0;
}

impl Default for MerlBrdfParam {
    fn default() -> Self {
        let mut zenith = DArr::zeros();
        for i in 0..Self::RES_THETA_H as usize {
            // Use the center of the bin as the zenith angle.
            zenith.as_mut_slice()[i] =
                Radians::HALF_PI * (i as f32 + 0.5) / Self::RES_THETA_H as f32;
        }
        let mut azimuth_d = DArr::zeros();
        for i in 0..Self::RES_PHI_D as usize {
            // Use the center of the bin as the azimuth angle.
            azimuth_d.as_mut_slice()[i] = Radians::PI * (i as f32 + 0.5) / Self::RES_PHI_D as f32;
        }

        Self {
            zenith_d: zenith.clone(),
            azimuth_d,
            zenith_h: zenith,
        }
    }
}

impl BrdfParam for MerlBrdfParam {
    fn kind() -> BrdfParamKind { BrdfParamKind::HalfVector }
}

/// BRDF from the MERL database: <http://www.merl.com/brdf/>
///
/// The original data of MERL BRDF is stored as row-major order in a 1D array
/// with dimensions: (channel, theta_h, theta_d, phi_d), where channel is the
/// index of the RGB colour channels (0: Red, 1: Green, 2: Blue). The right-most
/// index is the fastest varying index.
///
/// Inside the `MerlBrdf` structure, the data is stored as a 4D row-majored
/// array with dimensions: (theta_h, theta_d, phi_d, lambda), where lambda is
/// the wavelength index from smallest to largest. Therefore, the RGB channels
/// in the original data are reversed as BGR as Blue has the smallest
/// wavelength. The chosen wavelengths of the BGR channels are 435.8, 546.1, and
/// 700 nm, respectively.
pub type MerlBrdf = MeasuredBrdf<MerlBrdfParam, 4>;

unsafe impl Send for MerlBrdf {}
unsafe impl Sync for MerlBrdf {}

impl_any_measured_trait!(@single_level_brdf MerlBrdf);

impl MerlBrdf {
    /// Lookup the index of the azimuthal angle for the difference vector.
    pub fn phi_d_index(&self, phi_d: Radians) -> usize {
        // Make sure the angle is in the range [0, π].
        let phi_d = if phi_d < Radians::ZERO {
            phi_d + Radians::PI
        } else if phi_d >= Radians::PI {
            // Mirror the angle to the range [0, π].
            phi_d - Radians::PI
        } else {
            phi_d
        };

        let index = ((phi_d / Radians::PI) * MerlBrdfParam::RES_PHI_D as f32).round() as i32;
        index.clamp(0, MerlBrdfParam::RES_PHI_D as i32 - 1) as usize
    }

    /// Lookup the index of the zenith angle for the half-vector.
    pub fn theta_d_index(&self, theta_d: Radians) -> usize {
        assert!(theta_d >= Radians::ZERO && theta_d <= Radians::HALF_PI);
        let index =
            ((theta_d * 2.0 / Radians::PI) * MerlBrdfParam::RES_THETA_D as f32).round() as i32;
        index.clamp(0, MerlBrdfParam::RES_THETA_D as i32 - 1) as usize
    }

    /// Lookup the index of the zenith angle for the difference vector.
    ///
    /// The mapping is not linear.
    pub fn theta_h_index(&self, theta_h: Radians) -> usize {
        assert!(theta_h >= Radians::ZERO && theta_h <= Radians::HALF_PI);
        let theta_h_deg = theta_h * 0.5 / Radians::PI * MerlBrdfParam::RES_THETA_H as f32;
        let index = (theta_h_deg * MerlBrdfParam::RES_THETA_H as f32)
            .sqrt()
            .round() as i32;
        index.clamp(0, MerlBrdfParam::RES_THETA_H as i32 - 1) as usize
    }

    /// Lookup the sample of the BRDF at the given incident and outgoing
    /// directions.
    ///
    /// The sample is returned as a 3-element array, where each element is the
    /// value of the BRDF at the given incident and outgoing directions.
    ///
    /// # Arguments
    ///
    /// * `wi` - The incident direction.
    /// * `wo` - The outgoing direction.
    ///
    /// # Returns
    ///
    /// A 3-element array containing the value of the BRDF at the given incident
    /// and outgoing directions.
    pub fn sample_at(&self, wi: Sph2, wo: Sph2) -> [f32; 3] {
        let (wh, wd) = io2hd_sph(&wi, &wo);
        let theta_h_index = self.theta_h_index(wh.theta);
        let theta_d_index = self.theta_d_index(wd.theta);
        let phi_d_index = self.phi_d_index(wd.phi);
        let mut sample = [0.0f32; 3];
        let strides = self.samples.strides();
        let index =
            theta_h_index * strides[0] + theta_d_index * strides[1] + phi_d_index * strides[2];
        sample.copy_from_slice(&self.samples.as_slice()[index..index + 3]);
        sample
    }

    /// Loads the MERL BRDF from the given path.
    ///
    /// # Arguments
    ///
    /// * `filepath` - The path to the MERL BRDF data.
    #[cfg(feature = "bxdf_io")]
    pub fn load<P: AsRef<Path>>(filepath: P) -> Result<Self, VgonioError> {
        use std::{fs::File, io::Read};

        use crate::utils::medium::Medium;

        if !filepath.as_ref().exists() {
            return Err(VgonioError::new(
                format!(
                    "Can't read MERL BRDF from {:?}: file not found!",
                    filepath.as_ref()
                ),
                None,
            ));
        }

        let mut medium = Medium::Unknown;

        let splits = filepath
            .as_ref()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .split('-');
        for split in splits {
            if split == "aluminium" {
                medium = Medium::Aluminium;
                break;
            }
        }

        if medium == Medium::Unknown {
            return Err(VgonioError::new(
                format!(
                    "Can't read MERL BRDF from {:?}: unknown material!",
                    filepath.as_ref()
                ),
                None,
            ));
        }

        let mut file = File::open(filepath.as_ref())
            .map_err(|err| VgonioError::from_io_error(err, "Can't read MERL BRDF file!"))?;
        let mut count = 1u32;
        let mut buf = [0u8; size_of::<u32>()];

        log::info!("Reading MERL BRDF dismensions...");
        for _ in 0..3 {
            file.read_exact(&mut buf).map_err(|err| {
                VgonioError::from_io_error(err, "Can't read MERL BRDF dimensions!")
            })?;
            let n = u32::from_le_bytes(buf);
            count *= n;
        }

        if count != MerlBrdfParam::RES_TOTAL {
            return Err(VgonioError::new(
                format!(
                    "Can't read MERL BRDF from {:?}: invalid dimensions (expecting {}, actual {})!",
                    filepath.as_ref(),
                    MerlBrdfParam::RES_TOTAL,
                    count
                ),
                None,
            ));
        }

        // Read the data which is composited of 3 channels (RGB).
        let mut data = vec![0.0f64; count as usize * 3].into_boxed_slice();
        file.read_exact(unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                count as usize * 3 * size_of::<f64>(),
            )
        })
        .map_err(|err| VgonioError::from_io_error(err, "Can't read MERL BRDF data!"))?;

        // Rearrange the data into a 4D array.
        // Original data layout: (channel, theta_h, theta_d, phi_d)
        // New data layout: (theta_h, theta_d, phi_d, channel)
        // TODO: ideally, we should simply read the data into a 4D array directly
        // following the MERL BRDF layout then apply the dimension permutation with
        // multidimentional indexing.
        let mut samples = DyArr::zeros([
            MerlBrdfParam::RES_THETA_H as usize,
            MerlBrdfParam::RES_THETA_D as usize,
            MerlBrdfParam::RES_PHI_D as usize,
            3,
        ]);

        let stride_c = MerlBrdfParam::RES_TOTAL as usize;
        let stride_th = MerlBrdfParam::RES_THETA_D as usize * MerlBrdfParam::RES_PHI_D as usize;
        let stride_td = MerlBrdfParam::RES_PHI_D as usize;
        let scale = [
            MerlBrdfParam::R_SCALE,
            MerlBrdfParam::G_SCALE,
            MerlBrdfParam::B_SCALE,
        ];
        for i in 0..MerlBrdfParam::RES_THETA_H as usize {
            for j in 0..MerlBrdfParam::RES_THETA_D as usize {
                for k in 0..MerlBrdfParam::RES_PHI_D as usize {
                    for c in 0..3 {
                        let offset = c * stride_c + i * stride_th + j * stride_td + k;
                        samples[[i, j, k, c]] = (data[offset] * scale[c]) as f32;
                    }
                }
            }
        }

        let spectrum = DyArr::from_vec_1d(vec![nm!(435.8), nm!(546.1), nm!(700.0)]);
        let params = Box::new(MerlBrdfParam::default());

        Ok(Self {
            kind: MeasuredBrdfKind::Merl,
            origin: Origin::RealWorld,
            incident_medium: Medium::Air,
            transmitted_medium: medium,
            params,
            spectrum,
            samples,
        })
    }

    /// Returns the kind of the BRDF.
    pub fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Merl }
}

impl AnyMeasuredBrdf for MerlBrdf {
    crate::any_measured_brdf_trait_common_impl!(MerlBrdf, Merl);

    fn proxy(&self, iors: &IorRegistry) -> BrdfProxy {
        let iors_i = Cow::Owned(
            iors.ior_of_spectrum(self.incident_medium, self.spectrum.as_slice())
                .unwrap()
                .into_vec(),
        );
        let iors_t = Cow::Owned(
            iors.ior_of_spectrum(self.transmitted_medium, self.spectrum.as_slice())
                .unwrap()
                .into_vec(),
        );
        let theta_res = MerlBrdfParam::RES_THETA_H as usize;
        let phi_res = MerlBrdfParam::RES_PHI_D as usize;
        let i_thetas = Cow::Owned(DyArr::from_iterator(
            [-1],
            (0..theta_res).into_iter().map(|i| (i as f32).to_radians()),
        ));
        let o_thetas = Cow::Owned(DyArr::from_iterator(
            [-1],
            (0..theta_res).into_iter().map(|i| (i as f32).to_radians()),
        ));
        let i_phis = Cow::Owned(DyArr::from_iterator(
            [-1],
            (0..phi_res).into_iter().map(|i| (i as f32).to_radians()),
        ));
        let o_phis = Cow::Owned(DyArr::from_iterator(
            [-1],
            (0..phi_res).into_iter().map(|i| (i as f32).to_radians()),
        ));

        let mut resampled = DynArr::zeros(&[theta_res, phi_res, theta_res, phi_res, 3]);
        for i in 0..theta_res {
            for j in 0..phi_res {
                for k in 0..theta_res {
                    for l in 0..phi_res {
                        let wi = Sph2::new(
                            Radians::from_degrees(i as f32),
                            Radians::from_degrees(j as f32),
                        );
                        let wo = Sph2::new(
                            Radians::from_degrees(k as f32),
                            Radians::from_degrees(l as f32),
                        );
                        let sample = self.sample_at(wi, wo);
                        resampled[[i, j, k, l, 0]] = sample[0];
                        resampled[[i, j, k, l, 1]] = sample[1];
                        resampled[[i, j, k, l, 2]] = sample[2];
                    }
                }
            }
        }

        let o_dirs = OutgoingDirs::new_grid(o_thetas, o_phis);

        BrdfProxy {
            has_nan: false,
            source: ProxySource::Measured,
            brdf: self,
            i_thetas,
            i_phis,
            o_dirs,
            resampled: Cow::Owned(resampled),
            iors_i,
            iors_t,
        }
    }
}
