#[cfg(feature = "fitting")]
use crate::brdf::measured::AnalyticalFit;
use crate::brdf::{
    measured::{BrdfParameterisation, MeasuredBrdf, Origin, ParametrisationKind},
    Bxdf,
};
#[cfg(feature = "io")]
use base::error::VgonioError;
use base::{
    impl_measured_data_trait,
    medium::Medium,
    optics::ior::RefractiveIndexRegistry,
    units::{nm, Nanometres, Radians},
    ErrorMetric, MeasuredBrdfKind, MeasuredData, MeasurementKind, ResidualErrorMetric,
};
use jabr::array::{s, DArr, DyArr};
use std::any::Any;
#[cfg(feature = "io")]
use std::path::Path;

/// Parameterisation for a measured BRDF from the MERL database: <http://www.merl.com/brdf/>
#[derive(Debug, Clone, PartialEq)]
pub struct MerlBrdfParameterisation {
    /// The zenith angles difference between the incident and outgoing
    /// directions.
    zenith_d: DArr<Radians, s![90]>,
    /// The azimuthal angles difference between the incident and outgoing
    /// directions.
    azimuth_d: DArr<Radians, s![180]>,
    /// Zenith angles of the half-vector.
    zenith_h: DArr<Radians, s![90]>,
}

impl MerlBrdfParameterisation {
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
    pub const R_SCALE: f64 = 1.0 / 901500.0;
    /// The scale factor for the Green channel.
    pub const G_SCALE: f64 = 1.15 / 1500.0;
    /// The scale factor for the Blue channel.
    pub const B_SCALE: f64 = 1.66 / 1500.0;
}

impl Default for MerlBrdfParameterisation {
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

impl BrdfParameterisation for MerlBrdfParameterisation {
    fn kind() -> ParametrisationKind { ParametrisationKind::HalfVector }
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
pub type MerlBrdf = MeasuredBrdf<MerlBrdfParameterisation, 4>;

impl_measured_data_trait!(MerlBrdf, Bsdf, Some(MeasuredBrdfKind::Merl));

unsafe impl Send for MerlBrdf {}
unsafe impl Sync for MerlBrdf {}

impl MerlBrdf {
    /// Lookup the index of the azimuthal angle for the difference vector.
    pub fn phi_d_index(&self, phi_d: Radians) -> usize {
        // Make sure the angle is in the range [0, π].
        let phi_d = if phi_d < Radians::ZERO {
            phi_d + Radians::PI
        } else if phi_d >= Radians::PI{
            // Mirror the angle to the range [0, π].
            phi_d - Radians::PI
        } else {
            phi_d
        };

        let index = ((phi_d / Radians::PI) * MerlBrdfParameterisation::RES_PHI_D as f32).round() as i32;
        index.clamp(0, MerlBrdfParameterisation::RES_PHI_D as i32 - 1) as usize
    }

    /// Lookup the index of the zenith angle for the half-vector.
    pub fn theta_d_index(&self, theta_d: Radians) -> usize {
        assert!(theta_d >= Radians::ZERO && theta_d <= Radians::HALF_PI);
        let index = ((theta_d * 2.0 / Radians::PI) * MerlBrdfParameterisation::RES_THETA_D as f32).round() as i32;
        index.clamp(0, MerlBrdfParameterisation::RES_THETA_D as i32 - 1) as usize
    }

    /// Lookup the index of the zenith angle for the difference vector.
    ///
    /// The mapping is not linear.
    pub fn theta_h_index(&self, theta_h: Radians) -> usize {
        assert!(theta_h >= Radians::ZERO && theta_h <= Radians::HALF_PI);
        let theta_h_deg = theta_h * 0.5 / Radians::PI * BRDF_THETA_H as f32;
        let index = (theta_h_deg * MerlBrdfParameterisation::RES_THETA_H as f32).sqrt().round() as i32;
        index.clamp(0, MerlBrdfParameterisation::RES_THETA_H as i32 - 1) as usize
    }

    #[cfg(feature = "io")]
    pub fn load<P: AsRef<Path>>(filepath: P) -> Result<Self, VgonioError> {
        use std::{fs::File, io::Read};

        use base::medium::Medium;

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
            file.read_exact(&mut buf)
                .map_err(|err| VgonioError::from_io_error(err, "Can't read MERL BRDF dimensions!"))
                .unwrap();
            let n = u32::from_le_bytes(buf);
            count *= n;
        }

        if count != MerlBrdfParameterisation::RES_TOTAL {
            return Err(VgonioError::new(
                format!(
                    "Can't read MERL BRDF from {:?}: invalid dimensions (expecting {}, actual {})!",
                    filepath.as_ref(),
                    MerlBrdfParameterisation::RES_TOTAL,
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
        // following the MERL BRDF layout then apply the dimension permutation.
        let mut samples = DyArr::zeros([
            MerlBrdfParameterisation::RES_THETA_H as usize,
            MerlBrdfParameterisation::RES_THETA_D as usize,
            MerlBrdfParameterisation::RES_PHI_D as usize,
            3,
        ]);

        let stride_c = MerlBrdfParameterisation::RES_TOTAL as usize;
        let stride_th = MerlBrdfParameterisation::RES_THETA_D as usize
            * MerlBrdfParameterisation::RES_PHI_D as usize;
        let stride_td = MerlBrdfParameterisation::RES_PHI_D as usize;
        let scale = [
            MerlBrdfParameterisation::R_SCALE,
            MerlBrdfParameterisation::G_SCALE,
            MerlBrdfParameterisation::B_SCALE,
        ];
        for i in 0..MerlBrdfParameterisation::RES_THETA_H as usize {
            for j in 0..MerlBrdfParameterisation::RES_THETA_D as usize {
                for k in 0..MerlBrdfParameterisation::RES_PHI_D as usize {
                    for c in 0..3 {
                        let offset = c * stride_c + i * stride_th + j * stride_td + k;
                        samples[[i, j, k, c]] = (data[offset] * scale[c]) as f32;
                    }
                }
            }
        }

        let spectrum = DyArr::from_vec_1d(vec![nm!(435.8), nm!(546.1), nm!(700.0)]);
        let params = Box::new(MerlBrdfParameterisation::default());

        Ok(Self {
            origin: Origin::RealWorld,
            incident_medium: Medium::Air,
            transmitted_medium: medium,
            params,
            spectrum,
            samples,
        })
    }

    pub fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Merl }
}

#[cfg(feature = "fitting")]
impl AnalyticalFit for MerlBrdf {
    type Params = MerlBrdfParameterisation;

    impl_analytical_fit_trait!(self);

    fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Merl }

    fn new_analytical(
        medium_i: Medium,
        medium_t: Medium,
        spectrum: &[Nanometres],
        params: &Self::Params,
        model: &dyn Bxdf<Params = [f64; 2]>,
        iors: &RefractiveIndexRegistry,
    ) -> Self
    where
        Self: Sized,
    {
        let iors_i = iors.ior_of_spectrum(medium_i, spectrum).unwrap();
        let iors_t = iors.ior_of_spectrum(medium_t, spectrum).unwrap();
        let n_spectrum = 3;
        let mut samples = DyArr::<f32, 3>::zeros([
            MerlBrdfParameterisation::RES_THETA_H as usize,
            MerlBrdfParameterisation::RES_THETA_D as usize,
            MerlBrdfParameterisation::RES_PHI_D as usize,
            n_spectrum,
        ]);
        params.zenith_h.iter().enumerate().for_each(|(i, &theta_h)| {
            params.zenith_d.iter().enumerate().for_each(|(j, &theta_d)| {
                params.azimuth_d.iter().enumerate().for_each(|(k, &phi_d)| {
                    let wd = Sph2::new(theta_d, phi_d);
                    let d = wd.to_cartesian();
                    let wh = Sph2::new(theta_h, 0.0);
                    let (i, o) = hd2io(, d);
                });
            });
        });
    }

    fn distance(&self, other: &Self, metric: ErrorMetric, rmetric: ResidualErrorMetric) -> f64
    where
        Self: Sized,
    {
        todo!()
    }

    fn filtered_distance(
        &self,
        other: &Self,
        metric: ErrorMetric,
        rmetric: ResidualErrorMetric,
        limit: Radians,
    ) -> f64
    where
        Self: Sized,
    {
        todo!()
    }
}
