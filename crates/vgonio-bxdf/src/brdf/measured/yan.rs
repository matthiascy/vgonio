use crate::{
    brdf::measured::{
        BrdfParam, BrdfParamKind, BrdfSnapshot, BrdfSnapshotIterator, MeasuredBrdf, Origin,
    },
    fitting::brdf::{AnalyticalFit2, BrdfFittingProxy},
};
#[cfg(feature = "fitting")]
use crate::{
    brdf::{measured::AnalyticalFit, Bxdf},
    Scattering,
};
use base::{
    error::VgonioError,
    impl_measured_data_trait,
    math::{compute_bicubic_spline_coefficients, Sph2, Vec2, Vec3},
    medium::Medium,
    units::{deg, rad, Nanometres},
    MeasuredBrdfKind, MeasuredData, MeasurementKind,
};
#[cfg(feature = "fitting")]
use base::{math, optics::ior::IorRegistry, units::Radians, ErrorMetric, Weighting};
use jabr::array::DyArr;
use std::{cmp::Ordering, fmt::Debug, path::Path, str::FromStr};

/// Parameterisation of the BRDF simulated from the paper "Rendering Specular
/// Microgeometry with Wave Optics" by Yan et al. 2018.
#[derive(Debug, Clone, PartialEq)]
pub struct Yan2018BrdfParameterisation {
    /// Number of incident directions along the polar angle.
    pub n_zenith_i: usize,
    /// The incident directions of the BRDF. The directions are stored in
    /// spherical coordinates (theta, phi); the theta is the polar(zenith
    /// angle) and the phi is the azimuth angle; The zenith angle is in the
    /// range [0, pi] and the azimuth angle is in the range [0, 2pi]; the zenith
    /// angle increases from the top to the bottom, and inside, inside the data,
    /// the zenith angle increases first compared to the azimuth angle.
    pub incoming: DyArr<Sph2>,
    /// The outgoing directions of the BRDF. The directions are stored in
    /// spherical coordinates (theta, phi).
    pub outgoing: DyArr<Sph2>,
    /// The mapping from the pixel index to the outgoing direction index.
    /// The pixel index is the index of the pixel in the EXR file, and the
    /// outgoing direction index is the index of the outgoing direction in
    /// the BRDF. The mapping is a 2D array with the first dimension is the
    /// row index of the pixel, and the second dimension is the column index
    /// of the pixel.
    pub mapping: DyArr<u32, 2>,
    /// The width of the original EXR image.
    pub width: u32,
    /// The height of the original EXR image.
    pub height: u32,
}

impl BrdfParam for Yan2018BrdfParameterisation {
    fn kind() -> BrdfParamKind { BrdfParamKind::IncidentDirection }
}

impl Yan2018BrdfParameterisation {
    /// Returns the incoming directions in cartesian coordinates.
    pub fn incoming_cartesian(&self) -> DyArr<Vec3> {
        DyArr::from_iterator([-1], self.incoming.iter().map(|sph| sph.to_cartesian()))
    }

    ///  Returns the outgoing directions in cartesian coordinates.
    pub fn outgoing_cartesian(&self) -> DyArr<Vec3> {
        DyArr::from_iterator([-1], self.outgoing.iter().map(|sph| sph.to_cartesian()))
    }

    /// Returns the number of outgoing directions.
    pub fn n_wo(&self) -> usize { self.outgoing.len() }

    /// Returns the number of incident directions.
    pub fn n_wi(&self) -> usize { self.incoming.len() }
}

/// BRDF measured from the paper "Rendering Specular Microgeometry with Wave
/// Optics" by Yan et al. 2018.
///
/// The original generated data is in the form of an EXR file with RGBA
/// channels, which are converted from the raw measured wavelength-dependent
/// BRDF values into XYZ trisimulus values then to RGB space. Moreover, the
/// output exr file can store the measured data for only one incident direction.
///
/// To make the data usable in the VGonio framework, the following updates are
/// made:
///
/// - The measured data is stored directly without any conversion.
/// - The output exr file stores the measured data for all incident directions.
///
/// Inside the VGonio framework, the measured data is stored in the form of a
/// 3D array, where the first dimension is the incident direction, the second
/// dimension is the outgoing direction derived from the pixel coordinates, and
/// the third dimension is the wavelength.
pub type Yan2018Brdf = MeasuredBrdf<Yan2018BrdfParameterisation, 3>;

unsafe impl Send for Yan2018Brdf {}
unsafe impl Sync for Yan2018Brdf {}

impl_measured_data_trait!(Yan2018Brdf, Bsdf, Some(MeasuredBrdfKind::Yan2018));

impl Yan2018Brdf {
    /// Creates a new BRDF from the given measured data.
    pub fn new(
        incident_medium: Medium,
        transmitted_medium: Medium,
        params: Yan2018BrdfParameterisation,
        spectrum: DyArr<Nanometres>,
        samples: DyArr<f32, 3>,
    ) -> Self {
        MeasuredBrdf {
            origin: Origin::Simulated,
            incident_medium,
            transmitted_medium,
            params: Box::new(params),
            spectrum,
            samples,
        }
    }

    pub fn snapshots(&self) -> BrdfSnapshotIterator<'_, Yan2018BrdfParameterisation, 3> {
        BrdfSnapshotIterator {
            brdf: self,
            n_spectrum: self.spectrum.len(),
            n_incoming: self.params.n_wi(),
            n_outgoing: self.params.n_wo(),
            index: 0,
        }
    }

    /// Returns the sample values (per wavelength) of the BRDF at the given
    /// incident and outgoing directions.
    ///
    /// The exact incident direction must be provided; meanwhile, the sample
    /// values of the outgoing direction are interpolated from the recorded
    /// data using bicubic interpolation.
    pub fn sample_at(&self, wi: Sph2, wo: Sph2) -> Box<[f32]> {
        let i = self
            .params
            .incoming
            .iter()
            .position(|sph| sph == &wi)
            .unwrap_or_else(|| panic!("The incident direction {:?} is not found in the BRDF.", wi));
        let uv = Self::spherical_coord_to_uv(wo);
        self.bicubic_interpolate(i, uv)
    }

    /// Returns the sample values (per wavelength) of the BRDF with the given
    /// incident direction index and the pixel coordinates of the outgoing
    /// direction.
    pub fn sample_at_pixel_coord(&self, wi_idx: usize, r: usize, c: usize) -> Box<[f32]> {
        let n_spectrum = self.spectrum.len();
        let mut out = vec![0.0; n_spectrum];
        let n_wo = self.params.n_wo();
        let wo_idx = self.params.mapping[[r, c]];
        if wo_idx != u32::MAX {
            let offset = wi_idx * n_wo * n_spectrum + wo_idx as usize * n_spectrum;
            out.copy_from_slice(&self.samples.as_slice()[offset..offset + n_spectrum]);
        }
        out.into_boxed_slice()
    }

    /// Returns derivative of the sample values (per wavelength) respective to
    /// the vertical direction of the BRDF with the given incident direction
    /// index and the pixel coordinates of the outgoing direction.
    pub fn sample_dy_at(&self, wi_idx: usize, r: usize, c: usize) -> Box<[f32]> {
        let samples_prev = self.sample_at_pixel_coord(wi_idx, r.saturating_sub(1), c);
        let samples_next =
            self.sample_at_pixel_coord(wi_idx, (r + 1) % self.params.height as usize, c);
        let mut derivative = vec![0.0; self.spectrum.len()];
        for (i, (prev, next)) in samples_prev.iter().zip(samples_next.iter()).enumerate() {
            derivative[i] = (next - prev) / 2.0;
        }
        derivative.into_boxed_slice()
    }

    /// Returns derivative of the sample values (per wavelength) respective to
    /// the horizontal direction of the BRDF with the given incident
    /// direction index and the pixel coordinates of the outgoing direction.
    /// Note: x is the row index and y is the column index.
    pub fn sample_dx_at(&self, wi_idx: usize, r: usize, c: usize) -> Box<[f32]> {
        let samples_prev = self.sample_at_pixel_coord(wi_idx, r, c.saturating_sub(1));
        let samples_next =
            self.sample_at_pixel_coord(wi_idx, r, (c + 1) % self.params.width as usize);
        let mut derivative = vec![0.0; self.spectrum.len()];
        for (i, (prev, next)) in samples_prev.iter().zip(samples_next.iter()).enumerate() {
            derivative[i] = (next - prev) / 2.0;
        }
        derivative.into_boxed_slice()
    }

    /// Returns the derivative of the sample values (per wavelength) respective
    /// to both the vertical and horizontal directions of the BRDF with the
    /// given incident direction index and the pixel coordinates of the
    /// outgoing direction.
    pub fn sample_dxy_at(&self, wi_idx: usize, r: usize, c: usize) -> Box<[f32]> {
        let rnext = (r + 1) % self.params.height as usize;
        let cnext = (c + 1) % self.params.width as usize;
        let samples_rnext_cnext = self.sample_at_pixel_coord(wi_idx, rnext, cnext);
        let samples_rnext_c = self.sample_at_pixel_coord(wi_idx, rnext, c);
        let samples_r_cnext = self.sample_at_pixel_coord(wi_idx, r, cnext);
        let samples_r_c = self.sample_at_pixel_coord(wi_idx, r, c);
        let samples_rprev_cprev =
            self.sample_at_pixel_coord(wi_idx, r.saturating_sub(1), c.saturating_sub(1));
        let samples_rprev_c = self.sample_at_pixel_coord(wi_idx, r.saturating_sub(1), c);
        let samples_r_cprev = self.sample_at_pixel_coord(wi_idx, r, c.saturating_sub(1));
        let mut derivative = vec![0.0; self.spectrum.len()];
        for i in 0..self.spectrum.len() {
            derivative[i] = (samples_rnext_cnext[i] - samples_rnext_c[i] - samples_r_cnext[i]
                + 2.0 * samples_r_c[i]
                - samples_rprev_c[i]
                - samples_r_cprev[i]
                + samples_rprev_cprev[i])
                * 0.5;
        }
        derivative.into_boxed_slice()
    }

    /// Loads the BRDF from an EXR file.
    #[cfg(feature = "exr")]
    pub fn load_from_exr<P: AsRef<Path>>(
        filepath: P,
        incident_medium: Medium,
        transmitted_medium: Medium,
    ) -> Result<Self, VgonioError> {
        if !filepath.as_ref().exists() {
            return Err(VgonioError::new(
                format!(
                    "Can't load Yan 2018 BRDF model from {:?}: file not found.",
                    filepath.as_ref()
                ),
                None,
            ));
        }

        let extension = filepath
            .as_ref()
            .extension()
            .and_then(|s| s.to_str())
            .and_then(|s| if s == "exr" { Some(s) } else { None });

        if extension.is_none() {
            return Err(VgonioError::new(
                format!(
                    "Can't load Yan 2018 BRDF model from {:?}: invalid file extension.",
                    filepath.as_ref()
                ),
                None,
            ));
        }

        // Load the EXR file and convert the data to the BRDF format.
        let data = exr::prelude::read_all_flat_layers_from_file(&filepath).map_err(|e| {
            VgonioError::new(
                format!(
                    "Can't load Yan 2018 BRDF model from {:?}: {}",
                    filepath.as_ref(),
                    e
                ),
                None,
            )
        })?;

        // Retrieve the width and height of the EXR file, all the layers have the
        // same size.
        let w = data.attributes.display_window.size.width();
        let h = data.attributes.display_window.size.height();

        // Retrieve the wavelength data shared by all the layers of the BRDF
        // from the EXR file.
        let spectrum: DyArr<Nanometres> = DyArr::from_iterator(
            [-1],
            data.layer_data[0]
                .channel_data
                .list
                .iter()
                .map(|channel| Nanometres::from_str(&channel.name.to_string()).unwrap()),
        );

        // Compute the outgoing directions of the BRDF from pixel coordinates
        // and sort them in the order of the outgoing directions (colatitude
        // then latitude).
        // The index of the outgoing direction is the pixel index.
        // The first value stored inside the outgoing vector is the index of the
        // sorted outgoing direction(usize::MAX if the direction is invalid), and the
        // second value is the outgoing direction in spherical coordinates.
        let mut pixel_outgoing = vec![(usize::MAX, Sph2::zero()); w * h];
        for i in 0..h {
            for j in 0..w {
                let idx = i * w + j;
                let v = Vec2::new(
                    (i as f32 + 0.5) / h as f32 * 2.0 - 1.0,
                    (j as f32 + 0.5) / w as f32 * 2.0 - 1.0,
                );
                if v.length() > 1.0 {
                    continue;
                }

                let z = (1.0 - v.length_squared()).sqrt();
                let mut phi = v.y.atan2(v.x);
                if phi < 0.0 {
                    phi += 2.0 * std::f32::consts::PI;
                }
                let theta = z.acos();
                let sph = Sph2::new(rad!(theta), rad!(phi));
                pixel_outgoing[idx] = (idx, sph);
            }
        }

        pixel_outgoing.retain(|(idx, _)| *idx != usize::MAX);

        // Sort by first the colatitude then the latitude.
        pixel_outgoing.sort_by(|(_, a), (_, b)| {
            let t = a.theta.partial_cmp(&b.theta).unwrap();
            if t == Ordering::Equal {
                a.phi.partial_cmp(&b.phi).unwrap()
            } else {
                t
            }
        });

        // Extract the outgoing directions and the mapping from the pixel index
        // to the outgoing direction index.
        let mut mapping = vec![u32::MAX; h * w];
        pixel_outgoing
            .iter()
            .enumerate()
            .for_each(|(i, (idx, sph))| {
                mapping[*idx] = i as u32;
            });
        let outgoing: DyArr<Sph2> =
            DyArr::from_iterator([-1], pixel_outgoing.iter().map(|(_, sph)| *sph));

        // Read BRDF values following the order of the outgoing directions.
        let n_wi = data.layer_data.len();
        let n_wo = outgoing.len();
        let n_spectrum = spectrum.len();
        let mut samples = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        // Each layer of the EXR file corresponds to a different incident direction.
        data.layer_data.iter().enumerate().for_each(|(i, layer)| {
            for (k, channel) in layer.channel_data.list.iter().enumerate() {
                channel
                    .sample_data
                    .values_as_f32()
                    .enumerate()
                    .for_each(|(pixel, value)| {
                        let j = mapping[pixel];
                        if j == u32::MAX {
                            return;
                        }
                        samples[[i, j as usize, k]] = value;
                    });
            }
        });

        // Extract the incident directions from the layer names.
        let incoming = DyArr::from_iterator(
            [-1],
            data.layer_data.iter().map(|layer| {
                let name_text = layer.attributes.layer_name.clone().unwrap();
                let name = String::from_utf8_lossy(name_text.bytes()).into_owned();
                // Original text: θ29_00.φ0.00
                let mut split = name.split('.');
                let theta = split
                    .next()
                    .unwrap()
                    .replace("θ", "")
                    .replace("_", ".")
                    .parse::<f32>()
                    .unwrap();
                let phi = split
                    .next()
                    .unwrap()
                    .replace("φ", "")
                    .replace("_", ".")
                    .parse::<f32>()
                    .unwrap();
                Sph2::new(deg!(theta).into(), deg!(phi).into())
            }),
        );

        Ok(Self::new(
            incident_medium,
            transmitted_medium,
            Yan2018BrdfParameterisation {
                n_zenith_i: incoming
                    .iter()
                    .skip(1)
                    .position(|sph| sph.theta == incoming[0].theta)
                    .unwrap()
                    + 1,
                incoming,
                outgoing,
                mapping: DyArr::from_slice([h, w], &mapping),
                width: w as u32,
                height: h as u32,
            },
            spectrum,
            samples,
        ))
    }

    /// Converts the pixel coordinates to the outgoing direction in
    /// spherical coordinates (theta, phi).
    pub fn pixel_coord_to_spherical_coord(
        px: usize,
        py: usize,
        w: usize,
        h: usize,
    ) -> Option<Sph2> {
        let x = (px as f32 + 0.5) / w as f32 * 2.0 - 1.0;
        let y = (py as f32 + 0.5) / h as f32 * 2.0 - 1.0;

        if x * x + y * y > 1.0 {
            return None;
        }

        let z = (1.0 - x * x - y * y).sqrt();
        let phi = y.atan2(x);
        let theta = z.acos();

        if phi < 0.0 {
            Some(Sph2::new(
                rad!(theta),
                rad!(phi + 2.0 * std::f32::consts::PI),
            ))
        } else {
            Some(Sph2::new(rad!(theta), rad!(phi)))
        }
    }

    /// Converts the spherical coordinates (theta, phi) to the pixel
    /// coordinates.
    ///
    /// The returned value is not exactly the pixel coordinates
    /// in integer values but the pixel coordinates in the range [0, 1].
    /// To map back to the pixel coordinates, the returned value must be
    /// multiplied by the width and height of the image.
    pub fn spherical_coord_to_uv(sph: Sph2) -> [f32; 2] {
        let x = sph.phi.sin() * sph.theta.sin();
        let y = sph.phi.cos() * sph.theta.sin();
        let u = (x / 2.0 + 0.5).clamp(0.0, 1.0);
        let v = (y / 2.0 + 0.5).clamp(0.0, 1.0);
        [u, v]
    }

    /// Bicubic interpolation of the BRDF at the given UV coordinates.
    /// The UV coordinates are in the range [0, 1].
    ///
    /// Note: u is the horizontal coordinate and v is the vertical coordinate.
    pub fn bicubic_interpolate(&self, wi_idx: usize, uv: [f32; 2]) -> Box<[f32]> {
        let [u, v] = uv;
        let n_spectrum = self.spectrum.len();
        let x = u * self.params.width as f32;
        let y = v * self.params.height as f32;
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1) % self.params.width as usize;
        let y1 = (y0 + 1) % self.params.height as usize;
        let dx = x - x0 as f32;
        let dy = y - y0 as f32;
        let x0y0 = self.sample_at_pixel_coord(wi_idx, y0, x0);
        let x1y0 = self.sample_at_pixel_coord(wi_idx, y0, x1);
        let x0y1 = self.sample_at_pixel_coord(wi_idx, y1, x0);
        let x1y1 = self.sample_at_pixel_coord(wi_idx, y1, x1);
        let dy_x0y0 = self.sample_dy_at(wi_idx, y0, x0);
        let dy_x1y0 = self.sample_dy_at(wi_idx, y0, x1);
        let dy_x0y1 = self.sample_dy_at(wi_idx, y1, x0);
        let dy_x1y1 = self.sample_dy_at(wi_idx, y1, x1);
        let dx_x0y0 = self.sample_dx_at(wi_idx, y0, x0);
        let dx_x1y0 = self.sample_dx_at(wi_idx, y0, x1);
        let dx_x0y1 = self.sample_dx_at(wi_idx, y1, x0);
        let dx_x1y1 = self.sample_dx_at(wi_idx, y1, x1);
        let dxy_x0y0 = self.sample_dxy_at(wi_idx, y0, x0);
        let dxy_x1y0 = self.sample_dxy_at(wi_idx, y0, x1);
        let dxy_x0y1 = self.sample_dxy_at(wi_idx, y1, x0);
        let dxy_x1y1 = self.sample_dxy_at(wi_idx, y1, x1);
        let mut samples = vec![0.0; n_spectrum];

        for k in 0..n_spectrum {
            // The unknown bicubic interpolation coefficients.
            let mut alpha = [0.0; 16];
            // The values and derivatives of the BRDF at the four corners of the
            // interpolation region.
            #[rustfmt::skip]
            let x = [
                x0y0[k], x1y0[k], x0y1[k], x1y1[k],
                dx_x0y0[k], dx_x1y0[k], dx_x0y1[k], dx_x1y1[k],
                dy_x0y0[k], dy_x1y0[k], dy_x0y1[k], dy_x1y1[k],
                dxy_x0y0[k], dxy_x1y0[k], dxy_x0y1[k], dxy_x1y1[k],
            ];
            compute_bicubic_spline_coefficients(&mut alpha, &x);
            for i in 0..4 {
                for j in 0..4 {
                    samples[k] += alpha[i * 4 + j] * (dy.powi(i as i32) * dx.powi(j as i32));
                }
            }
        }

        samples.into_boxed_slice()
    }
}

#[cfg(feature = "fitting")]
impl AnalyticalFit for Yan2018Brdf {
    type Params = Yan2018BrdfParameterisation;

    impl_analytical_fit_trait!(self);

    fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Yan2018 }

    /// Creates a new analytical BRDF with the same parameterisation as the
    /// given measured BRDF.
    fn new_analytical(
        medium_i: Medium,
        medium_t: Medium,
        spectrum: &[Nanometres],
        params: &Self::Params,
        model: &dyn Bxdf<Params = [f64; 2]>,
        iors: &IorRegistry,
    ) -> Self
    where
        Self: Sized,
    {
        let iors_i = iors.ior_of_spectrum(medium_i, spectrum).unwrap();
        let iors_t = iors.ior_of_spectrum(medium_t, spectrum).unwrap();
        let n_spectrum = spectrum.len();
        let n_wi = params.n_wi();
        let n_wo = params.n_wo();
        let mut samples = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        for (i, wi_sph) in params.incoming.iter().enumerate() {
            let wi = wi_sph.to_cartesian();
            for (j, wo) in params.outgoing_cartesian().iter().enumerate() {
                let spectral_samples =
                    Scattering::eval_reflectance_spectrum(model, &wi, &wo, &iors_i, &iors_t)
                        .iter()
                        .map(|&x| x as f32)
                        .collect::<Box<_>>();
                let offset = i * n_wo * n_spectrum + j * n_spectrum;
                samples.as_mut_slice()[offset..offset + n_spectrum]
                    .copy_from_slice(&spectral_samples);
            }
        }

        Self {
            origin: Origin::Analytical,
            incident_medium: medium_i,
            transmitted_medium: medium_t,
            params: Box::new(params.clone()),
            spectrum: DyArr::from_slice([n_spectrum], spectrum),
            samples,
        }
    }

    fn distance(&self, other: &Self, metric: ErrorMetric, rmetric: Weighting) -> f64
    where
        Self: Sized,
    {
        assert_eq!(
            self.spectrum.len(),
            other.spectrum.len(),
            "The spectra must be equal."
        );
        if self.params != other.params {
            panic!("The BRDFs must have the same parameterisation.");
        }
        let factor = match metric {
            ErrorMetric::Mse => math::rcp_f64(self.samples().len() as f64),
            ErrorMetric::Nllsq => 0.5,
            ErrorMetric::L1 => todo!(),
            ErrorMetric::L2 => todo!(),
            ErrorMetric::Rmse => todo!(),
        };
        match rmetric {
            Weighting::None => self
                .samples
                .iter()
                .zip(other.samples.iter())
                .fold(0.0, |acc, (s1, s2)| {
                    acc + math::sqr(*s1 as f64 - *s2 as f64) * factor
                }),
            Weighting::LnCos => {
                self.snapshots()
                    .zip(other.snapshots())
                    .fold(0.0, |acc, (xs, ys)| {
                        let cos_theta_i = xs.wi.theta.cos() as f64;
                        acc + xs
                            .samples
                            .iter()
                            .zip(ys.samples.iter())
                            .fold(0.0, |acc, (s1, s2)| {
                                let d = (1.0 + *s1 as f64 * cos_theta_i).ln()
                                    - (1.0 + *s2 as f64 * cos_theta_i).ln();
                                acc + math::sqr(d) * factor
                            })
                    })
            },
        }
    }

    fn filtered_distance(
        &self,
        other: &Self,
        metric: ErrorMetric,
        rmetric: Weighting,
        limit: Radians,
    ) -> f64
    where
        Self: Sized,
    {
        assert_eq!(self.spectrum, other.spectrum, "The spectra must be equal.");
        if self.params != other.params {
            panic!("The BRDFs must have the same parameterisation.");
        }

        if (limit.as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6 {
            return self.distance(other, metric, rmetric);
        }
        let n_wo = self.params.n_wo();
        let n_wi_filtered = self
            .params
            .incoming
            .iter()
            .position(|sph| sph.theta >= limit)
            .unwrap_or(self.params.n_wi());
        let n_wo_filtered = self
            .params
            .outgoing
            .iter()
            .position(|sph| sph.theta >= limit)
            .unwrap_or(n_wo);
        let n_spectrum = self.spectrum.len();
        let n_samples = n_wi_filtered * n_wo_filtered * n_spectrum;

        let factor = match metric {
            ErrorMetric::Mse => math::rcp_f64(n_samples as f64),
            ErrorMetric::Nllsq => 0.5,
            ErrorMetric::L1 => todo!(),
            ErrorMetric::L2 => todo!(),
            ErrorMetric::Rmse => todo!(),
        };

        let mut dist = 0.0;
        for i in 0..n_wi_filtered {
            for j in 0..n_wo_filtered {
                for k in 0..n_spectrum {
                    let offset = i * n_wo * n_spectrum + j * n_spectrum + k;
                    let diff = self.samples[offset] as f64 - other.samples[offset] as f64;
                    dist += math::sqr(diff) * factor;
                }
            }
        }

        dist
    }
}

impl<'a> Iterator for BrdfSnapshotIterator<'a, Yan2018BrdfParameterisation, 3> {
    type Item = BrdfSnapshot<'a, f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.n_incoming {
            return None;
        }

        let snapshot = BrdfSnapshot {
            wi: self.brdf.params.incoming[self.index],
            n_spectrum: self.n_spectrum,
            samples: &self.brdf.samples.as_slice()[self.index * self.n_outgoing * self.n_spectrum
                ..(self.index + 1) * self.n_outgoing * self.n_spectrum],
        };
        self.index += 1;
        Some(snapshot)
    }
}

#[cfg(feature = "fitting")]
impl AnalyticalFit2 for Yan2018Brdf {
    fn proxy(&self, iors: &IorRegistry) -> BrdfFittingProxy<Self> { todo!() }

    fn spectrum(&self) -> &[Nanometres] { self.spectrum.as_slice() }
}
