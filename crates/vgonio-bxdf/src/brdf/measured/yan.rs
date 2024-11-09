use crate::brdf::measured::{
    BrdfParameterisation, BrdfSnapshot, BrdfSnapshotIterator, MeasuredBrdf, Origin,
    ParametrisationKind,
};
#[cfg(feature = "fitting")]
use crate::{
    brdf::{measured::AnalyticalFit, Bxdf},
    Scattering,
};
use base::{
    error::VgonioError,
    impl_measured_data_trait,
    math::{Sph2, Vec2, Vec3},
    medium::Medium,
    units::{deg, rad, Nanometres},
    MeasuredBrdfKind, MeasuredData, MeasurementKind,
};
#[cfg(feature = "fitting")]
use base::{
    math, optics::ior::RefractiveIndexRegistry, units::Radians, ErrorMetric, ResidualErrorMetric,
};
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
}

impl BrdfParameterisation for Yan2018BrdfParameterisation {
    fn kind() -> ParametrisationKind { ParametrisationKind::IncidentDirection }
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
        let mut mapping = vec![usize::MAX; w * h];
        pixel_outgoing
            .iter()
            .enumerate()
            .for_each(|(i, (idx, sph))| {
                mapping[*idx] = i;
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
                        if j == usize::MAX {
                            return;
                        }
                        samples[[i, j, k]] = value;
                    });
            }
        });

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
                n_zenith_i: n_wi,
                incoming,
                outgoing,
            },
            spectrum,
            samples,
        ))
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
        iors: &RefractiveIndexRegistry,
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

    fn distance(&self, other: &Self, metric: ErrorMetric, rmetric: ResidualErrorMetric) -> f64
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
        };
        match rmetric {
            ResidualErrorMetric::Identity => self
                .samples
                .iter()
                .zip(other.samples.iter())
                .fold(0.0, |acc, (s1, s2)| {
                    acc + math::sqr(*s1 as f64 - *s2 as f64) * factor
                }),
            ResidualErrorMetric::JLow => {
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
        rmetric: ResidualErrorMetric,
        limit: Radians,
    ) -> f64
    where
        Self: Sized,
    {
        assert_eq!(
            self.spectrum(),
            other.spectrum(),
            "The spectra must be equal."
        );
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
