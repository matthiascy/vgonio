//! BRDF measured in the paper "Investigation and Simulation of Diffraction on
//! Rough Surfaces" by O. Clausen, Y. Chen, A. Fuhrmann and R. Marroquim.
use crate::brdf::measured::{BrdfParameterisation, MeasuredBrdf, Origin, ParametrisationKind};
#[cfg(feature = "fitting")]
use crate::{brdf::measured::AnalyticalFit, Bxdf, Scattering};
use base::{
    error::VgonioError,
    impl_measured_data_trait,
    math::Sph2,
    medium::Medium,
    units::{nm, Nanometres, Radians},
    MeasuredBrdfKind, MeasuredData, MeasurementKind,
};
#[cfg(feature = "fitting")]
use base::{math, optics::ior::RefractiveIndexRegistry, ErrorMetric, ResidualErrorMetric};
use jabr::array::DyArr;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

/// Parameterisation for the Clausen BRDF.
///
/// BRDFs measured in the paper "Investigation and Simulation of Diffraction on
/// Rough Surfaces" are in-plane BRDFs where the incident direction and the
/// outgoing direction are in the same plane. Moreover, there are no
/// measurements at the positions where the incident and outgoing directions are
/// the same.
#[derive(Clone, PartialEq, Debug)]
pub struct ClausenBrdfParameterisation {
    /// The incident directions of the BRDF.
    pub incoming: DyArr<Sph2>,
    /// The outgoing directions of the BRDF for each incident direction.
    /// Directions are stored in a 2D array with dimensions: ωi, ωo.
    pub outgoing: DyArr<Sph2, 2>,
    /// The number of outgoing directions per incident direction.
    pub n_wo: usize,
}

impl BrdfParameterisation for ClausenBrdfParameterisation {
    fn kind() -> ParametrisationKind { ParametrisationKind::IncidentDirection }
}

impl ClausenBrdfParameterisation {
    /// Create a new parameterisation for the Clausen BRDF.
    pub fn new(incoming: DyArr<Sph2>, outgoing: DyArr<Sph2, 2>) -> Self {
        let num_outgoing_per_incoming = outgoing.shape()[1];
        Self {
            incoming,
            outgoing,
            n_wo: num_outgoing_per_incoming,
        }
    }

    /// Returns an iterator over the incident directions and the corresponding
    /// outgoing directions.
    pub fn wi_wos_iter(&self) -> impl Iterator<Item = (usize, (&Sph2, &[Sph2]))> {
        self.incoming
            .as_slice()
            .iter()
            .zip(self.outgoing.as_slice().chunks(self.n_wo))
            .enumerate()
    }

    /// Returns an iterator over all possible incidents and outgoing directions.
    pub fn all_wi_wo_iter(&self) -> impl Iterator<Item = (&Sph2, &Sph2)> {
        self.incoming
            .as_slice()
            .iter()
            .zip(self.outgoing.as_slice().chunks(self.n_wo))
            .flat_map(|(wi, wos)| wos.iter().map(move |wo| (wi, wo)))
    }
}

/// In-plane BRDF measured in the paper "Investigation and Simulation of
/// Diffraction on Rough Surfaces" by O. Clausen, Y. Chen, A. Fuhrmann and
/// R. Marroquim.
///
/// BRDF samples are stored in a 3D array with dimensions: ωi, ωo, λ.
pub type ClausenBrdf = MeasuredBrdf<ClausenBrdfParameterisation, 3>;

unsafe impl Send for ClausenBrdf {}
unsafe impl Sync for ClausenBrdf {}

impl_measured_data_trait!(ClausenBrdf, Bsdf, Some(MeasuredBrdfKind::Clausen));

impl ClausenBrdf {
    /// Creates a new Clausen BRDF. The BRDF is parameterised in the incident
    /// and outgoing directions.
    pub fn new(
        origin: Origin,
        incident_medium: Medium,
        transmitted_medium: Medium,
        params: Box<ClausenBrdfParameterisation>,
        spectrum: DyArr<Nanometres>,
        samples: DyArr<f32, 3>,
    ) -> Self {
        Self {
            origin,
            incident_medium,
            transmitted_medium,
            params,
            spectrum,
            samples,
        }
    }

    /// Returns the kind of the measured BRDF.
    pub fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Clausen }

    /// Return the number of incident directions in the measured BRDF.
    pub fn n_wi(&self) -> usize { self.params.incoming.len() }

    /// Return the number of outgoing directions for each incident direction.
    pub fn n_wo(&self) -> usize { self.params.n_wo }

    /// Computes the local BRDF maximum values for each snapshot, that is, for
    /// each incident direction (per wavelength).
    pub fn compute_max_values(&self) -> DyArr<f32, 2> {
        let (n_wi, n_wo, n_spectrum) = (self.n_wi(), self.n_wo(), self.spectrum.len());
        let mut max_values = DyArr::splat(-1.0f32, [n_wi, n_spectrum]);
        for i in 0..n_wi {
            for j in 0..n_wo {
                for k in 0..n_spectrum {
                    max_values[[i, k]] = f32::max(max_values[[i, k]], self.samples[[i, j, k]]);
                }
            }
        }
        max_values
    }

    /// Loads the BRDF from a file.
    #[cfg(feature = "io")]
    pub fn load<P: AsRef<Path>>(filepath: P) -> Result<Self, VgonioError> {
        if !filepath.as_ref().exists() {
            return Err(VgonioError::new(
                format!(
                    "Can't read Clausen BRDF from {:?}: file not found!",
                    filepath.as_ref()
                ),
                None,
            ));
        }

        match filepath.as_ref().extension() {
            None => Err(VgonioError::new(
                "Can't read Clausen BRDF from files without extension!".to_string(),
                None,
            )),
            // TODO: use OsStr::display() once it's stable
            Some(ext) => match ext.to_str().unwrap() {
                "json" => Self::load_from_reader(BufReader::new(File::open(filepath).unwrap())),
                _ => Err(VgonioError::new(
                    format!(
                        "Can't read Clausen BRDF from *.{} files!",
                        ext.to_str().unwrap()
                    ),
                    None,
                )),
            },
        }
    }

    /// Loads the BRDF from a reader.
    #[cfg(feature = "io")]
    pub fn load_from_reader<R: BufRead>(reader: R) -> Result<Self, VgonioError> {
        use serde_json::Value;

        // TODO: auto detect medium from file

        let content: Value = serde_json::from_reader(reader).map_err(|err| {
            VgonioError::new(
                "Load Clausen BRDF: failed to parse JSON file".to_string(),
                Some(Box::new(err)),
            )
        })?;

        let data_array = content.as_array().unwrap();
        let spectrum = DyArr::<Nanometres>::from_vec_1d(
            data_array[0]["wavelengths"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| nm!(v.as_f64().unwrap() as f32))
                .collect::<Vec<_>>(),
        );
        let n_spectrum = spectrum.len();
        let mut unordered_samples = Vec::new();
        let mut i = 1;
        loop {
            if i >= data_array.len() {
                break;
            }
            let measurement = &data_array[i];
            let wi = {
                let phi_i = measurement["phiIn"].as_f64().unwrap() as f32;
                let theta_i = measurement["thetaIn"].as_f64().unwrap() as f32;
                Sph2::new(Radians::from_degrees(theta_i), Radians::from_degrees(phi_i))
            };
            let wo = {
                let phi_o = measurement["phiOut"].as_f64().unwrap() as f32;
                let theta_o = measurement["thetaOut"].as_f64().unwrap() as f32;
                Sph2::new(Radians::from_degrees(theta_o), Radians::from_degrees(phi_o))
            };
            let spectrum_samples = measurement["spectrum"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Box<[f32]>>();
            assert_eq!(spectrum_samples.len(), n_spectrum);
            unordered_samples.push((wi, wo, spectrum_samples));
            i += 1;
        }
        unordered_samples.sort_by(|(wi_a, wo_a, _), (wi_b, wo_b, _)| {
            // first sort by theta_i
            if wi_a.theta < wi_b.theta {
                return std::cmp::Ordering::Less;
            } else if wi_a.theta > wi_b.theta {
                return std::cmp::Ordering::Greater;
            }

            // then sort by phi_i
            if wi_a.phi < wi_b.phi {
                return std::cmp::Ordering::Less;
            } else if wi_a.phi > wi_b.phi {
                return std::cmp::Ordering::Greater;
            }

            // then sort by theta_o
            if wo_a.theta < wo_b.theta {
                return std::cmp::Ordering::Less;
            } else if wo_a.theta > wo_b.theta {
                return std::cmp::Ordering::Greater;
            }

            // then sort by phi_o
            if wo_a.phi < wo_b.phi {
                std::cmp::Ordering::Less
            } else if wo_a.phi > wo_b.phi {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });

        // TODO: remove, unordered_samples ==> outgoing
        let mut wi_wo_pairs: Vec<(Sph2, Vec<Sph2>)> = Vec::new();
        unordered_samples.iter().for_each(|(wi, wo, _)| {
            if let Some((acc_wi, acc_wos)) = wi_wo_pairs.last_mut() {
                if *acc_wi == *wi {
                    acc_wos.push(*wo);
                } else {
                    wi_wo_pairs.push((*wi, vec![*wo]));
                }
            } else {
                wi_wo_pairs.push((*wi, vec![*wo]));
            }
        });

        let n_wi = wi_wo_pairs.len();
        let n_wo = wi_wo_pairs[0].1.len();

        #[cfg(debug_assertions)]
        log::debug!(
            "Load Clausen BRDF with {} wavelengths, {} incident directions, {} outgoing directions",
            n_spectrum,
            n_wi,
            n_wo
        );

        let samples = DyArr::from_iterator(
            [n_wi as isize, n_wo as isize, n_spectrum as isize],
            unordered_samples
                .iter()
                .flat_map(|(_, _, s)| s.iter())
                .cloned(),
        );

        let outgoing = DyArr::from_iterator(
            [n_wi as isize, n_wo as isize],
            wi_wo_pairs.iter().flat_map(|(_, wos)| wos.iter().cloned()),
        );

        let params = Box::new(ClausenBrdfParameterisation {
            incoming: DyArr::from_vec_1d(wi_wo_pairs.iter().map(|(wi, _)| *wi).collect::<Vec<_>>()),
            outgoing,
            n_wo,
        });

        debug_assert_eq!(params.incoming.len(), n_wi, "Number of incident directions");
        #[cfg(debug_assertions)]
        log::debug!(
            "Load Clausen BRDF with {} wavelengths, parameters: {:?}",
            n_spectrum,
            params
        );

        Ok(Self {
            origin: Origin::RealWorld,
            incident_medium: Medium::Air,
            transmitted_medium: Medium::Aluminium,
            params,
            spectrum,
            samples,
        })
    }
}

#[cfg(feature = "fitting")]
impl AnalyticalFit for ClausenBrdf {
    type Params = ClausenBrdfParameterisation;

    impl_analytical_fit_trait!(self);

    #[inline]
    fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Clausen }

    /// Creates a new Clausen BRDF with the same parameterisation as the given
    /// BRDF, but with new analytical samples.
    fn new_analytical(
        medium_i: Medium,
        medium_t: Medium,
        spectrum: &[Nanometres],
        params: &Self::Params,
        model: &dyn Bxdf<Params = [f64; 2]>,
        iors: &RefractiveIndexRegistry,
    ) -> Self {
        let iors_i = iors.ior_of_spectrum(medium_i, spectrum).unwrap();
        let iors_t = iors.ior_of_spectrum(medium_t, spectrum).unwrap();
        let n_wo = params.n_wo;
        let n_wi = params.incoming.len();
        let n_spectrum = spectrum.len();
        let mut samples = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        params
            .incoming
            .as_slice()
            .iter()
            .enumerate()
            .zip(params.outgoing.as_slice().chunks(n_wo))
            .for_each(|((i, wi), wos)| {
                let wi = wi.to_cartesian();
                wos.iter().enumerate().for_each(|(j, wo)| {
                    let wo = wo.to_cartesian();
                    let spectral_samples =
                        Scattering::eval_reflectance_spectrum(model, &wi, &wo, &iors_i, &iors_t);
                    for (k, sample) in spectral_samples.iter().enumerate() {
                        samples[[i, j, k]] = *sample as f32;
                    }
                });
            });

        Self {
            origin: Origin::Analytical,
            incident_medium: medium_i,
            transmitted_medium: medium_t,
            params: Box::new(params.clone()),
            spectrum: DyArr::from_slice([n_spectrum], spectrum),
            samples,
        }
    }

    /// Computes the distance between the measured data and the model.
    fn distance(&self, other: &Self, metric: ErrorMetric, rmetric: ResidualErrorMetric) -> f64 {
        assert_eq!(self.spectrum(), other.spectrum(), "Spectra must be equal!");
        if self.params() != other.params() {
            panic!("Parameterization must be the same!");
        }
        let factor = match metric {
            ErrorMetric::Mse => math::rcp_f64(self.samples().len() as f64),
            ErrorMetric::Nllsq => 0.5,
        };
        match rmetric {
            ResidualErrorMetric::Identity => self
                .samples()
                .iter()
                .zip(other.samples().iter())
                .fold(0.0f64, |acc, (a, b)| {
                    let diff = *a as f64 - *b as f64;
                    acc + math::sqr(diff) * factor
                }),
            ResidualErrorMetric::JLow => {
                let n_spectrum = self.n_spectrum();
                let n_wo = self.n_wo();
                self.params
                    .wi_wos_iter()
                    .fold(0.0f64, |acc, (i, (wi, wos))| {
                        let cos_theta_i = wi.theta.cos() as f64;
                        acc + wos.iter().enumerate().fold(0.0f64, |acc, (j, wo)| {
                            let offset = i * n_wo * n_spectrum + j * n_spectrum;
                            let a = &self.samples.as_slice()[offset..offset + n_spectrum];
                            let b = &other.samples.as_slice()[offset..offset + n_spectrum];
                            acc + a.iter().zip(b.iter()).fold(0.0, |acc, (a, b)| {
                                let diff = (*a as f64 * cos_theta_i + 1.0).ln()
                                    - (*b as f64 * cos_theta_i + 1.0).ln();
                                acc + math::sqr(diff) * factor
                            })
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
    ) -> f64 {
        log::debug!("Filtering distance with limit: {}", limit.prettified());
        assert_eq!(self.spectrum(), other.spectrum(), "Spectra must be equal!");
        if self.params() != other.params() {
            panic!("Parameterization must be the same!");
        }
        let n_spectrum = self.n_spectrum();
        let n_samples = self
            .params
            .all_wi_wo_iter()
            .filter(|(wi, wo)| wi.theta < limit && wo.theta < limit)
            .count()
            * n_spectrum;
        log::debug!(
            "Number of samples: {}, total: {}",
            n_samples,
            self.samples.len()
        );
        let factor = match metric {
            ErrorMetric::Mse => math::rcp_f64(n_samples as f64),
            ErrorMetric::Nllsq => 0.5,
        };
        self.params
            .all_wi_wo_iter()
            .zip(
                self.samples
                    .as_slice()
                    .chunks(n_spectrum)
                    .zip(other.samples.as_slice().chunks(n_spectrum)),
            )
            .fold(0.0, |acc, ((wi, wo), (s1, s2))| {
                if wi.theta < limit && wo.theta < limit {
                    let mut diff = 0.0f64;
                    for (a, b) in s1.iter().zip(s2.iter()) {
                        diff += math::sqr(*a as f64 - *b as f64) * factor;
                    }
                    acc + diff
                } else {
                    acc
                }
            })
    }
}
