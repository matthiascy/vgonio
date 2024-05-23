//! BRDF measured in the paper "Investigation and Simulation of Diffraction on
//! Rough Surfaces" by O. Clausen, Y. Chen, A. Fuhrmann and R. Marroquim.
use crate::{
    brdf::{
        measured::{
            AnalyticalFit, BrdfParameterisation, MeasuredBrdf, MeasuredBrdfKind, Origin,
            ParametrisationKind,
        },
        Bxdf,
    },
    Scattering,
};
use base::{
    error::VgonioError,
    math::Sph2,
    medium::Medium,
    optics::ior::RefractiveIndexRegistry,
    units::{nm, Nanometres, Radians},
    MeasuredData, MeasurementKind,
};
use jabr::array::DyArr;
use std::{
    any::Any,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

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
            .flat_map(move |wi| self.outgoing.as_slice().iter().map(move |wo| (wi, wo)))
    }
}

/// In-plane BRDF measured in the paper "Investigation and Simulation of
/// Diffraction on Rough Surfaces" by O. Clausen, Y. Chen, A. Fuhrmann and
/// R. Marroquim.
///
/// BRDF samples are stored in a 3D array with dimensions: ωi, ωo, λ.
pub type ClausenBrdf = MeasuredBrdf<ClausenBrdfParameterisation, 3>;

impl MeasuredData for ClausenBrdf {
    fn kind(&self) -> MeasurementKind { MeasurementKind::Bsdf }

    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

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

    /// Computes the local BRDF maximum values for each snapshot, i.e., for each
    /// incident direction (per wavelength).
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
            None => {
                return Err(VgonioError::new(
                    format!("Can't read Clausen BRDF from files without extension!"),
                    None,
                ))
            }
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

    #[cfg(feature = "io")]
    pub fn load_from_reader<R: BufRead>(reader: R) -> Result<Self, VgonioError> {
        use serde_json::Value;

        let content: Value = serde_json::from_reader(reader).map_err(|err| {
            VgonioError::new(
                format!("Load Clausen BRDF: failed to parse JSON file",),
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
                .collect::<Vec<f32>>()
                .into_boxed_slice();
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
                return std::cmp::Ordering::Less;
            } else if wo_a.phi > wo_b.phi {
                return std::cmp::Ordering::Greater;
            } else {
                return std::cmp::Ordering::Equal;
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
            incoming: DyArr::from_vec_1d(
                unordered_samples
                    .iter()
                    .map(|(wi, _, _)| *wi)
                    .collect::<Vec<_>>(),
            ),
            outgoing,
            n_wo: 0,
        });

        Ok(Self {
            origin: Origin::RealWorld,
            incident_medium: Medium::Vacuum,
            transmitted_medium: Medium::Vacuum,
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
}
