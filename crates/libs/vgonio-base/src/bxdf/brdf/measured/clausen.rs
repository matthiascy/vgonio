//! BRDF measured in the paper "Investigation and Simulation of Diffraction on
//! Rough Surfaces" by O. Clausen, Y. Chen, A. Fuhrmann and R. Marroquim.
#[cfg(feature = "bxdf_fit")]
use crate::optics::ior::IorRegistry;
use crate::{
    bxdf::{
        brdf::measured::{BrdfParam, BrdfParamKind, MeasuredBrdf, Origin},
        BrdfProxy, OutgoingDirs, ProxySource,
    },
    error::VgonioError,
    impl_any_measured_trait,
    math::Sph2,
    units::{nm, Nanometres, Radians},
    utils::medium::Medium,
    AnyMeasured, AnyMeasuredBrdf, BrdfLevel, MeasuredBrdfKind, MeasurementKind,
};
use jabr::array::{DyArr, DynArr};
use std::{
    borrow::Cow,
    f32,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

// TODO: relayout in theta_i, phi_i, theta_o, phi_o format
/// Parameterisation for the Clausen BRDF.
///
/// BRDFs measured in the paper "Investigation and Simulation of Diffraction on
/// Rough Surfaces" are in-plane BRDFs where the incident direction and the
/// outgoing direction are in the same plane. Moreover, there are no
/// measurements at the positions where the incident and outgoing directions are
/// the same.
#[derive(Clone, PartialEq, Debug)]
pub struct ClausenBrdfParameterisation {
    /// Polar angles of the incident directions.
    pub i_thetas: DyArr<f32>,
    /// Polar angles of the outgoing directions.
    pub o_thetas: DyArr<f32>,
    /// Azimuthal angles of the incident and outgoing directions.
    pub phis: DyArr<f32>,
    /// The incident directions of the BRDF in [theta, phi] format. Values are
    /// in always sorted. Increases first by phi, then by theta.
    pub incoming: DyArr<Sph2>,
    /// The outgoing directions of the BRDF for each incident direction.
    /// Directions are stored in a 2D array with dimensions: ωi, ωo.
    pub outgoing: DyArr<Sph2, 2>,
    /// The number of outgoing directions per incident direction.
    pub n_wo: usize,
}

impl BrdfParam for ClausenBrdfParameterisation {
    fn kind() -> BrdfParamKind { BrdfParamKind::InOutDirs }
}

impl ClausenBrdfParameterisation {
    // /// Create a new parameterisation for the Clausen BRDF.
    // pub fn new(incoming: DyArr<Sph2>, outgoing: DyArr<Sph2, 2>) -> Self {
    //     let num_outgoing_per_incoming = outgoing.shape()[1];
    //     Self {
    //         incoming,
    //         outgoing,
    //         n_wo: num_outgoing_per_incoming,
    //         i_thetas: incoming.as_slice().iter().map(|wi| wi.theta).collect(),
    //         o_thetas: outgoing.as_slice().iter().map(|wo| wo.theta).collect(),
    //         phis: incoming.as_slice().iter().map(|wi| wi.phi).collect(),
    //     }
    // }

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

impl_any_measured_trait!(@single_level_brdf ClausenBrdf);

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
            kind: MeasuredBrdfKind::Clausen,
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
    #[cfg(feature = "bxdf_io")]
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
            Some(ext) => match ext.to_str().unwrap_or("") {
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
    #[cfg(feature = "bxdf_io")]
    pub fn load_from_reader<R: BufRead>(reader: R) -> Result<Self, VgonioError> {
        use crate::units::Rads;
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
        // The measured data contains only in-plane measurements, so phi_i and
        // phi_o are always 0 and pi.
        let phis = DyArr::from_slice_1d(&[Rads::ZERO.as_f32(), Rads::PI.as_f32()]);
        let mut i_thetas: Vec<f32> = vec![];
        let mut o_thetas: Vec<f32> = vec![];
        let mut unordered_samples = Vec::new();
        let mut i = 1;
        // Collect all samples into memory.
        loop {
            if i >= data_array.len() {
                break;
            }
            let measurement = &data_array[i];
            let wi = {
                let phi_i = measurement["phiIn"].as_f64().unwrap() as f32;
                let theta_i = measurement["thetaIn"].as_f64().unwrap() as f32;
                if i_thetas
                    .iter()
                    .find(|&t| (*t - theta_i).abs() < 1e-6)
                    .is_none()
                {
                    i_thetas.push(theta_i);
                    i_thetas.sort_by(|a, b| a.partial_cmp(b).unwrap());
                }
                Sph2::new(Radians::from_degrees(theta_i), Radians::from_degrees(phi_i))
            };
            let wo = {
                let phi_o = measurement["phiOut"].as_f64().unwrap() as f32;
                let theta_o = measurement["thetaOut"].as_f64().unwrap() as f32;
                if o_thetas
                    .iter()
                    .find(|&t| (*t - theta_o).abs() < 1e-6)
                    .is_none()
                {
                    o_thetas.push(theta_o);
                    o_thetas.sort_by(|a, b| a.partial_cmp(b).unwrap());
                }
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
        let i_thetas = DyArr::from_iterator([-1], i_thetas.into_iter().map(f32::to_radians));
        let o_thetas = DyArr::from_iterator([-1], o_thetas.into_iter().map(f32::to_radians));

        // Sort the samples by incident direction first, then by outgoing direction.
        // This is done by sorting the incident directions first by theta, then by phi.
        // The outgoing directions are sorted by theta, then by phi.
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

        log::debug!("i_thetas: {:?}", i_thetas);
        log::debug!("o_thetas: {:?}", o_thetas);
        log::debug!("phis: {:?}", phis);
        let params = Box::new(ClausenBrdfParameterisation {
            incoming: DyArr::from_vec_1d(wi_wo_pairs.iter().map(|(wi, _)| *wi).collect::<Vec<_>>()),
            outgoing,
            n_wo,
            i_thetas,
            o_thetas,
            phis,
        });

        debug_assert_eq!(params.incoming.len(), n_wi, "Number of incident directions");
        #[cfg(debug_assertions)]
        log::debug!(
            "Load Clausen BRDF with {} wavelengths, parameters: {:?}",
            n_spectrum,
            params
        );

        Ok(Self {
            kind: MeasuredBrdfKind::Clausen,
            origin: Origin::RealWorld,
            incident_medium: Medium::Air,
            transmitted_medium: Medium::Aluminium,
            params,
            spectrum,
            samples,
        })
    }
}

impl AnyMeasuredBrdf for ClausenBrdf {
    crate::any_measured_brdf_trait_common_impl!(ClausenBrdf, Clausen);

    fn proxy(&self, iors: &IorRegistry) -> BrdfProxy {
        let iors_i = iors
            .ior_of_spectrum(self.incident_medium, self.spectrum.as_slice())
            .unwrap()
            .into_vec();
        let iors_t = iors
            .ior_of_spectrum(self.transmitted_medium, self.spectrum.as_slice())
            .unwrap()
            .into_vec();
        let o_thetas = Cow::Borrowed(&self.params.o_thetas);
        let phis = Cow::Borrowed(&self.params.phis);
        let o_dirs = OutgoingDirs::new_grid(o_thetas, phis.clone());
        let shape = [
            self.params.i_thetas.len(),
            self.params.phis.len(),
            self.params.o_thetas.len(),
            self.params.phis.len(),
            self.n_spectrum(),
        ];
        let n_spectrum = self.n_spectrum();
        let org_strides = self.samples.strides();
        let mut resampled = DynArr::splat(f32::NAN, &shape);
        let new_strides = resampled.strides().to_owned();
        // Fill in the resampled array.
        self.params.wi_wos_iter().for_each(|(i, (wi, wo))| {
            let theta_i = wi.theta.as_f32();
            let phi_i = wi.phi.as_f32();
            for (j, wo) in wo.iter().enumerate() {
                let theta_o = wo.theta.as_f32();
                let phi_o = wo.phi.as_f32();
                let org_offset = i * org_strides[0] + j * org_strides[1];
                let theta_i_idx = self
                    .params
                    .i_thetas
                    .iter()
                    .position(|t| (*t - theta_i).abs() < 1e-6)
                    .unwrap();
                let phi_i_idx = self
                    .params
                    .phis
                    .iter()
                    .position(|p| (*p - phi_i).abs() < 1e-6)
                    .unwrap();
                let theta_o_idx = self
                    .params
                    .o_thetas
                    .iter()
                    .position(|t| (*t - theta_o).abs() < 1e-6)
                    .unwrap();
                let phi_o_idx = self
                    .params
                    .phis
                    .iter()
                    .position(|p| (*p - phi_o).abs() < 1e-6)
                    .unwrap();
                let new_offset = theta_i_idx * new_strides[0]
                    + phi_i_idx * new_strides[1]
                    + theta_o_idx * new_strides[2]
                    + phi_o_idx * new_strides[3];
                resampled.as_mut_slice()[new_offset..new_offset + n_spectrum]
                    .copy_from_slice(&self.samples.as_slice()[org_offset..org_offset + n_spectrum]);
            }
        });

        BrdfProxy {
            has_nan: true,
            source: ProxySource::Measured,
            brdf: self,
            i_thetas: Cow::Borrowed(&self.params.i_thetas),
            i_phis: Cow::Borrowed(&self.params.phis),
            o_dirs,
            resampled: Cow::Owned(resampled),
            iors_i: Cow::Owned(iors_i),
            iors_t: Cow::Owned(iors_t),
        }
    }
}
