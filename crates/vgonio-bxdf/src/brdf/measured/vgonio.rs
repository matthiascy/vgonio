//! BRDF from the VGonio simulator.
use crate::{
    brdf::measured::{
        BrdfParam, BrdfParamKind, BrdfSnapshot, BrdfSnapshotIterator, MeasuredBrdf, Origin,
    },
    fitting::brdf::{AnalyticalFit2, BrdfFittingProxy, OutgoingDirs, ProxySource},
};

#[cfg(feature = "fitting")]
use crate::{
    brdf::{measured::AnalyticalFit, Bxdf},
    Scattering,
};
#[cfg(feature = "fitting")]
use base::{math, optics::ior::IorRegistry, units::Radians, ErrorMetric, Weighting};

use base::{
    error::VgonioError,
    math::{Sph2, Vec3},
    medium::Medium,
    partition::SphericalPartition,
    units::Nanometres,
    MeasuredBrdfKind,
};
#[cfg(feature = "exr")]
use chrono::{DateTime, Local};

use jabr::array::{DyArr, DynArr};
use std::{borrow::Cow, collections::HashMap, path::Path};

/// Parameterisation of the VGonio BRDF.
#[derive(Clone, PartialEq, Debug)]
pub struct VgonioBrdfParameterisation {
    /// Number of incident directions along the polar angle.
    pub n_zenith_i: usize,
    /// The incident directions of the BRDF. The directions are stored in
    /// spherical coordinates, i.e. azimuthal and zenith angles; the azimuthal
    /// angle is in the range `[0, 2π]` and the zenith angle is in the range
    /// `[0, π/2]`; the zenith angle increases first.
    pub incoming: DyArr<Sph2>,
    /// The outgoing directions of the BRDF.
    pub outgoing: SphericalPartition,
}

impl BrdfParam for VgonioBrdfParameterisation {
    fn kind() -> BrdfParamKind { BrdfParamKind::IncidentDirection }
}

impl VgonioBrdfParameterisation {
    /// Returns the incoming directions in cartesian coordinates.
    pub fn incoming_cartesian(&self) -> DyArr<Vec3> {
        DyArr::from_iterator([-1], self.incoming.iter().map(|sph| sph.to_cartesian()))
    }

    /// Returns the outgoing directions in cartesian coordinates.
    pub fn outgoing_cartesian(&self) -> DyArr<Vec3> {
        DyArr::from_iterator(
            [-1],
            self.outgoing
                .patches
                .iter()
                .map(|patch| patch.center().to_cartesian()),
        )
    }

    /// Returns the outgoing directions in spherical coordinates.
    pub fn outgoing_spherical(&self) -> DyArr<Sph2> {
        DyArr::from_iterator(
            [-1],
            self.outgoing.patches.iter().map(|patch| patch.center()),
        )
    }

    /// Returns the number of outgoing directions.
    pub fn n_wo(&self) -> usize { self.outgoing.patches.len() }

    /// Returns the number of incoming directions.
    pub fn n_wi(&self) -> usize { self.incoming.len() }

    /// Returns the number of incident directions along the zenith angle.
    pub fn n_wi_zenith(&self) -> usize { self.n_zenith_i }

    /// Returns the number of incident directions along the azimuthal angle.
    pub fn n_wi_azimuth(&self) -> usize { self.incoming.len() / self.n_zenith_i }
}

/// BRDF from the VGonio simulator.
///
/// Sampled BRDF data has three dimensions: ωi, ωo, λ.
/// NOTO: the actual dimensions are [n_phi_i, n_theta_i, n_theta_o, n_phi_o,
/// n_spectrum]. TODO: decompose the ωi into θi and φi.
pub type VgonioBrdf = MeasuredBrdf<VgonioBrdfParameterisation, 3>;

unsafe impl Send for VgonioBrdf {}
unsafe impl Sync for VgonioBrdf {}

// TODO: extract the common code to save data carried on the hemisphere to exr.

impl VgonioBrdf {
    /// Creates a new VGonio BRDF. The BRDF is parameterised in the incident
    /// and outgoing directions.
    pub fn new(
        origin: Origin,
        incident_medium: Medium,
        transmitted_medium: Medium,
        params: VgonioBrdfParameterisation,
        spectrum: DyArr<Nanometres>,
        samples: DyArr<f32, 3>,
    ) -> Self {
        Self {
            origin,
            incident_medium,
            transmitted_medium,
            params: Box::new(params),
            spectrum,
            samples,
        }
    }

    /// Returns the kind of the measured BRDF.
    pub fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Vgonio }

    /// Computes local BRDF maximum values for each snapshot, that is, for each
    /// incident direction (per wavelength).
    pub fn compute_max_values(&self) -> DyArr<f32, 2> {
        let n_wi = self.params.incoming.len();
        let n_spectrum = self.spectrum.len();
        let mut max_values = DyArr::<f32, 2>::zeros([n_wi, n_spectrum]);
        for (i, snapshot) in self.snapshots().enumerate() {
            snapshot
                .samples
                .chunks(n_spectrum)
                .for_each(|spectrum_samples| {
                    spectrum_samples.iter().enumerate().for_each(|(j, &value)| {
                        max_values[[i, j]] = max_values[[i, j]].max(value);
                    });
                });
        }
        max_values
    }

    /// Returns an iterator over the snapshots of the BRDF.
    pub fn snapshots(&self) -> BrdfSnapshotIterator<'_, VgonioBrdfParameterisation, 3> {
        BrdfSnapshotIterator {
            brdf: self,
            n_spectrum: self.spectrum.len(),
            n_incoming: self.params.incoming.len(),
            n_outgoing: self.params.outgoing.patches.len(),
            index: 0,
        }
    }

    #[cfg(feature = "exr")]
    pub fn write_as_exr(
        &self,
        filepath: &Path,
        timestamp: &DateTime<Local>,
        resolution: u32,
    ) -> Result<(), VgonioError> {
        use exr::prelude::*;
        let n_wi = self.params.incoming.len();
        let n_spectrum = self.spectrum.len();
        let (w, h) = (resolution as usize, resolution as usize);

        // The BSDF data is stored in a single flat row-major array, with the order of
        // the dimensions [wi, λ, y, x] where x is the width, y is the height, λ is the
        // wavelength, and wi is the incident direction.
        let mut bsdf_images = DyArr::<f32, 4>::zeros([n_wi, n_spectrum, w, h]);
        // Pre-compute the patch index for each pixel.
        let mut patch_indices = vec![0i32; w * h].into_boxed_slice();
        self.params.outgoing.compute_pixel_patch_indices(
            resolution,
            resolution,
            &mut patch_indices,
        );

        // Each snapshot is saved as a separate layer of the image.
        let mut layers = Box::new_uninit_slice(n_wi);
        // Each channel of the layer stores the BSDF data for a single wavelength.
        for (l, snapshot) in self.snapshots().enumerate() {
            let offset = l * n_spectrum * w * h;
            for i in 0..w {
                for j in 0..h {
                    let idx = patch_indices[i + j * w];
                    if idx < 0 {
                        continue;
                    }
                    for k in 0..n_spectrum {
                        bsdf_images[offset + k * w * h + j * w + i] = snapshot[[idx as usize, k]];
                    }
                }
            }
        }

        for (l, snapshot) in self.snapshots().enumerate() {
            let theta =
                format!("{:4.2}", snapshot.wi.theta.in_degrees().as_f32()).replace(".", "_");
            let phi = format!("{:4.2}", snapshot.wi.phi.in_degrees().as_f32()).replace(".", "_");
            let layer_attrib = LayerAttributes {
                owner: Text::new_or_none("vgonio"),
                capture_date: Text::new_or_none(base::utils::iso_timestamp_from_datetime(
                    timestamp,
                )),
                software_name: Text::new_or_none("vgonio"),
                other: HashMap::new(), // TODO: encode more info: self.params.to_exr_extra_info(),
                layer_name: Some(Text::new_or_panic(format!("θ{}.φ{}", theta, phi))),
                ..LayerAttributes::default()
            };
            let offset = l * w * h * n_spectrum;
            let channels = self
                .spectrum
                .as_slice()
                .iter()
                .enumerate()
                .map(|(i, wavelength)| {
                    let name = Text::new_or_panic(format!("{}", wavelength));
                    AnyChannel::new(
                        name,
                        FlatSamples::F32(Cow::Borrowed(
                            &bsdf_images.as_slice()[offset + i * w * h..offset + (i + 1) * w * h],
                        )),
                    )
                })
                .collect::<Vec<_>>();
            layers[l].write(Layer::new(
                (w, h),
                layer_attrib,
                Encoding::SMALL_LOSSLESS,
                AnyChannels {
                    list: SmallVec::from(channels),
                },
            ));
        }

        let layers = unsafe { layers.assume_init().into_vec() };
        let img_attrib = ImageAttributes::new(IntegerBounds::new((0, 0), (w, h)));
        let image = Image::from_layers(img_attrib, layers);
        image.write().to_file(filepath).map_err(|err| {
            VgonioError::new(
                format!(
                    "Failed to write BSDF measurement data to image file: {}",
                    err
                ),
                Some(Box::new(err)),
            )
        })?;

        Ok(())
    }
}

#[cfg(feature = "fitting")]
impl AnalyticalFit for VgonioBrdf {
    type Params = VgonioBrdfParameterisation;

    impl_analytical_fit_trait!(self);

    #[inline]
    fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::Vgonio }

    /// Creates a new VGonio BRDF with the same parameterisation as the given
    /// BRDF except the BRDF samples are generated from an analytical model.
    fn new_analytical(
        medium_i: Medium,
        medium_t: Medium,
        spectrum: &[Nanometres],
        params: &Self::Params,
        model: &dyn Bxdf<Params = [f64; 2]>,
        iors: &IorRegistry,
    ) -> Self {
        let iors_i = iors.ior_of_spectrum(medium_i, spectrum).unwrap();
        let iors_t = iors.ior_of_spectrum(medium_t, spectrum).unwrap();
        let n_spectrum = spectrum.len();
        let n_wi = params.n_wi();
        let n_wo = params.n_wo();
        let mut samples = DyArr::<f32, 3>::zeros([n_wi, n_wo, n_spectrum]);
        for (i, wi) in params.incoming.iter().enumerate() {
            let vi = wi.to_cartesian();
            for (j, vo) in params.outgoing_cartesian().iter().enumerate() {
                let spectral_samples =
                    Scattering::eval_reflectance_spectrum(model, &vi, &vo, &iors_i, &iors_t)
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

    /// Returns the distance between two VGonio BRDFs.
    fn distance(&self, other: &Self, metric: ErrorMetric, rmetric: Weighting) -> f64 {
        log::debug!("--distance");
        assert_eq!(self.spectrum, other.spectrum, "Spectra must be equal!");
        if self.params() != other.params() {
            panic!("Parameterization must be the same!");
        }
        let factor = match metric {
            ErrorMetric::Mse => math::rcp_f64(self.samples().len() as f64),
            ErrorMetric::Nllsq => 0.5,
            ErrorMetric::L1 => todo!(),
            ErrorMetric::L2 => todo!(),
            ErrorMetric::Rmse => todo!(),
        };
        log::debug!("factor@1: {}", factor);
        let sum = match rmetric {
            Weighting::None => {
                self.samples()
                    .iter()
                    .zip(other.samples().iter())
                    .fold(0.0f64, |acc, (a, b)| {
                        let diff = *a as f64 - *b as f64;
                        acc + math::sqr(diff) * factor
                    })
            },
            Weighting::LnCos => {
                self.snapshots()
                    .zip(other.snapshots())
                    .fold(0.0f64, |acc, (xs, ys)| {
                        let cos_theta_i = xs.wi.theta.cos() as f64;
                        acc + xs.samples.iter().zip(ys.samples.iter()).fold(
                            0.0f64,
                            |acc, (a, b)| {
                                let diff = (*a as f64 * cos_theta_i + 1.0).ln()
                                    - (*b as f64 * cos_theta_i + 1.0).ln();
                                acc + math::sqr(diff) * factor
                            },
                        )
                    })
            },
        };
        log::debug!("sum@1: {}", sum);
        sum
    }

    /// Returns the distance between two VGonio BRDFs considering only the
    /// samples that are lower than the given limit.
    fn filtered_distance(
        &self,
        other: &Self,
        metric: ErrorMetric,
        rmetric: Weighting,
        limit: Radians,
    ) -> f64 {
        assert_eq!(self.spectrum, other.spectrum, "Spectra must be equal!");
        if self.params != other.params {
            panic!("Parameterization must be the same!");
        }
        if (limit.as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6 {
            return self.distance(other, metric, rmetric);
        }
        let n_wo = self.params.n_wo();
        let wi_filtered = self
            .params
            .incoming
            .iter()
            .enumerate()
            .filter_map(|(i, sph)| if sph.theta < limit { Some(i) } else { None })
            .collect::<Box<_>>();
        let n_wi_filtered = wi_filtered.len();
        let n_wo_filtered = self
            .params
            .outgoing
            .patches
            .iter()
            .position(|patch| patch.center().theta >= limit)
            .unwrap_or(n_wo);
        let n_spectrum = self.spectrum.len();
        let n_samples = n_wi_filtered * n_wo_filtered * n_spectrum;

        let factor = match metric {
            ErrorMetric::Mse | ErrorMetric::Rmse => math::rcp_f64(n_samples as f64),
            ErrorMetric::Nllsq => 0.5,
            ErrorMetric::L1 => todo!(),
            ErrorMetric::L2 => todo!(),
        };

        let mut dist = 0.0;

        match rmetric {
            Weighting::None => {
                for i in wi_filtered {
                    for j in 0..n_wo_filtered {
                        for k in 0..n_spectrum {
                            let diff =
                                self.samples[[i, j, k]] as f64 - other.samples[[i, j, k]] as f64;
                            dist += math::sqr(diff) * factor;
                        }
                    }
                }
            },
            Weighting::LnCos => {
                for i in wi_filtered {
                    let cos_theta_i = self.params.incoming[i].theta.cos() as f64;
                    for j in 0..n_wo_filtered {
                        for k in 0..n_spectrum {
                            let a = self.samples[[i, j, k]] as f64;
                            let b = other.samples[[i, j, k]] as f64;
                            let diff = (a * cos_theta_i + 1.0).ln() - (b * cos_theta_i + 1.0).ln();
                            dist += math::sqr(diff) * factor;
                        }
                    }
                }
            },
        }

        if metric == ErrorMetric::Rmse {
            dist.sqrt()
        } else {
            dist
        }
    }
}

impl<'a> Iterator for BrdfSnapshotIterator<'a, VgonioBrdfParameterisation, 3> {
    type Item = BrdfSnapshot<'a, f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.n_incoming {
            return None;
        }
        let snapshot_range = self.index * self.n_outgoing * self.n_spectrum
            ..(self.index + 1) * self.n_outgoing * self.n_spectrum;
        let snapshot = BrdfSnapshot {
            wi: self.brdf.params.incoming[self.index],
            n_spectrum: self.n_spectrum,
            samples: &self.brdf.samples.as_slice()[snapshot_range],
        };
        self.index += 1;
        Some(snapshot)
    }
}

#[cfg(feature = "fitting")]
impl AnalyticalFit2 for VgonioBrdf {
    fn proxy(&self, iors: &IorRegistry) -> BrdfFittingProxy<Self> {
        let iors_i = iors
            .ior_of_spectrum(self.incident_medium, self.spectrum.as_slice())
            .unwrap()
            .into_vec();
        let iors_t = iors
            .ior_of_spectrum(self.transmitted_medium, self.spectrum.as_slice())
            .unwrap()
            .into_vec();

        let n_theta_i = self.params.n_wi_zenith();
        let n_phi_i = self.params.n_wi_azimuth();
        let i_thetas = DyArr::from_iterator(
            [-1],
            self.params
                .incoming
                .iter()
                .take(n_theta_i)
                .map(|sph| sph.theta.as_f32()),
        );
        let i_phis = DyArr::from_iterator(
            [-1],
            self.params
                .incoming
                .iter()
                .step_by(n_theta_i)
                .map(|sph| sph.phi.as_f32()),
        );
        let theta_o = DyArr::from_iterator(
            [-1],
            self.params
                .outgoing
                .rings
                .iter()
                .map(|ring| ring.zenith_center().as_f32()),
        );
        let n_theta_o = theta_o.len();
        let phi_o = DyArr::from_iterator(
            [-1],
            self.params
                .outgoing
                .patches
                .iter()
                .map(|patch| patch.center().phi.as_f32()),
        );
        let n_wo = self.params.outgoing.patches.len();
        let mut offsets = vec![0; n_theta_o + 1].into_boxed_slice();
        for (i, ring) in self.params.outgoing.rings.iter().enumerate() {
            offsets[i + 1] = ring.base_index + ring.patch_count;
        }
        let n_spectrum = self.spectrum.len();
        let o_dirs = OutgoingDirs::new_list(
            Cow::Owned(theta_o),
            Cow::Owned(phi_o),
            DyArr::from_boxed_slice([n_theta_o + 1], offsets),
        );
        // The original dimensions are [n_phi_i, n_theta_i, n_theta_o, n_phi_o,
        // n_spectrum] the target dimensions are [n_theta_i, n_phi_i, n_theta_o,
        // n_phi_o, n_spectrum].
        let mut resampled = DynArr::zeros(&[n_theta_i, n_phi_i, n_wo, n_spectrum]);
        let &[stride_theta_i, stride_phi_i, ..] = resampled.strides() else {
            panic!("Invalid strides!");
        };
        let org_n_theta_i = self.params.n_wi_zenith();
        let org_n_wo = self.params.n_wo();
        let org_slice = self.samples.as_slice();
        for theta_i in 0..n_theta_i {
            for phi_i in 0..n_phi_i {
                let new_offset = theta_i * stride_theta_i + phi_i * stride_phi_i;
                let org_offset =
                    phi_i * org_n_theta_i * org_n_wo * n_spectrum + theta_i * org_n_wo * n_spectrum;
                resampled.as_mut_slice()[new_offset..new_offset + n_wo * n_spectrum]
                    .copy_from_slice(&org_slice[org_offset..org_offset + n_wo * n_spectrum]);
            }
        }
        BrdfFittingProxy {
            has_nan: false,
            source: ProxySource::Measured,
            brdf: self,
            i_thetas: Cow::Owned(i_thetas),
            i_phis: Cow::Owned(i_phis),
            o_dirs: o_dirs,
            resampled: Cow::Owned(resampled),
            iors_i: Cow::Owned(iors_i),
            iors_t: Cow::Owned(iors_t),
        }
    }

    fn spectrum(&self) -> &[Nanometres] { self.spectrum.as_slice() }
}
