//! BRDF from the VGonio simulator.
use crate::brdf::measured::{
    BrdfParameterisation, MeasuredBrdf, MeasuredBrdfKind, Origin, ParametrisationKind,
};
use base::{
    error::VgonioError, math::Sph2, medium::Medium, partition::SphericalPartition,
    units::Nanometres,
};
use jabr::array::DyArr;
use std::{borrow::Cow, collections::HashMap, ops::Index, path::Path};

#[derive(Clone, PartialEq)]
pub struct VgonioBrdfParameterisation {
    /// The incident directions of the BRDF.
    pub incoming: DyArr<Sph2>,
    /// The outgoing directions of the BRDF.
    pub outgoing: SphericalPartition,
}

impl BrdfParameterisation for VgonioBrdfParameterisation {
    fn kind() -> ParametrisationKind { ParametrisationKind::IncidentDirection }
}

/// BRDF from the VGonio simulator.
///
/// Sampled BRDF data has three dimensions: ωi, ωo, λ.
pub type VgonioBrdf = MeasuredBrdf<VgonioBrdfParameterisation, 3>;

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

    /// Computes local BRDF maximum values for each snapshot, i.e., for each
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
    pub fn snapshots(&self) -> BrdfSnapshotIterator {
        BrdfSnapshotIterator {
            brdf: self,
            n_spectrum: self.spectrum.len(),
            n_wi: self.params.incoming.len(),
            n_wo: self.params.outgoing.patches.len(),
            idx: 0,
        }
    }

    #[cfg(feature = "exr")]
    pub fn write_as_exr(&self, filepath: &Path, resolution: u32) -> Result<(), VgonioError> {
        use exr::prelude::*;
        let n_wi = self.params.incoming.len();
        let n_spectrum = self.spectrum.len();
        let (w, h) = (resolution as usize, resolution as usize);
        let timestamp = chrono::Local::now();

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
                    &timestamp,
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
                Encoding::FAST_LOSSLESS,
                AnyChannels {
                    list: SmallVec::from(channels),
                },
            ));
        }

        let layers = unsafe { Vec::from_raw_parts(layers.assume_init().as_mut_ptr(), n_wi, n_wi) };

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

/// BRDF snapshot at a specific incident direction.
pub struct BrdfSnapshot<'a> {
    /// The incident direction of the snapshot.
    wi: Sph2,
    // TODO: use NdArray subslice which carries the shape information.
    /// Number of wavelengths.
    n_spectrum: usize,
    // TODO: use NdArray subslice which carries the shape information.
    /// The samples of the snapshot stored in a flat row-major array with
    /// dimension: ωo, λ.
    samples: &'a [f32],
}

// TODO: once the NdArray subslice is implemented, no need to implement Index
impl Index<[usize; 2]> for BrdfSnapshot<'_> {
    type Output = f32;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        // The first index is the outgoing direction index, and the second index is the
        // wavelength index.
        &self.samples[index[0] * self.n_spectrum + index[1]]
    }
}

/// An iterator over the snapshots of the BRDF.
pub struct BrdfSnapshotIterator<'a> {
    /// The Vgonio BRDF.
    brdf: &'a VgonioBrdf,
    /// Number of wavelengths.
    n_spectrum: usize,
    /// Number of incident directions.
    n_wi: usize,
    /// Number of outgoing directions.
    n_wo: usize,
    /// The current index of the snapshot.
    idx: usize,
}

impl<'a> Iterator for BrdfSnapshotIterator<'a> {
    type Item = BrdfSnapshot<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.n_wi {
            return None;
        }
        let snapshot_range =
            self.idx * self.n_wo * self.n_spectrum..(self.idx + 1) * self.n_wo * self.n_spectrum;
        let snapshot = BrdfSnapshot {
            wi: self.brdf.params.incoming[self.idx],
            n_spectrum: self.n_spectrum,
            samples: &self.brdf.samples.as_slice()[snapshot_range],
        };
        self.idx += 1;
        Some(snapshot)
    }
}

impl<'a> ExactSizeIterator for BrdfSnapshotIterator<'a> {
    fn len(&self) -> usize { self.n_wi - self.idx }
}
