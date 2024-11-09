use crate::measure::{
    bsdf::{MeasuredBrdfLevel, MeasuredBsdfData},
    mfd::{MeasuredGafData, MeasuredNdfData},
    params::NdfMeasurementMode,
    DataCarriedOnHemisphereSampler,
};
use base::{
    math::Sph2,
    medium::Medium,
    optics::ior::RefractiveIndexRegistry,
    range::{RangeByStepSizeExclusive, RangeByStepSizeInclusive},
    units::{rad, Degrees, Nanometres, Radians, Rads},
    MeasuredBrdfKind, MeasuredData,
};
use bxdf::{
    brdf::{
        analytical::microfacet::{BeckmannBrdf, TrowbridgeReitzBrdf},
        measured::{AnalyticalFit, ClausenBrdf, VgonioBrdf},
    },
    Scattering,
};
use exr::{
    image::{FlatImage, FlatSamples},
    prelude::Text,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyList};
use surf::MicroSurface;

pub fn plot_err(errs: &[f64], alpha_start: f64, alpha_stop: f64, alpha_step: f64) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_err")?
                .into();
        let errs = PyList::new_bound(py, errs);
        let args = (alpha_start, alpha_stop, alpha_step, errs);
        fun.call1(py, args)?;
        Ok(())
    })
}

/// Plot the comparison between two the VgonioBrdf and ClausenBrdf.
///
/// # Arguments
///
/// * `itrp` - The VgonioBrdf to plot, which is the interpolated BRDF to match
///   the ClausenBrdf.
/// * `meas` - The ClausenBrdf to plot, which is the measured BRDF.
/// * `dense` - Whether to sample 4x more points than the original data.
pub fn plot_brdf_vgonio_clausen(
    itrp: &ClausenBrdf,
    meas: &ClausenBrdf,
    dense: bool,
) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_brdf_comparison")?
                .into();
        let n_samples_itrp = itrp.n_samples();
        let n_samples_olaf = meas.n_samples();
        // Rearrange the data layout from [wi, wo, spectrum] to [spectrum, wi, wo]
        let mut data_itrp = vec![0.0; n_samples_itrp];
        let mut data_olaf = vec![0.0; n_samples_olaf];
        let n_wi = itrp.n_wi();
        let n_wo_itrp = itrp.n_wo();
        let n_wo_olaf = meas.n_wo();
        let n_spectrum_itrp = itrp.n_spectrum();
        let n_spectrum_olaf = meas.n_spectrum();
        for i in 0..n_wi {
            for j in 0..n_wo_itrp {
                for k in 0..n_spectrum_itrp {
                    let original_offset = i * n_wo_itrp * n_spectrum_itrp + j * n_spectrum_itrp + k;
                    let new_offset = k * n_wi * n_wo_itrp + i * n_wo_itrp + j;
                    data_itrp[new_offset] = itrp.samples[original_offset];
                }
            }
            for j in 0..n_wo_olaf {
                for k in 0..n_spectrum_olaf {
                    let original_offset = i * n_wo_olaf * n_spectrum_olaf + j * n_spectrum_olaf + k;
                    let new_offset = k * n_wi * n_wo_olaf + i * n_wo_olaf + j;
                    data_olaf[new_offset] = meas.samples[original_offset];
                }
            }
        }
        let wi_wo_pairs_olaf = meas
            .params
            .incoming
            .as_slice()
            .iter()
            .zip(meas.params.outgoing.as_slice().chunks(meas.params.n_wo))
            .map(|(wi, wos)| {
                (
                    (wi.theta.as_f32(), wi.phi.as_f32()),
                    PyList::new_bound(
                        py,
                        wos.iter()
                            .map(|wo| (wo.theta.as_f32(), wo.phi.as_f32()))
                            .collect::<Vec<_>>(),
                    ),
                )
            })
            .collect::<Vec<_>>();

        let wi_wo_pairs_itrp = itrp
            .params
            .incoming
            .as_slice()
            .iter()
            .zip(itrp.params.outgoing.as_slice().chunks(itrp.params.n_wo))
            .map(|(wi, wos)| {
                (
                    (wi.theta.as_f32(), wi.phi.as_f32()),
                    PyList::new_bound(
                        py,
                        wos.iter()
                            .map(|wo| (wo.theta.as_f32(), wo.phi.as_f32()))
                            .collect::<Vec<_>>(),
                    ),
                )
            })
            .collect::<Vec<_>>();
        let spectrum_itrp = itrp
            .spectrum
            .as_slice()
            .iter()
            .map(|x| x.as_f32())
            .collect::<Vec<_>>();
        let spectrum_olaf = meas
            .spectrum
            .as_slice()
            .iter()
            .map(|x| x.as_f32())
            .collect::<Vec<_>>();
        let max_itrp = itrp.compute_max_values().into_vec();
        let max_olaf = meas.compute_max_values().into_vec();
        let args = (
            n_wi,
            dense,
            n_wo_itrp,
            wi_wo_pairs_itrp,
            n_wo_olaf,
            wi_wo_pairs_olaf,
            data_itrp,
            spectrum_itrp,
            // n_lambda_itrp,
            max_itrp,
            data_olaf,
            spectrum_olaf,
            // n_lambda_olaf,
            max_olaf,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}

/// Plot the BRDF slice with the given incoming direction and outgoing
/// azimuthal angle.
///
/// # Arguments
///
/// * `brdf` - The measured BRDF to plot.
/// * `wi` - The incoming direction of the slice.
/// * `phi` - The azimuthal angle of the slice.
/// * `spectrum` - The spectrum of the BRDF.
pub fn plot_brdf_slice(
    brdf: &[(&VgonioBrdf, String)],
    wi: Sph2,
    phi_o: Radians,
    legend: bool,
    cmap: String,
    scale: f32,
    log: bool,
) -> PyResult<()> {
    let opposite_phi_o = (phi_o + Radians::PI).wrap_to_tau();
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_brdf_slice")?
                .into();
        let brdfs = PyList::new_bound(
            py,
            brdf.iter()
                .map(|(brdf, label)| {
                    let sampler = DataCarriedOnHemisphereSampler::new(*brdf).unwrap();
                    let theta = brdf
                        .params
                        .outgoing
                        .rings
                        .iter()
                        .map(|x| x.zenith_center().to_degrees().as_f32())
                        .collect::<Vec<_>>();
                    let n_spectrum = brdf.spectrum.len();
                    let spectrum =
                        PyArray1::from_iter_bound(py, brdf.spectrum.iter().map(|x| x.as_f32()));
                    let slice_phi = {
                        let data = sampler.sample_slice_at(wi, phi_o).unwrap();
                        let slice = PyArray2::zeros_bound(py, [theta.len(), n_spectrum], false);
                        unsafe {
                            slice.as_slice_mut().unwrap().copy_from_slice(&data);
                        }
                        slice
                    };
                    let slice_opposite_phi = {
                        let data = sampler
                            .sample_slice_at(wi, opposite_phi_o)
                            .unwrap()
                            .into_vec();
                        let slice = PyArray2::zeros_bound(py, [theta.len(), n_spectrum], false);
                        unsafe {
                            slice.as_slice_mut().unwrap().copy_from_slice(&data);
                        }
                        slice
                    };
                    (
                        slice_phi,
                        slice_opposite_phi,
                        PyArray1::from_vec_bound(py, theta),
                        spectrum,
                        label,
                    )
                })
                .collect::<Vec<_>>(),
        );
        let args = (
            phi_o.to_degrees().as_f32(),
            opposite_phi_o.to_degrees().as_f32(),
            brdfs,
            legend,
            cmap,
            scale,
            log,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}

/// Plot the BRDF slices on the plane with the given incoming azimuthal angle.
pub fn plot_brdf_slice_in_plane(brdf: &[&VgonioBrdf], phi: Radians) -> PyResult<()> {
    let phi_opp = (phi + Radians::PI).wrap_to_tau();
    log::debug!(
        "phi: {}, phi_opp: {}",
        phi.prettified(),
        phi_opp.prettified()
    );
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_brdf_slice_in_plane")?
                .into();
        let brdfs = PyList::new_bound(
            py,
            brdf.iter()
                .map(|&brdf| {
                    let sampler = DataCarriedOnHemisphereSampler::new(brdf).unwrap();
                    let mut theta_i = brdf
                        .params
                        .incoming
                        .iter()
                        .take(brdf.params.n_zenith_i)
                        .map(|x| x.theta.as_f32())
                        .collect::<Vec<_>>();
                    log::debug!("theta_i: {:?}", theta_i);
                    let theta_o = brdf
                        .params
                        .outgoing
                        .rings
                        .iter()
                        .map(|x| x.zenith_center().to_degrees().as_f32())
                        .collect::<Vec<_>>();
                    let wavelength = PyArray1::from_vec_bound(
                        py,
                        brdf.spectrum.iter().map(|x| x.as_f32()).collect::<Vec<_>>(),
                    );
                    let (slices_phi, slices_phi_opp): (Vec<Vec<_>>, Vec<Vec<_>>) = theta_i
                        .iter()
                        .map(|&theta_i| {
                            let wi = Sph2::new(Rads::new(theta_i), phi);
                            let slice_phi = sampler.sample_slice_at(wi, phi).unwrap().into_vec();
                            let slice_phi_opp =
                                sampler.sample_slice_at(wi, phi_opp).unwrap().into_vec();
                            (slice_phi, slice_phi_opp)
                        })
                        .unzip();
                    for theta in theta_i.iter_mut() {
                        *theta = theta.to_degrees();
                    }
                    (slices_phi, slices_phi_opp, theta_i, theta_o, wavelength)
                })
                .collect::<Vec<_>>(),
        );
        let args = (
            phi.to_degrees().as_f32(),
            phi_opp.to_degrees().as_f32(),
            brdfs,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}

pub fn plot_ndf(
    ndf: &[&MeasuredNdfData],
    phi: Radians,
    labels: Vec<String>,
    ylim: Option<f32>,
) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_ndf_slice")?
                .into();
        let slices = PyList::new_bound(
            py,
            ndf.iter()
                .enumerate()
                .map(|(i, &ndf)| {
                    let label = labels
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("NDF {}", i));
                    match ndf.params.mode {
                        NdfMeasurementMode::ByPoints { zenith, .. } => {
                            let theta = zenith
                                .values_wrapped()
                                .map(|x| x.as_f32())
                                .collect::<Vec<_>>();
                            let (spls, opp_spls) = ndf.slice_at(phi);
                            (
                                label,
                                numpy::PyArray1::from_vec_bound(py, theta),
                                numpy::PyArray1::from_slice_bound(py, spls),
                                numpy::PyArray1::from_slice_bound(py, opp_spls.unwrap()),
                            )
                        },
                        NdfMeasurementMode::ByPartition { .. } => {
                            let sampler = DataCarriedOnHemisphereSampler::new(ndf).unwrap();
                            let partition = sampler.extra.as_ref().unwrap();
                            let theta =
                                numpy::PyArray1::zeros_bound(py, [partition.n_rings()], false);
                            partition
                                .rings
                                .iter()
                                .enumerate()
                                .for_each(|(i, ring)| unsafe {
                                    theta.as_slice_mut().unwrap()[i] = ring.zenith_center().as_f32()
                                });
                            let phi_opp = (phi + Radians::PI).wrap_to_tau();
                            let slice_phi = numpy::PyArray1::from_vec_bound(
                                py,
                                sampler.sample_slice_at(phi).into_vec(),
                            );
                            let slice_phi_opp = numpy::PyArray1::from_vec_bound(
                                py,
                                sampler.sample_slice_at(phi_opp).into_vec(),
                            );
                            (label, theta, slice_phi, slice_phi_opp)
                        },
                    }
                })
                .collect::<Vec<_>>(),
        );
        let args = (
            phi.as_f32(),
            (phi + Radians::PI).wrap_to_tau().as_f32(),
            slices,
            ylim,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}

pub fn plot_gaf(
    gaf: &[&MeasuredGafData],
    wm: Sph2,
    phi_v: Radians,
    labels: Vec<String>,
    save: Option<String>,
) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_gaf_slice")?
                .into();
        let slices = PyList::new_bound(
            py,
            gaf.iter()
                .enumerate()
                .map(|(i, &gaf)| {
                    let label = labels
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("GAF {}", i));
                    let theta = gaf
                        .params
                        .zenith
                        .values_wrapped()
                        .map(|x| x.as_f32())
                        .collect::<Vec<_>>();

                    let (spls, opp_spls) = gaf.slice_at(wm.phi, wm.theta, phi_v);
                    (
                        label,
                        PyArray1::from_vec_bound(py, theta),
                        PyArray1::from_slice_bound(py, spls),
                        PyArray1::from_slice_bound(py, opp_spls.unwrap()),
                    )
                })
                .collect::<Vec<_>>(),
        );

        let args = (
            wm.theta.as_f32(),
            wm.phi.as_f32(),
            phi_v.as_f32(),
            (phi_v + Radians::PI).wrap_to_tau().as_f32(),
            slices,
            save,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}

pub fn plot_brdf_map(
    imgs: &[(FlatImage, String)],
    theta_i: Degrees,
    phi_i: Degrees,
    lamda: Nanometres,
    cmap: String,
    cbar: bool,
    coord: bool,
    diff: bool,
    fc: String,
    pstep: Degrees,
    tstep: Degrees,
    save: Option<String>,
) -> PyResult<()> {
    let theta_i_str = format!("{:4.2}", theta_i.as_f32()).replace(".", "_");
    let phi_i_str = format!("{:4.2}", phi_i.as_f32()).replace(".", "_");
    let layer_name = Text::new_or_panic(format!("θ{}.φ{}", theta_i_str, phi_i_str));
    let channel_name = Text::new_or_panic(format!("{}", lamda));
    log::debug!("layer_name: {}, channel_name: {}", layer_name, channel_name);
    Python::with_gil(|py| {
        PyModule::from_code_bound(
            py,
            include_str!("./pyplot/tone_mapping.py"),
            "tone_mapping.py",
            "tone_mapping",
        )
        .unwrap();
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")
                .unwrap()
                .getattr("plot_brdf_map")
                .unwrap()
                .into();
        let images = PyList::new_bound(
            py,
            imgs.iter()
                .map(|(img, name)| {
                    let size = img.attributes.display_window.size;
                    let layer = img
                        .layer_data
                        .iter()
                        .find(|layer| layer.attributes.layer_name.as_ref() == Some(&layer_name))
                        .unwrap();
                    let data = layer
                        .channel_data
                        .list
                        .iter()
                        .find(|chn| chn.name == channel_name)
                        .unwrap();

                    let pixels = match &data.sample_data {
                        FlatSamples::F32(pixels) => {
                            let img_data = PyArray2::zeros_bound(py, (size.0, size.1), false);
                            unsafe {
                                img_data.as_slice_mut().unwrap().copy_from_slice(pixels);
                            }
                            img_data
                        },
                        _ => panic!("Only f32 data is supported!"),
                    };
                    let name = format!(
                        "brdf_θ{:4.2}_φ{:4.2}_λ{:.2}_{}",
                        theta_i.as_f32(),
                        phi_i.as_f32(),
                        lamda.as_f32(),
                        name,
                    );
                    (name, (size.0 as u32, size.1 as u32), pixels)
                })
                .collect::<Vec<_>>(),
        );
        let args = (
            images,
            cmap.clone(),
            cbar,
            coord,
            diff,
            fc,
            pstep.as_f32(),
            tstep.as_f32(),
            save,
        );
        fun.call1(py, args).unwrap();
        Ok(())
    })
}

pub fn plot_surfaces(surfaces: &[&MicroSurface], downsample: u32, cmap: String) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_surfaces")?
                .into();
        let surfaces = PyList::new_bound(
            py,
            surfaces
                .iter()
                .map(|&surface| {
                    let samples =
                        numpy::PyArray2::zeros_bound(py, (surface.rows, surface.cols), false);
                    unsafe {
                        samples
                            .as_slice_mut()
                            .unwrap()
                            .copy_from_slice(&surface.samples);
                    }
                    log::debug!("rows: {}, cols: {}", surface.rows, surface.cols);
                    (surface.dv, surface.du, samples, surface.name.clone())
                })
                .collect::<Vec<_>>(),
        );
        fun.call1(py, (surfaces, cmap, downsample))?;
        Ok(())
    })
}

pub fn plot_brdf_3d(
    brdf: &VgonioBrdf,
    wi: Sph2,
    wavelength: Nanometres,
    cmap: String,
    scale: f32,
) -> PyResult<()> {
    let wavelength_idx = brdf.spectrum.iter().position(|&x| x == wavelength).unwrap();
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_brdf_3d")?
                .into();
        let theta = brdf
            .params
            .outgoing
            .rings
            .iter()
            .map(|x| x.zenith_center().as_f32())
            .collect::<Vec<_>>();
        let phi = RangeByStepSizeInclusive::new(0.0, 360.0, 5.0f32)
            .values()
            .map(|x| x.to_radians())
            .collect::<Vec<_>>();
        let n_theta = theta.len();
        let n_phi = phi.len();
        let n_spectrum = brdf.spectrum.len();
        let vals = {
            let sampler = DataCarriedOnHemisphereSampler::new(brdf).unwrap();
            let data = PyArray2::zeros_bound(py, [n_theta, n_phi], false);
            unsafe {
                let mut spectrum_samples = vec![0.0; n_spectrum].into_boxed_slice();
                let slice = data.as_slice_mut().unwrap();
                for (i, t) in theta.iter().enumerate() {
                    for (j, p) in phi.iter().enumerate() {
                        sampler.sample_point_at(
                            wi,
                            Sph2::new(Rads::new(*t), Rads::new(*p)),
                            &mut spectrum_samples,
                        );
                        slice[i * n_phi + j] = spectrum_samples[wavelength_idx];
                    }
                }
            }
            data
        };
        let args = (
            wi.theta.as_f32(),
            wi.phi.as_f32(),
            PyArray1::from_vec_bound(py, theta),
            PyArray1::from_vec_bound(py, phi),
            vals,
            cmap,
            scale,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}

pub fn plot_brdf_fitting(
    brdf: &Box<dyn MeasuredData>,
    roughness: (f64, f64),
    iors: &RefractiveIndexRegistry,
) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_brdf_fitting")?
                .into();
        if let Some(kind) = brdf.brdf_kind() {
            match kind {
                MeasuredBrdfKind::Vgonio => {
                    let measured = brdf.downcast_ref::<MeasuredBsdfData>().unwrap();
                    let model_bk = BeckmannBrdf::new(roughness.0, roughness.1);
                    let model_tr = TrowbridgeReitzBrdf::new(roughness.0, roughness.1);
                    let brdf = measured.brdf_at(MeasuredBrdfLevel::L0).unwrap();
                    let iors_i = iors
                        .ior_of_spectrum(brdf.incident_medium, brdf.spectrum())
                        .unwrap();
                    let iors_t = iors
                        .ior_of_spectrum(brdf.transmitted_medium, brdf.spectrum())
                        .unwrap();
                    let n_spectrum = brdf.spectrum.len();
                    let sampler = DataCarriedOnHemisphereSampler::new(brdf).unwrap();
                    let theta_i = brdf
                        .params
                        .incoming
                        .iter()
                        .take(brdf.params.n_wi_zenith())
                        .map(|x| x.theta.as_f32())
                        .collect::<Vec<_>>();
                    let phi_i = brdf
                        .params
                        .incoming
                        .iter()
                        .take(brdf.params.n_wi_azimuth())
                        .map(|x| x.phi.as_f32())
                        .collect::<Vec<_>>();
                    let theta_o = brdf
                        .params
                        .outgoing
                        .rings
                        .iter()
                        .map(|x| x.zenith_center().as_f32())
                        .collect::<Vec<_>>();
                    let phi_o = RangeByStepSizeExclusive::new(0.0f32, 360.0, 60.0)
                        .values()
                        .collect::<Vec<_>>();
                    let n_theta_i = theta_i.len();
                    let n_phi_i = phi_i.len();
                    let n_theta_o = brdf.params.outgoing.n_rings();
                    let n_phi_o = phi_o.len();
                    // n_theta_i x n_phi_i x n_phi_o x n_theta_o x n_spectrum
                    let samples = PyArray1::zeros_bound(
                        py,
                        [n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_bk = PyArray1::zeros_bound(
                        py,
                        [n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_tr = PyArray1::zeros_bound(
                        py,
                        [n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    for (i, t_i) in theta_i.iter().enumerate() {
                        for (j, p_i) in phi_i.iter().enumerate() {
                            for (k, p_o) in phi_o.iter().enumerate() {
                                let wi = Sph2::new(rad!(*t_i), rad!(*p_i));
                                let slice = sampler.sample_slice_at(wi, rad!(*p_o)).unwrap();
                                let offset = i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                    + j * n_phi_o * n_theta_o * n_spectrum
                                    + k * n_theta_o * n_spectrum;
                                unsafe {
                                    samples.as_slice_mut().unwrap()
                                        [offset..offset + n_theta_o * n_spectrum]
                                        .copy_from_slice(&slice);
                                }
                                for (l, t_o) in theta_o.iter().enumerate() {
                                    let wo = Sph2::new(rad!(*t_o), rad!(*p_o));
                                    let spectral_samples_bk = Scattering::eval_reflectance_spectrum(
                                        &model_bk,
                                        &wi.to_cartesian(),
                                        &wo.to_cartesian(),
                                        &iors_i,
                                        &iors_t,
                                    );
                                    let spectral_samples_tr = Scattering::eval_reflectance_spectrum(
                                        &model_tr,
                                        &wi.to_cartesian(),
                                        &wo.to_cartesian(),
                                        &iors_i,
                                        &iors_t,
                                    );
                                    let offset = i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                        + j * n_phi_o * n_theta_o * n_spectrum
                                        + k * n_theta_o * n_spectrum
                                        + l * n_spectrum;
                                    unsafe {
                                        fitted_bk.as_slice_mut().unwrap()
                                            [offset..offset + n_spectrum]
                                            .copy_from_slice(&spectral_samples_bk);
                                        fitted_tr.as_slice_mut().unwrap()
                                            [offset..offset + n_spectrum]
                                            .copy_from_slice(&spectral_samples_tr);
                                    }
                                }
                            }
                        }
                    }
                    let samples =
                        samples.reshape((n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let theta_o = PyArray1::from_vec_bound(py, theta_o);
                    let theta_i = PyArray1::from_vec_bound(py, theta_i);
                    let phi_i = PyArray1::from_vec_bound(py, phi_i);
                    let wavelength = PyArray1::from_vec_bound(
                        py,
                        brdf.spectrum.iter().map(|x| x.as_f32()).collect::<Vec<_>>(),
                    );
                    let fitted_bk =
                        fitted_bk.reshape((n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let fitted_tr =
                        fitted_tr.reshape((n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    fun.call1(
                        py,
                        (
                            samples, theta_i, phi_i, theta_o, wavelength, fitted_bk, fitted_tr,
                        ),
                    )?;
                },
                MeasuredBrdfKind::Yan2018 => {
                    let brdf = brdf.downcast_ref::<ClausenBrdf>().unwrap();
                    let n_wi = brdf.params.incoming.len();
                    let n_wo = brdf.params.outgoing.len();
                },
                MeasuredBrdfKind::Clausen => {},
                _ => {
                    log::error!("The BRDF kind is unknown!");
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "The BRDF kind is unknown!",
                    ));
                },
            }
        }
        Ok(())
    })
}
