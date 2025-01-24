use core::f64;

use crate::measure::{
    mfd::{MeasuredGafData, MeasuredNdfData},
    params::NdfMeasurementMode,
    DataCarriedOnHemisphereSampler,
};
use base::{
    bxdf::{
        brdf::{
            analytical::microfacet::{MicrofacetBrdfBK, MicrofacetBrdfTR},
            measured::{
                rgl::RglBrdf, ClausenBrdf, MeasuredBrdfKind, MerlBrdf, MerlBrdfParam, VgonioBrdf,
                Yan18Brdf,
            },
            AnalyticalBrdf,
        },
        OutgoingDirs, Scattering,
    },
    math::{self, Sph2},
    optics::ior::IorRegistry,
    units::{rad, Degrees, Nanometres, Radians, Rads},
    utils::range::{StepRangeExcl, StepRangeIncl},
    AnyMeasured, AnyMeasuredBrdf, BrdfLevel, ErrorMetric, MeasurementKind, Weighting,
};
use exr::{
    image::{FlatImage, FlatSamples},
    prelude::Text,
};
use jabr::array::{DyArr, DynArr};
use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::{ffi::c_str, prelude::*, types::PyList};
use std::{ffi::CString, fs::File, io::Read, path::Path};
use surf::MicroSurface;

/// Load the Python source code dynamically from the given path into a
/// `CString`. This function is an alternative to `include_str!` for loading
/// Python source code dynamically without recompiling the crate.
///
/// # Arguments
///
/// * `path` - The path to the Python source code.
pub fn load_python_source_code(path: &Path) -> Result<CString, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(CString::new(contents)?)
}

pub fn plot_err(errs: &[f64], alpha: &[f64], n_digits: u32) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/pyplot.py")),
            c_str!("pyplot.py"),
            c_str!("vgp"),
        )?
        .getattr("plot_err")?
        .into();
        let errs = PyArray1::from_vec(py, errs.to_vec());
        let alphas = PyArray1::from_vec(py, alpha.to_vec());
        let args = (alphas, errs, n_digits);
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
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/pyplot.py")),
            c_str!("pyplot.py"),
            c_str!("vgp"),
        )?
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
                    PyList::new(
                        py,
                        wos.iter()
                            .map(|wo| (wo.theta.as_f32(), wo.phi.as_f32()))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
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
                    PyList::new(
                        py,
                        wos.iter()
                            .map(|wo| (wo.theta.as_f32(), wo.phi.as_f32()))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
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
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/pyplot.py")),
            c_str!("pyplot.py"),
            c_str!("vgp"),
        )?
        .getattr("plot_brdf_slice")?
        .into();
        let brdfs = PyList::new(
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
                        PyArray1::from_iter(py, brdf.spectrum.iter().map(|x| x.as_f32()));
                    let slice_phi = {
                        let data = sampler.sample_slice_at(wi, phi_o).unwrap();
                        let slice = PyArray2::zeros(py, [theta.len(), n_spectrum], false);
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
                        let slice = PyArray2::zeros(py, [theta.len(), n_spectrum], false);
                        unsafe {
                            slice.as_slice_mut().unwrap().copy_from_slice(&data);
                        }
                        slice
                    };
                    (
                        slice_phi,
                        slice_opposite_phi,
                        PyArray1::from_vec(py, theta),
                        spectrum,
                        label,
                    )
                })
                .collect::<Vec<_>>(),
        )?;
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
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/pyplot.py")),
            c_str!("pyplot.py"),
            c_str!("vgp"),
        )?
        .getattr("plot_brdf_slice_in_plane")?
        .into();
        let brdfs = PyList::new(
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
                    let theta_o = brdf
                        .params
                        .outgoing
                        .rings
                        .iter()
                        .map(|x| x.zenith_center().to_degrees().as_f32())
                        .collect::<Vec<_>>();
                    let wavelength = PyArray1::from_vec(
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
        )?;
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
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/pyplot.py")),
            c_str!("pyplot.py"),
            c_str!("vgp"),
        )?
        .getattr("plot_ndf_slice")?
        .into();
        let slices = PyList::new(
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
                                numpy::PyArray1::from_vec(py, theta),
                                numpy::PyArray1::from_slice(py, spls),
                                numpy::PyArray1::from_slice(py, opp_spls.unwrap()),
                            )
                        },
                        NdfMeasurementMode::ByPartition { .. } => {
                            let sampler = DataCarriedOnHemisphereSampler::new(ndf).unwrap();
                            let partition = sampler.extra.as_ref().unwrap();
                            let theta = numpy::PyArray1::zeros(py, [partition.n_rings()], false);
                            partition
                                .rings
                                .iter()
                                .enumerate()
                                .for_each(|(i, ring)| unsafe {
                                    theta.as_slice_mut().unwrap()[i] = ring.zenith_center().as_f32()
                                });
                            let phi_opp = (phi + Radians::PI).wrap_to_tau();
                            let slice_phi = numpy::PyArray1::from_vec(
                                py,
                                sampler.sample_slice_at(phi).into_vec(),
                            );
                            let slice_phi_opp = numpy::PyArray1::from_vec(
                                py,
                                sampler.sample_slice_at(phi_opp).into_vec(),
                            );
                            (label, theta, slice_phi, slice_phi_opp)
                        },
                    }
                })
                .collect::<Vec<_>>(),
        )?;
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
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/pyplot.py")),
            c_str!("pyplot.py"),
            c_str!("vgp"),
        )?
        .getattr("plot_gaf_slice")?
        .into();
        let slices = PyList::new(
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
                        PyArray1::from_vec(py, theta),
                        PyArray1::from_slice(py, spls),
                        PyArray1::from_slice(py, opp_spls.unwrap()),
                    )
                })
                .collect::<Vec<_>>(),
        )?;

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
        PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/tone_mapping.py")),
            c_str!("tone_mapping.py"),
            c_str!("tone_mapping"),
        )?;
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/pyplot.py")),
            c_str!("pyplot.py"),
            c_str!("vgp"),
        )
        .unwrap()
        .getattr("plot_brdf_map")
        .unwrap()
        .into();
        let images = PyList::new(
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
                            let img_data = PyArray2::zeros(py, (size.0, size.1), false);
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
        )?;
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
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/pyplot.py")),
            c_str!("pyplot.py"),
            c_str!("vgp"),
        )?
        .getattr("plot_surfaces")?
        .into();
        let surfaces = PyList::new(
            py,
            surfaces
                .iter()
                .map(|&surface| {
                    let samples = numpy::PyArray2::zeros(py, (surface.rows, surface.cols), false);
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
        )?;
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
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(include_str!("./pyplot/pyplot.py")),
            c_str!("pyplot.py"),
            c_str!("vgp"),
        )?
        .getattr("plot_brdf_3d")?
        .into();
        let theta = brdf
            .params
            .outgoing
            .rings
            .iter()
            .map(|x| x.zenith_center().as_f32())
            .collect::<Vec<_>>();
        let phi = StepRangeIncl::new(0.0, 360.0, 5.0f32)
            .values()
            .map(|x| x.to_radians())
            .collect::<Vec<_>>();
        let n_theta = theta.len();
        let n_phi = phi.len();
        let n_spectrum = brdf.spectrum.len();
        let vals = {
            let sampler = DataCarriedOnHemisphereSampler::new(brdf).unwrap();
            let data = PyArray2::zeros(py, [n_theta, n_phi], false);
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
            PyArray1::from_vec(py, theta),
            PyArray1::from_vec(py, phi),
            vals,
            cmap,
            scale,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}

/// Helper struct to preproccess the data and generate/manage the figures for
/// the BRDF fitting process.
pub struct BrdfFittingPlotter {}

impl BrdfFittingPlotter {
    /// Plot the BRDF fitting process interactively.
    pub fn plot_interactive(
        brdf: &dyn AnyMeasuredBrdf,
        alphas: &[(f64, f64)],
        iors: &IorRegistry,
    ) -> PyResult<()> {
        log::info!("Plotting BRDF fitting...");
        let n_models = alphas.len();
        log::info!("Number of models: {}", n_models);
        let bk_models = alphas
            .iter()
            .map(|(alpha_x, alpha_y)| MicrofacetBrdfBK::new(*alpha_x, *alpha_y))
            .collect::<Box<_>>();
        let tr_models = alphas
            .iter()
            .map(|(alpha_x, alpha_y)| MicrofacetBrdfTR::new(*alpha_x, *alpha_y))
            .collect::<Box<_>>();
        Python::with_gil(|py| {
            let fun: Py<PyAny> = PyModule::from_code(
                py,
                c_str!(include_str!("./pyplot/pyplot.py")),
                c_str!("pyplot.py"),
                c_str!("vgp"),
            )?
            .getattr("plot_brdf_fitting")?
            .into();
            let alphas = PyArray1::from_vec(
                py,
                alphas
                    .iter()
                    .flat_map(|(x, y)| [*x, *y])
                    .collect::<Vec<_>>(),
            )
            .reshape((n_models, 2))?;
            match brdf.kind() {
                MeasuredBrdfKind::Vgonio => {
                    let brdf = brdf.as_any().downcast_ref::<VgonioBrdf>().unwrap();
                    let iors_i = iors
                        .ior_of_spectrum(brdf.incident_medium, brdf.spectrum.as_ref())
                        .unwrap();
                    let iors_t = iors
                        .ior_of_spectrum(brdf.transmitted_medium, brdf.spectrum.as_ref())
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
                        .step_by(brdf.params.n_wi_zenith())
                        .map(|x| x.phi.as_f32())
                        .collect::<Vec<_>>();
                    let theta_o = brdf
                        .params
                        .outgoing
                        .rings
                        .iter()
                        .map(|x| x.zenith_center().as_f32())
                        .collect::<Vec<_>>();
                    let phi_o = StepRangeExcl::new(0.0f32, 360.0, 15.0)
                        .values()
                        .map(|x| x.to_radians())
                        .collect::<Vec<_>>();
                    let n_theta_i = theta_i.len();
                    let n_phi_i = phi_i.len();
                    let n_theta_o = brdf.params.outgoing.n_rings();
                    let n_phi_o = phi_o.len();
                    // n_theta_i x n_phi_i x n_phi_o x n_theta_o x n_spectrum
                    let samples = PyArray1::zeros(
                        py,
                        [n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_bk = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_tr = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
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
                                    for (m, (bk, tr)) in
                                        bk_models.iter().zip(tr_models.iter()).enumerate()
                                    {
                                        let spectral_samples_bk =
                                            Scattering::eval_reflectance_spectrum(
                                                bk,
                                                &wi.to_cartesian(),
                                                &wo.to_cartesian(),
                                                &iors_i,
                                                &iors_t,
                                            );
                                        let spectral_samples_tr =
                                            Scattering::eval_reflectance_spectrum(
                                                tr,
                                                &wi.to_cartesian(),
                                                &wo.to_cartesian(),
                                                &iors_i,
                                                &iors_t,
                                            );
                                        let offset = m
                                            * n_theta_i
                                            * n_phi_i
                                            * n_phi_o
                                            * n_theta_o
                                            * n_spectrum
                                            + i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
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
                    }
                    let samples =
                        samples.reshape((n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let theta_o = PyArray1::from_vec(py, theta_o);
                    let theta_i = PyArray1::from_vec(py, theta_i);
                    let phi_i = PyArray1::from_vec(py, phi_i);
                    let phi_o = PyArray1::from_vec(py, phi_o);
                    let wavelength = PyArray1::from_vec(
                        py,
                        brdf.spectrum.iter().map(|x| x.as_f32()).collect::<Vec<_>>(),
                    );
                    let fitted_bk = fitted_bk
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let fitted_tr = fitted_tr
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    fun.call1(
                        py,
                        (
                            samples,
                            (theta_i, phi_i),
                            (theta_o, phi_o),
                            wavelength,
                            (fitted_bk, fitted_tr),
                            alphas,
                        ),
                    )?;
                },
                MeasuredBrdfKind::Yan2018 => {
                    let brdf = brdf.as_any().downcast_ref::<Yan18Brdf>().unwrap();
                    let iors_i = iors
                        .ior_of_spectrum(brdf.incident_medium, brdf.spectrum.as_ref())
                        .unwrap();
                    let iors_t = iors
                        .ior_of_spectrum(brdf.transmitted_medium, brdf.spectrum.as_ref())
                        .unwrap();
                    let n_spectrum = brdf.n_spectrum();
                    let theta_i = brdf
                        .params
                        .incoming
                        .iter()
                        .take(brdf.params.n_zenith_i)
                        .map(|x| x.theta.as_f32())
                        .collect::<Vec<_>>();
                    let phi_i = brdf
                        .params
                        .incoming
                        .iter()
                        .step_by(brdf.params.n_zenith_i)
                        .map(|x| x.phi.as_f32())
                        .collect::<Vec<_>>();
                    let theta_o = StepRangeExcl::new(0.0f32, 90.0, 1.0)
                        .values()
                        .map(|x| x.to_radians())
                        .collect::<Vec<_>>();
                    let phi_o = StepRangeExcl::new(0.0f32, 360.0, 60.0)
                        .values()
                        .map(|x| x.to_radians())
                        .collect::<Vec<_>>();
                    let n_theta_i = theta_i.len();
                    let n_phi_i = phi_i.len();
                    let n_theta_o = theta_o.len();
                    let n_phi_o = phi_o.len();
                    // n_theta_i x n_phi_i x n_phi_o x n_theta_o x n_spectrum
                    let samples = PyArray1::zeros(
                        py,
                        [n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_bk = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_tr = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    // Fill the samples and fitted data
                    for (i, t_i) in theta_i.iter().enumerate() {
                        for (j, p_i) in phi_i.iter().enumerate() {
                            for (k, t_o) in theta_o.iter().enumerate() {
                                for (l, p_o) in phi_o.iter().enumerate() {
                                    let wi = Sph2::new(rad!(*t_i), rad!(*p_i));
                                    let wo = Sph2::new(rad!(*t_o), rad!(*p_o));
                                    let resampled = brdf.sample_at(wi, wo);
                                    let offset = i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                        + j * n_phi_o * n_theta_o * n_spectrum
                                        + l * n_theta_o * n_spectrum
                                        + k * n_spectrum;
                                    unsafe {
                                        samples.as_slice_mut().unwrap()
                                            [offset..offset + n_spectrum]
                                            .copy_from_slice(&resampled);
                                    }
                                    for (m, (bk, tr)) in
                                        bk_models.iter().zip(tr_models.iter()).enumerate()
                                    {
                                        let spectral_samples_bk =
                                            Scattering::eval_reflectance_spectrum(
                                                bk,
                                                &wi.to_cartesian(),
                                                &wo.to_cartesian(),
                                                &iors_i,
                                                &iors_t,
                                            );
                                        let spectral_samples_tr =
                                            Scattering::eval_reflectance_spectrum(
                                                tr,
                                                &wi.to_cartesian(),
                                                &wo.to_cartesian(),
                                                &iors_i,
                                                &iors_t,
                                            );
                                        let offset = m
                                            * n_theta_i
                                            * n_phi_i
                                            * n_phi_o
                                            * n_theta_o
                                            * n_spectrum
                                            + i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                            + j * n_phi_o * n_theta_o * n_spectrum
                                            + l * n_theta_o * n_spectrum
                                            + k * n_spectrum;
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
                    }
                    let samples =
                        samples.reshape((n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let theta_o = PyArray1::from_vec(py, theta_o);
                    let theta_i = PyArray1::from_vec(py, theta_i);
                    let phi_i = PyArray1::from_vec(py, phi_i);
                    let phi_o = PyArray1::from_vec(py, phi_o);
                    let wavelength = PyArray1::from_vec(
                        py,
                        brdf.spectrum.iter().map(|x| x.as_f32()).collect::<Vec<_>>(),
                    );
                    let fitted_bk = fitted_bk
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let fitted_tr = fitted_tr
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    fun.call1(
                        py,
                        (
                            samples,
                            (theta_i, phi_i),
                            (theta_o, phi_o),
                            wavelength,
                            (fitted_bk, fitted_tr),
                            alphas,
                        ),
                    )?;
                },
                MeasuredBrdfKind::Clausen => {
                    let brdf = brdf.as_any().downcast_ref::<ClausenBrdf>().unwrap();
                    let iors_i = iors
                        .ior_of_spectrum(brdf.incident_medium, brdf.spectrum.as_ref())
                        .unwrap();
                    let iors_t = iors
                        .ior_of_spectrum(brdf.transmitted_medium, brdf.spectrum.as_ref())
                        .unwrap();
                    let wavelength = PyArray1::from_vec(
                        py,
                        brdf.spectrum.iter().map(|x| x.as_f32()).collect::<Vec<_>>(),
                    );
                    let n_spectrum = brdf.n_spectrum();
                    let mut theta_i = vec![];
                    let mut phi_i = vec![];
                    let mut theta_o = vec![];
                    let mut phi_o = vec![];
                    brdf.params.wi_wos_iter().for_each(|(i, (wi, wos))| {
                        if !theta_i.contains(&wi.theta.as_f32()) {
                            theta_i.push(wi.theta.as_f32());
                        }
                        if !phi_i.contains(&wi.phi.as_f32()) {
                            phi_i.push(wi.phi.as_f32());
                        }
                        wos.iter().enumerate().for_each(|(j, wo)| {
                            if !theta_o.contains(&wo.theta.as_f32()) {
                                theta_o.push(wo.theta.as_f32());
                            }
                            if !theta_i.contains(&wo.theta.as_f32()) {
                                theta_i.push(wo.theta.as_f32());
                            }
                            if !phi_o.contains(&wo.phi.as_f32()) {
                                phi_o.push(wo.phi.as_f32());
                            }
                        });
                    });

                    let n_theta_i = theta_i.len();
                    let n_phi_i = phi_i.len();
                    let n_theta_o = theta_o.len();
                    let n_phi_o = phi_o.len();
                    theta_i.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    phi_i.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    theta_o.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    phi_o.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    // n_theta_i x n_phi_i x n_theta_o x n_phi_o x n_spectrum
                    let mut samples: DyArr<f32, 5> = DyArr::splat(
                        f32::NAN,
                        [n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum],
                    );

                    brdf.params.wi_wos_iter().for_each(|(i, (wi, wos))| {
                        wos.iter().enumerate().for_each(|(j, wo)| {
                            let target_ti = theta_i
                                .iter()
                                .position(|x| (*x - wi.theta.as_f32()).abs() < 1e-6)
                                .unwrap();
                            let target_pi = phi_i
                                .iter()
                                .position(|x| (*x - wi.phi.as_f32()).abs() < 1e-6)
                                .unwrap();
                            let target_to = theta_o
                                .iter()
                                .position(|x| (*x - wo.theta.as_f32()).abs() < 1e-6)
                                .unwrap();
                            let target_po = phi_o
                                .iter()
                                .position(|x| (*x - wo.phi.as_f32()).abs() < 1e-6)
                                .unwrap();

                            if target_ti == target_to && target_pi == target_po {
                                return;
                            }

                            let target_offset =
                                target_ti * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                    + target_pi * n_phi_o * n_theta_o * n_spectrum
                                    + target_po * n_theta_o * n_spectrum
                                    + target_to * n_spectrum;
                            let offset = i * brdf.n_wo() * n_spectrum + j * n_spectrum;
                            samples.as_mut_slice()[target_offset..target_offset + n_spectrum]
                                .copy_from_slice(
                                    &brdf.samples.as_slice()[offset..offset + n_spectrum],
                                );

                            if target_to == 0 {
                                let target_offset =
                                    target_ti * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                        + target_pi * n_phi_o * n_theta_o * n_spectrum
                                        + (target_po + 1) * n_theta_o * n_spectrum
                                        + target_to * n_spectrum;
                                samples.as_mut_slice()[target_offset..target_offset + n_spectrum]
                                    .copy_from_slice(
                                        &brdf.samples.as_slice()[offset..offset + n_spectrum],
                                    );
                            }

                            // When wi == (0, 0), Olaf doesn't have the data for
                            // phi == 180, so we populate the data with the
                            // opposite direction.
                            if i == 0 {
                                let target_offset =
                                    target_ti * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                        + (target_pi + 1) * n_phi_o * n_theta_o * n_spectrum
                                        + target_po * n_theta_o * n_spectrum
                                        + target_to * n_spectrum;
                                samples.as_mut_slice()[target_offset..target_offset + n_spectrum]
                                    .copy_from_slice(
                                        &brdf.samples.as_slice()[offset..offset + n_spectrum],
                                    );
                            }
                        });
                    });

                    let fitted_bk = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_tr = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );

                    for (i, t_i) in theta_i.iter().enumerate() {
                        for (j, p_i) in phi_i.iter().enumerate() {
                            for (k, p_o) in phi_o.iter().enumerate() {
                                let wi = Sph2::new(rad!(*t_i), rad!(*p_i));
                                for (l, t_o) in theta_o.iter().enumerate() {
                                    let wo = Sph2::new(rad!(*t_o), rad!(*p_o));
                                    for (m, (bk, tr)) in
                                        bk_models.iter().zip(tr_models.iter()).enumerate()
                                    {
                                        let spectral_samples_bk =
                                            Scattering::eval_reflectance_spectrum(
                                                bk,
                                                &wi.to_cartesian(),
                                                &wo.to_cartesian(),
                                                &iors_i,
                                                &iors_t,
                                            );
                                        let spectral_samples_tr =
                                            Scattering::eval_reflectance_spectrum(
                                                tr,
                                                &wi.to_cartesian(),
                                                &wo.to_cartesian(),
                                                &iors_i,
                                                &iors_t,
                                            );
                                        let offset = m
                                            * n_theta_i
                                            * n_phi_i
                                            * n_phi_o
                                            * n_theta_o
                                            * n_spectrum
                                            + i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
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
                    }
                    let samples = PyArray1::from_vec(py, samples.into_vec());
                    let theta_i = PyArray1::from_vec(py, theta_i.into_iter().collect());
                    let phi_i = PyArray1::from_vec(py, phi_i.into_iter().collect());
                    let theta_o = PyArray1::from_vec(py, theta_o.into_iter().collect());
                    let phi_o = PyArray1::from_vec(py, phi_o.into_iter().collect());
                    let fitted_bk = fitted_bk
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let fitted_tr = fitted_tr
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    fun.call1(
                        py,
                        (
                            samples
                                .reshape((n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?,
                            (theta_i, phi_i),
                            (theta_o, phi_o),
                            wavelength,
                            (fitted_bk, fitted_tr),
                            alphas,
                        ),
                    )?;
                },
                MeasuredBrdfKind::Rgl => {
                    let brdf = brdf.as_any().downcast_ref::<RglBrdf>().unwrap();
                    let iors_i = iors
                        .ior_of_spectrum(brdf.incident_medium, brdf.spectrum.as_ref())
                        .unwrap();
                    let iors_t = iors
                        .ior_of_spectrum(brdf.transmitted_medium, brdf.spectrum.as_ref())
                        .unwrap();
                    let n_spectrum = brdf.n_spectrum();
                    let theta_i = brdf
                        .params
                        .incoming
                        .iter()
                        .take(brdf.params.n_zenith_i)
                        .map(|x| x.theta.as_f32())
                        .collect::<Vec<_>>();
                    let phi_i = brdf
                        .params
                        .incoming
                        .iter()
                        .step_by(brdf.params.n_zenith_i)
                        .map(|x| x.phi.as_f32())
                        .collect::<Vec<_>>();
                    let theta_o = StepRangeIncl::new(0.0f32, 80.0, 1.0)
                        .values()
                        .map(|x| x.to_radians())
                        .collect::<Vec<_>>();
                    let phi_o = StepRangeExcl::new(0.0f32, 360.0, 30.0)
                        .values()
                        .map(|x| x.to_radians())
                        .collect::<Vec<_>>();
                    let n_theta_i = theta_i.len();
                    let n_phi_i = phi_i.len();
                    let n_theta_o = theta_o.len();
                    let n_phi_o = phi_o.len();
                    // n_theta_i x n_phi_i x n_theta_o x n_phi_o x n_spectrum
                    let samples = PyArray1::zeros(
                        py,
                        [n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_bk = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_tr = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    // Fill the samples and fitted data
                    for (i, t_i) in theta_i.iter().enumerate() {
                        for (j, p_i) in phi_i.iter().enumerate() {
                            for (k, t_o) in theta_o.iter().enumerate() {
                                for (l, p_o) in phi_o.iter().enumerate() {
                                    let wi = Sph2::new(rad!(*t_i), rad!(*p_i));
                                    let wo = Sph2::new(rad!(*t_o), rad!(*p_o));
                                    let resampled =
                                        brdf.params.original.eval(*t_i, *p_i, *t_o, *p_o);
                                    let offset = i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                        + j * n_phi_o * n_theta_o * n_spectrum
                                        + l * n_theta_o * n_spectrum
                                        + k * n_spectrum;
                                    unsafe {
                                        samples.as_slice_mut().unwrap()
                                            [offset..offset + n_spectrum]
                                            .copy_from_slice(&resampled);
                                    }
                                    for (m, (bk, tr)) in
                                        bk_models.iter().zip(tr_models.iter()).enumerate()
                                    {
                                        let spectral_samples_bk =
                                            Scattering::eval_reflectance_spectrum(
                                                bk,
                                                &wi.to_cartesian(),
                                                &wo.to_cartesian(),
                                                &iors_i,
                                                &iors_t,
                                            );
                                        let spectral_samples_tr =
                                            Scattering::eval_reflectance_spectrum(
                                                tr,
                                                &wi.to_cartesian(),
                                                &wo.to_cartesian(),
                                                &iors_i,
                                                &iors_t,
                                            );
                                        let offset = m
                                            * n_theta_i
                                            * n_phi_i
                                            * n_phi_o
                                            * n_theta_o
                                            * n_spectrum
                                            + i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                            + j * n_phi_o * n_theta_o * n_spectrum
                                            + l * n_theta_o * n_spectrum
                                            + k * n_spectrum;

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
                    }
                    let samples =
                        samples.reshape((n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let theta_o = PyArray1::from_vec(py, theta_o);
                    let theta_i = PyArray1::from_vec(py, theta_i);
                    let phi_i = PyArray1::from_vec(py, phi_i);
                    let phi_o = PyArray1::from_vec(py, phi_o);
                    let wavelength = PyArray1::from_vec(
                        py,
                        brdf.spectrum.iter().map(|x| x.as_f32()).collect::<Vec<_>>(),
                    );
                    let fitted_bk = fitted_bk
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let fitted_tr = fitted_tr
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    fun.call1(
                        py,
                        (
                            samples,
                            (theta_i, phi_i),
                            (theta_o, phi_o),
                            wavelength,
                            (fitted_bk, fitted_tr),
                            alphas,
                        ),
                    )?;
                },
                MeasuredBrdfKind::Merl => {
                    let merl_brdf = brdf.as_any().downcast_ref::<MerlBrdf>().unwrap();
                    let iors_i = iors
                        .ior_of_spectrum(merl_brdf.incident_medium, merl_brdf.spectrum.as_ref())
                        .unwrap();
                    let iors_t = iors
                        .ior_of_spectrum(merl_brdf.transmitted_medium, merl_brdf.spectrum.as_ref())
                        .unwrap();
                    let n_spectrum = merl_brdf.n_spectrum();
                    let theta_res = MerlBrdfParam::RES_THETA_H as usize / 2;
                    let phi_res = MerlBrdfParam::RES_PHI_D as usize / 4;
                    let i_thetas = (0..theta_res)
                        .map(|i| (i as f32).to_radians())
                        .collect::<Vec<_>>();
                    let i_phis = (0..phi_res)
                        .map(|i| (i as f32).to_radians())
                        .collect::<Vec<_>>();
                    let o_thetas = i_thetas.clone();
                    let o_phis = i_phis.clone();
                    let n_theta_i = i_thetas.len();
                    let n_phi_i = i_phis.len();
                    let n_theta_o = o_thetas.len();
                    let n_phi_o = o_phis.len();
                    // n_theta_i x n_phi_i x n_theta_o x n_phi_o x n_spectrum
                    // dimensions: [n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum]
                    let samples = PyArray1::zeros(
                        py,
                        [n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_bk = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    let fitted_tr = PyArray1::zeros(
                        py,
                        [n_models * n_theta_i * n_phi_i * n_phi_o * n_theta_o * n_spectrum],
                        false,
                    );
                    // Fill the samples and fitted data
                    for (i, t_i) in i_thetas.iter().enumerate() {
                        for (j, p_i) in i_phis.iter().enumerate() {
                            for (k, t_o) in o_thetas.iter().enumerate() {
                                for (l, p_o) in o_phis.iter().enumerate() {
                                    let wi = Sph2::new(rad!(*t_i), rad!(*p_i));
                                    let wo = Sph2::new(rad!(*t_o), rad!(*p_o));
                                    let resampled = merl_brdf.sample_at(wi, wo);
                                    let offset = i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                        + j * n_phi_o * n_theta_o * n_spectrum
                                        + l * n_theta_o * n_spectrum
                                        + k * n_spectrum;
                                    unsafe {
                                        samples.as_slice_mut().unwrap()
                                            [offset..offset + n_spectrum]
                                            .copy_from_slice(&resampled);
                                    }
                                    for (m, (bk, tr)) in
                                        bk_models.iter().zip(tr_models.iter()).enumerate()
                                    {
                                        let samples_bk = Scattering::eval_reflectance_spectrum(
                                            bk,
                                            &wi.to_cartesian(),
                                            &wo.to_cartesian(),
                                            &iors_i,
                                            &iors_t,
                                        );
                                        let samples_tr = Scattering::eval_reflectance_spectrum(
                                            tr,
                                            &wi.to_cartesian(),
                                            &wo.to_cartesian(),
                                            &iors_i,
                                            &iors_t,
                                        );
                                        let offset = m
                                            * n_theta_i
                                            * n_phi_i
                                            * n_phi_o
                                            * n_theta_o
                                            * n_spectrum
                                            + i * n_phi_i * n_phi_o * n_theta_o * n_spectrum
                                            + j * n_phi_o * n_theta_o * n_spectrum
                                            + l * n_theta_o * n_spectrum
                                            + k * n_spectrum;
                                        unsafe {
                                            fitted_bk.as_slice_mut().unwrap()
                                                [offset..offset + n_spectrum]
                                                .copy_from_slice(&samples_bk);
                                            fitted_tr.as_slice_mut().unwrap()
                                                [offset..offset + n_spectrum]
                                                .copy_from_slice(&samples_tr);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let samples =
                        samples.reshape((n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let theta_o = PyArray1::from_vec(py, o_thetas);
                    let theta_i = PyArray1::from_vec(py, i_thetas);
                    let phi_i = PyArray1::from_vec(py, i_phis);
                    let phi_o = PyArray1::from_vec(py, o_phis);
                    let wavelength = PyArray1::from_vec(
                        py,
                        merl_brdf
                            .spectrum
                            .iter()
                            .map(|x| x.as_f32())
                            .collect::<Vec<_>>(),
                    );
                    let fitted_bk = fitted_bk
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    let fitted_tr = fitted_tr
                        .reshape((n_models, n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_spectrum))?;
                    fun.call1(
                        py,
                        (
                            samples,
                            (theta_i, phi_i),
                            (theta_o, phi_o),
                            wavelength,
                            (fitted_bk, fitted_tr),
                            alphas,
                        ),
                    )?;
                },
                _ => {
                    log::error!("The BRDF kind is unknown!");
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "The BRDF kind is unknown!",
                    ));
                },
            }
            Ok(())
        })
    }

    /// Plot the BRDF fitting process non-interactively.
    ///
    /// Generated plots include:
    ///
    ///   - Residuals maps for each wavelength: x-axis is ωo (ϑo, ϕo), y-axis is
    ///     ωi (ϑi, ϕi), pixel value: residuals
    ///
    ///   - MSE of per incident angle fitting results: x-axis is the wavelength,
    ///     y-axis is the incident angle, pixel value: MSE
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the measured BRDF.
    /// * `measured` - The measured BRDF.
    /// * `alphas` - The alphas of the BRDF models.
    /// * `metric` - The metric used to compute the residuals.
    /// * `weighting` - The weighting used to compute the residuals.
    /// * `iors` - The IOR registry.
    /// * `parallel` - Whether to plot the BRDF error map in parallel.
    pub fn plot_non_interactive(
        name: &str,
        measured: &Box<dyn AnyMeasured>,
        alphas: &[(f64, f64)],
        metric: ErrorMetric,
        weighting: Weighting,
        iors: &IorRegistry,
        parallel: bool,
    ) -> PyResult<()> {
        assert_eq!(
            measured.kind(),
            MeasurementKind::Bsdf,
            "Invalid data passed to the function!"
        );
        log::info!("Plotting BRDF error map...");
        let top_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let src_path = Path::new(&top_dir).join("src/pyplot/brdf_fitting_non_interactive.py");
        let src_code = load_python_source_code(&src_path)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        let func: Py<PyAny> = Python::with_gil(|py| {
            PyModule::from_code(
                py,
                src_code.as_c_str(),
                c_str!("brdf_fitting_non_interactive.py"),
                c_str!("vgplot"),
            )?
            .getattr("plot_brdf_fitting_errors")
            .map(|x| x.into())
        })?;
        let models: Box<[Box<dyn AnalyticalBrdf<Params = [f64; 2]>>]> = alphas
            .iter()
            .flat_map(|(x, y)| {
                [
                    Box::new(MicrofacetBrdfBK::new(*x, *y))
                        as Box<dyn AnalyticalBrdf<Params = [f64; 2]>>,
                    Box::new(MicrofacetBrdfTR::new(*x, *y))
                        as Box<dyn AnalyticalBrdf<Params = [f64; 2]>>,
                ]
            })
            .collect();
        let brdf = measured.as_any_brdf(BrdfLevel::L0).unwrap();
        let proxy = brdf.proxy(iors);
        let shape = proxy.samples().shape();

        Python::with_gil(|py| {
            let i_thetas = PyArray::from_slice(py, &proxy.i_thetas.as_slice());
            let i_phis = PyArray::from_slice(py, &proxy.i_phis.as_slice());
            let i_count = proxy.i_thetas.len() * proxy.i_phis.len();
            let (o_thetas, o_phis, offsets, o_count, is_grid) = match &proxy.o_dirs {
                OutgoingDirs::List {
                    o_thetas,
                    o_phis,
                    offsets,
                } => (
                    PyArray::from_slice(py, &o_thetas.as_slice()),
                    PyArray::from_slice(py, &o_phis.as_slice()),
                    Some(PyArray::from_slice(py, &offsets.as_slice())),
                    o_phis.len(),
                    false,
                ),
                OutgoingDirs::Grid { o_thetas, o_phis } => (
                    PyArray::from_slice(py, &o_thetas.as_slice()),
                    PyArray::from_slice(py, &o_phis.as_slice()),
                    None,
                    o_thetas.len() * o_phis.len(),
                    true,
                ),
            };
            let spectrum = PyArray::from_slice(py, {
                unsafe { std::mem::transmute::<&[Nanometres], &[f32]>(brdf.spectrum()) }
            });

            let mut residuals = DynArr::<f64>::splat(f64::NAN, shape);
            let n_spectrum = brdf.spectrum().len();
            let n_theta_i = proxy.i_thetas.len();
            let n_phi_i = proxy.i_phis.len();
            // Residual maps per wavelength
            let mut rmaps = DynArr::<f64>::zeros(&[n_spectrum, i_count, o_count]);
            // MSE of residuals per incident angle
            let mut mmaps = DynArr::<f64>::zeros(&[n_spectrum, i_count]);

            let rcp_o_count = math::rcp_f64(o_count as f64);

            for model in models.iter() {
                let modelled = proxy.generate_analytical(model.as_ref());
                proxy.residuals(&modelled, weighting, residuals.as_mut_slice());

                // Rearrange the residuals to the shape of the residual maps per wavelength
                // In case of the grid: ϑi, ϕi, ϑo, ϕo, λ -> λ, (ϑi, ϕi), (ϑo, ϕo)
                // In case of the list: ϑi, ϕi, ωo, λ -> λ, (ϑi, ϕi), ωo
                if is_grid {
                    let n_theta_o = o_thetas.len().unwrap();
                    let n_phi_o = o_phis.len().unwrap();
                    for i in 0..n_spectrum {
                        for j in 0..n_theta_i {
                            for k in 0..n_phi_i {
                                for l in 0..n_theta_o {
                                    for m in 0..n_phi_o {
                                        rmaps[[i, j * n_phi_i + k, l * n_phi_o + m]] =
                                            residuals[[j, k, l, m, i]];
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for i in 0..n_spectrum {
                        for j in 0..n_theta_i {
                            for k in 0..n_phi_i {
                                for l in 0..o_count {
                                    rmaps[[i, j * n_phi_i + k, l]] = residuals[[j, k, l, i]];
                                }
                            }
                        }
                    }
                }

                // Compute the MSE of residuals per incident angle from the residual maps
                // Pre-allocate the memory for the MSE maps computation
                let mut temp = DynArr::<f64>::zeros(&[o_count]);
                for i in 0..n_spectrum {
                    for j in 0..i_count {
                        for k in 0..o_count {
                            let r = rmaps[[i, j, k]];
                            if !r.is_nan() {
                                temp[k] = r.powi(2);
                            }
                        }
                        // Use pairwise summation to compute the MSE to avoid numerical instability
                        mmaps[[i, j]] = math::pairwise_sum(&temp.as_slice()) * rcp_o_count;
                    }
                }

                // TODO: avoid copying the data to Python
                let rs = PyArray::from_slice(py, residuals.as_slice());
                let rmaps_py = PyArray::from_slice(py, rmaps.as_slice());
                let mmaps_py = PyArray::from_slice(py, mmaps.as_slice());
                let res = func.call1(
                    py,
                    (
                        name,
                        (
                            rs.reshape(shape).unwrap(),
                            rmaps_py.reshape(rmaps.shape()).unwrap(),
                            mmaps_py.reshape(mmaps.shape()).unwrap(),
                        ),
                        model.name(),
                        metric.to_string(),
                        &i_thetas,
                        &i_phis,
                        &o_thetas,
                        &o_phis,
                        offsets.as_ref(),
                        &spectrum,
                        parallel,
                    ),
                );
                if let Err(e) = res {
                    e.print_and_set_sys_last_vars(py);
                }
                residuals.fill(f64::NAN);
                rmaps.fill(f64::NAN);
                mmaps.fill(f64::NAN);
            }
            Ok(())
        })
    }
}
