use crate::measure::DataCarriedOnHemisphereSampler;
use base::{
    math::Sph2,
    units::{Nanometres, Radians, Rads},
};
use bxdf::brdf::measured::{ClausenBrdf, VgonioBrdf};
use pyo3::{prelude::*, types::PyList};

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
    spectrum: Vec<Nanometres>,
) -> PyResult<()> {
    let opposite_phi_o = (phi_o + Radians::PI).wrap_to_tau();
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_brdf_slice")?
                .into();
        let brdfs = PyList::new_bound(
            py,
            brdf.into_iter()
                .map(|(brdf, legend)| {
                    let sampler = DataCarriedOnHemisphereSampler::new(*brdf);
                    let theta = brdf
                        .params
                        .outgoing
                        .rings
                        .iter()
                        .map(|x| x.zenith_center().to_degrees().as_f32())
                        .collect::<Vec<_>>();
                    let slice_phi = sampler.sample_slice_at(wi, phi_o).unwrap().into_vec();
                    let slice_opposite_phi = sampler
                        .sample_slice_at(wi, opposite_phi_o)
                        .unwrap()
                        .into_vec();
                    (slice_phi, slice_opposite_phi, theta, legend)
                })
                .collect::<Vec<_>>(),
        );
        let wavelengths = PyList::new_bound(py, spectrum.iter().map(|x| x.as_f32()));
        let args = (
            phi_o.to_degrees().as_f32(),
            opposite_phi_o.to_degrees().as_f32(),
            brdfs,
            wavelengths,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}

/// Plot the BRDF slices on the plane with the given incoming azimuthal angle.
pub fn plot_brdf_slice_in_plane(
    brdf: &[&VgonioBrdf],
    phi: Radians,
    spectrum: Vec<Nanometres>,
) -> PyResult<()> {
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
            brdf.into_iter()
                .map(|&brdf| {
                    let sampler = DataCarriedOnHemisphereSampler::new(brdf);
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
                    (slices_phi, slices_phi_opp, theta_i, theta_o)
                })
                .collect::<Vec<_>>(),
        );
        let wavelengths = PyList::new_bound(py, spectrum.iter().map(|x| x.as_f32()));
        let args = (
            phi.to_degrees().as_f32(),
            phi_opp.to_degrees().as_f32(),
            brdfs,
            wavelengths,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}
