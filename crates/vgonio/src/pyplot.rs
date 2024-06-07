use bxdf::brdf::measured::ClausenBrdf;
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

pub fn plot_brdf_vgonio_clausen(
    itrp: &ClausenBrdf,
    olaf: &ClausenBrdf,
    dense: bool,
) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_brdf_comparison")?
                .into();
        let n_samples_itrp = itrp.n_samples();
        let n_samples_olaf = olaf.n_samples();
        // Rearrange the data layout from [wi, wo, spectrum] to [spectrum, wi, wo]
        let mut data_itrp = vec![0.0; n_samples_itrp];
        let mut data_olaf = vec![0.0; n_samples_olaf];
        let n_wi = itrp.n_wi();
        let n_wo_itrp = itrp.n_wo();
        let n_wo_olaf = olaf.n_wo();
        let n_spectrum_itrp = itrp.n_spectrum();
        let n_spectrum_olaf = olaf.n_spectrum();
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
                    data_olaf[new_offset] = olaf.samples[original_offset];
                }
            }
        }
        let wi_wo_pairs_olaf = olaf
            .params
            .incoming
            .as_slice()
            .iter()
            .zip(olaf.params.outgoing.as_slice().chunks(olaf.params.n_wo))
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
        let spectrum_olaf = olaf
            .spectrum
            .as_slice()
            .iter()
            .map(|x| x.as_f32())
            .collect::<Vec<_>>();
        let max_itrp = itrp.compute_max_values().into_vec();
        let max_olaf = olaf.compute_max_values().into_vec();
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
