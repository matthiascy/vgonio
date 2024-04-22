use crate::measure::data::SampledBrdf;
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

pub fn plot_brdf(itrp: &SampledBrdf, olaf: &SampledBrdf, dense: bool) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_brdf_comparison")?
                .into();
        // Rearrange the data layout from [wi, wo, spectrum] to [spectrum, wi, wo]
        let mut data_itrp = vec![0.0; itrp.samples.len()];
        let mut data_olaf = vec![0.0; olaf.samples.len()];
        let n_wi = itrp.wi_wo_pairs.len();
        let n_wo_itrp = itrp.n_wo();
        let n_wo_olaf = olaf.n_wo();
        let n_lambda_itrp = itrp.spectrum.len();
        let n_lambda_olaf = olaf.spectrum.len();
        for (i, (_, wos)) in itrp.wi_wo_pairs.iter().enumerate() {
            for (j, _) in wos.iter().enumerate() {
                for (k, _) in itrp.spectrum.iter().enumerate() {
                    let original_offset = i * n_wo_itrp * n_lambda_itrp + j * n_lambda_itrp + k;
                    let new_offset = k * n_wi * n_wo_itrp + i * n_wo_itrp + j;
                    data_itrp[new_offset] = itrp.samples[original_offset];
                }
            }
        }
        for (i, (_, wos)) in olaf.wi_wo_pairs.iter().enumerate() {
            for (j, _) in wos.iter().enumerate() {
                for (k, _) in olaf.spectrum.iter().enumerate() {
                    let original_offset = i * n_wo_olaf * n_lambda_olaf + j * n_lambda_olaf + k;
                    let new_offset = k * n_wi * n_wo_olaf + i * n_wo_olaf + j;
                    data_olaf[new_offset] = olaf.samples[original_offset];
                }
            }
        }
        let wi_wo_pairs_olaf = olaf
            .wi_wo_pairs
            .iter()
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
            .wi_wo_pairs
            .iter()
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
        let wavelengths_itrp = itrp.spectrum.iter().map(|w| w.as_f32()).collect::<Vec<_>>();
        let wavelengths_olaf = olaf.spectrum.iter().map(|w| w.as_f32()).collect::<Vec<_>>();
        let max_itrp = itrp.max_values.iter().map(|v| *v).collect::<Vec<_>>();
        let max_olaf = olaf.max_values.iter().map(|v| *v).collect::<Vec<_>>();
        let args = (
            n_wi,
            dense,
            n_wo_itrp,
            wi_wo_pairs_itrp,
            n_wo_olaf,
            wi_wo_pairs_olaf,
            data_itrp,
            wavelengths_itrp,
            // n_lambda_itrp,
            max_itrp,
            data_olaf,
            wavelengths_olaf,
            // n_lambda_olaf,
            max_olaf,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}
