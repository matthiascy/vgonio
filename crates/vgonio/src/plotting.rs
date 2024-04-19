use crate::measure::data::SampledBrdf;
use core::slice::SlicePattern;
use pyo3::{
    prelude::*,
    types::{IntoPyDict, PyList},
};

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

pub fn plot_brdf(interpolated: &SampledBrdf, measured: &SampledBrdf) -> PyResult<()> {
    Python::with_gil(|py| {
        let fun: Py<PyAny> =
            PyModule::from_code_bound(py, include_str!("./pyplot/pyplot.py"), "pyplot.py", "vgp")?
                .getattr("plot_brdf_comparison")?
                .into();
        // Rearrange the data layout from [wi, wo, spectrum] to [spectrum, wi, wo]
        let mut interpolated_data = vec![0.0; interpolated.samples.len()];
        let mut measured_data = vec![0.0; measured.samples.len()];
        let n_wi = interpolated.wi_wo_pairs.len();
        let n_wo = interpolated.wi_wo_pairs[0].1.len();
        let n_lambda_interpolated = interpolated.spectrum.len();
        let n_lambda_measured = measured.spectrum.len();
        for (i, (wi, wos)) in interpolated.wi_wo_pairs.iter().enumerate() {
            for (j, _) in wos.iter().enumerate() {
                for (k, _) in interpolated.spectrum.iter().enumerate() {
                    let original_offset =
                        i * n_wo * n_lambda_interpolated + j * n_lambda_interpolated + k;
                    let new_offset = k * n_wi * n_wo + i * n_wo + j;
                    interpolated_data[new_offset] = interpolated.samples[original_offset];
                }
                for (k, _) in measured.spectrum.iter().enumerate() {
                    let original_offset = i * n_wo * n_lambda_measured + j * n_lambda_measured + k;
                    let new_offset = k * n_wi * n_wo + i * n_wo + j;
                    measured_data[new_offset] = measured.samples[original_offset];
                }
            }
        }
        let wi_wo_pairs = measured
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
        let interpolated_wavelengths = interpolated
            .spectrum
            .iter()
            .map(|w| w.as_f32())
            .collect::<Vec<_>>();
        let measured_wavelengths = measured
            .spectrum
            .iter()
            .map(|w| w.as_f32())
            .collect::<Vec<_>>();
        let interpolated_max = interpolated
            .max_values
            .iter()
            .map(|v| *v)
            .collect::<Vec<_>>();
        let measured_max = measured.max_values.iter().map(|v| *v).collect::<Vec<_>>();
        let args = (
            n_wi,
            n_wo,
            wi_wo_pairs,
            interpolated_data,
            interpolated_wavelengths,
            n_lambda_interpolated,
            interpolated_max,
            measured_data,
            measured_wavelengths,
            n_lambda_measured,
            measured_max,
        );
        fun.call1(py, args)?;
        Ok(())
    })
}
