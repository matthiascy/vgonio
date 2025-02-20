#[cfg(feature = "pybind")]
mod pybind {
    use crate::surface::MicroSurface;
    use pyo3::prelude::*;

    #[pymodule]
    fn vgonio_surf(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<MicroSurface>()?;
        Ok(())
    }
}
