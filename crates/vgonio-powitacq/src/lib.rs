#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("vgonio-powitacq/cxx/powitacq.h");

        type BRDF;

        fn load_brdf(path: &str) -> UniquePtr<BRDF>;
        fn brdf_wavelengths(brdf: &BRDF) -> Vec<f32>;
        fn brdf_eval(brdf: &BRDF, theta_i: f32, phi_i: f32, theta_r: f32, phi_r: f32) -> Vec<f32>;
        fn brdf_n_wavelengths(brdf: &BRDF) -> u32;
    }
}

pub struct BrdfData {
    inner: cxx::UniquePtr<ffi::BRDF>,
}

impl BrdfData {
    /// Load a BRDF from the given file.
    pub fn new(path: &str) -> Self {
        BrdfData {
            inner: ffi::load_brdf(path),
        }
    }

    pub fn n_wavelengths(&self) -> u32 { ffi::brdf_n_wavelengths(&self.inner) }

    /// Get the wavelengths of the BRDF.
    pub fn wavelengths(&self) -> Vec<f32> { ffi::brdf_wavelengths(&self.inner) }

    /// Evaluate the BRDF at the given angles in radians.
    pub fn eval(&self, theta_i: f32, phi_i: f32, theta_o: f32, phi_o: f32) -> Vec<f32> {
        ffi::brdf_eval(&self.inner, theta_i, phi_i, theta_o, phi_o)
    }
}
