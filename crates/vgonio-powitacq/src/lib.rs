//! This crate provides Rust bindings to the Powitacq library for loading and
//! evaluating measured BRDF data. It uses the `cxx` crate to interface with the
//! C++ code that implements the actual BRDF loading and evaluation logic. The
//! main struct is `BrdfData`, which holds a shared pointer to the underlying
//! C++ BRDF data and provides methods for accessing the wavelengths, evaluating
//! the BRDF at given angles, and getting the number of wavelengths.
use std::path::Path;

/// Access to the C++ code is provided through the `ffi` module, which defines the
/// C++ types and functions that we need to call from Rust. The `BrdfData
/// struct` is a safe wrapper around the C++ BRDF data, and it implements `Debug` and
/// `PartialEq` for convenience.
/// The `BrdfData` struct is also `Send` and `Sync`, allowing it to be safely shared
/// across threads.
/// The build script `build.rs` is responsible for compiling the C++ code and linking it
/// with the Rust code. It uses the `cxx_build` crate to compile the C
/// ++ code and generate the necessary bindings for the `cxx` crate to work.
/// Overall, this crate provides a clean and safe Rust interface to the Powitacq library
/// for working with measured BRDF data, while keeping the C++ implementation details
/// encapsulated and hidden from the Rust code.
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("vgonio-powitacq/cxx/powitacq.h");

        type BRDF;

        fn load_brdf(path: &str) -> SharedPtr<BRDF>;
        fn brdf_wavelengths(brdf: &BRDF) -> Vec<f32>;
        fn brdf_eval(brdf: &BRDF, theta_i: f32, phi_i: f32, theta_r: f32, phi_r: f32) -> Vec<f32>;
        fn brdf_n_wavelengths(brdf: &BRDF) -> u32;
        fn brdf_eq(brdf1: &BRDF, brdf2: &BRDF) -> bool;
    }
}

#[derive(Clone)]
pub struct BrdfData {
    /// The inner BRDF data loaded from the C++ code. This is a shared pointer
    /// to allow for cheap cloning and sharing of the data across threads.
    inner: cxx::SharedPtr<ffi::BRDF>,
}

unsafe impl Send for BrdfData {}
unsafe impl Sync for BrdfData {}

impl BrdfData {
    /// Load a BRDF from the given file.
    ///
    /// # Panics
    /// Panics if the file cannot be loaded or if the path is not valid UTF-8.
    #[must_use]
    pub fn new(path: &Path) -> Self {
        BrdfData {
            inner: ffi::load_brdf(path.as_os_str().to_str().unwrap()),
        }
    }

    /// Get the number of wavelengths in the BRDF.
    #[must_use]
    pub fn n_wavelengths(&self) -> u32 { ffi::brdf_n_wavelengths(&self.inner) }

    /// Get the wavelengths of the BRDF.
    #[must_use]
    pub fn wavelengths(&self) -> Vec<f32> { ffi::brdf_wavelengths(&self.inner) }

    /// Evaluate the BRDF at the given angles in radians.
    #[must_use]
    pub fn eval(&self, theta_i: f32, phi_i: f32, theta_o: f32, phi_o: f32) -> Vec<f32> {
        ffi::brdf_eval(&self.inner, theta_i, phi_i, theta_o, phi_o)
    }
}

impl std::fmt::Debug for BrdfData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrdfData").finish()
    }
}

// Workaround to be able to but the type inside the MeasuredBrdf struct.
impl PartialEq for BrdfData {
    fn eq(&self, other: &Self) -> bool { ffi::brdf_eq(&self.inner, &other.inner) }
}
