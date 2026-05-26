#![feature(associated_type_defaults)]
#![feature(downcast_unchecked)]
#![feature(allocator_api)]
#![feature(adt_const_params)]
//! Bxdf models and utilities.

use crate::brdf::{measured::MeasuredBrdfKind, AnalyticalBrdf};
#[cfg(feature = "fitting")]
use crate::fitting::proxy::BrdfProxy;
use std::fmt::Debug;
use vgn_core::{
    math::{cos_theta, Vec3},
    optics::{fresnel, Ior},
    units::Nanometres,
    utils::medium::MediumId,
    BrdfLevel, MeasurementKind,
};

pub mod brdf;
pub mod distro;

#[cfg(feature = "fitting")]
pub mod fitting;

#[cfg(feature = "fitting")]
use vgn_core::optics::IorReg;

/// Different kinds of BRDFs.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BrdfFamily {
    /// Microfacet-based BRDF.
    Microfacet,
    /// Lambertian BRDF.
    Lambert,
    /// MERL BRDF.
    Merl,
    /// UTIA BRDF.
    Utia,
}

/// Common interface for measured BRDFs.
pub trait AnyMeasuredBrdf: Sync + Send {
    /// Returns the kind of the measured `BxDF`.
    fn kind(&self) -> MeasuredBrdfKind;

    /// Returns the wavelengths at which the `BxDF` is measured.
    fn spectrum(&self) -> &[Nanometres];

    /// Returns the transmitted medium.
    fn transmitted_medium(&self) -> MediumId;

    /// Returns the incident medium.
    fn incident_medium(&self) -> MediumId;

    /// Returns a proxy for the measured BRDF.
    #[cfg(feature = "fitting")]
    fn proxy(&self, iors: &IorReg) -> BrdfProxy;

    /// Casts the measured BRDF to any type for later downcasting.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Casts the measured BRDF to any mutable type for later downcasting.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Boilerplate macro for implementing the `AnyMeasuredBrdf` trait for a type.
#[macro_export]
macro_rules! any_measured_brdf_trait_common_impl {
    ($t:ty, $kind:ident) => {
        fn kind(&self) -> MeasuredBrdfKind { MeasuredBrdfKind::$kind }

        fn spectrum(&self) -> &[Nanometres] { &self.spectrum.as_ref() }

        fn transmitted_medium(&self) -> vgn_core::utils::medium::MediumId {
            self.transmitted_medium
        }

        fn incident_medium(&self) -> vgn_core::utils::medium::MediumId { self.incident_medium }

        fn as_any(&self) -> &dyn std::any::Any { self }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    };
}

/// Trait for the different kinds of measurement data.
///
/// Measurement data can be of different kinds, such as
/// - Normal Distribution Function (NDF)
/// - Masking Shadowing Function (MSF)
/// - Slope Distribution Function (SDF)
/// - Bidirectional Scattering Distribution Function (BSDF)
pub trait AnyMeasured: Debug + Send + Sync {
    /// Returns the kind of the measurement.
    fn kind(&self) -> MeasurementKind;

    /// Returns true if the measurement contains multiple levels of BRDF data.
    fn has_multiple_levels(&self) -> bool { false }

    /// Casts the measurement data to a trait object for downcasting to the
    /// concrete type.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Casts the measurement data to a mutable trait object for downcasting to
    /// the concrete type.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Returns the BRDF data at the given level if the measurement data
    /// contains multiple levels of BRDF data.
    #[allow(unused)]
    fn as_any_brdf(&self, level: BrdfLevel) -> Option<&dyn AnyMeasuredBrdf> { None }
}

impl dyn AnyMeasured {
    /// Downcasts the measurement data to the concrete type.
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: AnyMeasured + 'static,
    {
        self.as_any().downcast_ref()
    }

    /// Downcasts the measurement data to the mutable concrete type.
    pub fn downcast_mut<T>(&mut self) -> Option<&mut T>
    where
        T: AnyMeasured + 'static,
    {
        self.as_any_mut().downcast_mut()
    }
}

#[macro_export]
/// Boilerplate macro for implementing the `AnyMeasured` trait for a type.
///
/// This macro is used to implement the `AnyMeasured` trait for a type.
///
/// The macro takes two arguments:
/// - `$t`: The type to implement the `AnyMeasured` trait for.
/// - `$kind`: The kind of the measurement.
///
/// NOTE: The macro doesn't cover the case where the type has multiple levels of
/// BRDF data. Please implement the `AnyMeasured` trait manually for such types.
macro_rules! impl_any_measured_trait {
    // Non-BRDF types.
    ($t:ty, $kind:ident) => {
        impl AnyMeasured for $t {
            fn kind(&self) -> MeasurementKind { MeasurementKind::$kind }

            fn as_any(&self) -> &dyn std::any::Any { self }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        }
    };
    // Single-level BRDF types.
    (@single_level_brdf $t:ty) => {
        impl AnyMeasured for $t {
            fn kind(&self) -> MeasurementKind { MeasurementKind::Bsdf }

            fn as_any(&self) -> &dyn std::any::Any { self }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

            fn as_any_brdf(&self, _level: BrdfLevel) -> Option<&dyn AnyMeasuredBrdf> { Some(self) }
        }
    };
}

/// Structure for evaluating the reflectance combining the BRDF evaluation and
/// Fresnel term.
pub struct Scattering;

impl Scattering {
    /// Evaluates the reflectance of the given BRDF model.
    ///
    /// # Arguments
    ///
    /// * `brdf` - The BRDF model.
    /// * `wi` - The incident direction, assumed to be normalized, pointing away from the surface.
    /// * `wo` - The outgoing direction, assumed to be normalized, pointing away from the surface.
    /// * `ior_i` - The refractive index of the incident medium.
    /// * `ior_t` - The refractive index of the transmitted medium.
    pub fn eval_reflectance<P: 'static>(
        brdf: &dyn AnalyticalBrdf<P>,
        vi: &Vec3,
        vo: &Vec3,
        ior_i: &Ior,
        ior_t: &Ior,
    ) -> f64 {
        f64::from(fresnel::reflectance(cos_theta(&(-*vi)), ior_i, ior_t)) * brdf.eval(vi, vo)
    }

    /// Evaluates the reflectance of the given BRDF model for a spectrum.
    ///
    /// # Arguments
    ///
    /// * `brdf` - The BRDF model.
    /// * `vi` - The incident direction, assumed to be normalized, pointing away from the surface.
    /// * `vo` - The outgoing direction, assumed to be normalized, pointing away from the surface.
    /// * `iors_i` - The refractive indices of the incident media.
    /// * `iors_t` - The refractive indices of the transmitted media.
    pub fn eval_reflectance_spectrum<P: 'static>(
        brdf: &dyn AnalyticalBrdf<P>,
        vi: &Vec3,
        vo: &Vec3,
        iors_i: &[Ior],
        iors_t: &[Ior],
    ) -> Box<[f64]> {
        debug_assert_eq!(iors_i.len(), iors_t.len(), "IOR pair count mismatch");
        let mut reflectances = Box::new_uninit_slice(iors_i.len());
        for ((ior_i, ior_t), refl) in iors_i
            .iter()
            .zip(iors_t.iter())
            .zip(reflectances.iter_mut())
        {
            refl.write(Scattering::eval_reflectance(brdf, vi, vo, ior_i, ior_t));
        }
        unsafe { reflectances.assume_init() }
    }
}
