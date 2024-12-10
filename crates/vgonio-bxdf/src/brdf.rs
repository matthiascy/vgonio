use crate::distro::MicrofacetDistroKind;
use base::{
    math::{Sph2, Vec3},
    Isotropy,
};
use num_traits::Float;
use std::fmt::Debug;

pub mod analytical;
pub mod lambert;
pub mod measured;

#[cfg(feature = "fitting")]
use base::optics::ior::Ior;

/// Different kinds of BRDFs.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BxdfFamily {
    /// Microfacet-based BxDF.
    Microfacet,
    /// Lambertian BRDF.
    Lambert,
    /// MERL BRDF.
    Merl,
    /// UTIA BRDF.
    Utia,
}

/// Common interface for BxDFs.
pub trait Bxdf: Send + Sync + Debug + 'static {
    /// The type of the parameters of the BRDF model.
    type Params;

    /// Returns the kind of the BRDF.
    fn family(&self) -> BxdfFamily;
    /// Returns the kind of the microfacet distribution function.
    fn distro(&self) -> Option<MicrofacetDistroKind> { None }
    /// Tells whether the BRDF model is isotropic or not.
    fn isotropic(&self) -> bool;
    /// Returns the isotropy of the BRDF model.
    fn isotropy(&self) -> Isotropy {
        if self.isotropic() {
            Isotropy::Isotropic
        } else {
            Isotropy::Anisotropic
        }
    }

    /// Returns the parameters of the BRDF model.
    fn params(&self) -> Self::Params;

    /// Sets the parameters of the BRDF model.
    fn set_params(&mut self, params: &Self::Params);

    /// Evaluates the BRDF ($f_r$) with the classical parameterization for any
    /// incident and outgoing direction located on the hemisphere.
    ///
    /// # Arguments
    ///
    /// * `i` - The incident direction (normalized).
    /// * `o` - The outgoing direction (normalized).
    fn eval(&self, i: &Vec3, o: &Vec3) -> f64;

    #[rustfmt::skip]
    /// Evaluates the BRDF ($f_r$) with the Rusinkiewicz parameterization.
    ///
    /// Szymon M Rusinkiewicz. A new change of variables for efficient BRDF
    /// representation. In Rendering Techniques '98, pages 11-22. Springer, 1998.
    ///
    /// See [`hd2io`] and [`io2hd`] for conversion between the incident and outgoing direction and
    /// the half and difference vector (incident direction in the frame of reference in which the
    /// halfway vector is at the North Pole).
    ///
    /// # Arguments
    ///
    /// * `h` - The half vector.
    /// * `d` - The difference vector.
    fn eval_hd(&self, h: &Vec3, d: &Vec3) -> f64;

    /// Evaluates the projected BRDF with the classical parameterization.
    ///
    /// $f_r \cdot \cos \theta_i$
    ///
    /// # Arguments
    ///
    /// * `i` - The incident direction.
    /// * `o` - The outgoing direction.
    fn evalp(&self, i: &Vec3, o: &Vec3) -> f64;

    /// Evaluates the projected BRDF with the Rusinkiewicz parameterization.
    ///
    /// $f_r \cdot \cos \theta_i$
    ///
    /// # Arguments
    ///
    /// * `h` - The half vector.
    /// * `d` - The difference vector.
    fn evalp_hd(&self, h: &Vec3, d: &Vec3) -> f64;

    /// Evaluates the projected BRDF with importance sampling.
    fn evalp_is(&self, u: f32, v: f32, o: &Vec3, i: &mut Vec3, pdf: &mut f32) -> f64;

    /// Importance sample f_r * cos_theta_i using two uniform variates.
    ///
    /// # Arguments
    ///
    /// * `u` - The first uniform variate.
    /// * `v` - The second uniform variate.
    /// * `o` - The outgoing direction.
    fn sample(&self, u: f32, v: f32, o: &Vec3) -> f64;

    /// Evaluates the PDF of a sample.
    fn pdf(&self, i: &Vec3, o: &Vec3) -> f64;

    #[cfg(feature = "fitting")]
    /// Computes the partial derivatives of the BRDF model with respect to the
    /// roughness parameters of the model.
    ///
    /// # Arguments
    ///
    /// * `i` - The incident directions.
    /// * `o` - The outgoing directions.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the BRDF model with respect to the roughness
    /// parameters of the model for each incident and outgoing direction pair.
    ///
    /// # Note
    ///
    /// The returned vector has the length of the number of parameters times the
    /// length of `i` times the length of `o`. For each incident direction
    /// `i`, the derivatives with respect to params are evaluated for each
    /// outgoing direction `o`.
    fn pds(&self, i: &[Vec3], o: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]>;

    /// Computes the partial derivatives of the BRDF model with respect to the
    /// roughness parameters of the model for a single incident and outgoing
    /// direction pair.
    #[cfg(feature = "fitting")]
    fn pd(&self, i: &Vec3, o: &Vec3, ior_i: &Ior, ior_t: &Ior) -> [f64; 2];

    #[cfg(feature = "fitting")]
    /// Computes the partial derivatives of the BRDF model with respect to the
    /// roughness parameters of the model for isotropic materials.
    ///
    /// # Arguments
    ///
    /// * `i` - The incident directions.
    /// * `o` - The outgoing directions.
    ///
    /// # Returns
    ///
    /// The partial derivatives of the BRDF model with respect to the roughness
    /// parameters of the model for each incident and outgoing direction pair.
    ///
    /// # Note
    ///
    /// For each incident direction *i*, the derivatives with respect to params
    /// are evaluated for each outgoing direction *o*.
    /// The returned vector has the length of number of parameters times the
    /// length of `i` times the length of `o`.
    fn pds_iso(&self, i: &[Vec3], o: &[Vec3], ior_i: &Ior, ior_t: &Ior) -> Box<[f64]>;

    /// Computes the partial derivatives of the BRDF model with respect to the
    /// roughness parameters of the model for a single incident and outgoing
    /// direction pair for isotropic materials.
    #[cfg(feature = "fitting")]
    fn pd_iso(&self, i: &Vec3, o: &Vec3, ior_i: &Ior, ior_t: &Ior) -> f64;
}

impl<P: 'static + Clone> Clone for Box<dyn Bxdf<Params = P>> {
    fn clone(&self) -> Box<dyn Bxdf<Params = P>> { self.clone() }
}

#[rustfmt::skip]
/// Converts the incident and outgoing direction to the half and difference
/// vector.
///
/// $$
/// \begin{cases}
///   \mathbf{\omega_h} = \frac{\mathbf{\omega_i} + \mathbf{\omega_o}}{||\mathbf{\omega_i} + \mathbf{\omega_o}||} \\\\
///   \mathbf{\omega_d} = Rot_y(-\theta_h) \cdot Rot_z(-\phi_h) \cdot \mathbf{\omega_i}
/// \end{cases}
/// $$
///
/// where $\theta_h$ and $\phi_h$ are the polar and azimuthal angles of the half vector.
/// 
/// # Arguments
///
/// * `i` - The incident direction.
/// * `o` - The outgoing direction.
pub fn io2hd(i: &Vec3, o: &Vec3) -> (Vec3, Vec3) {
    todo!("io2hd")
}

#[rustfmt::skip]
/// Converts the half and difference vectors to the incident and outgoing
/// direction.
///
/// $$
/// \begin{cases}
///   \mathbf{\omega_i} = Rot_z(+\theta_h) \cdot Rot_y(+\phi_h) \cdot \mathbf{\omega_i} \\\\
///   \mathbf{\omega_o} = 2 \cdot (\mathbf{\omega_h} \cdot \mathbf{n})\mathbf{\omega_h} - \mathbf{\omega_i}
/// \end{cases}
/// $$
///
/// where $\theta_h$ and $\phi_h$ are the polar and azimuthal angles of the half vector.
///
/// # Arguments
///
/// * `h` - The half vector.
/// * `d` - The difference vector.
pub fn hd2io(h: &Vec3, d: &Vec3) -> (Vec3, Vec3) { todo!("hd2io") }

/// The data generated by the analytical BxDF.
pub struct AnalyticalBxdfData<F: Float> {
    pub family: BxdfFamily,
    pub samples: Box<[F]>,
}
