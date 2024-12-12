use crate::distro::MicrofacetDistroKind;
use base::{
    math::{Mat3, Sph2, Vec3},
    Isotropy,
};
use num_traits::Float;
use std::fmt::Debug;

pub mod analytical;
pub mod lambert;
pub mod measured;

#[cfg(feature = "fitting")]
use base::optics::ior::Ior;

#[cfg(feature = "fitting")]
pub mod fitting;

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

/// Common interface for BxDFs (analytical BRDF models).
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
///   \mathbf{\h} = \frac{\mathbf{\i} + \mathbf{\o}}{||\mathbf{\i} + \mathbf{\o}||} \\\\
///   \mathbf{\d} = Rot_y(-\theta_h) \cdot Rot_z(-\phi_h) \cdot \mathbf{\i}
/// \end{cases}
/// $$
///
/// where $\theta_h$ and $\phi_h$ are the polar and azimuthal angles of the half vector.
/// 
/// # Arguments
///
/// * `i` - The incident direction.
/// * `o` - The outgoing direction.
///
/// # Returns
///
/// (half, difference)
pub fn io2hd(i: &Vec3, o: &Vec3) -> (Vec3, Vec3) {
    let h = (*i + *o).normalize();
    let wh = Sph2::from_cartesian(h);
    let rot_y = Mat3::from_rotation_y(-wh.theta.as_f32());
    let rot_z = Mat3::from_rotation_z(-wh.phi.as_f32());
    let d = rot_y * rot_z * *i;
    (h, d)
}

/// Converts the incident and outgoing direction to half vector and difference
/// vector in spherical coordinates.
///
/// See [`io2hd`] for the conversion between the incident and outgoing direction
/// and the half and difference vector in Cartesian coordinates.
///
/// # Arguments
///
/// * `wi` - The incident direction in spherical coordinates.
/// * `wo` - The outgoing direction in spherical coordinates.
///
/// # Returns
///
/// (half, difference)
pub fn io2hd_sph(wi: &Sph2, wo: &Sph2) -> (Sph2, Sph2) {
    let (h, d) = io2hd(&wi.to_cartesian(), &wo.to_cartesian());
    (Sph2::from_cartesian(h), Sph2::from_cartesian(d))
}

#[rustfmt::skip]
/// Converts the half and difference vectors to the incident and outgoing
/// direction.
///
/// $$
/// \begin{cases}
///   \mathbf{\i} = Rot_z(+\phi_h) \cdot Rot_y(+\theta_h) \cdot \mathbf{\d} \\\\
///   \mathbf{\o} = 2 \cdot (\mathbf{\i} \cdot \mathbf{h})\mathbf{\h} - \mathbf{\i}
/// \end{cases}
/// $$
///
/// where $\theta_h$ and $\phi_h$ are the polar and azimuthal angles of the half vector.
///
/// # Arguments
///
/// * `h` - The half vector.
/// * `d` - The difference vector.
///
/// # Returns
///
/// (incident, outgoing)
pub fn hd2io(h: &Vec3, d: &Vec3) -> (Vec3, Vec3) {
    let wh = Sph2::from_cartesian(*h);
    let phi_h = wh.phi;
    let theta_h = wh.theta;
    let rot_y = Mat3::from_rotation_y(theta_h.as_f32());
    let rot_z = Mat3::from_rotation_z(phi_h.as_f32());
    let i = rot_z * rot_y * *d;
    let o = 2.0 * i.dot(*h) * *h - i;
    (i, o)
}

/// Converts the half and difference vectors to the incident and outgoing
/// direction in spherical coordinates.
///
/// # Arguments
///
/// * `wh` - The half vector in spherical coordinates.
/// * `wd` - The difference vector in spherical coordinates.
///
/// # Returns
///
/// (incident, outgoing)
pub fn hd2io_sph(wh: &Sph2, wd: &Sph2) -> (Sph2, Sph2) {
    let (i, o) = hd2io(&wh.to_cartesian(), &wd.to_cartesian());
    (Sph2::from_cartesian(i), Sph2::from_cartesian(o))
}

/// The data generated by the analytical BxDF.
pub struct AnalyticalBxdfData<F: Float> {
    pub family: BxdfFamily,
    pub samples: Box<[F]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use base::{math::Vec3, units::Radians};

    #[track_caller]
    fn assert_vec3_eq(v1: &Vec3, v2: &Vec3) {
        assert!((v1.x - v2.x).abs() < 1e-6, "x: {} != {}", v1.x, v2.x);
        assert!((v1.y - v2.y).abs() < 1e-6, "y: {} != {}", v1.y, v2.y);
        assert!((v1.z - v2.z).abs() < 1e-6, "z: {} != {}", v1.z, v2.z);
    }

    #[test]
    fn test_io2hd() {
        let i = Vec3::new(1.0, 0.0, 0.0);
        let o = Vec3::new(0.0, 1.0, 0.0);
        let (h, d) = io2hd(&i, &o);
        assert_eq!(h, Vec3::new(1.0, 1.0, 0.0).normalize());
        let expected_d = Vec3::new(0.0, -1.0, 1.0).normalize();
        assert_vec3_eq(&d, &expected_d);

        {
            let i_converted = hd2io(&h, &d).0;
            assert_vec3_eq(&i_converted, &i);
            let o_converted = hd2io(&h, &d).1;
            assert_vec3_eq(&o_converted, &o);
        }

        let wi = Sph2::new(Radians::PI / 4.0, Radians::PI / 3.0);
        let wo = Sph2::new(Radians::PI / 3.0, Radians::PI / 2.0);
        let (h, d) = io2hd_sph(&wi, &wo);
        let h_expected = Vec3::new(0.182143, 0.761639, 0.621876);
        let d_expected = Vec3::new(
            -0.13223252786478978393,
            -0.20142696029304382743,
            0.97053682992537293472,
        );
        assert_vec3_eq(&h.to_cartesian(), &h_expected);
        assert_vec3_eq(&d.to_cartesian(), &d_expected);

        {
            let i_converted = hd2io_sph(&h, &d).0;
            assert_vec3_eq(&i_converted.to_cartesian(), &wi.to_cartesian());
            let o_converted = hd2io_sph(&h, &d).1;
            assert_vec3_eq(&o_converted.to_cartesian(), &wo.to_cartesian());
        }
    }
}
