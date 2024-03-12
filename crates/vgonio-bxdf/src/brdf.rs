use base::math::Vec3;

mod lambert;
mod merl;
pub mod microfacet;
mod utia;

use crate::MicrofacetDistribution;

/// Common interface for BRDFs.
pub trait Brdf {
    /// Evaluates the BRDF ($f_r$) with the classical parameterization for any
    /// incident and outgoing direction located on the hemisphere.
    ///
    /// # Arguments
    ///
    /// * `wi` - The incident direction (normalized).
    /// * `wo` - The outgoing direction (normalized).
    fn eval(&self, wi: &Vec3, wo: &Vec3) -> f32;

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
    /// * `wh` - The half vector.
    /// * `wd` - The difference vector.
    fn eval_hd(&self, wh: &Vec3, wd: &Vec3) -> f32;

    /// Evaluates the projected BRDF with the classical parameterization.
    ///
    /// $f_r \cdot \cos \theta_i$
    ///
    /// # Arguments
    ///
    /// * `wi` - The incident direction.
    /// * `wo` - The outgoing direction.
    fn evalp(&self, wi: &Vec3, wo: &Vec3) -> f32;

    /// Evaluates the projected BRDF with the Rusinkiewicz parameterization.
    ///
    /// $f_r \cdot \cos \theta_i$
    ///
    /// # Arguments
    ///
    /// * `wh` - The half vector.
    /// * `wd` - The difference vector.
    fn evalp_hd(&self, wh: &Vec3, wd: &Vec3) -> f32;

    /// Evaluates the projected BRDF with importance sampling.
    fn evalp_is(&self, u: f32, v: f32, o: &Vec3, i: &mut Vec3, pdf: &mut f32) -> f32;

    /// Importance sample f_r * cos_theta_i using two uniform variates.
    ///
    /// # Arguments
    ///
    /// * `u` - The first uniform variate.
    /// * `v` - The second uniform variate.
    /// * `wo` - The outgoing direction.
    fn sample(&self, u: f32, v: f32, wo: &Vec3) -> f32;

    /// Evaluates the PDF of a sample.
    fn pdf(&self, wi: &Vec3, wo: &Vec3) -> f32;
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
/// * `wi` - The incident direction.
/// * `wo` - The outgoing direction.
pub fn io2hd(wi: &Vec3, wo: &Vec3) -> (Vec3, Vec3) {
    todo!()
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
/// * `wh` - The half vector.
/// * `wd` - The difference vector.
pub fn hd2io(wh: &Vec3, wd: &Vec3) -> (Vec3, Vec3) { todo!("hd2io") }
