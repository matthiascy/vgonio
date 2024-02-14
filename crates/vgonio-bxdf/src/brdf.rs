use base::math::Vec3;

mod beckmann;
mod trowbridge_reitz;

pub use beckmann::*;
pub use trowbridge_reitz::*;

/// Common interface for BRDFs.
pub trait Brdf {
    /// Evaluates the BRDF (f_r) with the classical parameterisation.
    ///
    /// # Arguments
    ///
    /// * `i` - The incident direction.
    /// * `o` - The outgoing direction.
    fn eval(&self, i: &Vec3, o: &Vec3) -> Vec3;

    /// Evaluates the BRDF (f_r) with the Rusinkiewicz parameterisation.
    ///
    /// # Arguments
    ///
    /// * `h` - The half vector.
    /// * `d` - The difference vector.
    fn eval_hd(&self, h: &Vec3, d: &Vec3) -> Vec3;

    /// Evaluates the projected BRDF with the classical parameterisation.
    ///
    /// $f_r \cdot \cos \theta_i$
    ///
    /// # Arguments
    ///
    /// * `i` - The incident direction.
    /// * `o` - The outgoing direction.
    fn evalp(&self, i: &Vec3, o: &Vec3) -> Vec3;

    /// Evaluates the projected BRDF with the Rusinkiewicz parameterisation.
    ///
    /// $f_r \cdot \cos \theta_i$
    ///
    /// # Arguments
    ///
    /// * `h` - The half vector.
    /// * `d` - The difference vector.
    fn evalp_hd(&self, h: &Vec3, d: &Vec3) -> Vec3;

    /// Evaluates the projected BRDF with importance sampling.
    fn evalp_is(&self, u: f32, v: f32, o: &Vec3, i: &mut Vec3, pdf: &mut f32) -> Vec3;

    /// Importance sample f_r * cos using two uniform variates.
    ///
    /// # Arguments
    ///
    /// * `u` - The first uniform variate.
    /// * `v` - The second uniform variate.
    /// * `o` - The outgoing direction.
    fn sample(&self, u: f32, v: f32, o: &Vec3) -> Vec3;

    /// Evaluates the PDF of a sample.
    fn pdf(&self, i: &Vec3, o: &Vec3) -> f32;
}

pub fn io2hd(i: &Vec3, o: &Vec3) -> (Vec3, Vec3) { todo!("io2hd") }

pub fn hd2io(h: &Vec3, d: &Vec3) -> (Vec3, Vec3) { todo!("hd2io") }

pub trait MicrofacetBrdf {
    /// Parameters of the microfacet BRDF.
    type Params;
}
