use crate::fitting::{
    MicrofacetAreaDistributionModel, MicrofacetMaskingShadowingModel, MicrofacetModelFamily,
    ReflectionModelFamily,
};

/// Beckmann-Spizzichino microfacet area distribution function.
///
/// Based on the Gaussian distribution of microfacet slopes. If σ is the
/// RMS slope of the microfacets, then the alpha parameter of the Beckmann
/// distribution is given by: $\alpha = \sqrt{2} \sigma$.
#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoADF {
    /// Roughness parameter of the NDF.
    pub alpha: f64,
    #[cfg(feature = "adf-fitting-scaling")]
    /// The scale factor of the NDF.
    pub scale: Option<f64>,
}

impl BeckmannSpizzichinoADF {
    pub fn default() -> Self {
        BeckmannSpizzichinoADF {
            alpha: 0.1,
            #[cfg(feature = "adf-fitting-scaling")]
            scale: None,
        }
    }

    #[cfg(feature = "adf-fitting-scaling")]
    pub fn default_with_scale() -> Self {
        BeckmannSpizzichinoADF {
            alpha: 0.1,
            scale: Some(1.0),
        }
    }
}

impl MicrofacetAreaDistributionModel for BeckmannSpizzichinoADF {
    fn name(&self) -> &'static str {
        #[cfg(feature = "adf-fitting-scaling")]
        if self.scale.is_none() {
            "Beckmann-Spizzichino ADF"
        } else {
            "Scaled Beckmann-Spizzichino ADF"
        }
        #[cfg(not(feature = "adf-fitting-scaling"))]
        "Beckmann-Spizzichino ADF"
    }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::BeckmannSpizzichino)
    }

    fn is_isotropic(&self) -> bool { true }

    fn params(&self) -> [f64; 2] { [self.alpha, self.alpha] }

    fn set_params(&mut self, params: [f64; 2]) { self.alpha = params[0]; }

    #[cfg(feature = "adf-fitting-scaling")]
    fn scale(&self) -> Option<f64> { self.scale }

    #[cfg(feature = "adf-fitting-scaling")]
    fn set_scale(&mut self, scale: f64) {
        #[cfg(debug_assertions)]
        if self.scale.is_none() {
            panic!("Trying to set the scale on a non-scaled Beckmann Spizzichino ADF");
        }
        self.scale.replace(scale);
    }

    fn eval_with_cos_theta_m(&self, cos_theta_m: f64) -> f64 {
        let alpha2 = self.alpha * self.alpha;
        let cos_theta_m2 = cos_theta_m * cos_theta_m;
        let tan_theta_m2 = (1.0 - cos_theta_m2) / cos_theta_m2;
        let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
        (-tan_theta_m2 / alpha2).exp() / (std::f64::consts::PI * alpha2 * cos_theta_m4)
    }

    fn clone_box(&self) -> Box<dyn MicrofacetAreaDistributionModel> { Box::new(*self) }
}

#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoAnisotropicADF {
    /// Roughness parameter along the horizontal axis of the NDF.
    pub roughness_u: f32,
    /// Roughness parameter along the vertical axis of the NDF.
    pub roughness_v: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct BeckmannSpizzichinoMSF {
    /// Roughness parameter of the MSF.
    pub roughness: f64,
}

impl MicrofacetMaskingShadowingModel for BeckmannSpizzichinoMSF {
    fn name(&self) -> &'static str { "Beckmann-Spizzichino MSF" }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::BeckmannSpizzichino)
    }

    fn param(&self) -> f64 { self.roughness }

    fn set_param(&mut self, param: f64) { self.roughness = param; }

    fn eval_with_cos_theta_v(&self, cos_theta_v: f64) -> f64 {
        let alpha = self.roughness;
        let tan_theta_v = (1.0 - cos_theta_v * cos_theta_v).sqrt() / cos_theta_v;
        let cot_theta_v = 1.0 / tan_theta_v;
        let a = cot_theta_v / alpha;
        let denominator =
            1.0 + libm::erf(a) + (-a * a).exp() * alpha * tan_theta_v / std::f64::consts::PI.sqrt();
        2.0 / denominator
    }

    fn clone_box(&self) -> Box<dyn MicrofacetMaskingShadowingModel> { Box::new(*self) }
}

#[test]
fn beckmann_spizzichino_family() {
    let madf = BeckmannSpizzichinoADF { alpha: 0.1 };
    assert_eq!(
        madf.family(),
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::BeckmannSpizzichino)
    );
    let mmsf = BeckmannSpizzichinoMSF { roughness: 0.1 };
    assert_eq!(
        mmsf.family(),
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::BeckmannSpizzichino)
    );
}
