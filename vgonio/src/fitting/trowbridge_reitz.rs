use crate::fitting::{
    AreaDistributionFittingMode, MicrofacetAreaDistributionModel, MicrofacetMaskingShadowingModel,
    MicrofacetModelFamily, ReflectionModelFamily,
};

/// Trowbridge-Reitz(GGX) microfacet area distribution function.
///
/// $$ D(\mathbf{m}) = \frac{\alpha^2}{\pi \cos^4 \theta_m (\alpha^2 + \tan^2
/// \theta_m)^2} $$
///
/// where $\alpha$ is the width parameter of the NDF, $\theta_m$ is the angle
/// between the microfacet normal and the normal of the surface.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzNDF {
    /// Width parameter of the NDF.
    pub alpha: f64,
    #[cfg(feature = "scaled-ndf-fitting")]
    /// The scale factor of the NDF.
    pub scale: Option<f64>,
}

impl TrowbridgeReitzNDF {
    pub fn default() -> Self {
        TrowbridgeReitzNDF {
            alpha: 0.1,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale: None,
        }
    }

    pub fn default_with_scale() -> Self {
        TrowbridgeReitzNDF {
            alpha: 0.1,
            #[cfg(feature = "scaled-ndf-fitting")]
            scale: Some(1.0),
        }
    }
}

impl MicrofacetAreaDistributionModel for TrowbridgeReitzNDF {
    fn name(&self) -> &'static str {
        #[cfg(feature = "scaled-ndf-fitting")]
        if self.scale.is_none() {
            "Trowbridge-Reitz ADF"
        } else {
            "Scaled Trowbridge-Reitz ADF"
        }
        #[cfg(not(feature = "scaled-ndf-fitting"))]
        "Scaled Trowbridge-Reitz ADF"
    }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::TrowbridgeReitz)
    }

    fn is_isotropic(&self) -> bool { true }

    fn params(&self) -> [f64; 2] { [self.alpha, self.alpha] }

    fn set_params(&mut self, params: [f64; 2]) { self.alpha = params[0]; }

    #[cfg(feature = "scaled-ndf-fitting")]
    fn scale(&self) -> Option<f64> { self.scale }

    #[cfg(feature = "scaled-ndf-fitting")]
    fn set_scale(&mut self, scale: f64) {
        #[cfg(debug_assertions)]
        if self.scale.is_none() {
            panic!("Trying to set the scale on a non-scaled Trowbridge Reitz ADF");
        }
        self.scale.replace(scale);
    }

    fn eval_with_cos_theta_m(&self, cos_theta_m: f64) -> f64 {
        let alpha2 = self.alpha * self.alpha;
        let cos_theta_m2 = cos_theta_m * cos_theta_m;
        let tan_theta_m2 = (1.0 - cos_theta_m2) / cos_theta_m2;
        let cos_theta_m4 = cos_theta_m2 * cos_theta_m2;
        alpha2 / (std::f64::consts::PI * cos_theta_m4 * (alpha2 + tan_theta_m2).powi(2))
    }

    fn clone_box(&self) -> Box<dyn MicrofacetAreaDistributionModel> { Box::new(*self) }
}

/// Trowbridge-Reitz model is also known as GGX model.
pub type GGXIsotropicNDF = TrowbridgeReitzNDF;

/// Anisotropic Trowbridge-Reitz microfacet area distribution function.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzAnisotropicNDF {
    /// Width parameter along the horizontal axis of the NDF.
    pub alpha_x: f32,
    /// Width parameter along the vertical axis of the NDF.
    pub alpha_y: f32,
}

/// Trowbridge-Reitz microfacet masking-shadowing function.
#[derive(Debug, Copy, Clone)]
pub struct TrowbridgeReitzMSF {
    /// Width parameter of the MSF.
    pub width: f64,
}

impl MicrofacetMaskingShadowingModel for TrowbridgeReitzMSF {
    fn name(&self) -> &'static str { "Trowbridge-Reitz MSF" }

    fn family(&self) -> ReflectionModelFamily {
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::TrowbridgeReitz)
    }

    fn param(&self) -> f64 { self.width }

    fn set_param(&mut self, param: f64) { self.width = param; }

    fn eval_with_cos_theta_v(&self, cos_theta_v: f64) -> f64 {
        let alpha2 = self.width * self.width;
        let cos_theta_v2 = cos_theta_v * cos_theta_v;
        let tan_theta_v2 = (1.0 - cos_theta_v2) / cos_theta_v2;
        2.0 / (1.0 + (1.0 + alpha2 * tan_theta_v2).sqrt())
    }

    fn clone_box(&self) -> Box<dyn MicrofacetMaskingShadowingModel> { Box::new(*self) }
}

#[test]
fn trowbridge_reitz_family() {
    let mndf = TrowbridgeReitzNDF::default();
    assert_eq!(
        mndf.family(),
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::TrowbridgeReitz)
    );
    let mmsf = TrowbridgeReitzMSF { width: 0.1 };
    assert_eq!(
        mmsf.family(),
        ReflectionModelFamily::Microfacet(MicrofacetModelFamily::TrowbridgeReitz)
    );
}
