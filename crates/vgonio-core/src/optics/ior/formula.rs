//! refractiveindex.info dispersion formulas (forms 1-9) for computing the index of
//! refraction of a material at a given wavelength.
//!
//! Each form yields the *real* part of the refractive index η as a function of wavelength λ
//! in micrometers (μm). The extinction coefficient (imaginary part) κ for materials described
//! by a formula, when present, is supplied as a separate tabulated table by
//! [`super::IorData::Dispersion`].
//!
//! The per-variant coefficient order mirros the upstream `coefficients` list.
use serde::{Deserialize, Serialize};

/// A dispersion formula for the real refractive index η(λ)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DispersionFormula {
    /// Form 1 - Sellmeier: η²−1 = c0 + Σ aᵢ·λ²/(λ²−bᵢ²).
    Sellmeier {
        /// Constant term `c0`.
        c0: f64,
        /// Sum terms `(aᵢ, bᵢ)`.
        terms: Vec<(f64, f64)>,
    },
    /// Form 2 - Sellmeier-2: η²−1 = c0 + Σ aᵢ·λ²/(λ²−bᵢ).
    Sellmeier2 {
        /// Constant term `c0`.
        c0: f64,
        /// Sum terms `(aᵢ, bᵢ)`.
        terms: Vec<(f64, f64)>,
    },
    /// Form 3 - Polynomial: η² = c0 + Σ aᵢ·λ^bᵢ.
    Polynomial {
        /// Constant term `c0`.
        c0: f64,
        /// Sum terms `(aᵢ, bᵢ)`.
        terms: Vec<(f64, f64)>,
    },
    /// Form 4 - refractiveindex.info full form (≤17 coefficients, trailing terms => 0):
    /// η² = c0 + c1·λ^c2/(λ²−c3^c4) + c5·λ^c6/(λ²−c7^c8) + c9·λ^c10 + c11·λ^c12 + c13·λ^c14 +
    /// c15·λ^c16.
    RiiFull {
        /// Coefficients `c0..c16`; trailing missing entries are treated as `0`.
        coeffs: Vec<f64>,
    },
    /// Form 5 - Cauchy: η = c0 + Σ aᵢ·λ^bᵢ.
    Cauchy {
        /// Constant term `c0`.
        c0: f64,
        /// Sum terms `(aᵢ, bᵢ)`.
        terms: Vec<(f64, f64)>,
    },
    /// Form 6 - Gases: η−1 = c0 + Σ aᵢ/(bᵢ − λ⁻²).
    Gases {
        /// Constant term `c0`.
        c0: f64,
        /// Sum terms `(aᵢ, bᵢ)`.
        terms: Vec<(f64, f64)>,
    },
    /// Form 7 - Herzberger: η = c0 + c1/(λ²−0.028) + c2/(λ²−0.028)² + c3·λ² + c4·λ⁴ + c5·λ⁶.
    Herzberger {
        /// Coefficients `[c0, c1, c2, c3, c4, c5]`.
        coeffs: [f64; 6],
    },
    /// Form 8 - Retro: (η²−1)/(η²+2) = c0 + c1·λ²/(λ²−c2) + c3·λ².
    Retro {
        /// Coefficients `[c0, c1, c2, c3]`.
        coeffs: [f64; 4],
    },
    /// Form 9 - Exotic: η² = c0 + c1/(λ²−c2) + c3·(λ−c4)/((λ−c4)²+c5).
    Exotic {
        /// Coefficients `[c0, c1, c2, c3, c4, c5]`.
        coeffs: [f64; 6],
    },
}

impl DispersionFormula {
    /// Evaluates η at wavelength `lambda_um` (micrometres).
    pub fn eval_eta(&self, lambda_um: f64) -> f64 {
        let l = lambda_um;
        let l2 = l * l;
        match self {
            DispersionFormula::Sellmeier { c0, terms } => {
                let s: f64 = terms.iter().map(|&(a, b)| a * l2 / (l2 - b * b)).sum();
                (1.0 + c0 + s).sqrt()
            },
            DispersionFormula::Sellmeier2 { c0, terms } => {
                let s: f64 = terms.iter().map(|&(a, b)| a * l2 / (l2 - b)).sum();
                (1.0 + c0 + s).sqrt()
            },
            DispersionFormula::Polynomial { c0, terms } => {
                let s: f64 = terms.iter().map(|&(a, b)| a * l.powf(b)).sum();
                (c0 + s).sqrt()
            },
            DispersionFormula::RiiFull { coeffs } => {
                // c[i] is 0 when absent.
                let c = |i: usize| coeffs.get(i).copied().unwrap_or(0.0);
                let n2 =
                    c(0) + if c(2) == 0.0 {
                        0.0
                    } else {
                        c(1) * l.powf(c(2)) / (l2 - c(3).powf(c(4)))
                    } + if c(6) == 0.0 {
                        0.0
                    } else {
                        c(5) * l.powf(c(6)) / (l2 - c(7).powf(c(8)))
                    } + c(9) * l.powf(c(10))
                        + c(11) * l.powf(c(12))
                        + c(13) * l.powf(c(14))
                        + c(15) * l.powf(c(16));
                n2.sqrt()
            },
            DispersionFormula::Cauchy { c0, terms } => {
                c0 + terms.iter().map(|&(a, b)| a * l.powf(b)).sum::<f64>()
            },
            DispersionFormula::Gases { c0, terms } => {
                1.0 + c0 + terms.iter().map(|&(a, b)| a / (b - 1.0 / l2)).sum::<f64>()
            },
            DispersionFormula::Herzberger { coeffs } => {
                let d = l2 - 0.028;
                coeffs[0]
                    + coeffs[1] / d
                    + coeffs[2] / (d * d)
                    + coeffs[3] * l2
                    + coeffs[4] * l2 * l2
                    + coeffs[5] * l2 * l2 * l2
            },
            DispersionFormula::Retro { coeffs } => {
                let rhs = coeffs[0] + coeffs[1] * l2 / (l2 - coeffs[2]) + coeffs[3] * l2;
                // (η²−1)/(η²+2) = rhs  ⇒  η² = (1 + 2·rhs)/(1 − rhs)
                ((1.0 + 2.0 * rhs) / (1.0 - rhs)).sqrt()
            },
            DispersionFormula::Exotic { coeffs } => {
                let n2 = coeffs[0]
                    + coeffs[1] / (l2 - coeffs[2])
                    + coeffs[3] * (l - coeffs[4]) / ((l - coeffs[4]).powi(2) + coeffs[5]);
                n2.sqrt()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "expected {b}, got {a} (eps {eps})");
    }

    // Fused silica, Malitson 1965, refractiveindex.info `formula 1`
    // coefficients: 0  0.6961663 0.0684043  0.4079426 0.1162414  0.8974794 9.896161
    // η(0.5876 µm) ≈ 1.458404.
    #[test]
    fn sellmeier_fused_silica() {
        let f = DispersionFormula::Sellmeier {
            c0: 0.0,
            terms: vec![
                (0.6961663, 0.0684043),
                (0.4079426, 0.1162414),
                (0.8974794, 9.896161),
            ],
        };
        approx(f.eval_eta(0.58756), 1.458404, 5e-4);
    }

    // Schott N-BK7, refractiveindex.info `formula 2`
    // coefficients: 0  1.03961212 0.00600069867  0.231792344 0.0200179144  1.01046945 103.560653
    // η(0.58756 µm) ≈ 1.51680.
    #[test]
    fn sellmeier2_n_bk7() {
        let f = DispersionFormula::Sellmeier2 {
            c0: 0.0,
            terms: vec![
                (1.03961212, 0.00600069867),
                (0.231792344, 0.0200179144),
                (1.01046945, 103.560653),
            ],
        };
        approx(f.eval_eta(0.58756), 1.51680, 5e-4);
    }

    // Cauchy: η = 1.5 + 0.005/λ² (λ in µm) => η(0.5) = 1.5 + 0.005/0.25 = 1.52.
    #[test]
    fn cauchy_basic() {
        let f = DispersionFormula::Cauchy {
            c0: 1.5,
            terms: vec![(0.005, -2.0)],
        };
        approx(f.eval_eta(0.5), 1.52, 1e-12);
    }

    // Polynomial: η² = 2.25 + 0.0 => η = 1.5.
    #[test]
    fn polynomial_constant() {
        let f = DispersionFormula::Polynomial {
            c0: 2.25,
            terms: vec![],
        };
        approx(f.eval_eta(0.6), 1.5, 1e-12);
    }

    // RiiFull with only c0 present: η² = 2.25 => η = 1.5; trailing absent coeffs are 0.
    #[test]
    fn rii_full_trailing_zero() {
        let f = DispersionFormula::RiiFull { coeffs: vec![2.25] };
        approx(f.eval_eta(1.0), 1.5, 1e-12);
    }

    // Gases (dry air, Ciddor-like simple check): with c0=0 and one term a/(b−λ⁻²),
    // pick a=1e-6, b=200 => η−1 ≈ 1e-6/(200 − 1/λ²); at λ=0.6 µm: 1/λ²≈2.7778,
    // η ≈ 1 + 1e-6/197.222 ≈ 1.0000000050706.
    #[test]
    fn gases_basic() {
        let f = DispersionFormula::Gases {
            c0: 0.0,
            terms: vec![(1e-6, 200.0)],
        };
        approx(f.eval_eta(0.6), 1.0 + 1e-6 / (200.0 - 1.0 / 0.36), 1e-15);
    }

    // Herzberger: only c0 => η = c0.
    #[test]
    fn herzberger_constant() {
        let f = DispersionFormula::Herzberger {
            coeffs: [1.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        approx(f.eval_eta(0.7), 1.5, 1e-12);
    }

    // Retro: rhs = c0 only; with c0 = 1/4 => (η²−1)/(η²+2) = 1/4 => η² = (1+0.5)/(1−0.25) = 2 => η
    // = √2.
    #[test]
    fn retro_basic() {
        let f = DispersionFormula::Retro {
            coeffs: [0.25, 0.0, 1.0, 0.0],
        };
        approx(f.eval_eta(0.5), 2.0_f64.sqrt(), 1e-12);
    }

    // Exotic: only c0 => η² = c0 => η = √c0; with c2=1.0 to keep the (λ²−c2) term inert when c1=0.
    #[test]
    fn exotic_basic() {
        let f = DispersionFormula::Exotic {
            coeffs: [2.25, 0.0, 1.0, 0.0, 0.0, 1.0],
        };
        approx(f.eval_eta(0.6), 1.5, 1e-12);
    }
}
