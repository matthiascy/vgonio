use crate::{
    bxdf::{
        brdf::{
            analytical::microfacet::{MicrofacetBrdfBK, MicrofacetBrdfTR},
            AnalyticalBrdf,
        },
        distro::MicrofacetDistroKind,
        BrdfProxy, OutgoingDirs,
    },
    math::Sph2,
    optics::ior::Ior,
    units::{rad, Radians},
    utils::range::StepRangeIncl,
    Symmetry, Weighting,
};
use jabr::array::{
    shape::{compute_index_from_strides, compute_strides},
    MemLayout,
};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{Dyn, Matrix, Owned, VecStorage, Vector, U1, U2};
use rayon::iter::{ParallelBridge, ParallelIterator};

/// Initialises the microfacet based BRDF models with the given range of
/// roughness parameters as the initial guess.
///
/// # Arguments
///
/// * `range` - The range of roughness parameters.
/// * `target` - The target microfacet distribution kind.
/// * `symmetry` - The symmetry of the microfacet distribution.
pub(crate) fn init_microfacet_brdf_models(
    range: StepRangeIncl<f64>,
    target: MicrofacetDistroKind,
    symmetry: Symmetry,
) -> Box<[Box<dyn AnalyticalBrdf<Params = [f64; 2]>>]> {
    let count = range.step_count();
    match symmetry {
        Symmetry::Isotropic => (0..count)
            .map(|i| {
                let alpha = range.start + range.step_size * i as f64;
                match target {
                    MicrofacetDistroKind::TrowbridgeReitz => {
                        Box::new(MicrofacetBrdfTR::new(alpha, alpha)) as _
                    },
                    MicrofacetDistroKind::Beckmann => {
                        Box::new(MicrofacetBrdfBK::new(alpha, alpha)) as _
                    },
                }
            })
            .collect(),
        Symmetry::Anisotropic => (0..count)
            .flat_map(|i| {
                let ax = range.start + range.step_size * i as f64;
                (0..count).map(move |j| {
                    let ay = range.start + range.step_size * j as f64;
                    match target {
                        MicrofacetDistroKind::TrowbridgeReitz => {
                            Box::new(MicrofacetBrdfTR::new(ax, ay)) as _
                        },
                        MicrofacetDistroKind::Beckmann => {
                            Box::new(MicrofacetBrdfBK::new(ax, ay)) as _
                        },
                    }
                })
            })
            .collect(),
    }
}

/// A proxy for the BRDF fitting problem using the NLLSQ algorithm.
pub struct NllsqBrdfFittingProxy<'a, const I: Symmetry> {
    /// The proxy for the BRDF data.
    proxy: &'a BrdfProxy<'a>,
    /// Cached IORs for the incident medium.
    iors_i: &'a [Ior],
    /// Cached IORs for the transmitted medium.
    iors_t: &'a [Ior],
    /// The target model being fitted to the measured data.
    pub(crate) model: Box<dyn AnalyticalBrdf<Params = [f64; 2]>>,
    /// The weighting function.
    weighting: Weighting,
    /// The maximum incident angle.
    max_theta_i: Option<Radians>,
    /// The maximum transmitted angle.
    max_theta_o: Option<Radians>,
}

impl<'a, const I: Symmetry> NllsqBrdfFittingProxy<'a, I> {
    /// Creates a new proxy for the BRDF fitting problem using the NLLSQ
    /// algorithm.
    pub fn new(
        proxy: &'a BrdfProxy,
        model: Box<dyn AnalyticalBrdf<Params = [f64; 2]>>,
        weighting: Weighting,
        max_theta_i: Option<Radians>,
        max_theta_o: Option<Radians>,
    ) -> Self {
        let proxy = Self {
            proxy,
            iors_i: proxy.iors_i.as_ref(),
            iors_t: proxy.iors_t.as_ref(),
            model,
            weighting,
            max_theta_i,
            max_theta_o,
        };
        proxy
    }

    /// Returns true if the fitting is applied to a filtered range of incident
    /// and outgoing angles.
    pub fn filtered(&self) -> bool { self.max_theta_i.is_some() || self.max_theta_o.is_some() }

    /// Returns the fitted model.
    pub fn fitted_model(&self) -> &Box<dyn AnalyticalBrdf<Params = [f64; 2]>> { &self.model }

    /// Computes the residuals between the measured and modelled BRDF data.
    fn residuals(&self) -> Vector<f64, Dyn, VecStorage<f64, Dyn, U1>> {
        let jac_shape = self.proxy.filtered_shape(
            self.max_theta_i.unwrap_or(rad!(90.0)).as_f32(),
            self.max_theta_o.unwrap_or(rad!(90.0)).as_f32(),
        );
        let jac_len = jac_shape.iter().product();
        let mut residuals = Vector::<f64, Dyn, VecStorage<f64, Dyn, U1>>::zeros(jac_len);
        let modelled = self.proxy.generate_analytical(&*self.model);
        log::debug!("filtered residuals: {:?}", self.filtered());
        if self.filtered() {
            self.proxy.residuals_filtered(
                &modelled,
                self.weighting,
                residuals.as_mut_slice(),
                self.max_theta_i.unwrap().as_f32(),
                self.max_theta_o.unwrap().as_f32(),
            );
        } else {
            self.proxy
                .residuals(&modelled, self.weighting, residuals.as_mut_slice());
        }
        residuals
    }
}

/// Specialisation for the isotropic case.
impl<'a> NllsqBrdfFittingProxy<'a, { Symmetry::Isotropic }> {
    /// Computes the Jacobian matrix for the isotropic case.
    fn jacobian(&self) -> Matrix<f64, Dyn, U1, Owned<f64, Dyn, U1>> {
        let mut jacobian =
            Matrix::<f64, Dyn, U1, Owned<f64, Dyn, U1>>::zeros(self.proxy.resampled.len());
        let strides = self.proxy.resampled.strides();
        match &self.proxy.o_dirs {
            OutgoingDirs::Grid {
                o_thetas: theta_o,
                o_phis: phi_o,
            } => {
                jacobian
                    .as_mut_slice()
                    .chunks_mut(strides[0])
                    .zip(self.proxy.i_thetas.as_slice().iter())
                    .par_bridge()
                    .for_each(|(per_theta_i, theta_i)| {
                        self.proxy
                            .i_phis
                            .iter()
                            .zip(per_theta_i.chunks_mut(strides[1]))
                            .for_each(|(phi_i, per_phi_i)| {
                                let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                                theta_o
                                    .iter()
                                    .zip(per_phi_i.chunks_mut(strides[2]))
                                    .for_each(|(theta_o, per_theta_o)| {
                                        phi_o
                                            .iter()
                                            .zip(per_theta_o.chunks_mut(strides[3]))
                                            .for_each(|(phi_o, per_phi_o)| {
                                                let vo = Sph2::new(rad!(*theta_o), rad!(*phi_o))
                                                    .to_cartesian();
                                                self.iors_i
                                                    .iter()
                                                    .zip(self.iors_t.iter())
                                                    .zip(per_phi_o.iter_mut())
                                                    .for_each(|((ior_i, ior_t), jac)| {
                                                        *jac = self
                                                            .model
                                                            .pd_iso(&vi, &vo, ior_i, ior_t);
                                                    });
                                            });
                                    });
                            });
                    });
            },
            OutgoingDirs::List {
                o_thetas: theta_o,
                o_phis: phi_o,
                offsets,
            } => {
                jacobian
                    .as_mut_slice()
                    .chunks_mut(strides[0])
                    .zip(self.proxy.i_thetas.as_slice().iter())
                    .par_bridge()
                    .for_each(|(jac_per_theta_i, theta_i)| {
                        for (j, phi_i) in self.proxy.i_phis.iter().enumerate() {
                            let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                            for (k, theta_o) in theta_o.iter().enumerate() {
                                let mut wo_idx = 0;
                                for l in offsets[k]..offsets[k + 1] {
                                    let vo =
                                        Sph2::new(rad!(*theta_o), rad!(phi_o[l])).to_cartesian();
                                    self.iors_i
                                        .iter()
                                        .zip(self.iors_t.iter())
                                        .enumerate()
                                        .for_each(|(m, (ior_i, ior_t))| {
                                            let idx = compute_index_from_strides(
                                                &[j, wo_idx, m],
                                                &strides[1..],
                                            );
                                            jac_per_theta_i[idx] =
                                                self.model.pd_iso(&vi, &vo, ior_i, ior_t);
                                        });
                                    wo_idx += 1;
                                }
                            }
                        }
                    });
            },
        }

        jacobian
    }

    /// Computes the Jacobian matrix for the isotropic case with a filtered
    /// range of incident and outgoing angles.
    fn jacobian_filtered(&self) -> Matrix<f64, Dyn, U1, Owned<f64, Dyn, U1>> {
        let jac_shape = self.proxy.filtered_shape(
            self.max_theta_i.unwrap_or(rad!(90.0)).as_f32(),
            self.max_theta_o.unwrap_or(rad!(90.0)).as_f32(),
        );
        let mut jac_strides = vec![1; jac_shape.len()].into_boxed_slice();
        compute_strides(&jac_shape, &mut jac_strides, MemLayout::RowMajor);
        let jac_len = jac_shape.iter().product();
        let mut jacobian = Matrix::<f64, Dyn, U1, Owned<f64, Dyn, U1>>::zeros(jac_len);

        match &self.proxy.o_dirs {
            OutgoingDirs::Grid {
                o_thetas: theta_o,
                o_phis: phi_o,
            } => {
                jacobian
                    .as_mut_slice()
                    .chunks_mut(jac_strides[0])
                    .zip(self.proxy.i_thetas.as_slice().iter())
                    .par_bridge()
                    .for_each(|(per_theta_i, theta_i)| {
                        self.proxy
                            .i_phis
                            .iter()
                            .zip(per_theta_i.chunks_mut(jac_strides[1]))
                            .for_each(|(phi_i, per_phi_i)| {
                                let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                                theta_o
                                    .iter()
                                    .zip(per_phi_i.chunks_mut(jac_strides[2]))
                                    .for_each(|(theta_o, per_theta_o)| {
                                        phi_o
                                            .iter()
                                            .zip(per_theta_o.chunks_mut(jac_strides[3]))
                                            .for_each(|(phi_o, per_phi_o)| {
                                                let vo = Sph2::new(rad!(*theta_o), rad!(*phi_o))
                                                    .to_cartesian();
                                                self.iors_i
                                                    .iter()
                                                    .zip(self.iors_t.iter())
                                                    .zip(per_phi_o.iter_mut())
                                                    .for_each(|((ior_i, ior_t), jac)| {
                                                        *jac = self
                                                            .model
                                                            .pd_iso(&vi, &vo, ior_i, ior_t);
                                                    });
                                            });
                                    });
                            });
                    });
            },
            OutgoingDirs::List {
                o_thetas: theta_o,
                o_phis: phi_o,
                offsets,
            } => {
                let n_theta_o = theta_o
                    .as_slice()
                    .partition_point(|&x| x < self.max_theta_o.unwrap().as_f32());
                jacobian
                    .as_mut_slice()
                    .chunks_mut(jac_strides[0])
                    .zip(self.proxy.i_thetas.as_slice().iter())
                    .par_bridge()
                    .for_each(|(jac_per_theta_i, theta_i)| {
                        for (j, phi_i) in self.proxy.i_phis.iter().enumerate() {
                            let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                            for (k, theta_o) in theta_o.iter().take(n_theta_o).enumerate() {
                                let mut wo_idx = 0;
                                for l in offsets[k]..offsets[k + 1] {
                                    let vo =
                                        Sph2::new(rad!(*theta_o), rad!(phi_o[l])).to_cartesian();
                                    self.iors_i
                                        .iter()
                                        .zip(self.iors_t.iter())
                                        .enumerate()
                                        .for_each(|(m, (ior_i, ior_t))| {
                                            let idx = compute_index_from_strides(
                                                &[j, wo_idx, m],
                                                &jac_strides[1..],
                                            );
                                            jac_per_theta_i[idx] =
                                                self.model.pd_iso(&vi, &vo, ior_i, ior_t);
                                        });
                                    wo_idx += 1;
                                }
                            }
                        }
                    });
            },
        }

        jacobian
    }
}

/// Specialisation for the anisotropic case.
impl<'a> NllsqBrdfFittingProxy<'a, { Symmetry::Anisotropic }> {
    /// Computes the Jacobian matrix for the anisotropic case.
    fn jacobian(&self) -> Matrix<f64, Dyn, U2, Owned<f64, Dyn, U2>> {
        let shape = self.proxy.resampled.shape();
        let mut jac_shape = vec![1; shape.len() + 1].into_boxed_slice();
        jac_shape[..shape.len()].copy_from_slice(shape);
        jac_shape[shape.len()] = 2;
        let mut jac_strides = vec![1; jac_shape.len()].into_boxed_slice();
        compute_strides(&jac_shape, &mut jac_strides, MemLayout::RowMajor);
        let jac_len = jac_shape.iter().product();
        let mut jacobian = Matrix::<f64, Dyn, U2, Owned<f64, Dyn, U2>>::zeros(jac_len);
        match &self.proxy.o_dirs {
            OutgoingDirs::Grid {
                o_thetas: theta_o,
                o_phis: phi_o,
            } => {
                assert_eq!(jac_strides.len(), 6);
                jacobian
                    .as_mut_slice()
                    .chunks_mut(jac_strides[0])
                    .zip(self.proxy.i_thetas.as_slice().iter())
                    .par_bridge()
                    .for_each(|(per_theta_i, theta_i)| {
                        self.proxy
                            .i_phis
                            .iter()
                            .zip(per_theta_i.chunks_mut(jac_strides[1]))
                            .for_each(|(phi_i, per_phi_i)| {
                                let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                                theta_o
                                    .iter()
                                    .zip(per_phi_i.chunks_mut(jac_strides[2]))
                                    .for_each(|(theta_o, per_theta_o)| {
                                        phi_o
                                            .iter()
                                            .zip(per_theta_o.chunks_mut(jac_strides[3]))
                                            .for_each(|(phi_o, per_phi_o)| {
                                                let vo = Sph2::new(rad!(*theta_o), rad!(*phi_o))
                                                    .to_cartesian();
                                                self.iors_i
                                                    .iter()
                                                    .zip(self.iors_t.iter())
                                                    .zip(per_phi_o.chunks_mut(jac_strides[4]))
                                                    .for_each(|((ior_i, ior_t), jac)| {
                                                        jac.copy_from_slice(
                                                            &self.model.pd(&vi, &vo, ior_i, ior_t),
                                                        );
                                                    });
                                            });
                                    });
                            });
                    });
            },
            OutgoingDirs::List {
                o_thetas: theta_o,
                o_phis: phi_o,
                offsets,
            } => {
                assert_eq!(jac_strides.len(), 5);
                jacobian
                    .as_mut_slice()
                    .chunks_mut(jac_strides[0])
                    .zip(self.proxy.i_thetas.as_slice().iter())
                    .par_bridge()
                    .for_each(|(jac_per_theta_i, theta_i)| {
                        for (j, phi_i) in self.proxy.i_phis.iter().enumerate() {
                            let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                            for (k, theta_o) in theta_o.iter().enumerate() {
                                let mut wo_idx = 0;
                                for l in offsets[k]..offsets[k + 1] {
                                    let vo =
                                        Sph2::new(rad!(*theta_o), rad!(phi_o[l])).to_cartesian();
                                    self.iors_i
                                        .iter()
                                        .zip(self.iors_t.iter())
                                        .enumerate()
                                        .for_each(|(m, (ior_i, ior_t))| {
                                            let idx = compute_index_from_strides(
                                                &[j, wo_idx, m],
                                                &jac_strides[1..],
                                            );
                                            jac_per_theta_i[idx..idx + 2].copy_from_slice(
                                                &self.model.pd(&vi, &vo, ior_i, ior_t),
                                            );
                                        });
                                    wo_idx += 1;
                                }
                            }
                        }
                    });
            },
        }

        jacobian
    }

    /// Computes the Jacobian matrix for the anisotropic case with a filtered
    /// range of incident and outgoing angles.
    fn jacobian_filtered(&self) -> Matrix<f64, Dyn, U2, Owned<f64, Dyn, U2>> {
        let jac_shape = {
            let shape = self.proxy.filtered_shape(
                self.max_theta_i.unwrap_or(rad!(90.0)).as_f32(),
                self.max_theta_o.unwrap_or(rad!(90.0)).as_f32(),
            );
            let mut jac_shape = vec![1; shape.len() + 1].into_boxed_slice();
            jac_shape[..shape.len()].copy_from_slice(&shape);
            jac_shape[shape.len()] = 2;
            shape
        };
        let mut jac_strides = vec![1; jac_shape.len()].into_boxed_slice();
        compute_strides(&jac_shape, &mut jac_strides, MemLayout::RowMajor);
        let mut jacobian =
            Matrix::<f64, Dyn, U2, Owned<f64, Dyn, U2>>::zeros(jac_shape.iter().product());
        match &self.proxy.o_dirs {
            OutgoingDirs::Grid {
                o_thetas: theta_o,
                o_phis: phi_o,
            } => {
                assert_eq!(jac_strides.len(), 6);
                jacobian
                    .as_mut_slice()
                    .chunks_mut(jac_strides[0])
                    .zip(self.proxy.i_thetas.as_slice().iter())
                    .par_bridge()
                    .for_each(|(per_theta_i, theta_i)| {
                        self.proxy
                            .i_phis
                            .iter()
                            .zip(per_theta_i.chunks_mut(jac_strides[1]))
                            .for_each(|(phi_i, per_phi_i)| {
                                let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                                theta_o
                                    .iter()
                                    .zip(per_phi_i.chunks_mut(jac_strides[2]))
                                    .for_each(|(theta_o, per_theta_o)| {
                                        phi_o
                                            .iter()
                                            .zip(per_theta_o.chunks_mut(jac_strides[3]))
                                            .for_each(|(phi_o, per_phi_o)| {
                                                let vo = Sph2::new(rad!(*theta_o), rad!(*phi_o))
                                                    .to_cartesian();
                                                self.iors_i
                                                    .iter()
                                                    .zip(self.iors_t.iter())
                                                    .zip(per_phi_o.chunks_mut(jac_strides[4]))
                                                    .for_each(|((ior_i, ior_t), jac)| {
                                                        jac.copy_from_slice(
                                                            &self.model.pd(&vi, &vo, ior_i, ior_t),
                                                        );
                                                    });
                                            });
                                    });
                            });
                    });
            },
            OutgoingDirs::List {
                o_thetas: theta_o,
                o_phis: phi_o,
                offsets,
            } => {
                assert_eq!(jac_strides.len(), 5);
                jacobian
                    .as_mut_slice()
                    .chunks_mut(jac_strides[0])
                    .zip(self.proxy.i_thetas.as_slice().iter())
                    .par_bridge()
                    .for_each(|(jac_per_theta_i, theta_i)| {
                        for (j, phi_i) in self.proxy.i_phis.iter().enumerate() {
                            let vi = Sph2::new(rad!(*theta_i), rad!(*phi_i)).to_cartesian();
                            for (k, theta_o) in theta_o.iter().enumerate() {
                                let mut wo_idx = 0;
                                for l in offsets[k]..offsets[k + 1] {
                                    let vo =
                                        Sph2::new(rad!(*theta_o), rad!(phi_o[l])).to_cartesian();
                                    self.iors_i
                                        .iter()
                                        .zip(self.iors_t.iter())
                                        .enumerate()
                                        .for_each(|(m, (ior_i, ior_t))| {
                                            let idx = compute_index_from_strides(
                                                &[j, wo_idx, m],
                                                &jac_strides[1..],
                                            );
                                            jac_per_theta_i[idx..idx + 2].copy_from_slice(
                                                &self.model.pd(&vi, &vo, ior_i, ior_t),
                                            );
                                        });
                                    wo_idx += 1;
                                }
                            }
                        }
                    });
            },
        }

        jacobian
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U1> for NllsqBrdfFittingProxy<'a, { Symmetry::Isotropic }> {
    type ResidualStorage = VecStorage<f64, Dyn, U1>;

    type JacobianStorage = Owned<f64, Dyn, U1>;

    type ParameterStorage = Owned<f64, U1, U1>;

    fn set_params(&mut self, x: &Vector<f64, U1, Self::ParameterStorage>) {
        self.model.set_params(&[x[0], x[0]]);
    }

    fn params(&self) -> Vector<f64, U1, Self::ParameterStorage> {
        Vector::<f64, U1, Self::ParameterStorage>::new(self.model.params()[0])
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(self.residuals())
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U1, Self::JacobianStorage>> {
        if self.filtered() {
            log::debug!("filtered jacobian");
            Some(self.jacobian_filtered())
        } else {
            log::debug!("unfiltered jacobian");
            Some(self.jacobian())
        }
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, U2>
    for NllsqBrdfFittingProxy<'a, { Symmetry::Anisotropic }>
{
    type ResidualStorage = VecStorage<f64, Dyn, U1>;

    type JacobianStorage = Owned<f64, Dyn, U2>;

    type ParameterStorage = Owned<f64, U2, U1>;

    fn set_params(&mut self, x: &Vector<f64, U2, Self::ParameterStorage>) {
        self.model.set_params(&[x[0], x[1]]);
    }

    fn params(&self) -> Vector<f64, U2, Self::ParameterStorage> {
        Vector::<f64, U2, Self::ParameterStorage>::from(self.model.params())
    }

    fn residuals(&self) -> Option<Vector<f64, Dyn, Self::ResidualStorage>> {
        Some(self.residuals())
    }

    fn jacobian(&self) -> Option<Matrix<f64, Dyn, U2, Self::JacobianStorage>> {
        if self.filtered() {
            Some(self.jacobian_filtered())
        } else {
            Some(self.jacobian())
        }
    }
}
