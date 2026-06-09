//! Fitting orchestration. CLI shape (clap) stays in
//! [`crate::app::cli::cmd_fit`]; Phase 2 moves this module into a capability
//! crate.
//!
//! The capability contract is [`FitRequest`]. It is built from `FitOptions`
//! via `TryFrom` at the CLI boundary (path resolution, per-wavelength file
//! reads, mode flattening), and is the only thing the runtime sees. CLI
//! artifacts (the `"auto"` output magic, raw `[f64; 3]` triplets, the
//! `ndf`/`clausen` flags, the per-wavelength file paths) are absent from the
//! request by design: This module will be moved into `vgonio-fitting`
//! where those concepts do not exist.
//!
//! # Progress reporting (DIST Plan Task 1.15, done)
//!
//! This module is reporter-clean: it emits [`vgn_job_api::progress::Activity`]
//! events through `ctx.progress` rather than calling `vgn_core::cli`, so it can
//! move into a capability crate where `vgn_core::cli` is unreachable. There is
//! no top-level ADAPTER banner here (unlike [`vgn_measurement::orchestration`];
//! `fit` never printed an `Indent::ROOT` "Executing 'vgonio fit'…" line).
//!
//! Phase headings ("Fitting to distribution @...", "Fitting to model
//! ...@...", "Fitting with brute force method...") are
//! [`Activity::PhaseBegin`] + [`Activity::PhaseEnd`] keyed on the
//! `fit.*` namespace ([`PhaseKind::fit_ndf`], [`PhaseKind::fit_brdf`],
//! [`PhaseKind::fit_brdf_measured`], [`PhaseKind::fit_brdf_brute_force`]
//! and friends). Timing rides on `duration_micros` of the corresponding
//! `PhaseEnd` (the old "Took: ..." note is gone). Free-form info
//! ("λ = ...", per-wavelength banners) is [`Activity::Message`] under the
//! active brute / nllsq phase. The unknown-source arm emits
//! [`Activity::Warning`] (it does not abort). Fatal errors return
//! `Err(VgonioError)` and surface through
//! [`vgn_job_api::progress::Lifecycle::Failed`], which reconciles any open
//! phase.
//!
//! [`Activity::PhaseBegin`]: vgn_job_api::progress::Activity::PhaseBegin
//! [`Activity::PhaseEnd`]: vgn_job_api::progress::Activity::PhaseEnd
//! [`Activity::Message`]: vgn_job_api::progress::Activity::Message
//! [`Activity::Warning`]: vgn_job_api::progress::Activity::Warning
//! [`PhaseKind::fit_ndf`]: vgn_job_api::progress::PhaseKind::fit_ndf
//! [`PhaseKind::fit_brdf`]: vgn_job_api::progress::PhaseKind::fit_brdf
//! [`PhaseKind::fit_brdf_measured`]: vgn_job_api::progress::PhaseKind::fit_brdf_measured
//! [`PhaseKind::fit_brdf_brute_force`]: vgn_job_api::progress::PhaseKind::fit_brdf_brute_force

use crate::{measure::bsdf::BsdfMeasurement, pyplot::plot_err, FitOptions};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
use vgn_core::{
    config::Config,
    error::VgonioError,
    optics::IorReg,
    units::{Nanometres, Radians, Rads},
    utils::range::StepRangeIncl,
    BrdfLevel, ErrorMetric, Symmetry, Weighting,
};
use vgn_job_api::{
    context::JobContext,
    progress::{Activity, PhaseInstanceId, PhaseKind, PhaseOutcome},
};

use crate::{
    app::cache::{Cache, UiCache},
    fitting::{MfdFittingData, MicrofacetDistributionFittingProblem},
    measure::mfd::MeasuredNdfData,
    pyplot::plot_per_wavelength_err,
};
use vgn_bxdf::{
    brdf::{
        measured::{
            merl::MerlBrdf, rgl::RglBrdf, yan::Yan18Brdf, ClausenBrdf, MeasuredBrdfKind, VgonioBrdf,
        },
        AnalyticalBrdf,
    },
    distro::MicrofacetDistroKind,
    fitting::{proxy::BrdfProxy, FittingProblem, FittingReport, Roughness as BxdfRoughness},
    AnyMeasured, AnyMeasuredBrdf, BrdfFamily,
};
