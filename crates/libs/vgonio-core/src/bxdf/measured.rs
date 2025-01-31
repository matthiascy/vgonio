/// The kind of the measured BRDF.
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasuredBrdfKind {
    #[cfg_attr(feature = "cli", clap(name = "clausen"))]
    /// The measured BRDF by Clausen.
    Clausen,
    #[cfg_attr(feature = "cli", clap(name = "merl"))]
    /// The MERL BRDF dataset.
    Merl,
    #[cfg_attr(feature = "cli", clap(name = "utia"))]
    /// The measured BRDF by UTIA at Czech Technical University.
    Utia,
    #[cfg_attr(feature = "cli", clap(name = "rgl"))]
    /// The measured BRDF by Dupuy and Jakob in RGL at EPFL.
    Rgl,
    #[cfg_attr(feature = "cli", clap(name = "vgonio"))]
    /// The simulated BRDF by vgonio.
    Vgonio,
    #[cfg_attr(feature = "cli", clap(name = "yan2018"))]
    /// The BRDF model by Yan et al. 2018.
    Yan2018,
    #[cfg_attr(feature = "cli", clap(name = "unknown"))]
    /// Unknown.
    Unknown,
}

/// The origin of the measured BRDF.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum Origin {
    /// Measured in the real world.
    RealWorld,
    /// Measured in a simulation.
    Simulated,
    /// Analytically defined, the data is generated from a mathematical model.
    /// In this case, the measured BRDF is just a collection of samples of
    /// the analytical model.
    Analytical,
}

/// The parameterisation kind of the measured BRDF.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum BrdfParamKind {
    /// The BRDF is parameterised in the half-vector domain.
    HalfVector,
    /// The BRDF is parameterised in the incident and outgoing directions.
    InOutDirs,
}

/// Measured BRDF parameterisation.
pub trait BrdfParam: PartialEq {
    /// Returns the kind of the parameterisation.
    fn kind() -> BrdfParamKind;
}
