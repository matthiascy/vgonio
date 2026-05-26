//! VGONIO EXR Conventions v1 -- attribute keys, encoding tags, layer/channel
//! naming.

/// Conventions version emitted in EXR attributes.
pub const CONVENTIONS_V1: &str = "v1";

/// Top-level EXR attribute keys.
pub mod attr_key {
    /// Conventions version key.
    pub const CONVENTIONS: &str = "vgonio.conventions";
    /// Name and version of the writing software.
    pub const SOFTWARE: &str = "vgonio.software";
    /// File creation timestamp.
    pub const CREATED: &str = "vgonio.created";
    /// Discriminator for the contained output ([`OutputKind`]).
    pub const OUTPUT_KIND: &str = "vgonio.output_kind";
    /// Outgoing-domain encoding ([`OutgoingEncoding`]).
    pub const ENCODING: &str = "vgonio.encoding";
    /// Reference to the partition descriptor file.
    pub const PARTITION_REF: &str = "vgonio.partition_ref";

    /// Projection used for the disc encoding.
    pub const DISC_PROJECTION: &str = "vgonio.disc.projection";
    /// Pixel resolution of the disc encoding.
    pub const DISC_RESOLUTION: &str = "vgonio.disc.resolution";

    /// Number of θ samples in the thetaphi encoding.
    pub const THETAPHI_N_THETA: &str = "vgonio.thetaphi.n_theta";
    /// Number of φ samples in the thetaphi encoding.
    pub const THETAPHI_N_PHI: &str = "vgonio.thetaphi.n_phi";
    /// Inclusive `[min, max]` θ range (radians) of the thetaphi encoding.
    pub const THETAPHI_THETA_RANGE: &str = "vgonio.thetaphi.theta_range_rad";
    /// Inclusive `[min, max]` φ range (radians) of the thetaphi encoding.
    pub const THETAPHI_PHI_RANGE: &str = "vgonio.thetaphi.phi_range_rad";

    /// Number of patches in the patches encoding.
    pub const PATCHES_N_PATCHES: &str = "vgonio.patches.n_patches";
}
/// Conventions version handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConventionsVersion {
    /// Version 1 of the EXR conventions.
    V1,
}

impl ConventionsVersion {
    /// Returns the textual representation written to EXR attributes.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V1 => CONVENTIONS_V1,
        }
    }

    /// Parses the textual representation back into a version handle.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "v1" => Some(Self::V1),
            _ => None,
        }
    }
}

/// Output kind discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputKind {
    /// Bidirectional Scattering Distribution Function.
    Bsdf,
    /// Normal Distribution Function.
    Ndf,
    /// Masking-Shadowing Function.
    Msf,
    /// Slope Distribution Function.
    Sdf,
    /// Heightfield data.
    Heightfield,
}

impl OutputKind {
    /// Returns the textual tag written to EXR attributes.
    pub fn as_str(self) -> &'static str {
        match self {
            OutputKind::Bsdf => "bsdf",
            OutputKind::Ndf => "ndf",
            OutputKind::Msf => "msf",
            OutputKind::Sdf => "sdf",
            OutputKind::Heightfield => "heightfield",
        }
    }

    /// Parses the textual tag back into an [`OutputKind`].
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "bsdf" => Some(OutputKind::Bsdf),
            "ndf" => Some(OutputKind::Ndf),
            "msf" => Some(OutputKind::Msf),
            "sdf" => Some(OutputKind::Sdf),
            "heightfield" => Some(OutputKind::Heightfield),
            _ => None,
        }
    }
}

/// Outgoing-domain encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutgoingEncoding {
    /// Outgoing directions projected onto a 2D disc image.
    Disc,
    /// Outgoing directions parameterised as a (θ, φ) grid.
    Thetaphi,
    /// Outgoing directions binned by partition patch.
    Patches,
}

impl OutgoingEncoding {
    /// Returns the textual tag written to EXR attributes.
    pub fn as_str(self) -> &'static str {
        match self {
            OutgoingEncoding::Disc => "disc",
            OutgoingEncoding::Thetaphi => "thetaphi",
            OutgoingEncoding::Patches => "patches",
        }
    }

    /// Parses the textual tag back into an [`OutgoingEncoding`].
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "disc" => Some(OutgoingEncoding::Disc),
            "thetaphi" => Some(OutgoingEncoding::Thetaphi),
            "patches" => Some(OutgoingEncoding::Patches),
            _ => None,
        }
    }
}

/// Format an incident-direction layer name: `θ{theta_deg}.φ{phi_deg}` with
/// underscores replacing decimal points, two-decimal precision.
///
/// Matches the existing convention in `vgonio-bxdf`.
pub fn layer_name_for_wi(theta_deg: f32, phi_deg: f32) -> String {
    let theta = format!("{:5.2}", theta_deg).replace('.', "_");
    let phi = format!("{:5.2}", phi_deg).replace('.', "_");
    format!("θ{}.φ{}", theta.trim_start(), phi.trim_start())
}

/// Format a channel name for a wavelength, e.g. 400.0 → "400nm".
pub fn channel_name_for_wavelength_nm(nm: f32) -> String {
    if nm.fract() == 0.0 {
        format!("{:.0}nm", nm)
    } else {
        format!("{}nm", nm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conventions_version_parse() {
        assert_eq!(
            ConventionsVersion::parse("v1"),
            Some(ConventionsVersion::V1)
        );
        assert_eq!(ConventionsVersion::parse("v2"), None);
        assert_eq!(ConventionsVersion::V1.as_str(), "v1");
    }

    #[test]
    fn output_kind_round_trip() {
        for k in [
            OutputKind::Bsdf,
            OutputKind::Ndf,
            OutputKind::Msf,
            OutputKind::Sdf,
            OutputKind::Heightfield,
        ] {
            assert_eq!(OutputKind::parse(k.as_str()), Some(k));
        }
    }

    #[test]
    fn outgoing_encoding_round_trip() {
        for e in [
            OutgoingEncoding::Disc,
            OutgoingEncoding::Thetaphi,
            OutgoingEncoding::Patches,
        ] {
            assert_eq!(OutgoingEncoding::parse(e.as_str()), Some(e));
        }
    }

    #[test]
    fn layer_name_format() {
        assert_eq!(layer_name_for_wi(0.0, 0.0), "θ0_00.φ0_00");
        assert_eq!(layer_name_for_wi(15.0, 90.0), "θ15_00.φ90_00");
        assert_eq!(layer_name_for_wi(45.5, 180.25), "θ45_50.φ180_25");
    }

    #[test]
    fn channel_name_format() {
        assert_eq!(channel_name_for_wavelength_nm(400.0), "400nm");
        assert_eq!(channel_name_for_wavelength_nm(632.8), "632.8nm");
    }
}
