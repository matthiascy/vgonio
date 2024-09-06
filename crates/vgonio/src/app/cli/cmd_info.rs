use crate::{
    app::{
        args::{PrintInfoKind, PrintInfoOptions},
        Config,
    },
    measure::params::{
        BsdfMeasurementParams, GafMeasurementParams, MeasurementDescription, MeasurementParams,
        NdfMeasurementMode, NdfMeasurementParams, SurfacePath,
    },
};
use base::error::VgonioError;
use std::{
    fmt::{Display, Formatter},
    path::PathBuf,
};
use surf::subdivision::Subdivision;

impl Display for NdfMeasurementParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.mode {
            NdfMeasurementMode::ByPoints { azimuth, zenith } => {
                write!(
                    f,
                    "MicrofacetDistributionMeasurement\n    - by-points\n    - azimuthal angle: \
                     {} ~ {} per {}, {} bins\n    - polar angle    : {} ~ {} per {}, {} bins\n",
                    azimuth.start.prettified(),
                    azimuth.stop.prettified(),
                    azimuth.step_size.prettified(),
                    azimuth.step_count_wrapped(),
                    zenith.start.prettified(),
                    zenith.stop.prettified(),
                    zenith.step_size.prettified(),
                    zenith.step_count_wrapped(),
                )
            },
            NdfMeasurementMode::ByPartition { precision } => {
                write!(
                    f,
                    "MicrofacetDistributionMeasurement\n    - by-partition\n    - scheme: \
                     Beckers\n    - precision: {}",
                    precision.prettified()
                )
            },
        }
    }
}

impl Display for GafMeasurementParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MicrofacetShadowingMaskingMeasurement\n    - azimuthal angle: {} ~ {} per {}, {} \
             bins\n    - polar angle    : {} ~ {} per {}, {} bins\n    - resolution     : {} x {}",
            self.azimuth.start.prettified(),
            self.azimuth.stop.prettified(),
            self.azimuth.step_size.prettified(),
            self.azimuth.step_count_wrapped(),
            self.zenith.start.prettified(),
            self.zenith.stop.prettified(),
            self.zenith.step_size.prettified(),
            self.zenith.step_count_wrapped(),
            self.resolution,
            self.resolution
        )
    }
}

/// Prints Vgonio's current configurations.
/// TODO: print default parameters for each measurement
pub fn print_info(opts: PrintInfoOptions, config: Config) -> Result<(), VgonioError> {
    let mut prints = [false, false, false];
    match opts.kind {
        Some(kind) => match kind {
            PrintInfoKind::Config => {
                prints[0] = true;
            },
            PrintInfoKind::Defaults => {
                prints[1] = true;
            },
            PrintInfoKind::MeasurementDescription => {
                prints[2] = true;
            },
        },
        None => {
            prints = [true, true, true];
        },
    };

    if prints[0] {
        println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        println!("Current configurations:\n\n{config}");
    }

    if prints[1] {
        println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        println!(
            "Microfacet distribution default parameters:\n\n{}",
            NdfMeasurementParams::default()
        );
        println!(
            "Microfacet shadowing and masking default parameters:\n\n{}",
            GafMeasurementParams::default()
        );
        // TODO: print default parameters for brdf measurement (prettified
        // version)
    }

    if prints[2] {
        println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        [
            MeasurementDescription {
                params: MeasurementParams::Ndf(NdfMeasurementParams::default()),
                surfaces: vec![
                    SurfacePath::new(PathBuf::from("path/to/surface1"), None),
                    SurfacePath::new(
                        PathBuf::from("path/to/surface2"),
                        Some(Subdivision::Curved(3)),
                    ),
                ],
            },
            MeasurementDescription {
                params: MeasurementParams::Gaf(GafMeasurementParams::default()),
                surfaces: vec![
                    SurfacePath::new(PathBuf::from("path/to/surface1"), None),
                    SurfacePath::new(
                        PathBuf::from("path/to/surface2"),
                        Some(Subdivision::Wiggly {
                            level: 2,
                            offset: 100,
                        }),
                    ),
                ],
            },
            MeasurementDescription {
                params: MeasurementParams::Bsdf(BsdfMeasurementParams::default()),
                surfaces: vec![
                    SurfacePath::new(PathBuf::from("path/to/surface1"), None),
                    SurfacePath::new(
                        PathBuf::from("path/to/surface2"),
                        Some(Subdivision::Curved(3)),
                    ),
                ],
            },
        ]
        .into_iter()
        .for_each(|m| {
            print!("---\n{}", serde_yaml::to_string(&m).unwrap());
        });
    }

    Ok(())
}
