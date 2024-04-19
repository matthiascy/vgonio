use crate::{
    app::{
        args::{PrintInfoKind, PrintInfoOptions},
        Config,
    },
    measure::params::{
        AdfMeasurementMode, AdfMeasurementParams, BsdfMeasurementParams, Measurement,
        MeasurementParams, MsfMeasurementParams,
    },
};
use base::{error::VgonioError, partition::PartitionScheme};
use std::{
    fmt::{Display, Formatter},
    path::PathBuf,
};

impl Display for AdfMeasurementParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.mode {
            AdfMeasurementMode::ByPoints { azimuth, zenith } => {
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
            }
            AdfMeasurementMode::ByPartition { precision, scheme } => {
                write!(
                    f,
                    "MicrofacetDistributionMeasurement\n    - by-partition\n    - scheme: {:?}\n    \
                     - precision: {}",
                    scheme,
                    if scheme == &PartitionScheme::EqualAngle {
                        precision.to_string()
                    } else {
                        precision.theta.prettified()
                    }
                )
            }
        }
    }
}

impl Display for MsfMeasurementParams {
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
            }
            PrintInfoKind::Defaults => {
                prints[1] = true;
            }
            PrintInfoKind::MeasurementDescription => {
                prints[2] = true;
            }
        },
        None => {
            prints = [true, true, true];
        }
    };

    if prints[0] {
        println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        println!("Current configurations:\n\n{config}");
    }

    if prints[1] {
        println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        println!(
            "Microfacet distribution default parameters:\n\n{}",
            AdfMeasurementParams::default()
        );
        println!(
            "Microfacet shadowing and masking default parameters:\n\n{}",
            MsfMeasurementParams::default()
        );
        // TODO: print default parameters for brdf measurement (prettified
        // version)
    }

    if prints[2] {
        println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        [
            Measurement {
                params: MeasurementParams::Adf(AdfMeasurementParams::default()),
                surfaces: vec![
                    PathBuf::from("path/to/surface1"),
                    PathBuf::from("path/to/surface2"),
                ],
            },
            Measurement {
                params: MeasurementParams::Msf(MsfMeasurementParams::default()),
                surfaces: vec![
                    PathBuf::from("path/to/surface1"),
                    PathBuf::from("path/to/surface2"),
                ],
            },
            Measurement {
                params: MeasurementParams::Bsdf(BsdfMeasurementParams::default()),
                surfaces: vec![
                    PathBuf::from("path/to/surface1"),
                    PathBuf::from("path/to/surface2"),
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
