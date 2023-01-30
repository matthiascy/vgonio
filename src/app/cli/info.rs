use crate::{
    app::{
        args::{PrintInfoKind, PrintInfoOptions},
        Config,
    },
    error::Error,
    measure::measurement::{
        BsdfMeasurement, Measurement, MeasurementKind, MicrofacetDistributionMeasurement,
        MicrofacetShadowingMaskingMeasurement,
    },
};
use std::{
    fmt::{Display, Formatter},
    path::PathBuf,
};

impl Display for MicrofacetDistributionMeasurement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MicrofacetDistributionMeasurement\n    - azimuthal angle: {} ~ {} per {}, {} bins\n    - \
             polar angle    : {} ~ {} per {}, {} bins\n",
            self.azimuth.start.prettified(),
            self.azimuth.stop.prettified(),
            self.azimuth.step_size.prettified(),
            self.azimuth_step_count(),
            self.zenith.start.prettified(),
            self.zenith.stop.prettified(),
            self.zenith.step_size.prettified(),
            self.zenith_step_count()
        )
    }
}

impl Display for MicrofacetShadowingMaskingMeasurement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MicrofacetShadowingMaskingMeasurement\n    - azimuthal angle: {} ~ {} per {}, {} \
             bins\n    - polar angle    : {} ~ {} per {}, {} bins\n    - resolution     : {} x {}",
            self.azimuth.start.prettified(),
            self.azimuth.stop.prettified(),
            self.azimuth.step_size.prettified(),
            self.azimuth_step_count(),
            self.zenith.start.prettified(),
            self.zenith.stop.prettified(),
            self.zenith.step_size.prettified(),
            self.zenith_step_count(),
            self.resolution,
            self.resolution
        )
    }
}

/// Prints Vgonio's current configurations.
/// TODO: print default parameters for each measurement
pub fn print(opts: PrintInfoOptions, config: Config) -> Result<(), Error> {
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
        println!("Current configurations:\n{config}");
    }

    if prints[1] {
        println!(
            "Microfacet distribution default parameters:\n\n{}",
            MicrofacetDistributionMeasurement::default()
        );
        println!(
            "Microfacet shadowing and masking default parameters:\n\n{}",
            MicrofacetShadowingMaskingMeasurement::default()
        );
    }

    if prints[2] {
        [
            Measurement {
                kind: MeasurementKind::MicrofacetDistribution(
                    MicrofacetDistributionMeasurement::default(),
                ),
                surfaces: vec![
                    PathBuf::from("path/to/surface1"),
                    PathBuf::from("path/to/surface2"),
                ],
            },
            Measurement {
                kind: MeasurementKind::MicrofacetShadowingMasking(
                    MicrofacetShadowingMaskingMeasurement::default(),
                ),
                surfaces: vec![
                    PathBuf::from("path/to/surface1"),
                    PathBuf::from("path/to/surface2"),
                ],
            },
            Measurement {
                kind: MeasurementKind::Bsdf(BsdfMeasurement::default()),
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
