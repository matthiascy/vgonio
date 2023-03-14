use crate::{
    app::{
        args::{PrintInfoKind, PrintInfoOptions},
        Config,
    },
    error::Error,
    measure::measurement::{
        BsdfMeasurement, Measurement, MeasurementKind, MicrofacetMaskingShadowingMeasurement,
        MicrofacetNormalDistributionMeasurement,
    },
};
use std::{
    fmt::{Display, Formatter},
    path::PathBuf,
};

impl Display for MicrofacetNormalDistributionMeasurement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MicrofacetDistributionMeasurement\n    - azimuthal angle: {} ~ {} per {}, {} bins\n    - \
             polar angle    : {} ~ {} per {}, {} bins\n",
            self.azimuth.start.prettified(),
            self.azimuth.stop.prettified(),
            self.azimuth.step_size.prettified(),
            self.azimuth_step_count_inclusive(),
            self.zenith.start.prettified(),
            self.zenith.stop.prettified(),
            self.zenith.step_size.prettified(),
            self.zenith_step_count_inclusive()
        )
    }
}

impl Display for MicrofacetMaskingShadowingMeasurement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MicrofacetShadowingMaskingMeasurement\n    - azimuthal angle: {} ~ {} per {}, {} \
             bins\n    - polar angle    : {} ~ {} per {}, {} bins\n    - resolution     : {} x {}",
            self.azimuth.start.prettified(),
            self.azimuth.stop.prettified(),
            self.azimuth.step_size.prettified(),
            self.azimuth_step_count_inclusive(),
            self.zenith.start.prettified(),
            self.zenith.stop.prettified(),
            self.zenith.step_size.prettified(),
            self.zenith_step_count_inclusive(),
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
        println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        println!("Current configurations:\n\n{config}");
    }

    if prints[1] {
        println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        println!(
            "Microfacet distribution default parameters:\n\n{}",
            MicrofacetNormalDistributionMeasurement::default()
        );
        println!(
            "Microfacet shadowing and masking default parameters:\n\n{}",
            MicrofacetMaskingShadowingMeasurement::default()
        );
        // TODO: print default parameters for brdf measurement (prettified
        // version)
    }

    if prints[2] {
        println!("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
        [
            Measurement {
                kind: MeasurementKind::Mndf(MicrofacetNormalDistributionMeasurement::default()),
                surfaces: vec![
                    PathBuf::from("path/to/surface1"),
                    PathBuf::from("path/to/surface2"),
                ],
            },
            Measurement {
                kind: MeasurementKind::Mmsf(MicrofacetMaskingShadowingMeasurement::default()),
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
