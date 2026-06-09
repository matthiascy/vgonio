use crate::{
    app::args::{PrintInfoKind, PrintInfoOptions},
    measure::params::{
        BsdfMeasurementParams, GafMeasurementParams, MeasurementDescription, MeasurementParams,
        NdfMeasurementParams, SurfacePath,
    },
};
use std::path::PathBuf;
use vgn_core::{config::Config, error::VgonioError};
use vgn_io::subdivision::Subdivision;

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
