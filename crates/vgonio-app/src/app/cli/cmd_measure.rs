use crate::{
    app::{args::OutputFormat, cache::Cache},
    io::{OutputFileFormatOption, OutputOptions},
    measure,
    measure::params::{MeasurementDescription, MeasurementParams, NdfMeasurementMode},
};
use std::{path::PathBuf, time::Instant};
use vgn_core::{
    cli::{cli_error, cli_note, cli_step, cli_success, Indent},
    config::Config,
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
};

/// Measure different metrics of the micro-surface.
pub fn measure(opts: MeasureOptions, config: Config) -> Result<(), VgonioError> {
    log::info!("{:#?}", config);

    if let Some(nthreads) = opts.nthreads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads as usize)
            .build()
            .map_err(|err| {
                VgonioError::new(
                    &format!(
                        "Failed to create measurement thread pool with {} threads: {}",
                        nthreads, err
                    ),
                    None,
                )
            })?;
        pool.install(|| measure_impl(opts, config))
    } else {
        measure_impl(opts, config)
    }
}

fn measure_impl(opts: MeasureOptions, config: Config) -> Result<(), VgonioError> {
    cli_step!(
        Indent::ROOT,
        "Executing 'vgonio measure' with a thread pool of size: {}",
        rayon::current_num_threads()
    );

    cli_step!(Indent::SECTION, "Reading measurement description files...");
    let measurements = opts
        .inputs
        .iter()
        .flat_map(|meas_path| {
            config.resolve_path(meas_path).map(|resolved| {
                match MeasurementDescription::load(&resolved) {
                    Ok(meas) => Some(meas),
                    Err(err) => {
                        log::warn!("Failed to load measurement description file: {}", err);
                        None
                    },
                }
            })
        })
        .filter_map(|meas| meas)
        .flatten()
        .collect::<Vec<_>>();
    cli_success!(Indent::SUBSECTION, "{} measurement(s)", measurements.len());

    let cache = Cache::new(config.cache_dir());

    let (tasks, num_surfs) = cache.write(|cache| {
        // Load data files: refractive indices, spd etc. if needed.
        if measurements.iter().any(|meas| meas.params.is_bsdf()) {
            cli_step!(
                Indent::SECTION,
                "Loading data files (refractive indices, spd etc.)..."
            );
            cache.load_ior_database(&config);
            cli_success!(Indent::SUBSECTION, "Successfully loaded data files");
        }

        cli_step!(Indent::SECTION, "Resolving and loading micro-surfaces...");
        let tasks = measurements
            .into_iter()
            .filter_map(|meas| {
                cache
                    .load_micro_surfaces(&config, &meas.surfaces, config.user.triangulation)
                    .ok()
                    .map(|surfaces| (meas, surfaces))
            })
            .collect::<Vec<_>>();
        cli_success!(
            Indent::SUBSECTION,
            "{} micro-surface(s) loaded",
            cache.num_micro_surfaces()
        );

        #[cfg(debug_assertions)]
        cache
            .loaded_micro_surface_paths()
            .unwrap()
            .iter()
            .for_each(|s| cli_note!(Indent::DETAIL, "{}", s.display()));

        (tasks, cache.num_micro_surfaces())
    });

    if num_surfs == 0 {
        cli_error!(Indent::SECTION, "No micro-surface to measure. Exiting...");
        return Ok(());
    }

    let start_time = Instant::now();
    for (desc, surfaces) in tasks {
        let measurement_start_time = std::time::SystemTime::now();
        let measured = match desc.params {
            MeasurementParams::Bsdf(params) => {
                if let Err(reason) = params.check_supported_for_measurement() {
                    return Err(VgonioError::new(
                        &format!(
                            "Unsupported simulation method for this build: {}. Rebuild with the \
                             required backend feature or choose another simulation kind.",
                            reason
                        ),
                        None,
                    ));
                }
                cli_step!(
                    Indent::SECTION,
                    "Launch BSDF measurement at {}
    • parameters:
      + incident medium: {:?}
      + transmitted medium: {:?}
      + emitter:
        - num rays: {}
        - num sectors: {}
        - max bounces: {}
        - spectrum: {}
        - polar angle: {}
        - azimuthal angle: {}",
                    chrono::DateTime::<chrono::Utc>::from(measurement_start_time),
                    params.incident_medium,
                    params.transmitted_medium,
                    params.emitter.num_rays,
                    params.emitter.num_sectors,
                    params.emitter.max_bounces,
                    params.emitter.spectrum,
                    params.emitter.zenith.pretty_print(),
                    params.emitter.azimuth.pretty_print()
                );
                for receiver in &params.receivers {
                    println!(
                        "      + receiver:
        - domain: {}
        - scheme: {:?}
        - precision: {}",
                        receiver.domain, receiver.scheme, receiver.precision
                    );
                }
                cache.read(|cache| measure::bsdf::measure_bsdf_rt(params, &surfaces, cache))
            },
            MeasurementParams::Ndf(measurement) => {
                match &measurement.mode {
                    NdfMeasurementMode::ByPoints { azimuth, zenith } => {
                        cli_step!(
                            Indent::SECTION,
                            "Measuring microfacet area distribution:
    • parameters:
      + mode: by points
        + azimuth: {}
        + zenith: {}",
                            azimuth.pretty_print(),
                            zenith.pretty_print()
                        );
                    },
                    NdfMeasurementMode::ByPartition { precision } => {
                        cli_step!(
                            Indent::SECTION,
                            "Measuring microfacet area distribution:
    • parameters:
       + mode: by partition
           + scheme: Beckers
           + precision: {}",
                            precision.prettified()
                        );
                    },
                }
                cache.read(|cache| {
                    measure::mfd::measure_area_distribution(measurement, &surfaces, cache)
                })
            },
            MeasurementParams::Gaf(measurement) => {
                cli_step!(
                    Indent::SECTION,
                    "Measuring microfacet masking-shadowing function:
    • parameters:
      + azimuth: {}
      + zenith: {}
      + resolution: {} x {}",
                    measurement.azimuth.pretty_print(),
                    measurement.zenith.pretty_print(),
                    measurement.resolution,
                    measurement.resolution
                );

                #[cfg(debug_assertions)]
                log::warn!(
                    "Debug mode is enabled. Measuring MMSF in debug mode is not recommended."
                );
                cache.read(|cache| {
                    measure::mfd::measure_masking_shadowing_function(measurement, &surfaces, cache)
                })
            },
            MeasurementParams::Sdf(params) => {
                cli_step!(Indent::SECTION, "Measuring slope distribution function...");
                cache.read(|cache| {
                    measure::mfd::measure_slope_distribution(&surfaces, params, cache)
                })
            },
        };

        cli_success!(
            Indent::SUBSECTION,
            "Measurement finished in {} secs.",
            measurement_start_time.elapsed().unwrap().as_secs_f32()
        );

        let formats = match opts.output_format {
            OutputFormat::Vgmo => vec![OutputFileFormatOption::Vgmo {
                encoding: opts.encoding,
                compression: opts.compression,
            }]
            .into_boxed_slice(),
            OutputFormat::Exr => vec![OutputFileFormatOption::Exr {
                resolution: opts.resolution,
            }]
            .into_boxed_slice(),
            OutputFormat::VgmoExr => vec![
                OutputFileFormatOption::Vgmo {
                    encoding: opts.encoding,
                    compression: opts.compression,
                },
                OutputFileFormatOption::Exr {
                    resolution: opts.resolution,
                },
            ]
            .into_boxed_slice(),
        };

        crate::io::write_measured_data_to_file(
            &measured,
            &cache,
            &config,
            OutputOptions {
                dir: opts.output.clone(),
                formats,
            },
        )?;

        cli_success!(Indent::SUBSECTION, "Done!");
    }

    cli_success!(
        Indent::SUBSECTION,
        "Finished in {:.2} s",
        start_time.elapsed().as_secs_f32()
    );

    Ok(())
}

/// Options for the `measure` command.
#[derive(clap::Args, Debug)]
#[clap(about = "Measure different aspects of the micro-surface.")]
pub struct MeasureOptions {
    #[arg(
        short,
        long,
        required = true,
        num_args(1..),
        help = "The measurement description files or directories."
    )]
    pub inputs: Vec<PathBuf>,

    #[arg(
        short,
        long,
        help = "The path where stores the simulation data. Use // at the start of the\npath to \
                set the output path relative to the input file location.\nOutput path can also be \
                specified in configuration file."
    )]
    pub output: Option<PathBuf>,

    #[arg(
        short = 'f',
        long,
        default_value_t = OutputFormat::Vgmo,
        help = "The format of the measurement output. If not specified, the format\nwill be the \
                vgonio internal file format.",
    )]
    pub output_format: OutputFormat,

    #[arg(
        short,
        long,
        default_value_t = 512,
        help = "The resolution of the measurement output in case the output is image.\nIf not \
                specified, the resolution will be 512."
    )]
    pub resolution: u32,

    #[arg(
        short,
        long,
        required_if_eq("output_format", "vgms"),
        default_value_t = FileEncoding::Binary,
        help = "Data format for the measurement output.\nOnly used when output format is vgms."
    )]
    pub encoding: FileEncoding,

    #[arg(
    short,
    long,
    required_if_eq("output_format", "vgms"),
    default_value_t = CompressionScheme::None,
    help = "Data compression for the measurement output."
    )]
    pub compression: CompressionScheme,

    #[arg(
        short,
        long = "num-threads",
        help = "The number of threads in the thread pool"
    )]
    pub nthreads: Option<u32>,

    #[clap(
        long,
        help = "Show detailed statistics about memory and time\nusage during the measurement"
    )]
    pub print_stats: bool,
}
