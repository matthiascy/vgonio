use std::{path::PathBuf, time::Instant};

use crate::{
    acq::{
        self,
        measurement::{
            BsdfMeasurement, Measurement, MeasurementDetails, MicrofacetDistributionMeasurement,
            MicrofacetShadowingMaskingMeasurement, SimulationKind,
        },
        util::SphericalPartition,
        CollectorScheme, RtcMethod,
    },
    app::{
        args::{MeasureOptions, MeasurementKind, SubCommand},
        cache::{resolve_path, Cache, MicroSurfaceHandle},
        Config,
    },
    msurf::AxisAlignment,
    Error,
};

pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
pub const BRIGHT_RED: &str = "\u{001b}[31m";
pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
pub const RESET: &str = "\u{001b}[0m";

/// Execute the given command.
/// This function is the entry point of vgonio CLI.
pub fn execute(cmd: SubCommand, config: Config) -> Result<(), Error> {
    match cmd {
        SubCommand::Measure(opts) => measure(opts, config),
        SubCommand::Info => print_info(config),
    }
}

/// Measure different metrics of the micro-surface.
fn measure(opts: MeasureOptions, config: Config) -> Result<(), Error> {
    log::info!("{:#?}", config);
    // Configure thread pool for parallelism.
    if let Some(nthreads) = opts.nthreads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads as usize)
            .build_global()
            .unwrap();
    }
    println!(
        "{BRIGHT_YELLOW}>{RESET} Executing 'vgonio measure' with a thread pool of size: {}",
        rayon::current_num_threads()
    );

    let mut cache = Cache::new(config.cache_dir());
    println!("  {BRIGHT_YELLOW}>{RESET} Reading measurement description files...");
    let measurements = {
        match opts.fast {
            None => opts
                .inputs
                .iter()
                .filter_map(|meas_path| {
                    let resolved = resolve_path(&config.cwd, Some(meas_path));
                    match Measurement::load(&resolved) {
                        Ok(meas) => Some(meas),
                        Err(err) => {
                            log::warn!("Failed to load measurement description file: {}", err);
                            None
                        }
                    }
                })
                .flatten()
                .collect::<Vec<_>>(),
            Some(kinds) => kinds
                .iter()
                .map(|kind| match kind {
                    MeasurementKind::Bsdf => Measurement {
                        details: MeasurementDetails::Bsdf(BsdfMeasurement::default()),
                        surfaces: opts.inputs.clone(),
                    },
                    MeasurementKind::MicrofacetDistribution => Measurement {
                        details: MeasurementDetails::MicrofacetDistribution(
                            MicrofacetDistributionMeasurement::default(),
                        ),
                        surfaces: opts.inputs.clone(),
                    },
                    MeasurementKind::MicrofacetShadowingMasking => Measurement {
                        details: MeasurementDetails::MicrofacetShadowMasking(
                            MicrofacetShadowingMaskingMeasurement::default(),
                        ),
                        surfaces: opts.inputs.clone(),
                    },
                })
                .collect::<Vec<_>>(),
        }
    };
    println!(
        "    {BRIGHT_CYAN}✓{RESET} {} measurement(s)",
        measurements.len()
    );

    if measurements.iter().any(|meas| meas.details.is_bsdf()) {
        // Load data files: refractive indices, spd etc.
        println!("  {BRIGHT_YELLOW}>{RESET} Loading data files (refractive indices, spd etc.)...");
        cache.load_ior_database(&config);
        println!("    {BRIGHT_CYAN}✓{RESET} Successfully load data files");
    }

    println!("  {BRIGHT_YELLOW}>{RESET} Resolving and loading micro-surfaces...");
    let tasks = measurements
        .into_iter()
        .filter_map(|meas| {
            cache
                .load_micro_surfaces(&config, &meas.surfaces, Some(AxisAlignment::XZ))
                .ok()
                .map(|surfaces| (meas, surfaces))
        })
        .collect::<Vec<_>>();
    println!(
        "    {BRIGHT_CYAN}✓{RESET} {} micro-surface(s) loaded",
        cache.num_micro_surfaces()
    );
    for (measurement, surfaces) in tasks {
        match measurement.details {
            MeasurementDetails::Bsdf(measurement) => {
                let start = std::time::SystemTime::now();
                let collector_info = match measurement.collector.scheme {
                    CollectorScheme::Partitioned { domain, partition } => match partition {
                        SphericalPartition::EqualAngle { zenith, azimuth } => {
                            format!(
                                "        - domain: {}\n        - partition: {}\n          - polar \
                                 angle: {} ~ {}, per {}\n          - azimuthal angle {} ~ {}, per \
                                 {}",
                                domain,
                                "equal angle",
                                zenith.start.prettified(),
                                zenith.stop.prettified(),
                                zenith.step_size.prettified(),
                                azimuth.stop.prettified(),
                                azimuth.stop.prettified(),
                                azimuth.step_size.prettified()
                            )
                        }
                        SphericalPartition::EqualArea { zenith, azimuth } => {
                            format!(
                                "        - domain: {}\n        - partition: {}\n          - polar \
                                 angle: {} ~ {}, {} steps\n          - azimuthal angle: {} ~ {}, \
                                 per {}",
                                domain,
                                "equal area",
                                zenith.start.prettified(),
                                zenith.stop.prettified(),
                                zenith.step_count,
                                azimuth.start.prettified(),
                                azimuth.stop.prettified(),
                                azimuth.step_size.prettified()
                            )
                        }
                        SphericalPartition::EqualProjectedArea { zenith, azimuth } => {
                            format!(
                                "        - domain: {}\n        - partition: {}\n          - polar \
                                 angle: {} - {}, {} steps\n          - azimuthal angle {} - {}, \
                                 per {}",
                                domain,
                                "equal projected area",
                                zenith.start.prettified(),
                                zenith.stop.prettified(),
                                zenith.step_count,
                                azimuth.start.prettified(),
                                azimuth.stop.prettified(),
                                azimuth.step_size.prettified()
                            )
                        }
                    },
                    CollectorScheme::Individual {
                        domain,
                        shape,
                        zenith,
                        azimuth,
                    } => {
                        format!(
                            "        - domain: {domain}\n- shape: {shape:?}\n- polar angle: \
                             {zenith:?}\n- azimuthal angle {azimuth:?}\n",
                        )
                    }
                };

                let surfaces_str = surfaces
                    .iter()
                    .map(|s| format!("       - {}", s.path.display()))
                    .collect::<Vec<_>>()
                    .join("\n");

                println!(
                    "  {BRIGHT_YELLOW}>{RESET} Launch BSDF measurement at {}
    • parameters:
      + incident medium: {:?}
      + transmitted medium: {:?}
      + surfaces:\n {}
      + emitter:
        - radius: {}
        - num rays: {}
        - max bounces: {}
        - spectrum: {} ~ {} per {}
        - polar angle: {} ~ {} per {}
        - azimuthal angle: {} ~ {} per {}
      + collector:
        - radius: {}\n{}",
                    chrono::DateTime::<chrono::Utc>::from(start),
                    measurement.incident_medium,
                    measurement.transmitted_medium,
                    surfaces_str,
                    measurement.emitter.radius(),
                    measurement.emitter.num_rays,
                    measurement.emitter.max_bounces,
                    measurement.emitter.spectrum.start,
                    measurement.emitter.spectrum.stop,
                    measurement.emitter.spectrum.step_size,
                    measurement.emitter.zenith.start.prettified(),
                    measurement.emitter.zenith.stop.prettified(),
                    measurement.emitter.zenith.step_size.prettified(),
                    measurement.emitter.azimuth.start.prettified(),
                    measurement.emitter.azimuth.stop.prettified(),
                    measurement.emitter.azimuth.step_size.prettified(),
                    measurement.collector.radius,
                    collector_info
                );
                match measurement.sim_kind {
                    SimulationKind::GeomOptics { method } => {
                        println!(
                            "    {BRIGHT_YELLOW}>{RESET} Measuring {} with geometric optics...",
                            measurement.bsdf_kind
                        );
                        match method {
                            #[cfg(feature = "embree")]
                            RtcMethod::Standard => {
                                acq::bsdf::measure_bsdf_embree_rt(measurement, &cache, &surfaces);
                            }
                            RtcMethod::Grid => {
                                acq::bsdf::measure_bsdf_grid_rt(measurement, &cache, &surfaces);
                            }
                        }
                    }
                    SimulationKind::WaveOptics => {
                        println!(
                            "    {BRIGHT_YELLOW}>{RESET} Measuring {} with wave optics...",
                            measurement.bsdf_kind
                        );
                        // TODO: implement
                    }
                }
                println!("  {BRIGHT_YELLOW}>{RESET} Saving results...");
                // todo: save to file
                println!(
                    "    {BRIGHT_CYAN}✓{RESET} Successfully saved to \"{}\"",
                    config.output_dir().display()
                );
                println!(
                    "    {BRIGHT_CYAN}✓{RESET} Finished in {:.2} s",
                    start.elapsed().unwrap().as_secs_f32()
                );
            }
            MeasurementDetails::MicrofacetDistribution(measurement) => {
                measure_microfacet_distribution(
                    measurement,
                    &surfaces,
                    &cache,
                    &config,
                    &opts.output,
                )
                .map_err(|err| {
                    eprintln!("  {BRIGHT_RED}✗{RESET} {err}");
                    err
                })?;
            }
            MeasurementDetails::MicrofacetShadowMasking(measurement) => {
                measure_microfacet_shadowing_masking(
                    measurement,
                    &surfaces,
                    &cache,
                    &config,
                    &opts.output,
                )
                .map_err(|err| {
                    eprintln!("  {BRIGHT_RED}✗{RESET} {err}");
                    err
                })?;
            }
        }
    }
    Ok(())
}

/// Prints Vgonio's current configurations.
/// TODO: print default parameters for each measurement
fn print_info(config: Config) -> Result<(), Error> {
    println!("\n{config}");
    Ok(())
}

/// Measures the microfacet distribution of the given micro-surface and saves
/// the result to the given output directory.
fn measure_microfacet_distribution(
    measurement: MicrofacetDistributionMeasurement,
    surfaces: &[MicroSurfaceHandle],
    cache: &Cache,
    config: &Config,
    output: &Option<PathBuf>,
) -> Result<(), Error> {
    println!(
        "  {BRIGHT_YELLOW}>{RESET} Measuring microfacet distribution:
    • parameters:
      + azimuth: {} ~ {} per {}
      + zenith: {} ~ {} per {}",
        measurement.azimuth.start.prettified(),
        measurement.azimuth.stop.prettified(),
        measurement.azimuth.step_size.prettified(),
        measurement.zenith.start.prettified(),
        measurement.zenith.stop.prettified(),
        measurement.zenith.step_size.prettified()
    );
    let start_time = Instant::now();
    let distributions =
        acq::microfacet::measure_microfacet_distribution(measurement, surfaces, cache);
    let duration = Instant::now() - start_time;
    println!(
        "    {BRIGHT_CYAN}✓{RESET} Measurement finished in {} secs.",
        duration.as_secs_f32()
    );
    let output_dir = resolve_output_dir(config, output)?;
    println!("    {BRIGHT_YELLOW}>{RESET} Saving measurement data...");
    for (distrib, surface) in distributions.iter().zip(surfaces.iter()) {
        let filename = format!(
            "microfacet-distribution-{}.txt",
            surface
                .path
                .file_stem()
                .unwrap()
                .to_ascii_lowercase()
                .to_str()
                .unwrap()
        );
        let filepath = output_dir.join(filename);
        println!(
            "      {BRIGHT_CYAN}-{RESET} Saving to \"{}\"",
            filepath.display()
        );
        distrib.save_ascii(&filepath).unwrap_or_else(|err| {
            eprintln!(
                "        {BRIGHT_RED}!{RESET} Failed to save to \"{}\": {}",
                filepath.display(),
                err
            );
        })
    }
    println!("    {BRIGHT_CYAN}✓{RESET} Done!");
    Ok(())
}

/// Measures the microfacet shadowing-masking function of the given
/// micro-surface and saves the result to the given output directory.
fn measure_microfacet_shadowing_masking(
    measurement: MicrofacetShadowingMaskingMeasurement,
    surfaces: &[MicroSurfaceHandle],
    cache: &Cache,
    config: &Config,
    output: &Option<PathBuf>,
) -> Result<(), Error> {
    println!(
        "  {BRIGHT_YELLOW}>{RESET} Measuring microfacet shadowing-masking function:
    • parameters:
      + azimuth: {} ~ {} per {}
      + zenith: {} ~ {} per {}",
        measurement.azimuth.start.prettified(),
        measurement.azimuth.stop.prettified(),
        measurement.azimuth.step_size.prettified(),
        measurement.zenith.start.prettified(),
        measurement.zenith.stop.prettified(),
        measurement.zenith.step_size.prettified()
    );
    let start_time = Instant::now();
    let distributions =
        acq::microfacet::measure_microfacet_shadowing_masking(measurement, surfaces, cache);
    let duration = Instant::now() - start_time;
    println!(
        "    {BRIGHT_CYAN}✓{RESET} Measurement finished in {} secs.",
        duration.as_secs_f32()
    );
    let output_dir = resolve_output_dir(config, output)?;
    println!("    {BRIGHT_YELLOW}>{RESET} Saving measurement data...");
    for (distrib, surface) in distributions.iter().zip(surfaces.iter()) {
        let filename = format!(
            "microfacet-shdowing-masking-{}.txt",
            surface
                .path
                .file_stem()
                .unwrap()
                .to_ascii_lowercase()
                .to_str()
                .unwrap()
        );
        let filepath = output_dir.join(filename);
        println!(
            "      {BRIGHT_CYAN}-{RESET} Saving to \"{}\"",
            filepath.display()
        );
        distrib.save_ascii(&filepath).unwrap_or_else(|err| {
            eprintln!(
                "        {BRIGHT_RED}!{RESET} Failed to save to \"{}\": {}",
                filepath.display(),
                err
            );
        })
    }
    println!("    {BRIGHT_CYAN}✓{RESET} Done!");
    Ok(())
}

/// Returns the output directory in canonical form.
/// If the output directory is not specified, returns config's output directory.
///
/// # Arguments
///
/// * `config` - The configuration of the current Vgonio session.
/// * `output` - The output directory specified by the user.
fn resolve_output_dir(config: &Config, output_dir: &Option<PathBuf>) -> Result<PathBuf, Error> {
    match output_dir {
        Some(dir) => {
            let path = resolve_path(config.cwd(), Some(dir));
            if !path.is_dir() {
                return Err(Error::InvalidOutputDir(path));
            }
            Ok(path)
        }
        None => Ok(config.output_dir().to_path_buf()),
    }
}
