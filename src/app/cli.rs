use std::{fs::OpenOptions, path::PathBuf, time::Instant};

use crate::{
    app::{
        args::{ConvertKind, ConvertOptions, FastMeasurementKind, MeasureOptions, SubCommand},
        cache::{resolve_path, Cache},
        Config,
    },
    io::{CompressionScheme, FileEncoding},
    measure::{
        self,
        measurement::{
            BsdfMeasurement, Measurement, MeasurementKindDescription,
            MicrofacetMaskingShadowingMeasurement, MicrofacetNormalDistributionMeasurement,
            SimulationKind,
        },
        CollectorScheme, RtcMethod,
    },
    msurf::MicroSurface,
    Error, Handedness, SphericalPartition,
};

use super::{args::GenerateOptions, cache::Handle};

pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
pub const BRIGHT_RED: &str = "\u{001b}[31m";
pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
pub const RESET: &str = "\u{001b}[0m";

mod info;

/// Entry point of vgonio CLI.
pub fn run(cmd: SubCommand, config: Config) -> Result<(), Error> {
    match cmd {
        SubCommand::Measure(opts) => measure(opts, config),
        SubCommand::PrintInfo(opts) => info::print(opts, config),
        SubCommand::Generate(opts) => generate(opts, config),
        SubCommand::Convert(opts) => convert(opts, config),
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
        match opts.fast_measurement {
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
                    FastMeasurementKind::Bsdf => Measurement {
                        desc: MeasurementKindDescription::Bsdf(BsdfMeasurement::default()),
                        surfaces: opts.inputs.clone(),
                    },
                    FastMeasurementKind::MicrofacetNormalDistribution => Measurement {
                        desc: MeasurementKindDescription::Mndf(
                            MicrofacetNormalDistributionMeasurement::default(),
                        ),
                        surfaces: opts.inputs.clone(),
                    },
                    FastMeasurementKind::MicrofacetMaskingShadowing => Measurement {
                        desc: MeasurementKindDescription::Mmsf(
                            MicrofacetMaskingShadowingMeasurement::default(),
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

    // Load data files: refractive indices, spd etc. if needed.
    if measurements.iter().any(|meas| meas.desc.is_bsdf()) {
        println!("  {BRIGHT_YELLOW}>{RESET} Loading data files (refractive indices, spd etc.)...");
        cache.load_ior_database(&config);
        println!("    {BRIGHT_CYAN}✓{RESET} Successfully load data files");
    }

    println!("  {BRIGHT_YELLOW}>{RESET} Resolving and loading micro-surfaces...");
    let tasks = measurements
        .into_iter()
        .filter_map(|meas| {
            cache
                .load_micro_surfaces(&config, &meas.surfaces)
                .ok()
                .map(|surfaces| (meas, surfaces))
        })
        .collect::<Vec<_>>();
    println!(
        "    {BRIGHT_CYAN}✓{RESET} {} micro-surface(s) loaded",
        cache.num_micro_surfaces()
    );

    if cache.num_micro_surfaces() == 0 {
        println!("  {BRIGHT_RED}✗{RESET} No micro-surface to measure. Exiting...");
        return Ok(());
    }

    cache
        .loaded_micro_surface_paths()
        .unwrap()
        .iter()
        .for_each(|s| println!("      {BRIGHT_CYAN}-{RESET} {}", s.display()));
    for (measurement, surfaces) in tasks {
        match measurement.desc {
            MeasurementKindDescription::Bsdf(measurement) => {
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
                    CollectorScheme::SingleRegion {
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

                println!(
                    "  {BRIGHT_YELLOW}>{RESET} Launch BSDF measurement at {}
    • parameters:
      + incident medium: {:?}
      + transmitted medium: {:?}
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
                    SimulationKind::GeomOptics(method) => {
                        println!(
                            "    {BRIGHT_YELLOW}>{RESET} Measuring {} with geometric optics...",
                            measurement.kind
                        );
                        match method {
                            #[cfg(feature = "embree")]
                            RtcMethod::Embree => {
                                println!("      {BRIGHT_YELLOW}>{RESET} Using Embree ray tracing");
                                measure::bsdf::measure_bsdf_embree_rt(
                                    measurement,
                                    &cache,
                                    &surfaces,
                                );
                            }
                            #[cfg(feature = "optix")]
                            RtcMethod::Optix => {
                                println!("      {BRIGHT_YELLOW}>{RESET} Using OptiX ray tracing");
                                todo!("optix");
                            }
                            RtcMethod::Grid => {
                                println!(
                                    "      {BRIGHT_YELLOW}>{RESET} Using customised grid ray \
                                     tracing"
                                );
                                measure::bsdf::measure_bsdf_grid_rt(measurement, &cache, &surfaces);
                            }
                        }
                    }
                    SimulationKind::WaveOptics => {
                        println!(
                            "    {BRIGHT_YELLOW}>{RESET} Measuring {} with wave optics...",
                            measurement.kind
                        );
                        todo!("wave optics");
                    }
                }
                println!("    {BRIGHT_YELLOW}>{RESET} Saving results...");
                // todo: save to file
                println!(
                    "      {BRIGHT_CYAN}✓{RESET} Successfully saved to \"{}\"",
                    config.output_dir().display()
                );
                println!(
                    "    {BRIGHT_CYAN}✓{RESET} Finished in {:.2} s",
                    start.elapsed().unwrap().as_secs_f32()
                );
            }
            MeasurementKindDescription::Mndf(measurement) => {
                measure_microfacet_normal_distribution(
                    measurement,
                    &surfaces,
                    &cache,
                    &config,
                    &opts.output,
                    opts.encoding,
                    opts.compression,
                )
                .map_err(|err| {
                    eprintln!("  {BRIGHT_RED}✗{RESET} {err}");
                    err
                })?;
            }
            MeasurementKindDescription::Mmsf(measurement) => {
                measure_microfacet_masking_shadowing(
                    measurement,
                    &surfaces,
                    &cache,
                    &config,
                    &opts.output,
                    opts.encoding,
                    opts.compression,
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

/// Generates a micro-surface using 2D gaussian distribution.
fn generate(opts: GenerateOptions, config: Config) -> Result<(), Error> {
    let mut data: Vec<f32> = Vec::with_capacity((opts.res_x * opts.res_y) as usize);
    for i in 0..opts.res_y {
        for j in 0..opts.res_x {
            let x = ((i as f32 / opts.res_x as f32) * 2.0 - 1.0) * opts.sigma_x * 4.0;
            let y = ((j as f32 / opts.res_y as f32) * 2.0 - 1.0) * opts.sigma_y * 4.0;
            data.push(
                opts.amplitude
                    * (-(x - opts.mean_x) * (x - opts.mean_x)
                        / (2.0 * opts.sigma_x * opts.sigma_x)
                        - (y - opts.mean_y) * (y - opts.mean_y)
                            / (2.0 * opts.sigma_y * opts.sigma_y))
                        .exp(),
            );
        }
    }

    let path = {
        let p = if let Some(path) = opts.output {
            path
        } else {
            PathBuf::from("micro-surface.txt")
        };
        resolve_path(&config.cwd, Some(&p))
    };

    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&path)
        .map_err(|err| {
            eprintln!("  {BRIGHT_RED}✗{RESET} Failed to open file: {err}");
            err
        })?;

    use std::io::Write;

    writeln!(file, "AsciiMatrix {} {} 0.5 0.5", opts.res_x, opts.res_y).map_err(|err| {
        eprintln!("  {BRIGHT_RED}✗{RESET} Failed to write to file: {err}");
        err
    })?;

    for i in 0..opts.res_y {
        for j in 0..opts.res_x {
            write!(file, "{} ", data[(i * opts.res_x + j) as usize])?;
        }
        writeln!(file)?;
    }

    Ok(())
}

/// Measures the microfacet normal distribution of the given micro-surface and
/// saves the result to the given output directory.
fn measure_microfacet_normal_distribution(
    measurement: MicrofacetNormalDistributionMeasurement,
    surfaces: &[Handle<MicroSurface>],
    cache: &Cache,
    config: &Config,
    output: &Option<PathBuf>,
    encoding: FileEncoding,
    compression: CompressionScheme,
) -> Result<(), Error> {
    println!(
        "  {BRIGHT_YELLOW}>{RESET} Measuring microfacet normal distribution:
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
        measure::microfacet::measure_normal_distribution(measurement, surfaces, cache);
    let duration = Instant::now() - start_time;
    println!(
        "    {BRIGHT_CYAN}✓{RESET} Measurement finished in {} secs.",
        duration.as_secs_f32()
    );
    let output_dir = resolve_output_dir(config, output)?;
    println!("    {BRIGHT_YELLOW}>{RESET} Saving measurement data...");
    for (distrib, surface) in distributions.iter().zip(surfaces.iter()) {
        let filename = format!(
            "microfacet-normal-distribution-{}.vgmo",
            cache
                .get_micro_surface_filepath(*surface)
                .unwrap()
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
        distrib
            .write_to_file(&filepath, encoding, compression)
            .unwrap_or_else(|err| {
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

/// Measures the microfacet masking-shadowing function of the given
/// micro-surface and saves the result to the given output directory.
fn measure_microfacet_masking_shadowing(
    measurement: MicrofacetMaskingShadowingMeasurement,
    surfaces: &[Handle<MicroSurface>],
    cache: &Cache,
    config: &Config,
    output: &Option<PathBuf>,
    encoding: FileEncoding,
    compression: CompressionScheme,
) -> Result<(), Error> {
    println!(
        "  {BRIGHT_YELLOW}>{RESET} Measuring microfacet masking-shadowing function:
    • parameters:
      + azimuth: {} ~ {} per {}
      + zenith: {} ~ {} per {}
      + resolution: {} x {}",
        measurement.azimuth.start.prettified(),
        measurement.azimuth.stop.prettified(),
        measurement.azimuth.step_size.prettified(),
        measurement.zenith.start.prettified(),
        measurement.zenith.stop.prettified(),
        measurement.zenith.step_size.prettified(),
        measurement.resolution,
        measurement.resolution
    );
    let start_time = Instant::now();
    let distributions = measure::microfacet::measure_masking_shadowing(
        measurement,
        surfaces,
        cache,
        Handedness::RightHandedYUp,
    );
    let duration = Instant::now() - start_time;
    println!(
        "    {BRIGHT_CYAN}✓{RESET} Measurement finished in {} secs.",
        duration.as_secs_f32()
    );
    let output_dir = resolve_output_dir(config, output)?;
    println!("    {BRIGHT_YELLOW}>{RESET} Saving measurement data...");
    for (distrib, surface) in distributions.iter().zip(surfaces.iter()) {
        let filename = format!(
            "microfacet-masking-shadowing-{}.vgmo",
            cache
                .get_micro_surface_filepath(*surface)
                .unwrap()
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
        distrib
            .write_to_file(&filepath, encoding, compression)
            .unwrap_or_else(|err| {
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

fn convert(opts: ConvertOptions, config: Config) -> Result<(), Error> {
    let output_dir = resolve_output_dir(&config, &opts.output)?;
    for input in opts.inputs {
        let resolved = resolve_path(&config.cwd, Some(&input));
        match opts.kind {
            ConvertKind::MicroSurfaceProfile => {
                let profile = MicroSurface::read_from_file(&resolved, None)?;
                let filename = format!(
                    "{}.vgms",
                    resolved
                        .file_stem()
                        .unwrap()
                        .to_ascii_lowercase()
                        .to_str()
                        .unwrap()
                );
                println!(
                    "{BRIGHT_YELLOW}>{RESET} Converting {:?} to {:?}...",
                    resolved, output_dir
                );
                profile
                    .write_to_file(&output_dir.join(filename), opts.encoding, opts.compression)
                    .unwrap_or_else(|err| {
                        eprintln!(
                            "  {BRIGHT_RED}!{RESET} Failed to save to \"{}\": {}",
                            resolved.display(),
                            err
                        );
                    });
                println!("{BRIGHT_CYAN}✓{RESET} Done!",);
            }
        }
    }
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
