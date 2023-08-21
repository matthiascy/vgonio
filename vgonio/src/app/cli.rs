use egui::output;
use std::{fs::OpenOptions, path::PathBuf, time::Instant};
use vgcore::{
    error::VgonioError,
    io::{CompressionScheme, FileEncoding},
    math::Handedness,
    units::LengthUnit,
};
use vgsurf::{MicroSurface, RandomGenMethod, SurfGenKind};

use crate::{
    app::{
        args::{ConvertKind, ConvertOptions, FastMeasurementKind, MeasureOptions, SubCommand},
        cache::{resolve_path, Cache},
        cli::info::print_info,
        Config,
    },
    error::RuntimeError,
    measure::{
        self,
        measurement::{
            BsdfMeasurementParams, MadfMeasurementParams, Measurement, MeasurementData,
            MeasurementKind, MeasurementParams, MmsfMeasurementParams,
        },
        CollectorScheme,
    },
    SphericalPartition,
};

use super::{args::GenerateOptions, cache::Handle};

pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
pub const BRIGHT_RED: &str = "\u{001b}[31m";
pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
pub const RESET: &str = "\u{001b}[0m";

mod info;

/// Entry point of vgonio CLI.
pub fn run(cmd: SubCommand, config: Config) -> Result<(), VgonioError> {
    match cmd {
        SubCommand::Measure(opts) => measure(opts, config),
        SubCommand::PrintInfo(opts) => print_info(opts, config),
        SubCommand::Generate(opts) => generate(opts, config),
        SubCommand::Convert(opts) => convert(opts, config),
    }
}

/// Measure different metrics of the micro-surface.
fn measure(opts: MeasureOptions, config: Config) -> Result<(), VgonioError> {
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
                        params: MeasurementParams::Bsdf(BsdfMeasurementParams::default()),
                        surfaces: opts.inputs.clone(),
                    },
                    FastMeasurementKind::MicrofacetAreaDistribution => Measurement {
                        params: MeasurementParams::Madf(MadfMeasurementParams::default()),
                        surfaces: opts.inputs.clone(),
                    },
                    FastMeasurementKind::MicrofacetMaskingShadowing => Measurement {
                        params: MeasurementParams::Mmsf(MmsfMeasurementParams::default()),
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
    if measurements.iter().any(|meas| meas.params.is_bsdf()) {
        println!("  {BRIGHT_YELLOW}>{RESET} Loading data files (refractive indices, spd etc.)...");
        cache.load_ior_database(&config);
        println!("    {BRIGHT_CYAN}✓{RESET} Successfully load data files");
    }

    println!("  {BRIGHT_YELLOW}>{RESET} Resolving and loading micro-surfaces...");
    let tasks = measurements
        .into_iter()
        .filter_map(|meas| {
            cache
                .load_micro_surfaces(&config, &meas.surfaces, config.user.triangulation)
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

    let start_time = Instant::now();
    for (measurement, surfaces) in tasks {
        let measurement_start_time = std::time::SystemTime::now();
        let measured_data = match measurement.params {
            MeasurementParams::Bsdf(measurement) => {
                let collector_info = match measurement.collector.scheme {
                    CollectorScheme::Partitioned { partition } => match partition {
                        SphericalPartition::EqualAngle { zenith, azimuth } => {
                            format!(
                                "        - partition: {}\n          - polar angle: {}\n          \
                                 - azimuthal angle: {}",
                                "equal angle",
                                zenith.pretty_print(),
                                azimuth.pretty_print(),
                            )
                        }
                        SphericalPartition::EqualArea { zenith, azimuth } => {
                            format!(
                                "        - partition: {}\n          - polar angle: {}\n          \
                                 - azimuthal angle: {}",
                                "equal area",
                                zenith.pretty_print(),
                                azimuth.pretty_print(),
                            )
                        }
                        SphericalPartition::EqualProjectedArea { zenith, azimuth } => {
                            format!(
                                "        - partition: {}\n          - polar angle: {}\n          \
                                 - azimuthal angle: {}",
                                "equal projected area",
                                zenith.pretty_print(),
                                azimuth.pretty_print(),
                            )
                        }
                    },
                    CollectorScheme::SingleRegion {
                        shape,
                        zenith,
                        azimuth,
                    } => {
                        format!(
                            "        - shape: {shape:?}\n- polar angle: {zenith:?}\n- azimuthal \
                             angle {azimuth:?}\n",
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
        - spectrum: {}
        - polar angle: {}
        - azimuthal angle: {}
      + collector:
        - radius: {}\n{}",
                    chrono::DateTime::<chrono::Utc>::from(measurement_start_time),
                    measurement.incident_medium,
                    measurement.transmitted_medium,
                    measurement.emitter.orbit_radius(),
                    measurement.emitter.num_rays,
                    measurement.emitter.max_bounces,
                    measurement.emitter.spectrum,
                    measurement.emitter.zenith.pretty_print(),
                    measurement.emitter.azimuth.pretty_print(),
                    measurement.collector.radius,
                    collector_info
                );
                measure::bsdf::measure_bsdf_rt(measurement, &surfaces, measurement.sim_kind, &cache)
            }
            MeasurementParams::Madf(measurement) => {
                println!(
                    "  {BRIGHT_YELLOW}>{RESET} Measuring microfacet area distribution:
    • parameters:
      + azimuth: {}
      + zenith: {}",
                    measurement.azimuth.pretty_print(),
                    measurement.zenith.pretty_print(),
                );
                measure::microfacet::measure_area_distribution(measurement, &surfaces, &cache)
            }
            MeasurementParams::Mmsf(measurement) => {
                println!(
                    "  {BRIGHT_YELLOW}>{RESET} Measuring microfacet masking-shadowing function:
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

                measure::microfacet::measure_masking_shadowing(
                    measurement,
                    &surfaces,
                    &cache,
                    Handedness::RightHandedYUp,
                )
            }
        };

        println!(
            "    {BRIGHT_CYAN}✓{RESET} Measurement finished in {} secs.",
            measurement_start_time.elapsed().unwrap().as_secs_f32()
        );

        write_measured_data_to_file(
            &measured_data,
            &surfaces,
            &cache,
            &config,
            opts.encoding,
            opts.compression,
            &opts.output,
        )?;

        println!("    {BRIGHT_CYAN}✓{RESET} Done!");
    }

    println!(
        "    {BRIGHT_CYAN}✓{RESET} Finished in {:.2} s",
        start_time.elapsed().as_secs_f32()
    );

    Ok(())
}

/// Generates a micro-surface.
fn generate(opts: GenerateOptions, config: Config) -> Result<(), VgonioError> {
    let (res_x, res_y) = (opts.res[0], opts.res[1]);
    let (du, dv) = (opts.spacing[0], opts.spacing[1]);
    println!(
        "  {BRIGHT_YELLOW}>{RESET} Generating surface with resolution {}x{}...",
        res_x, res_y
    );

    let surf = match opts.kind {
        SurfGenKind::Random => match opts.method.unwrap() {
            RandomGenMethod::WhiteNoise => {
                println!("    {BRIGHT_YELLOW}>{RESET} Generating surface from white noise...");
                MicroSurface::from_white_noise(
                    res_y as usize,
                    res_x as usize,
                    du,
                    dv,
                    opts.max_height,
                    LengthUnit::UM,
                )
            }
            _ => {
                todo!()
            }
        },
        SurfGenKind::Gaussian2D => {
            println!(
                "  {BRIGHT_YELLOW}>{RESET} Generating surface from 2D gaussian distribution..."
            );
            let mut data: Vec<f32> = Vec::with_capacity((res_x * res_y) as usize);
            let (sigma_x, sigma_y) = (opts.sigma_x.unwrap(), opts.sigma_y.unwrap());
            let (mean_x, mean_y) = (opts.mean_x.unwrap(), opts.mean_y.unwrap());
            let amp = opts.amplitude.unwrap();
            MicroSurface::new_by(
                res_y as usize,
                res_x as usize,
                du,
                dv,
                LengthUnit::UM,
                |r, c| {
                    let x = ((c as f32 / res_x as f32) * 2.0 - 1.0) * sigma_x * 4.0;
                    let y = ((r as f32 / res_y as f32) * 2.0 - 1.0) * sigma_y * 4.0;
                    amp * (-(x - mean_x) * (x - mean_x) / (2.0 * sigma_x * sigma_x)
                        - (y - mean_y) * (y - mean_y) / (2.0 * sigma_y * sigma_y))
                        .exp()
                },
            )
        }
    };

    let path = {
        let base = resolve_output_dir(&config, &opts.output)?;
        let p = if let Some(path) = opts.output {
            path
        } else {
            PathBuf::from(
                format!(
                    "msurf-{:?}_{}.vgms",
                    opts.kind,
                    chrono::Local::now().format("%d-%m-%H-%M-%S")
                )
                .to_lowercase(),
            )
        };
        resolve_path(&base, Some(&p))
    };

    println!("    {BRIGHT_CYAN}✓{RESET} Surface generated");
    println!(
        "    {BRIGHT_YELLOW}>{RESET} Saving to \"{}\"",
        path.display()
    );
    surf.write_to_file(&path, opts.encoding, opts.compression)
}

/// Writes the measured data to a file.
fn write_measured_data_to_file(
    data: &[MeasurementData],
    surfaces: &[Handle<MicroSurface>],
    cache: &Cache,
    config: &Config,
    encoding: FileEncoding,
    compression: CompressionScheme,
    output: &Option<PathBuf>,
) -> Result<(), VgonioError> {
    let output_dir = resolve_output_dir(config, output)?;
    println!("    {BRIGHT_YELLOW}>{RESET} Saving measurement data...");
    for (measurement, surface) in data.iter().zip(surfaces.iter()) {
        let filename = match measurement.kind() {
            MeasurementKind::Madf => {
                format!(
                    "microfacet-area-distribution-{}{}.vgmo",
                    cache
                        .get_micro_surface_filepath(*surface)
                        .unwrap()
                        .file_stem()
                        .unwrap()
                        .to_ascii_lowercase()
                        .to_str()
                        .unwrap(),
                    if measurement
                        .measured
                        .madf_data()
                        .unwrap()
                        .params
                        .single_slice
                    {
                        "-single"
                    } else {
                        ""
                    }
                )
            }
            MeasurementKind::Mmsf => {
                format!(
                    "microfacet-masking-shadowing-{}.vgmo",
                    cache
                        .get_micro_surface_filepath(*surface)
                        .unwrap()
                        .file_stem()
                        .unwrap()
                        .to_ascii_lowercase()
                        .to_str()
                        .unwrap()
                )
            }
            MeasurementKind::Bsdf => {
                format!(
                    "bsdf-{}.vgmo",
                    cache
                        .get_micro_surface_filepath(*surface)
                        .unwrap()
                        .file_stem()
                        .unwrap()
                        .to_ascii_lowercase()
                        .to_str()
                        .unwrap()
                )
            }
        };
        let filepath = output_dir.join(filename);
        println!(
            "      {BRIGHT_CYAN}-{RESET} Saving to \"{}\"",
            filepath.display()
        );
        measurement
            .write_to_file(&filepath, encoding, compression)
            .unwrap_or_else(|err| {
                eprintln!(
                    "        {BRIGHT_RED}!{RESET} Failed to save to \"{}\": {}",
                    filepath.display(),
                    err
                );
            });
        println!(
            "      {BRIGHT_CYAN}✓{RESET} Successfully saved to \"{}\"",
            output_dir.display()
        );
    }
    Ok(())
}

fn convert(opts: ConvertOptions, config: Config) -> Result<(), VgonioError> {
    let output_dir = resolve_output_dir(&config, &opts.output)?;
    for input in opts.inputs {
        let resolved = resolve_path(&config.cwd, Some(&input));
        match opts.kind {
            ConvertKind::MicroSurfaceProfile => {
                let (profile, filename) = {
                    let loaded = MicroSurface::read_from_file(&resolved, None)?;
                    let (w, h) = if let Some(new_size) = opts.resize.as_ref() {
                        let (w, h) = (new_size[0] as usize, new_size[1] as usize);
                        println!("  {BRIGHT_YELLOW}>{RESET} Resizing to {}x{}...", w, h);
                        (w, h)
                    } else {
                        (loaded.cols, loaded.rows)
                    };

                    let (w, h) = if opts.squaring {
                        let s = w.min(h);
                        println!("  {BRIGHT_YELLOW}>{RESET} Squaring to {}x{}...", s, s);
                        (s, s)
                    } else {
                        (w, h)
                    };

                    let filename = format!(
                        "{}_converted.vgms",
                        resolved
                            .file_stem()
                            .unwrap()
                            .to_ascii_lowercase()
                            .to_str()
                            .unwrap()
                    );
                    (loaded.resize(h, w), filename)
                };
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
fn resolve_output_dir(
    config: &Config,
    output_dir: &Option<PathBuf>,
) -> Result<PathBuf, VgonioError> {
    match output_dir {
        Some(dir) => {
            let path = resolve_path(config.cwd(), Some(dir));
            if !path.is_dir() {
                return Err(VgonioError::new(
                    format!("{} is not a directory", path.display()),
                    Some(Box::new(RuntimeError::InvalidOutputDir)),
                ));
            }
            Ok(path)
        }
        None => Ok(config.output_dir().to_path_buf()),
    }
}
