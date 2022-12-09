use crate::{
    acq::{
        measurement::{Measurement, MeasurementKind},
        RayTracingMethod,
    },
    app::{
        cache::{Cache, VgonioDatafiles},
        Config, ExtractOptions, MeasureOptions,
    },
    util, Error,
};
use crate::acq::CollectorScheme;
use crate::acq::measurement::SimulationKind;
use crate::acq::util::SphericalPartition;

pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
pub const RESET: &str = "\u{001b}[0m";

/// Measure different metrics of the micro-surface.
pub fn measure(opts: MeasureOptions, config: Config) -> Result<(), Error> {
    log::info!("{:?}", config);
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

    // Load data files: refractive indices, spd etc.
    println!("  {BRIGHT_YELLOW}>{RESET} Loading data files (refractive indices, spd etc.)...");
    let mut cache = Cache::new(config.cache_dir.clone());
    let mut db = VgonioDatafiles::new();
    db.load_ior_database(&config);
    println!("    {BRIGHT_CYAN}✓{RESET} Successfully load data files");

    println!("  {BRIGHT_YELLOW}>{RESET} Reading measurement description file...");
    let measurements = Measurement::load_from_file(&opts.input_path)?;
    println!(
        "    {BRIGHT_YELLOW}✓{RESET} {} measurement(s)",
        measurements.len()
    );

    println!("    - Resolving and loading surfaces...");
    let all_surfaces = measurements
        .iter()
        .map(|desc| {
            let to_be_loaded = desc
                .surfaces
                .iter()
                .map(|s| {
                    if let Ok(stripped) = s.strip_prefix("user://") {
                        config.user_config.data_files_dir.join(stripped)
                    } else if let Ok(stripped) = s.strip_prefix("local://") {
                        config.data_files_dir.join(stripped)
                    } else {
                        util::resolve_file_path(&opts.input_path, Some(s))
                    }
                })
                .collect::<Vec<_>>();
            let handles = cache.load_surfaces_from_files(&to_be_loaded).unwrap();
            cache.triangulate_surfaces(&handles);
            handles
        })
        .collect::<Vec<_>>();

    println!("    {BRIGHT_CYAN}✓{RESET} Successfully read scene description file");

    let start = std::time::SystemTime::now();
    for (measurement, surfaces) in measurements.into_iter().zip(all_surfaces.iter()) {
        let collector_info = match measurement.collector.scheme {
            CollectorScheme::Partitioned { domain, partition } => {
                match partition {
                    SphericalPartition::EqualAngle {
                        zenith, azimuth
                    } => {
                        format!("        - domain: {}\
                        \n        - partition: {}\
                        \n          - polar angle: {} ~ {}, per {}\
                        \n          - azimuthal angle {} ~ {}, per {}", domain, "equal angle",
                                zenith.start.prettified(),
                                zenith.stop.prettified(),
                                zenith.step_size.prettified(),
                                azimuth.stop.prettified(),
                                azimuth.stop.prettified(),
                                azimuth.step_size.prettified())
                    },
                    SphericalPartition::EqualArea {
                        zenith, azimuth
                    } => {
                        format!("        - domain: {}\
                        \n        - partition: {}\
                        \n          - polar angle: {} ~ {}, {} steps\
                        \n          - azimuthal angle: {} ~ {}, per {}", domain, "equal area",
                                zenith.start.prettified(),
                                zenith.stop.prettified(),
                                zenith.step_count,
                                azimuth.start.prettified(),
                                azimuth.stop.prettified(),
                                azimuth.step_size.prettified())
                    },
                    SphericalPartition::EqualProjectedArea {
                        zenith, azimuth
                    } => {
                        format!("        - domain: {}\
                        \n        - partition: {}\
                        \n          - polar angle: {} - {}, {} steps\
                        \n          - azimuthal angle {} - {}, per {}", domain, "equal projected area",
                                zenith.start.prettified(),
                                zenith.stop.prettified(),
                                zenith.step_count,
                                azimuth.start.prettified(),
                                azimuth.stop.prettified(),
                                azimuth.step_size.prettified())
                    }
                }
            }
            CollectorScheme::Individual { domain, shape, zenith, azimuth } => {
                format!("        - domain: {}\n\
                        - shape: {:?}\n\
                        - polar angle: {:?}\n\
                        - azimuthal angle {:?}\n", domain, shape, zenith, azimuth)
            }
        };

        let surfaces_str = surfaces
            .iter()
            .map(|s| format!("       - {}", s.path.display()))
            .collect::<Vec<_>>()
            .join("\n");
        match measurement.kind {
            MeasurementKind::Bsdf(kind) => {
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
                    measurement.emitter.radius,
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
                println!("    {BRIGHT_YELLOW}>{RESET} Measuring {}...", kind);
                match measurement.sim_kind {
                    SimulationKind::GeomOptics { method } => {
                        match method {
                            RayTracingMethod::Standard => {
                                crate::acq::bsdf::measure_bsdf_embree_rt(measurement, &cache, &db, surfaces);
                            }
                            RayTracingMethod::Grid => {
                                crate::acq::bsdf::measure_bsdf_grid_rt(measurement, &cache, &db, surfaces);
                            }
                        }
                    }
                    SimulationKind::WaveOptics => {
                        // TODO: implement
                    }
                }
            }
            MeasurementKind::Ndf => {
                // todo: measure ndf
                todo!()
            }
        };
    }

    println!(
        "    {BRIGHT_CYAN}✓{RESET} Finished in {:.2} s",
        start.elapsed().unwrap().as_secs_f32()
    );

    println!("  {BRIGHT_YELLOW}>{RESET} Saving results...");
    // todo: save to file
    println!(
        "    {BRIGHT_CYAN}✓{RESET} Successfully saved to \"{}\"",
        config.user_config.output_dir.display()
    );

    Ok(())
}

pub fn extract(_opts: ExtractOptions, _config: Config) -> Result<(), Error> { Ok(()) }
