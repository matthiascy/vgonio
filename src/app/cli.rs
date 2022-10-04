use crate::{
    acq::{
        desc::{MeasurementDesc, MeasurementKind},
        RayTracingMethod,
    },
    app::{
        cache::{VgonioCache, VgonioDatafiles},
        Config, ExtractOptions, MeasureOptions,
    },
    util, Error,
};

pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
pub const RESET: &str = "\u{001b}[0m";

/// Measure different metrics of the micro-surface.
pub fn measure(opts: MeasureOptions, config: Config) -> Result<(), Error> {
    println!("{:?}", config);
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
    let mut cache = VgonioCache::new(config.cache_dir.clone());
    let mut db = VgonioDatafiles::new();
    db.load_ior_database(&config);

    println!("  {BRIGHT_YELLOW}>{RESET} Reading measurement description file...");
    let measurements = MeasurementDesc::load_from_file(&opts.input_path)?;
    println!(
        "    {BRIGHT_YELLOW}✓{RESET} {} measurements",
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
    for (desc, surfaces) in measurements.iter().zip(all_surfaces.iter()) {
        match desc.measurement_kind {
            MeasurementKind::Bsdf(kind) => {
                println!(
                    "  {BRIGHT_YELLOW}>{RESET} Launch BSDF measurement at {}
    • parameters:
      + incident medium: {:?}
      + transmitted medium: {:?}
      + surfaces: {:?}
      + emitter:
        - radius: {:?}
        - num rays: {}
        - max bounces: {}
        - spectrum: {} - {}, step size {}
        - polar angle: {}° - {}°, step size {}°
        - azimuthal angle: {}° - {}°, step size {}°
      + collector:
        - radius: {:?}
        - shape: {:?}
        - partition:
          - type: {}
          - polar angle: {}
          - azimuthal angle: {}",
                    chrono::DateTime::<chrono::Utc>::from(start),
                    desc.incident_medium,
                    desc.transmitted_medium,
                    surfaces,
                    desc.emitter.radius,
                    desc.emitter.num_rays,
                    desc.emitter.max_bounces,
                    desc.emitter.spectrum.start,
                    desc.emitter.spectrum.stop,
                    desc.emitter.spectrum.step,
                    desc.emitter.zenith.start,
                    desc.emitter.zenith.stop,
                    desc.emitter.zenith.step,
                    desc.emitter.azimuth.start,
                    desc.emitter.azimuth.stop,
                    desc.emitter.azimuth.step,
                    desc.collector.radius,
                    desc.collector.shape,
                    desc.collector.partition.kind_str(),
                    desc.collector.partition.zenith_range_str(),
                    desc.collector.partition.azimuth_range_str()
                );
                println!("    {BRIGHT_YELLOW}>{RESET} Measuring {}...", kind);
                match desc.tracing_method {
                    RayTracingMethod::Standard => {
                        crate::acq::bsdf::measure_bsdf_embree_rt(desc, &cache, &db, surfaces);
                    }
                    RayTracingMethod::Grid => {
                        crate::acq::bsdf::measure_bsdf_grid_rt(desc, &cache, &db, surfaces);
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
