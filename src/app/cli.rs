use std::time::Instant;

use crate::{
    acq::{
        self,
        measurement::{Measurement, SimulationKind},
        util::SphericalPartition,
        CollectorScheme, RtcMethod,
    },
    app::{
        cache::{resolve_path, Cache, MicroSurfaceHandle, VgonioDatafiles},
        ExtractOptions, MeasureOptions, VgonioCommand, VgonioConfig,
    },
    htfld::AxisAlignment,
    mesh::TriangulationMethod,
    Error,
};

pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
pub const BRIGHT_RED: &str = "\u{001b}[31m";
pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
pub const RESET: &str = "\u{001b}[0m";

/// Execute the given command.
/// This function is the entry point of vgonio CLI.
pub fn execute(cmd: VgonioCommand, config: VgonioConfig) -> Result<(), Error> {
    match cmd {
        VgonioCommand::Measure(opts) => measure(opts, config),
        VgonioCommand::Extract(opts) => extract(&opts, &config),
        VgonioCommand::Info => print_info(config),
    }
}

/// Measure different metrics of the micro-surface.
fn measure(opts: MeasureOptions, config: VgonioConfig) -> Result<(), Error> {
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

    // Load data files: refractive indices, spd etc.
    println!("  {BRIGHT_YELLOW}>{RESET} Loading data files (refractive indices, spd etc.)...");
    let mut cache = Cache::new(config.cache_dir());

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
            // Load surfaces height field from files and cache them.
            let handles = cache
                .load_micro_surfaces(&config, &desc.surfaces, Some(AxisAlignment::XZ))
                .expect("Failed to load surfaces from files"); // TODO: better error handling
                                                               // Triangulate all surfaces.
            {
                let ref mut this = cache;
                let handles: &[MicroSurfaceHandle] = &handles;
                let mut meshes = vec![];
                for handle in handles {
                    for surface in this.micro_surfaces.values() {
                        if surface.uuid == handle.uuid {
                            if let std::collections::hash_map::Entry::Vacant(e) =
                                this.triangle_meshes.entry(surface.uuid)
                            {
                                let mesh = surface.triangulate(TriangulationMethod::Regular);
                                e.insert(mesh);
                                meshes.push(surface.uuid);
                            } else {
                                meshes.push(surface.uuid);
                            }
                            break;
                        }
                    }
                }
                meshes
            };
            handles
        })
        .collect::<Vec<_>>();

    println!("    {BRIGHT_CYAN}✓{RESET} Successfully read scene description file");

    let start = std::time::SystemTime::now();
    for (measurement, surfaces) in measurements.into_iter().zip(all_surfaces.iter()) {
        let collector_info = match measurement.collector.scheme {
            CollectorScheme::Partitioned { domain, partition } => match partition {
                SphericalPartition::EqualAngle { zenith, azimuth } => {
                    format!(
                        "        - domain: {}\n        - partition: {}\n          - polar angle: \
                         {} ~ {}, per {}\n          - azimuthal angle {} ~ {}, per {}",
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
                        "        - domain: {}\n        - partition: {}\n          - polar angle: \
                         {} ~ {}, {} steps\n          - azimuthal angle: {} ~ {}, per {}",
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
                        "        - domain: {}\n        - partition: {}\n          - polar angle: \
                         {} - {}, {} steps\n          - azimuthal angle {} - {}, per {}",
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
                    "        - domain: {}\n- shape: {:?}\n- polar angle: {:?}\n- azimuthal angle \
                     {:?}\n",
                    domain, shape, zenith, azimuth
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
                    measurement.kind
                );
                match method {
                    RtcMethod::Standard => {
                        crate::acq::bsdf::measure_bsdf_embree_rt(
                            measurement,
                            &cache,
                            &db,
                            surfaces,
                        );
                    }
                    RtcMethod::Grid => {
                        crate::acq::bsdf::measure_bsdf_grid_rt(measurement, &cache, &db, surfaces);
                    }
                }
            }
            SimulationKind::WaveOptics => {
                println!(
                    "    {BRIGHT_YELLOW}>{RESET} Measuring {} with wave optics...",
                    measurement.kind
                );
                // TODO: implement
            }
        }
    }

    println!(
        "    {BRIGHT_CYAN}✓{RESET} Finished in {:.2} s",
        start.elapsed().unwrap().as_secs_f32()
    );

    println!("  {BRIGHT_YELLOW}>{RESET} Saving results...");
    // todo: save to file
    println!(
        "    {BRIGHT_CYAN}✓{RESET} Successfully saved to \"{}\"",
        config.output_dir().display()
    );

    Ok(())
}

/// Extracts micro-surface intrinsic properties.
fn extract(opts: &ExtractOptions, config: &VgonioConfig) -> Result<(), Error> {
    println!("{BRIGHT_YELLOW}>{RESET} Measuring micro-surface intrinsic properties...");
    let mut cache = Cache::new(config.cache_dir());
    let micro_surfaces =
        cache.load_micro_surfaces(&config, &opts.inputs, Some(AxisAlignment::XZ))?;
    match opts.property {
        super::MicroSurfaceProperty::MicroFacetNormal => todo!(),
        super::MicroSurfaceProperty::MicroFacetDistribution => {
            println!("  {BRIGHT_YELLOW}>{RESET} Measuring micro facet distribution...");
            let start_time = Instant::now();
            let distributions =
                acq::facet_distrb::measure_micro_facet_distribution(&micro_surfaces, &cache);
            let duration = Instant::now() - start_time;
            println!(
                "    {BRIGHT_CYAN}✓{RESET} Measurement finished in {} secs.",
                duration.as_secs_f32()
            );
            let output_dir = if let Some(ref output) = opts.output {
                let path = resolve_path(config.cwd(), Some(&output));
                if !path.is_dir() {
                    return Err(Error::InvalidOutputDir(path));
                }
                path
            } else {
                config.output_dir().to_path_buf()
            };
            println!("  {BRIGHT_YELLOW}>{RESET} Saving measurement data...");
            for (distrb, surface) in distributions.iter().zip(micro_surfaces.iter()) {
                let filename = format!(
                    "facet-distrb-{}.txt",
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
                    "    {BRIGHT_CYAN}-{RESET} Saving to \"{}\"...",
                    filepath.display()
                );
                distrb.save_ascii(&filepath).unwrap_or_else(|err| {
                    eprintln!(
                        "      {BRIGHT_RED}!{RESET} Failed to save to \"{}\": {}",
                        filepath.display(),
                        err
                    );
                })
            }
            println!("    {BRIGHT_CYAN}✓{RESET} Done!");
            Ok(())
        }
        super::MicroSurfaceProperty::MicroFacetMaskingShadowing => {
            let geometric_terms =
                acq::facet_shadow::measure_micro_facet_masking_shadowing(&micro_surfaces, &cache);
            // TODO: output to file
            Ok(())
        }
    }
}

/// Prints Vgonio's current configurations.
fn print_info(config: VgonioConfig) -> Result<(), Error> {
    println!("\n{}", config);
    Ok(())
}
