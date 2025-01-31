use crate::{app::cache::RawCache, measure::params::SdfMeasurementParams};
use std::{borrow::Cow, path::Path};
use surf::MicroSurface;
use vgonio_core::{
    error::VgonioError,
    impl_any_measured_trait, math,
    math::{IVec2, Vec2},
    res::{Handle, RawDataStore},
    units::{rad, Radians},
    utils::range::StepRangeIncl,
    AnyMeasured, MeasurementKind,
};

/// Measures the slope distribution function (SDF) of the given microsurfaces.
pub fn measure_slope_distribution(
    handles: &[Handle],
    params: SdfMeasurementParams,
    cache: &RawCache,
) -> Box<[Measurement]> {
    #[cfg(feature = "bench")]
    let start = std::time::Instant::now();

    log::info!("Measuring the slope distribution function (SDF)...");
    let surfaces = cache.get_micro_surfaces(handles);
    let meshes = cache.get_micro_surface_meshes_by_surfaces(handles);
    let measurements: Vec<_> = handles
        .iter()
        .zip(surfaces.iter())
        .zip(meshes.iter())
        .filter_map(|((hdl, surf), mesh)| {
            if surf.is_none() || mesh.is_none() {
                log::debug!("Skipping a surface because it is not loaded {:?}.", mesh);
                return None;
            }
            let mesh = mesh.unwrap();
            let slopes = {
                let mut slopes = Box::new_uninit_slice(mesh.facet_normals.len());
                // Iterate over all facet normals to compute the slopes.
                for (n, s) in mesh.facet_normals.iter().zip(slopes.iter_mut()) {
                    // Compute the slope of the microfacet normal.
                    s.write(Vec2::new(-n.x, -n.y) / n.z);
                }
                unsafe { slopes.assume_init() }
            };

            Some(Measurement {
                name: format!("sdf-{}", surf.unwrap().file_stem().unwrap()),
                source: MeasurementSource::Measured(*hdl),
                timestamp: chrono::Local::now(),
                measured: Box::new(MeasuredSdfData { params, slopes }),
            })
        })
        .collect();

    #[cfg(feature = "bench")]
    {
        let elapsed = start.elapsed();
        log::info!("SDF measurement took {} ms.", elapsed.as_millis());
    }

    measurements.into_boxed_slice()
}
