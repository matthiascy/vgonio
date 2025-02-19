#[cfg(feature = "fitting")]
use crate::fitting::FittedModels;
use crate::{
    app::cache::RawCache,
    measure::{Measurement, MeasurementSource},
};
use base::{units::LengthUnit, utils::handle::Handle, MeasurementKind};
use chrono::{DateTime, Local};
use std::collections::HashMap;
use surf::{
    subdivision::{Subdivision, SubdivisionKind},
    MicroSurface,
};

use super::outliner::OutlinerItem;

/// Micro surface properties.
#[derive(Clone, Debug)]
pub struct MicroSurfaceProp {
    /// The name of the micro surface.
    pub name: String,
    /// Whether the micro surface is visible.
    pub visible: bool,
    /// The scale factor of the micro surface.
    pub scale: f32,
    /// The length unit of the micro surface.
    pub unit: LengthUnit,
    /// The lowest value of the micro surface.
    pub min: f32,
    /// The highest value of the micro surface.
    pub max: f32,
    /// The offset along the y-axis.
    pub height_offset: f32,
    /// Size of the micro-surface.
    pub size: (u32, u32),
    /// The distance between two adjacent samples.
    pub spacing: (f32, f32),
    /// Macro surface area.
    pub macro_area: f32,
    /// Micro surface area.
    pub micro_area: f32,
    /// Root-mean-square of the height.
    pub rms_height: f32,
    /// Root-mean-square of the slope along the x-axis.
    pub rms_slope_x: f32,
    /// Root-mean-square of the slope along the y-axis.
    pub rms_slope_y: f32,
    /// Level of subdivision.
    pub subdivision_level: u32,
    /// The kind of subdivision.
    pub subdivision_kind: SubdivisionKind,
    /// The offset along the z-axis. This is used for the wiggly subdivision.
    /// See [`Subdivision::Wiggly`].
    pub subdivision_offset: u32,
    /// The micro surface area with different levels of subdivision.
    pub subdivided_micro_area: Vec<(Subdivision, f32)>,
}

/// Measured data properties.
#[derive(Debug, Clone)]
pub struct MeasurementProp {
    /// The kind of the measured data.
    pub kind: MeasurementKind,
    /// The name of the measured data.
    pub name: String,
    /// Timestamp of the measured data.
    pub timestamp: DateTime<Local>,
    /// Source of the measured data.
    pub source: MeasurementSource,
    #[cfg(feature = "fitting")]
    /// Fitted model.
    pub fitted: FittedModels,
}

/// Property data.
///
/// This is the data that is shared between different instances of the property
/// inspector and the outliner.
#[derive(Debug)]
pub struct PropertyData {
    /// Micro surface properties.
    pub surfaces: HashMap<Handle<MicroSurface>, MicroSurfaceProp>,
    /// Measured data properties.
    pub measured: HashMap<Handle<Measurement>, MeasurementProp>,
    /// The currently selected item.
    pub selected: Option<OutlinerItem>,
}

impl PropertyData {
    /// Create a new instance of the property data.
    pub fn new() -> Self {
        Self {
            surfaces: HashMap::default(),
            measured: HashMap::default(),
            selected: None,
        }
    }

    /// Update the micro surface properties.
    ///
    /// This should be called whenever the micro surface cache is updated (new
    /// micro surfaces are added or removed)
    pub fn update_surfaces(&mut self, surfs: &[Handle<MicroSurface>], cache: &RawCache) {
        for hdl in surfs {
            if let std::collections::hash_map::Entry::Vacant(e) = self.surfaces.entry(*hdl) {
                let record = cache.get_micro_surface_record(*hdl).unwrap();
                let surf = cache.get_micro_surface(*e.key()).unwrap();
                let mesh = cache.get_micro_surface_mesh(record.mesh).unwrap();
                e.insert(MicroSurfaceProp {
                    name: record.name().to_string(),
                    visible: false,
                    scale: 1.0,
                    unit: surf.unit,
                    min: surf.min,
                    max: surf.max,
                    height_offset: mesh.height_offset,
                    size: (surf.rows as u32, surf.cols as u32),
                    spacing: (surf.du, surf.dv),
                    macro_area: surf.macro_area(),
                    micro_area: mesh.facet_total_area, /* TODO: maybe get this from the micro
                                                        * surface? */
                    rms_height: surf.rms_height(),
                    rms_slope_x: surf.rms_slope_x(),
                    rms_slope_y: surf.rms_slope_y(),
                    subdivision_level: 0,
                    subdivision_kind: SubdivisionKind::Curved,
                    subdivision_offset: 100,
                    subdivided_micro_area: vec![],
                });
            }
        }
    }

    /// Tests whether any of the micro surfaces are visible.
    pub fn any_visible_surfaces(&self) -> bool { self.surfaces.iter().any(|(_, s)| s.visible) }

    /// Returns a list of visible micro surfaces.
    pub fn visible_surfaces_with_props(&self) -> Vec<(&Handle<MicroSurface>, &MicroSurfaceProp)> {
        self.surfaces.iter().filter(|(_, s)| s.visible).collect()
    }

    pub fn visible_surfaces(&self) -> Vec<Handle<MicroSurface>> {
        self.surfaces
            .iter()
            .filter(|(_, s)| s.visible)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Updates the list of measurement data.
    pub fn update_measurement_data(
        &mut self,
        measurements: &[Handle<Measurement>],
        cache: &RawCache,
    ) {
        for meas in measurements {
            if let std::collections::hash_map::Entry::Vacant(e) = self.measured.entry(*meas) {
                let data = cache.get_measurement(*meas).unwrap();
                e.insert(MeasurementProp {
                    kind: data.kind(),
                    source: data.source.clone(),
                    timestamp: data.timestamp,
                    name: data.name.clone(),
                    #[cfg(feature = "fitting")]
                    fitted: FittedModels::default(),
                });
            }
        }
    }

    pub fn on_item_selected(&mut self, item: OutlinerItem) { self.selected = Some(item); }
}
