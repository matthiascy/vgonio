use crate::{
    app::cache::{Cache, Handle},
    msurf::MicroSurface,
    units::LengthUnit,
};
use std::collections::HashMap;

/// States of one item in the outliner.
#[derive(Clone, Debug)]
pub struct PerMicroSurfaceState {
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
    /// Offset along the Y axis without scaling.
    pub y_offset: f32,
}

/// Outliner is a widget that displays the scene graph of the current scene.
///
/// It will reads the micro-surfaces from the cache and display them in a tree
/// structure. The user can toggle the visibility of the micro surfaces.
pub struct Outliner {
    /// States of the micro surfaces, indexed by their ids.
    states: HashMap<Handle<MicroSurface>, PerMicroSurfaceState>,
    headers: HashMap<Handle<MicroSurface>, SurfaceCollapsableHeader>,
}

impl Default for Outliner {
    fn default() -> Self { Self::new() }
}

impl Outliner {
    /// Creates a new outliner.
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            headers: Default::default(),
        }
    }

    /// Returns an iterator over all the visible micro surfaces.
    pub fn visible_surfaces(&self) -> Vec<(&Handle<MicroSurface>, &PerMicroSurfaceState)> {
        self.states
            .iter()
            .filter(|(_, s)| s.visible)
            .map(|(id, s)| (id, s))
            .collect()
    }

    pub fn any_visible_surfaces(&self) -> bool { self.states.iter().any(|(_, s)| s.visible) }

    /// Updates the outliner according to the cache.
    pub fn update_surfaces(&mut self, surfs: &[Handle<MicroSurface>], cache: &Cache) {
        for hdl in surfs {
            if let std::collections::hash_map::Entry::Vacant(e) = self.states.entry(*hdl) {
                let record = cache.get_micro_surface_record(*hdl).unwrap();
                let surf = cache.get_micro_surface(*e.key()).unwrap();
                e.insert(PerMicroSurfaceState {
                    name: record.name().to_string(),
                    visible: false,
                    scale: 1.0,
                    unit: surf.unit,
                    min: surf.min,
                    max: surf.max,
                    y_offset: 0.0,
                });
                self.headers
                    .insert(*hdl, SurfaceCollapsableHeader { selected: false });
            }
        }
    }
}

struct SurfaceCollapsableHeader {
    selected: bool,
}

impl SurfaceCollapsableHeader {
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut PerMicroSurfaceState) {
        let id = ui.make_persistent_id(&state.name);
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
            .show_header(ui, |ui| {
                ui.vertical_centered_justified(|ui| {
                    ui.horizontal(|ui| {
                        if ui.selectable_label(self.selected, &state.name).clicked() {
                            self.selected = !self.selected;
                        }
                    })
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.checkbox(&mut state.visible, "");
                })
            })
            .body(|ui| {
                // Scale
                egui::Grid::new("my_grid")
                    .num_columns(3)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.add(egui::Label::new("Min"));
                        ui.add(egui::Label::new(format!("{:.4} {}", state.min, state.unit)));
                        ui.end_row();
                        ui.add(egui::Label::new("Max"));
                        ui.add(egui::Label::new(format!("{:.4} {}", state.max, state.unit)));
                        ui.end_row();
                        ui.add(egui::Label::new(format!("Y Offset ({})", state.unit)))
                            .on_hover_text(
                                "Offset along the Y axis without scaling. (Visual only - does not \
                                 affect the actual surface)",
                            );
                        ui.add(
                            egui::Slider::new(&mut state.y_offset, -100.0..=100.0)
                                .trailing_fill(true),
                        );
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .add(egui::Button::new("Median"))
                                .on_hover_text(
                                    "Sets the Y offset to median value of the surface heights",
                                )
                                .clicked()
                            {
                                state.y_offset = -(state.min + state.max) * 0.5;
                            }
                            if ui
                                .add(egui::Button::new("Ground"))
                                .on_hover_text(
                                    "Adjusts its position so that the minimum height value is at \
                                     the ground level.",
                                )
                                .clicked()
                            {
                                state.y_offset = -state.min;
                            }
                        });
                        ui.end_row();
                        ui.add(egui::Label::new("Scale")).on_hover_text(
                            "Scales the surface visually. Doest not affect the actual surface.",
                        );
                        ui.add(egui::Slider::new(&mut state.scale, 0.05..=1.5).trailing_fill(true));
                    });
            });
    }
}

// GUI related functions
impl Outliner {
    /// Creates the ui for the outliner.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.set_min_size(egui::Vec2::new(460.0, 200.0));
        egui::CollapsingHeader::new("Surfaces")
            .default_open(true)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    for ((_, state), (_, header)) in
                        self.states.iter_mut().zip(self.headers.iter_mut())
                    {
                        header.ui(ui, state);
                    }
                });
            });
    }

    /// Represents the outliner as a window.
    pub fn show(&mut self, ctx: &egui::Context) {
        {
            let mut open = true;
            egui::Window::new("Outliner")
                .open(&mut open)
                //.title_bar(false)
                .collapsible(true)
                .vscroll(true)
                .anchor(egui::Align2::RIGHT_TOP, (0.0, 0.0))
                .show(ctx, |ui| self.ui(ui));
        }
    }
}
