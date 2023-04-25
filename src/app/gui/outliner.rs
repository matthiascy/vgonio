use crate::{
    app::cache::{Cache, Handle},
    msurf::MicroSurface,
};
use egui::Widget;
use std::{cell::RefCell, collections::HashMap, ops::Deref, sync::Arc};

/// States of one item in the outliner.
#[derive(Clone, Debug)]
struct State {
    /// The name of the micro surface.
    pub name: String,
    /// Whether the micro surface is visible.
    pub visible: bool,
    /// The scale factor of the micro surface.
    pub scale: f32,
}

/// Outliner is a widget that displays the scene graph of the current scene.
///
/// It will reads the micro-surfaces from the cache and display them in a tree
/// structure. The user can toggle the visibility of the micro surfaces.
pub struct Outliner {
    /// States of the micro surfaces, indexed by their ids.
    states: HashMap<Handle<MicroSurface>, State>,
    headers: HashMap<Handle<MicroSurface>, SurfaceCollapsableHeader>,
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
    pub fn visible_surfaces(&self) -> impl Iterator<Item = (&Handle<MicroSurface>, f32)> {
        self.states
            .iter()
            .filter(|(_, s)| s.visible)
            .map(|(id, s)| (id, s.scale))
    }

    /// Updates the outliner according to the cache.
    pub fn update_surfaces(&mut self, cache: &Cache) {
        let records = cache.records.iter().map(|(_, record)| record);
        for record in records {
            if !self.states.contains_key(&record.surf) {
                self.states.insert(
                    record.surf,
                    State {
                        name: record.name().to_string(),
                        visible: false,
                        scale: 1.0,
                    },
                );
                self.headers
                    .insert(record.surf, SurfaceCollapsableHeader { selected: false });
            }
        }

        println!("Cache records: {:?}", cache.records);

        println!("Outliner: {:?}", self.states);
    }
}

struct SurfaceCollapsableHeader {
    selected: bool,
}

impl SurfaceCollapsableHeader {
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &mut State) {
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
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.add(egui::Label::new("Scale"));
                        ui.add(egui::Slider::new(&mut state.scale, 0.05..=1.5).trailing_fill(true));
                    });
            });
    }
}

// GUI related functions
impl Outliner {
    /// Creates the ui for the outliner.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Outliner");
        ui.separator();

        ui.set_min_width(200.0);
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
                .title_bar(false)
                .resizable(false)
                .anchor(egui::Align2::RIGHT_TOP, (0.0, 0.0))
                .show(ctx, |ui| self.ui(ui));
        }
    }
}
