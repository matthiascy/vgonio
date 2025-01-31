use egui::Color32;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};
use vgonio_core::res::Handle;

/// The selection mode for surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    /// Only one single surface can be selected.
    Single,
    /// Multiple surfaces can be selected.
    Multiple,
}

/// A helper struct used in GUI to select surfaces.
#[derive(Debug, Clone)]
pub struct SurfaceSelector {
    /// The selection mode.
    mode: SelectionMode,
    /// The selected surfaces.
    selected: HashSet<Handle>,
    /// The surfaces that can be selected.
    surfaces: HashMap<Handle, String>,
    /// Selection changed.
    changed: bool,
}

impl SurfaceSelector {
    /// Creates a new surface selector allowing single selection.
    pub fn single() -> Self {
        SurfaceSelector {
            mode: SelectionMode::Single,
            selected: Default::default(),
            surfaces: Default::default(),
            changed: false,
        }
    }

    /// Creates a new surface selector allowing multiple selection.
    pub fn multiple() -> Self {
        SurfaceSelector {
            mode: SelectionMode::Multiple,
            selected: Default::default(),
            surfaces: Default::default(),
            changed: false,
        }
    }

    #[allow(dead_code)]
    /// Checks if any surface is selected.
    pub fn any_selected(&self) -> bool { !self.selected.is_empty() }

    /// Returns the selected surfaces.
    pub fn selected(&self) -> impl ExactSizeIterator<Item = Handle> + '_ {
        self.selected.iter().copied()
    }

    /// Returns the first selected surface.
    pub fn first_selected(&self) -> Option<Handle> { self.selected.iter().next().copied() }

    /// Returns the number of selected surfaces.
    pub fn selected_count(&self) -> usize { self.selected.len() }

    /// Returns the single selected surface only if the mode of selection is
    /// single.
    pub fn single_selected(&self) -> Option<Handle> {
        if self.mode == SelectionMode::Single {
            self.selected.iter().next().copied()
        } else {
            None
        }
    }

    /// Returns whether the selection changed.
    pub fn selection_changed(&mut self) -> bool { self.changed }

    /// Updates the list of surfaces.
    pub fn update<'a>(&mut self, surfs: impl IntoIterator<Item = (Handle, &'a str)>) {
        for (hdl, name) in surfs {
            if self.surfaces.contains_key(&hdl) {
                continue;
            }
            self.surfaces.insert(hdl, name.to_string());
        }
    }

    /// Ui for the surface selector.
    pub fn ui(&mut self, id_salt: impl Hash, ui: &mut egui::Ui) {
        let mut to_be_added: Option<Handle> = None;
        egui::Grid::new("surface_selector_grid")
            .num_columns(2)
            .show(ui, |ui| {
                let selected = self.selected.clone();
                for hdl in selected.into_iter() {
                    ui.label(self.surfaces.get(&hdl).unwrap());
                    if ui
                        .add(
                            egui::Button::new("\u{2716}")
                                .fill(Color32::TRANSPARENT)
                                .rounding(5.0),
                        )
                        .clicked()
                    {
                        self.selected.remove(&hdl);
                    }
                    ui.end_row();
                }
                egui::ComboBox::from_id_salt(id_salt)
                    .selected_text("Select micro-surface")
                    .show_ui(ui, |ui| {
                        for (hdl, name) in &self.surfaces {
                            ui.selectable_value(&mut to_be_added, Some(*hdl), name);
                        }
                        if let Some(hdl) = to_be_added.take() {
                            if self.selected.len() == 1 && self.mode == SelectionMode::Single {
                                self.selected.clear();
                            }
                            if self.selected.insert(hdl) {
                                self.changed = true;
                            }
                        }
                    });
                if ui.button("Clear").clicked() {
                    self.selected.clear();
                }
                ui.end_row();
            });
    }
}
