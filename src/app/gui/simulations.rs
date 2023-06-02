mod bsdf;
mod madf;
mod mmsf;

use crate::{
    app::{
        cache::{Cache, Handle},
        gui::{
            simulations::{bsdf::BsdfSimulation, madf::MadfSimulation, mmsf::MmsfSimulation},
            widgets::ToggleSwitch,
            VgonioEvent,
        },
    },
    measure::measurement::{
        BsdfMeasurementParams, MadfMeasurementParams, Measurement, MmsfMeasurementParams,
    },
    msurf::MicroSurface,
};
use egui::{epaint::ahash::HashSet, Color32};
use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
    hash::Hash,
    path::Path,
    sync::{Arc, Mutex},
};
use winit::event_loop::EventLoopProxy;

/// A helper struct used in GUI to select surfaces.
#[derive(Debug, Clone, Default)]
pub(crate) struct SurfaceSelector {
    pub selected: HashSet<Handle<MicroSurface>>,
    pub surfaces: HashMap<Handle<MicroSurface>, String>,
}

impl SurfaceSelector {
    /// Updates the list of surfaces.
    pub fn update(&mut self, surfs: &[Handle<MicroSurface>], cache: &Cache) {
        let surfs = surfs
            .iter()
            .filter(|hdl| !self.surfaces.iter().any(|(s, _)| s == *hdl));
        for record in cache.get_micro_surface_records(surfs) {
            self.surfaces.insert(record.surf, record.name().to_string());
        }
    }

    pub fn ui(&mut self, id_source: impl Hash, ui: &mut egui::Ui) {
        let mut to_be_added: Option<Handle<MicroSurface>> = None;
        ui.columns(1, |uis| {
            let ui = &mut uis[0];
            let selected = self.selected.clone();
            for hdl in selected.into_iter() {
                ui.horizontal_wrapped(|ui| {
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
                    ui.label(self.surfaces.get(&hdl).unwrap());
                });
            }
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_id_source(id_source)
                    .selected_text("Add micro-surface")
                    .show_ui(ui, |ui| {
                        for (hdl, name) in &self.surfaces {
                            ui.selectable_value(&mut to_be_added, Some(*hdl), name);
                        }
                        if let Some(hdl) = to_be_added.take() {
                            self.selected.insert(hdl);
                        }
                    });
                if ui.button("Clear").clicked() {
                    self.selected.clear();
                }
            });
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SimulationTab {
    Bsdf,
    Madf,
    Mmsf,
}

pub struct Simulations {
    pub bsdf_sim: BsdfSimulation,
    pub madf_sim: MadfSimulation,
    pub mmsf_sim: MmsfSimulation,
    pub states: [bool; 3],
    tab: SimulationTab,
}

impl Simulations {
    pub fn new(event_loop: EventLoopProxy<VgonioEvent>) -> Self {
        Simulations {
            bsdf_sim: BsdfSimulation::new(event_loop.clone()),
            madf_sim: MadfSimulation::new(event_loop.clone()),
            mmsf_sim: MmsfSimulation::new(event_loop.clone()),
            states: [false; 3],
            tab: SimulationTab::Madf,
        }
    }

    pub fn show_bsdf_sim(&mut self, ctx: &egui::Context) {
        egui::Window::new("BSDF Simulation")
            .open(&mut self.states[0])
            .show(ctx, |ui| self.bsdf_sim.ui(ui));
    }

    pub fn show_madf_sim(&mut self, ctx: &egui::Context) {
        egui::Window::new("NDF Simulation")
            .open(&mut self.states[1])
            .resizable(false)
            .show(ctx, |ui| self.madf_sim.ui(ui));
    }

    pub fn show_mmsf_sim(&mut self, ctx: &egui::Context) {
        egui::Window::new("Masking/Shadowing Simulation")
            .open(&mut self.states[2])
            .show(ctx, |ui| self.mmsf_sim.ui(ui));
    }

    pub fn show_all(&mut self, ctx: &egui::Context) {
        self.show_bsdf_sim(ctx);
        self.show_madf_sim(ctx);
        self.show_mmsf_sim(ctx);
    }

    pub fn open_bsdf_sim(&mut self) { self.states[0] = true; }

    pub fn open_madf_sim(&mut self) { self.states[1] = true; }

    pub fn open_mmsf_sim(&mut self) { self.states[2] = true; }

    pub fn update_loaded_surfaces(&mut self, surfs: &[Handle<MicroSurface>], cache: &Cache) {
        self.madf_sim.selector.update(surfs, cache);
        self.mmsf_sim.selector.update(surfs, cache);
        self.bsdf_sim.selector.update(surfs, cache);
    }

    pub fn ui(&mut self, ctx: &egui::Context) {
        egui::Window::new("Simulations").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_label(self.tab == SimulationTab::Bsdf, "BSDF")
                    .clicked()
                    .then(|| self.tab = SimulationTab::Bsdf);
                ui.selectable_label(self.tab == SimulationTab::Madf, "NDF")
                    .clicked()
                    .then(|| self.tab = SimulationTab::Madf);
                ui.selectable_label(self.tab == SimulationTab::Mmsf, "M/S")
                    .clicked()
                    .then(|| self.tab = SimulationTab::Mmsf);
            });
            ui.separator();
            match self.tab {
                SimulationTab::Bsdf => self.show_bsdf_sim(ctx),
                SimulationTab::Madf => self.show_madf_sim(ctx),
                SimulationTab::Mmsf => self.show_mmsf_sim(ctx),
            }
        });
    }
}
