mod bsdf;
mod madf;
mod mmsf;

use crate::{
    app::{
        cache::{Cache, Handle},
        gui::{
            simulations::{bsdf::BsdfSimulation, madf::MadfSimulation, mmsf::MmsfSimulation},
            VgonioEventLoop,
        },
    },
    msurf::MicroSurface,
};

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
    pub fn new(event_loop: VgonioEventLoop) -> Self {
        log::info!("Initializing simulations");
        Simulations {
            bsdf_sim: BsdfSimulation::new(event_loop.clone()),
            madf_sim: MadfSimulation::new(event_loop.clone()),
            mmsf_sim: MmsfSimulation::new(event_loop),
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
