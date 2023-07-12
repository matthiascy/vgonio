use super::state::InputState;
use egui::{Id, Ui, WidgetText};
use egui_dock::NodeIndex;
use std::{fmt, fmt::Debug, time::Duration};
use uuid::Uuid;

/// A trait for ui elements that can be docked.
pub trait Dockable {
    /// The title of the dockable.
    fn title(&self) -> WidgetText;

    /// Updates the dockable with the given input state.
    fn update_with_input_state(&mut self, _input: &InputState, _dt: Duration) {}

    /// The content of the dockable.
    fn ui(&mut self, ui: &mut Ui);

    /// The unique id of the dockable. Used to avoid ID collisions in egui.
    fn uuid(&self) -> Uuid;
}

pub(crate) struct DockableString {
    pub string: String,
    pub uuid: Uuid,
}

impl DockableString {
    pub fn new(string: String) -> Self {
        Self {
            string,
            uuid: Uuid::new_v4(),
        }
    }
}

/// Implenents `Dockable` for `String` only showing the string as a label.
/// Only available in debug builds for testing purposes.
#[cfg(debug_assertions)]
impl Dockable for DockableString {
    fn title(&self) -> WidgetText { self.string.clone().into() }

    fn ui(&mut self, ui: &mut Ui) { ui.label(&self.string); }

    fn uuid(&self) -> Uuid { self.uuid }
}

/// The kind of a tab in the dock tree.
#[derive(Debug)]
pub enum TabKind {
    Outliner,
    SurfViewer,
    BsdfViewer,
    #[cfg(debug_assertions)]
    Regular,
}

/// A tab in the dock tree.
pub struct Tab {
    /// The kind of the tab.
    pub kind: TabKind,
    /// The node index of the tab in the dock tree.
    pub index: NodeIndex,
    /// The dockable content of the tab.
    pub dockable: Box<dyn Dockable>,
}

impl Debug for Tab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tab")
            .field("kind", &self.kind)
            .field("index", &self.index)
            .finish()
    }
}

pub struct TabInfo {
    kind: TabKind,
    node: NodeIndex,
}

/// Viewer for the tabs in the dock tree.
pub(crate) struct TabViewer<'a> {
    pub to_be_added: &'a mut Vec<ToBeAdded>,
}

pub struct ToBeAdded {
    pub kind: TabKind,
    pub parent: NodeIndex,
}

impl<'a> egui_dock::TabViewer for TabViewer<'a> {
    type Tab = Tab;

    fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) { tab.dockable.ui(ui); }

    fn title(&mut self, tab: &mut Self::Tab) -> WidgetText { tab.dockable.title() }

    fn id(&mut self, tab: &mut Self::Tab) -> Id { Id::new(tab.dockable.uuid()) }

    fn add_popup(&mut self, ui: &mut Ui, node: NodeIndex) {
        ui.set_min_width(120.0);

        #[cfg(debug_assertions)]
        if ui.button("Regular").clicked() {
            self.to_be_added.push(ToBeAdded {
                kind: TabKind::Regular,
                parent: node,
            });
        }

        if ui.button("SurfViewer").clicked() {
            self.to_be_added.push(ToBeAdded {
                kind: TabKind::SurfViewer,
                parent: node,
            });
        }

        if ui.button("Outliner").clicked() {
            self.to_be_added.push(ToBeAdded {
                kind: TabKind::Outliner,
                parent: node,
            });
        }
    }
}
