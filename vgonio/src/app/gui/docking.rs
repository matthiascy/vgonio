use std::fmt::{self, Debug};

/// Docking space for widgets.
pub struct DockSpace {
    inner: egui_dock::Tree<DockingTab>,
}

/// Common docking functionality for all dockable widgets
pub trait Dockable {
    /// Unique identifier of the widget.
    fn uuid(&self) -> uuid::Uuid;

    /// Title of the widget.
    fn title(&self) -> egui::WidgetText;

    /// Actual content of the widget.
    fn ui(&mut self, ui: &mut egui::Ui);

    /// Kind of the widget.
    fn kind(&self) -> WidgetKind;
}

/// Kind of a possible widget.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WidgetKind {
    #[cfg(debug_assertions)]
    String,
    Outliner,
    SurfViewer,
    BsdfViewer,
}

/// A tab that can be stored in the dock space.
pub struct DockingTab {
    /// Index of the tab in the dock space inner tree.
    pub index: usize,
    /// Widget to be displayed in the tab.
    pub widget: Box<dyn Dockable>,
}

impl Debug for DockingTab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DockingTab")
            .field("index", &self.index)
            .field("kind", &self.widget.kind())
            .field("widget", &self.widget.uuid())
            .finish()
    }
}

/// New tab to be added to the dock space.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NewTab {
    /// Widget to be created.
    pub kind: WidgetKind,
    /// Index of the parent tab in the dock space inner tree.
    pub parent: usize,
}

/// Utility structure to render the tabs in the dock space.
pub struct DockingDisplay<'a> {
    pub added: &'a mut Vec<NewTab>,
}

impl<'a> egui_dock::TabViewer for DockingDisplay<'a> {
    type Tab = DockingTab;

    fn id(&mut self, tab: &mut Self::Tab) -> egui::Id { egui::Id::new(tab.widget.uuid()) }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) { tab.widget.ui(ui) }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText { tab.widget.title() }

    fn add_popup(&mut self, ui: &mut egui::Ui, parent: egui_dock::NodeIndex) {
        ui.set_min_width(120.0);

        if ui.button("SurfViewer").clicked() {
            self.added.push(NewTab {
                kind: WidgetKind::SurfViewer,
                parent: parent.0,
            });
        }

        if ui.button("BsdfViewer").clicked() {
            self.added.push(NewTab {
                kind: WidgetKind::BsdfViewer,
                parent: parent.0,
            });
        }

        if ui.button("Outliner").clicked() {
            self.added.push(NewTab {
                kind: WidgetKind::Outliner,
                parent: parent.0,
            });
        }

        #[cfg(debug_assertions)]
        if ui.button("String").clicked() {
            self.added.push(NewTab {
                kind: WidgetKind::String,
                parent: parent.0,
            });
        }
    }
}

#[cfg(debug_assertions)]
pub struct DockableString {
    pub uuid: uuid::Uuid,
    pub content: String,
}

#[cfg(debug_assertions)]
impl Dockable for DockableString {
    fn uuid(&self) -> uuid::Uuid { self.uuid }

    fn title(&self) -> egui::WidgetText { "Dockable String".into() }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label(&self.content);
        ui.label(&format!("uuid: {}, content: {}", self.uuid, self.content));
    }

    fn kind(&self) -> WidgetKind { WidgetKind::String }
}
