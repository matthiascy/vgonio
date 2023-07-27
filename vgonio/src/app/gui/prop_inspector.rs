use std::sync::{Arc, RwLock};

use crate::app::gui::docking::{Dockable, WidgetKind};

use crate::app::gui::{data::PropertyData, theme::ThemeVisuals};

/// The property inspector.
///
/// The property inspector is a dockable widget that shows the properties of
/// selected objects. It is also used to edit the properties of objects.
pub struct PropertyInspector {
    /// The unique identifier of the property inspector.
    uuid: uuid::Uuid,
    /// Property inspector data.
    data: Arc<RwLock<PropertyData>>,
}

impl PropertyInspector {
    pub fn new() -> Self {
        Self {
            uuid: uuid::Uuid::new_v4(),
            data: Arc::new(RwLock::new(PropertyData::new())),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) { ui.label("Properties"); }
}

impl Dockable for PropertyInspector {
    fn kind(&self) -> WidgetKind { WidgetKind::Properties }

    fn title(&self) -> egui::WidgetText { "Properties".into() }

    fn uuid(&self) -> uuid::Uuid { self.uuid }

    fn ui(&mut self, ui: &mut egui::Ui) { self.ui(ui); }
}
