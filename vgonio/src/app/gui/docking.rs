use crate::app::{
    cache::Cache,
    gfx::GpuContext,
    gui::{data::PropertyData, event::EventLoopProxy},
};
use std::{
    fmt::{self, Debug},
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock},
};

use crate::app::gui::{
    event::{SurfaceViewerEvent, VgonioEvent},
    outliner::Outliner,
    prop_insp::PropertyInspector,
    state::GuiRenderer,
    surf_viewer::SurfaceViewer,
    theme::ThemeKind,
};

/// Docking space for widgets.
pub struct DockSpace {
    gui: Arc<RwLock<GuiRenderer>>,   // TODO: remove
    cache: Arc<RwLock<Cache>>,       // TODO: remove
    data: Arc<RwLock<PropertyData>>, // TODO: remove
    /// GPU context.
    gpu: Arc<GpuContext>,
    /// Event loop proxy.
    event_loop: EventLoopProxy,
    /// Inner tree of the dock space.
    inner: egui_dock::Tree<DockingWidget>,
    /// New widgets to be added to the dock space.
    added: Vec<NewWidget>,
}

impl Deref for DockSpace {
    type Target = egui_dock::Tree<DockingWidget>;

    fn deref(&self) -> &Self::Target { &self.inner }
}

impl DerefMut for DockSpace {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
}

impl DockSpace {
    /// Create a new dock space with a default layout.
    ///
    /// |   | 1 |
    /// | 0 | - |
    /// |   | 2 |
    pub fn default_layout(
        gpu: Arc<GpuContext>,
        gui: Arc<RwLock<GuiRenderer>>,
        cache: Arc<RwLock<Cache>>,
        data: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        let mut inner = egui_dock::Tree::new(vec![DockingWidget {
            index: 0,
            dockable: Box::new(DockableString::new(String::from("Hello world"))),
        }]);
        let [_, r] = inner.split_right(
            egui_dock::NodeIndex::root(),
            0.8,
            vec![DockingWidget {
                index: 1,
                dockable: Box::new(Outliner::new(data.clone(), event_loop.clone())),
            }],
        );
        inner.split_below(
            r,
            0.5,
            vec![DockingWidget {
                index: 2,
                dockable: Box::new(PropertyInspector::new()),
            }],
        );
        Self {
            gui,
            cache,
            data,
            gpu,
            event_loop,
            inner,
            added: Vec::new(),
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        data: Arc<RwLock<PropertyData>>,
        theme_kind: ThemeKind,
    ) {
        use egui_dock::NodeIndex;

        egui_dock::DockArea::new(&mut self.inner)
            .show_add_buttons(true)
            .show_add_popup(true)
            .show(
                ctx,
                &mut DockSpaceView {
                    added: &mut self.added,
                },
            );

        self.added.drain(..).for_each(|new_tab| {
            // Focus the node that we want to add a tab to.
            self.inner.set_focused_node(NodeIndex(new_tab.parent));
            // Allocate a new index for the tab.
            let index = self.inner.num_tabs();
            // Add the widget to the dock space.
            let widget: Box<dyn Dockable> = match new_tab.kind {
                WidgetKind::Outliner => {
                    Box::new(Outliner::new(data.clone(), self.event_loop.clone()))
                }
                WidgetKind::SurfViewer => {
                    let widget = Box::new(SurfaceViewer::new(
                        self.gpu.clone(),
                        self.gui.clone(),
                        256,
                        256,
                        wgpu::TextureFormat::Bgra8UnormSrgb,
                        self.cache.clone(),
                        theme_kind,
                        self.event_loop.clone(),
                        self.data.clone(),
                    ));
                    self.event_loop
                        .send_event(VgonioEvent::SurfaceViewer(SurfaceViewerEvent::Create {
                            uuid: widget.uuid(),
                            tex_id: widget.color_attachment_id(),
                        }))
                        .unwrap();
                    widget
                }
                _ => Box::new(DockableString::new(String::from("Hello world"))),
            };
            self.inner.push_to_focused_leaf(DockingWidget {
                index,
                dockable: widget,
            });
        });
    }
}

/// Common docking functionality for all dockable widgets
pub trait Dockable {
    /// Kind of the widget.
    fn kind(&self) -> WidgetKind;

    /// Title of the widget.
    fn title(&self) -> egui::WidgetText;

    /// Unique identifier of the widget.
    fn uuid(&self) -> uuid::Uuid;

    /// Actual content of the widget.
    fn ui(&mut self, ui: &mut egui::Ui);
}

/// Kind of a possible widget.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WidgetKind {
    #[cfg(debug_assertions)]
    String,
    Outliner,
    SurfViewer,
    BsdfViewer,
    Properties,
}

/// A tab that can be stored in the dock space.
pub struct DockingWidget {
    /// Index of the tab in the dock space inner tree.
    pub index: usize,
    /// Widget to be displayed in the tab.
    pub dockable: Box<dyn Dockable>,
}

impl Debug for DockingWidget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DockingTab")
            .field("index", &self.index)
            .field("kind", &self.dockable.kind())
            .field("widget", &self.dockable.uuid())
            .finish()
    }
}

/// New tab to be added to the dock space.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NewWidget {
    /// Widget to be created.
    pub kind: WidgetKind,
    /// Index of the parent tab in the dock space inner tree.
    pub parent: usize,
}

/// Utility structure to render the tabs in the dock space.
pub struct DockSpaceView<'a> {
    pub added: &'a mut Vec<NewWidget>,
}

impl<'a> egui_dock::TabViewer for DockSpaceView<'a> {
    type Tab = DockingWidget;

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) { tab.dockable.ui(ui) }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText { tab.dockable.title() }

    fn id(&mut self, tab: &mut Self::Tab) -> egui::Id { egui::Id::new(tab.dockable.uuid()) }

    fn add_popup(&mut self, ui: &mut egui::Ui, parent: egui_dock::NodeIndex) {
        ui.set_min_width(120.0);

        if ui.button("SurfViewer").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::SurfViewer,
                parent: parent.0,
            });
        }

        if ui.button("BsdfViewer").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::BsdfViewer,
                parent: parent.0,
            });
        }

        if ui.button("Outliner").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::Outliner,
                parent: parent.0,
            });
        }

        #[cfg(debug_assertions)]
        if ui.button("String").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::String,
                parent: parent.0,
            });
        }
    }
}

/// Test utility structure to render a string tab in the dock space.
#[cfg(debug_assertions)]
pub struct DockableString {
    pub uuid: uuid::Uuid,
    pub content: String,
}

impl DockableString {
    pub fn new(content: String) -> Self { Self::from(content) }
}

impl From<String> for DockableString {
    fn from(content: String) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4(),
            content,
        }
    }
}

#[cfg(debug_assertions)]
impl Dockable for DockableString {
    fn kind(&self) -> WidgetKind { WidgetKind::String }

    fn title(&self) -> egui::WidgetText { "Dockable String".into() }

    fn uuid(&self) -> uuid::Uuid { self.uuid }

    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label(&self.content);
        ui.label(&format!("uuid: {}, content: {}", self.uuid, self.content));
    }
}
