use crate::app::{
    cache::Cache,
    gui::{
        data::PropertyData,
        event::{EventLoopProxy, SurfaceViewerEvent, VgonioEvent},
        outliner::Outliner,
        plotter::PlotInspector,
        prop_inspector::PropertyInspector,
        surf_viewer::SurfaceViewer,
    },
};
use egui_dock::DockState;
use std::{
    fmt::{self, Debug},
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock},
};
use uuid::Uuid;
use uxtk::UiRenderer;

/// Docking space for widgets.
pub struct DockSpace {
    gui: Arc<RwLock<UiRenderer>>,
    cache: Cache,
    data: Arc<RwLock<PropertyData>>,
    /// Event loop proxy.
    event_loop: EventLoopProxy,
    /// Inner tree of the dock space.
    inner: DockState<DockingWidget>,
    /// New widgets to be added to the dock space.
    added: Vec<NewWidget>,
}

impl Deref for DockSpace {
    type Target = DockState<DockingWidget>;

    fn deref(&self) -> &Self::Target { &self.inner }
}

impl DerefMut for DockSpace {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
}

// TODO: deal with the case where the tab is closed

impl DockSpace {
    /// Create a new dock space with a default layout.
    ///
    /// |   | 1 |
    /// | 0 | - |
    /// |   | 2 |
    pub fn default_layout(
        gui: Arc<RwLock<UiRenderer>>,
        cache: Cache,
        data: Arc<RwLock<PropertyData>>,
        event_loop: EventLoopProxy,
    ) -> Self {
        log::info!("Creating default dock space layout");
        let surf_viewer = Box::new(SurfaceViewer::new(gui.clone(), event_loop.clone()));
        log::info!("Created surface viewer");
        event_loop.send_event(VgonioEvent::SurfaceViewer(SurfaceViewerEvent::Create {
            uuid: surf_viewer.uuid(),
            tex_id: surf_viewer.color_attachment_id(),
        }));
        let mut inner = DockState::new(vec![DockingWidget {
            index: 0,
            dockable: surf_viewer,
        }]);
        let [_, r] = inner.main_surface_mut().split_right(
            egui_dock::NodeIndex::root(),
            0.8,
            vec![DockingWidget {
                index: 1,
                dockable: Box::new(Outliner::new(data.clone(), event_loop.clone())),
            }],
        );
        inner.main_surface_mut().split_below(
            r,
            0.5,
            vec![DockingWidget {
                index: 2,
                dockable: Box::new(PropertyInspector::new(
                    event_loop.clone(),
                    cache.clone(),
                    data.clone(),
                )),
            }],
        );
        Self {
            gui,
            cache,
            data,
            event_loop,
            inner,
            added: Vec::new(),
        }
    }

    pub fn surface_viewers(&self) -> Vec<Uuid> {
        self.inner
            .main_surface()
            .tabs()
            .filter(|t| t.dockable.kind() == WidgetKind::SurfViewer)
            .map(|t| t.dockable.uuid())
            .collect()
    }

    pub fn show(&mut self, ctx: &egui::Context, data: Arc<RwLock<PropertyData>>) {
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
            self.inner
                .main_surface_mut()
                .set_focused_node(NodeIndex(new_tab.parent));
            // Allocate a new index for the tab.
            let index = self.inner.main_surface().num_tabs();
            // Add the widget to the dock space.
            let widget: Box<dyn Dockable> = match new_tab.kind {
                WidgetKind::Outliner => {
                    Box::new(Outliner::new(data.clone(), self.event_loop.clone()))
                },
                WidgetKind::SurfViewer => {
                    let widget = Box::new(SurfaceViewer::new(
                        self.gui.clone(),
                        self.event_loop.clone(),
                    ));
                    self.event_loop.send_event(VgonioEvent::SurfaceViewer(
                        SurfaceViewerEvent::Create {
                            uuid: widget.uuid(),
                            tex_id: widget.color_attachment_id(),
                        },
                    ));
                    widget
                },
                WidgetKind::Properties => Box::new(PropertyInspector::new(
                    self.event_loop.clone(),
                    self.cache.clone(),
                    data.clone(),
                )),
                WidgetKind::Plotting => Box::new(PlotInspector::new(
                    "New Plot",
                    self.cache.clone(),
                    self.data.clone(),
                    self.event_loop.clone(),
                )),
            };
            self.inner.push_to_focused_leaf(DockingWidget {
                index,
                dockable: widget,
            });
        });
    }

    pub fn add_existing_widget(&mut self, widget: Box<dyn Dockable>) {
        if self
            .inner
            .main_surface()
            .tabs()
            .any(|t| t.dockable.uuid() == widget.uuid())
        {
            return;
        }

        self.inner.push_to_first_leaf(DockingWidget {
            index: self.inner.main_surface().num_tabs(),
            dockable: widget,
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
    fn uuid(&self) -> Uuid;

    /// Actual content of the widget.
    fn ui(&mut self, ui: &mut egui::Ui);
}

/// Kind of a possible widget.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WidgetKind {
    Outliner,
    SurfViewer,
    // BsdfViewer, TODO:
    Properties,
    Plotting,
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

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText { tab.dockable.title() }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) { tab.dockable.ui(ui) }

    fn id(&mut self, tab: &mut Self::Tab) -> egui::Id { egui::Id::new(tab.dockable.uuid()) }

    fn add_popup(
        &mut self,
        ui: &mut egui::Ui,
        surface: egui_dock::SurfaceIndex,
        parent: egui_dock::NodeIndex,
    ) {
        if !surface.is_main() {
            return;
        }

        ui.set_min_width(120.0);

        if ui.button("SurfViewer").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::SurfViewer,
                parent: parent.0,
            });
        }

        // if ui.button("BsdfViewer").clicked() {
        //     self.added.push(NewWidget {
        //         kind: WidgetKind::BsdfViewer,
        //         parent: parent.0,
        //     });
        // }

        if ui.button("Outliner").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::Outliner,
                parent: parent.0,
            });
        }

        if ui.button("Properties").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::Properties,
                parent: parent.0,
            });
        }

        if ui.button("Plotting").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::Plotting,
                parent: parent.0,
            });
        }
    }
}
