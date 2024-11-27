use egui_dock::DockState;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    fmt::Debug,
    ops::{Deref, DerefMut},
};

/// Common trait for all dockable widgets.
pub trait Dockable {
    /// Kind of the widget.
    fn kind(&self) -> WidgetKind;

    /// UUID of the widget.
    fn uuid(&self) -> uuid::Uuid;

    /// Title of the widget.
    /// This title is displayed in the tab of the widget.
    fn title(&self) -> egui::WidgetText;

    /// Draw the widget.
    fn draw(&mut self, ui: &mut egui::Ui);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DummyDockable {
    uuid: uuid::Uuid,
    kind: WidgetKind,
}

impl DummyDockable {
    pub fn new(kind: WidgetKind) -> Self {
        Self {
            uuid: uuid::Uuid::new_v4(),
            kind,
        }
    }
}

impl Dockable for DummyDockable {
    fn kind(&self) -> WidgetKind { self.kind }

    fn uuid(&self) -> uuid::Uuid { self.uuid }

    fn title(&self) -> egui::WidgetText {
        match self.kind {
            WidgetKind::SurfViewer => "SurfViewer".into(),
            WidgetKind::BsdfViewer => "BsdfViewer".into(),
            WidgetKind::Outliner => "Outliner".into(),
            WidgetKind::Properties => "Properties".into(),
            WidgetKind::Plotting => "Plotting".into(),
        }
    }

    fn draw(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("This is a dummy widget of type {:?}", self.kind));
    }
}

/// A tab that can be stored in a dock space.
pub struct DockWidget {
    /// Index of the tab in the dock space state.
    pub index: u32,
    /// Widget to be displayed in the tab.
    pub dockable: Box<dyn Dockable>,
}

impl Debug for DockWidget {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "DockWidget {{ index: {}, dockable: {:?}@{:?}  }}",
            self.index,
            self.dockable.uuid(),
            self.dockable.kind()
        )
    }
}

/// Docking space for widgets.
pub struct DockSpace {
    /// The underlying tree of the dock space.
    state: DockState<DockWidget>,
    /// New widgets to be added to the dock space.
    to_add: Vec<NewWidget>,
}

impl Deref for DockSpace {
    type Target = DockState<DockWidget>;

    fn deref(&self) -> &Self::Target { &self.state }
}

impl DerefMut for DockSpace {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.state }
}

impl DockSpace {
    /// Creates a new dock space with the default layout.
    ///
    /// |   | 1 |
    /// | 0 | - |
    /// |   | 2 |
    pub fn default_layout() -> Self {
        let surf_viewer = Box::new(DummyDockable::new(WidgetKind::SurfViewer));
        let mut state = DockState::new(vec![DockWidget {
            index: 0,
            dockable: surf_viewer,
        }]);
        let [_, r] = state.main_surface_mut().split_right(
            egui_dock::NodeIndex::root(),
            0.8,
            vec![DockWidget {
                index: 1,
                dockable: Box::new(DummyDockable::new(WidgetKind::Outliner)),
            }],
        );
        state.main_surface_mut().split_below(
            r,
            0.5,
            vec![DockWidget {
                index: 2,
                dockable: Box::new(DummyDockable::new(WidgetKind::Properties)),
            }],
        );
        Self {
            state,
            to_add: Vec::new(),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        use egui_dock::NodeIndex;

        egui_dock::DockArea::new(&mut self.state)
            .show_add_buttons(true)
            .show_add_popup(true)
            .show(
                ctx,
                &mut DockSpaceView {
                    added: &mut self.to_add,
                },
            );

        self.to_add.drain(..).for_each(|new_tab| {
            // Focus the node that we want to add a tab to.
            self.state
                .main_surface_mut()
                .set_focused_node(NodeIndex(new_tab.parent as usize));
            // Allocate a new index for the tab.
            let index = self.state.main_surface().num_tabs();
            // Add the widget to the dock space.
            let widget: Box<dyn Dockable> = match new_tab.kind {
                WidgetKind::Outliner => Box::new(DummyDockable::new(WidgetKind::Outliner)),
                WidgetKind::SurfViewer => Box::new(DummyDockable::new(WidgetKind::SurfViewer)),
                WidgetKind::Properties => Box::new(DummyDockable::new(WidgetKind::Properties)),
                WidgetKind::Plotting => Box::new(DummyDockable::new(WidgetKind::Plotting)),
                WidgetKind::BsdfViewer => Box::new(DummyDockable::new(WidgetKind::BsdfViewer)),
            };
            self.state.push_to_focused_leaf(DockWidget {
                index: index as u32,
                dockable: widget,
            });
        });
    }

    pub fn add_existing_widget(&mut self, widget: Box<dyn Dockable>) {
        if self
            .state
            .main_surface()
            .tabs()
            .any(|t| t.dockable.uuid() == widget.uuid())
        {
            return;
        }

        self.state.push_to_first_leaf(DockWidget {
            index: self.state.main_surface().num_tabs() as u32,
            dockable: widget,
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetKind {
    SurfViewer,
    BsdfViewer,
    Outliner,
    Properties,
    Plotting,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct NewWidget {
    pub kind: WidgetKind,
    pub parent: u32,
}

/// Utility struct to
pub struct DockSpaceView<'a> {
    pub added: &'a mut Vec<NewWidget>,
}

impl<'a> egui_dock::TabViewer for DockSpaceView<'a> {
    type Tab = DockWidget;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText { tab.dockable.title() }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) { tab.dockable.draw(ui) }

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
                parent: parent.0 as u32,
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
                parent: parent.0 as u32,
            });
        }

        if ui.button("Properties").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::Properties,
                parent: parent.0 as u32,
            });
        }

        if ui.button("Plotting").clicked() {
            self.added.push(NewWidget {
                kind: WidgetKind::Plotting,
                parent: parent.0 as u32,
            });
        }
    }
}
