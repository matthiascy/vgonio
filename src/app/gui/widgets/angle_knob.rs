// TODO: angle knob widget implementation

use egui::{Color32, Pos2, Response, Sense, Shape, Stroke, Ui, Vec2, Widget};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Winding {
    Clockwise,
    CounterClockwise,
}

impl Winding {
    pub(crate) fn to_float(self) -> f32 {
        match self {
            Winding::Clockwise => 1.0,
            Winding::CounterClockwise => -1.0,
        }
    }
}

pub struct AngleKnob<'a> {
    value: &'a mut f32,
    diameter: f32,
    interactive: bool,
    winding: Winding,
    min: Option<f32>,
    max: Option<f32>,
    snap: Option<f32>,
    show_axes: bool,
    axis_count: u32,
}

impl<'a> AngleKnob<'a> {
    pub const RESOLUTION: usize = 32;

    pub fn new(value: &'a mut f32) -> Self {
        Self {
            value,
            diameter: 32.0,
            interactive: true,
            winding: Winding::Clockwise,
            min: None,
            max: None,
            snap: None,
            show_axes: true,
            axis_count: 4,
        }
    }

    pub fn interactive(mut self, interactive: bool) -> Self {
        self.interactive = interactive;
        self
    }

    pub fn diameter(mut self, diameter: f32) -> Self {
        self.diameter = diameter;
        self
    }

    pub fn winding(mut self, winding: Winding) -> Self {
        self.winding = winding;
        self
    }

    pub fn min(mut self, min: Option<f32>) -> Self {
        self.min = min;
        self
    }

    pub fn max(mut self, max: Option<f32>) -> Self {
        self.max = max;
        self
    }

    pub fn snap(mut self, snap: Option<f32>) -> Self {
        self.snap = snap;
        self
    }

    pub fn show_axes(mut self, show_axes: bool) -> Self {
        self.show_axes = show_axes;
        self
    }

    pub fn axis_count(mut self, axis_count: u32) -> Self {
        self.axis_count = axis_count;
        self
    }
}
