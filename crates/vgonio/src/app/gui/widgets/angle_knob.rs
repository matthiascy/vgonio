use egui::{emath::Rot2, Color32, Pos2, Response, Sense, Shape, Stroke, Ui, Vec2, Widget};
use std::f32::consts::TAU;
use vgcore::units::{rad, Radians};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AngleKnobWinding {
    Clockwise,
    CounterClockwise,
}

impl AngleKnobWinding {
    pub(crate) fn to_float(self) -> f32 {
        match self {
            AngleKnobWinding::Clockwise => 1.0,
            AngleKnobWinding::CounterClockwise => -1.0,
        }
    }
}

/// The orientation of the angle knob.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub enum AngleKnobOrientation {
    #[default]
    Right,
    Top,
    Left,
    Bottom,
    Custom(f32),
}

impl AngleKnobOrientation {
    /// Returns the 2D rotation matrix for the orientation.
    pub fn rot2(self) -> Rot2 {
        match self {
            AngleKnobOrientation::Right => Rot2::from_angle(0.0),
            AngleKnobOrientation::Top => Rot2::from_angle(std::f32::consts::FRAC_PI_2),
            AngleKnobOrientation::Left => Rot2::from_angle(std::f32::consts::PI),
            AngleKnobOrientation::Bottom => Rot2::from_angle(-std::f32::consts::FRAC_PI_2),
            AngleKnobOrientation::Custom(angle) => Rot2::from_angle(angle),
        }
    }
}

#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AngleKnobShape {
    Circle,
    Square,
}

impl AngleKnobShape {
    pub const RESOLUTION: usize = 32;

    /// Returns the radius factor for the given angle according to the shape.
    pub fn radius_factor(&self, theta: f32) -> f32 {
        match self {
            AngleKnobShape::Circle => 1.0,
            AngleKnobShape::Square => (1.0 / theta.cos().abs()).min(1.0 / theta.sin().abs()),
        }
    }

    pub fn paint(&self, ui: &mut Ui, center: Pos2, radius: f32, stroke: Stroke, rotation: Rot2) {
        let outline_points = (0..Self::RESOLUTION)
            .map(move |i| {
                let angle = (i as f32 / Self::RESOLUTION as f32) * TAU;
                let factor = self.radius_factor(angle - (rotation * Vec2::RIGHT).angle());
                center + Vec2::angled(angle) * radius * factor
            })
            .collect::<Vec<_>>();
        ui.painter().add(Shape::closed_line(outline_points, stroke));
    }
}

pub struct AngleKnob<'a> {
    angle: &'a mut Radians,
    diameter: f32,
    interactive: bool,
    winding: AngleKnobWinding,
    shape: AngleKnobShape,
    orientation: AngleKnobOrientation,
    min: Option<Radians>,
    max: Option<Radians>,
    snap: Option<Radians>,
    show_axes: bool,
    axis_count: u32,
}

impl<'a> AngleKnob<'a> {
    pub fn new(value: &'a mut Radians) -> Self {
        Self {
            angle: value,
            diameter: 32.0,
            interactive: true,
            winding: AngleKnobWinding::Clockwise,
            shape: AngleKnobShape::Circle,
            orientation: AngleKnobOrientation::Right,
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

    pub fn winding(mut self, winding: AngleKnobWinding) -> Self {
        self.winding = winding;
        self
    }

    pub fn min(mut self, min: Option<Radians>) -> Self {
        self.min = min;
        self
    }

    pub fn max(mut self, max: Option<Radians>) -> Self {
        self.max = max;
        self
    }

    pub fn snap(mut self, snap: Option<Radians>) -> Self {
        self.snap = snap;
        self
    }

    #[allow(dead_code)]
    pub fn show_axes(mut self, show_axes: bool) -> Self {
        self.show_axes = show_axes;
        self
    }

    pub fn axis_count(mut self, axis_count: u32) -> Self {
        self.axis_count = axis_count;
        self
    }

    #[allow(dead_code)]
    pub fn orientation(mut self, orientation: AngleKnobOrientation) -> Self {
        self.orientation = orientation;
        self
    }

    #[allow(dead_code)]
    pub fn shape(mut self, shape: AngleKnobShape) -> Self {
        self.shape = shape;
        self
    }
}

impl<'a> Widget for AngleKnob<'a> {
    fn ui(self, ui: &mut Ui) -> Response {
        let desired_size = Vec2::splat(self.diameter);
        let (rect, mut response) = ui.allocate_exact_size(
            desired_size,
            if self.interactive {
                Sense::click_and_drag()
            } else {
                Sense::hover()
            },
        );
        let rot = self.orientation.rot2();

        if response.clicked() || response.dragged() {
            let new_val = (rot.inverse()
                * (response.interact_pointer_pos().unwrap() - rect.center()))
            .angle()
                * self.winding.to_float();
            *self.angle =
                constrain_angle_with_snap_wrap(rad!(new_val), self.snap, self.min, self.max);
            response.mark_changed();
        }

        if ui.is_rect_visible(rect) {
            let visuals = *ui.style().interact(&response);
            let radius = self.diameter * 0.5;
            let angle_to_shape_outline = |angle: f32| {
                rot * Vec2::angled(angle * self.winding.to_float())
                    * radius
                    * self.shape.radius_factor(angle * self.winding.to_float())
            };
            self.shape
                .paint(ui, rect.center(), radius, visuals.fg_stroke, rot);

            // Paint axes
            {
                let paint_axis = |axis_angle| {
                    ui.painter().add(Shape::dashed_line(
                        &[
                            rect.center(),
                            rect.center() + angle_to_shape_outline(axis_angle),
                        ],
                        visuals.fg_stroke,
                        1.0,
                        2.0,
                    ));
                };

                let min = self.min.unwrap_or(rad!(0.0));
                let max = self.max.unwrap_or(rad!(TAU));

                if self.show_axes {
                    for axis in 0..self.axis_count {
                        let angle = axis as f32 * (TAU / (self.axis_count as f32));
                        if (min..=max).contains(&rad!(angle)) {
                            paint_axis(angle);
                        }
                    }
                }
            }

            // Paint stop point
            {
                let paint_stop = |stop_position: f32| {
                    let stop_stroke = {
                        let stop_alpha = 1.0
                            - ((stop_position - self.angle.value()).abs() / (TAU * 0.75))
                                .clamp(0.0, 1.0)
                                .powf(5.0);
                        // TODO: Semantically correct color
                        Stroke::new(
                            visuals.fg_stroke.width,
                            visuals.fg_stroke.color.linear_multiply(stop_alpha),
                        )
                    };

                    ui.painter().line_segment(
                        [
                            rect.center(),
                            rect.center() + angle_to_shape_outline(stop_position),
                        ],
                        stop_stroke,
                    );
                };

                if let Some(min) = self.min {
                    paint_stop(min.value());
                }

                if let Some(max) = self.max {
                    paint_stop(max.value());
                }
            }

            {
                ui.painter().line_segment(
                    [
                        rect.center(),
                        rect.center() + angle_to_shape_outline(self.angle.value()),
                    ],
                    Stroke::new(2.0, Color32::LIGHT_GREEN),
                );

                ui.painter().circle(
                    rect.center(),
                    self.diameter / 24.0,
                    visuals.text_color(),
                    visuals.fg_stroke,
                );

                ui.painter().circle(
                    rect.center() + angle_to_shape_outline(self.angle.value()),
                    self.diameter / 24.0,
                    Color32::LIGHT_GREEN,
                    Stroke::new(2.0, Color32::LIGHT_GREEN),
                );
            }
        }

        response
    }
}

/// Constrains an angle to the range `[min, max]` and snaps it to the nearest
/// multiple of `snap`.
pub(crate) fn constrain_angle_with_snap_wrap(
    angle: Radians,
    snap: Option<Radians>,
    min: Option<Radians>,
    max: Option<Radians>,
) -> Radians {
    let mut val = angle.wrap_to_tau();

    if let Some(snap_angle) = snap {
        debug_assert!(snap_angle > rad!(0.0), "snap angle must be positive");
        val = (val / snap_angle).round() * snap_angle;
    }

    if let Some(min_angle) = min {
        val = val.max(min_angle);
    }

    if let Some(max_angle) = max {
        val = val.min(max_angle);
    }
    val
}
