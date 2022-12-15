mod analysis;
mod context;
mod gizmo;
mod plotter;
mod simulation;
pub mod state;
mod tools;
mod ui;
mod widgets;

use crate::{
    acq::{Ray, RtcMethod},
    error::Error,
};
pub use context::*;
use glam::IVec2;
pub(crate) use tools::{trace_ray_grid_dbg, trace_ray_standard_dbg, VisualDebugger};
pub use ui::VgonioGui;

use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::WindowBuilder,
};

/// Initial window width.
const WIN_INITIAL_WIDTH: u32 = 1600;
/// Initial window height.
const WIN_INITIAL_HEIGHT: u32 = 900;

/// Events used by Vgonio application.
#[derive(Debug)]
pub enum VgonioEvent {
    Quit,
    RequestRedraw,
    OpenFile(std::path::PathBuf),
    ToggleGrid,
    UpdateSurfaceScaleFactor(f32),
    UpdateDepthMap,
    TraceRayDbg {
        ray: Ray,
        max_bounces: u32,
        method: RtcMethod,
    },
    ToggleDebugDrawing,
    ToggleSurfaceVisibility,
    UpdateDebugT(f32),
    UpdatePrimId(u32),
    UpdateCellPos(IVec2),
}

use super::Config;

/// Launches Vgonio GUI application.
pub fn launch(config: Config) -> Result<(), Error> {
    use state::VgonioGuiState;

    let event_loop = EventLoopBuilder::<VgonioEvent>::with_user_event().build();
    let window = WindowBuilder::new()
        .with_decorations(true)
        .with_resizable(true)
        .with_transparent(false)
        .with_inner_size(PhysicalSize {
            width: WIN_INITIAL_WIDTH,
            height: WIN_INITIAL_HEIGHT,
        })
        .with_title("vgonio")
        .build(&event_loop)
        .unwrap();

    let mut vgonio = pollster::block_on(VgonioGuiState::new(config, &window, &event_loop))?;

    let mut last_frame_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let now = std::time::Instant::now();
        let dt = now - last_frame_time;
        last_frame_time = now;

        match event {
            Event::UserEvent(VgonioEvent::Quit) => {
                *control_flow = ControlFlow::Exit;
            }
            Event::UserEvent(event) => vgonio.handle_user_event(event),
            Event::WindowEvent {
                window_id,
                ref event,
            } if window_id == window.id() => {
                if !vgonio.handle_input(event) {
                    match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

                        WindowEvent::Resized(new_size) => vgonio.resize(*new_size),

                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            vgonio.resize(**new_inner_size);
                        }

                        _ => {}
                    }
                }
            }

            Event::RedrawRequested(window_id) if window_id == window.id() => {
                vgonio.update(dt);
                match vgonio.render(&window) {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => vgonio.resize(PhysicalSize {
                        width: vgonio.surface_width(),
                        height: vgonio.surface_height(),
                    }),
                    // The system is out of memory, we should quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }

            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually request it.
                window.request_redraw()
            }

            _ => {}
        }
    })
}
