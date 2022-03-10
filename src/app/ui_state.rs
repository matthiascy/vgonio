// File partially ported from https://github.com/egui_wgpu_backend

use crate::app::utils;
use crate::error::Error;
use copypasta::{ClipboardContext, ClipboardProvider};
use egui::epaint::ClippedShape;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, Device, Texture};
use winit::dpi::PhysicalSize;
use winit::event::{Event, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent};

#[derive(Debug)]
pub struct SizedBuffer {
    raw: wgpu::Buffer,
    size: usize,
}

#[derive(Debug)]
enum BufferType {
    Vertex,
    Index,
    Uniform,
}

/// The size of the window on which the UI is presented.
#[derive(Debug, Copy, Clone)]
pub struct WindowSize {
    /// Inner width of the window, in pixels.
    pub physical_width: u32,
    /// Inner height of the window, in pixels.
    pub physical_height: u32,
    /// HiDPI scale factor.
    pub scale_factor: f32,
}

/// Uniforms uploaded to the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UiUniform {
    pub screen_size: [f32; 2],
}

unsafe impl bytemuck::Pod for UiUniform {}
unsafe impl bytemuck::Zeroable for UiUniform {}

pub struct UiContext {
    raw_context: egui::Context,
    raw_input: egui::RawInput,
    pub window_size: WindowSize,
    clipboard: ClipboardContext,
    cursor_pos: Option<egui::Pos2>,
    cursor_icon: egui::CursorIcon,
    modifiers_state: winit::event::ModifiersState,
}

impl UiContext {
    pub fn new(
        window_size: WindowSize,
        font_config: egui::FontDefinitions,
        style: egui::Style,
    ) -> Self {
        // Create egui context
        let context = egui::Context::default();
        context.set_fonts(font_config);
        context.set_style(style);

        // Create egui raw input
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::default(),
                egui::vec2(window_size.physical_width as f32, window_size.height as f32)
                    / window_size.scale_factor as f32,
            )),
            pixels_per_point: Some(window_size.scale_factor as f32),
            ..Default::default()
        };

        // Create clipboard context
        let clipboard = ClipboardContext::new().expect("Failed to initialise ClipboardContext.");

        Self {
            raw_context: context,
            raw_input: input,
            modifiers_state: winit::event::ModifiersState::empty(),
            cursor_pos: Some(egui::Pos2::default()),
            window_size,
            clipboard,
            cursor_icon: Default::default(),
        }
    }

    pub fn raw_context(&self) -> &egui::Context {
        &self.raw_context
    }

    pub fn process_event<T>(&mut self, winit_event: &winit::event::Event<T>) {
        match winit_event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(PhysicalSize {
                    width: 0,
                    height: 0,
                }) => {}
                WindowEvent::Resized(physical_size) => {
                    self.window_size.physical_width = physical_size.width;
                    self.window_size.height = physical_size.height;

                    self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
                        Default::default(),
                        egui::vec2(
                            self.window_size.physical_width as f32,
                            self.window_size.height as f32,
                        ) / self.window_size.scale_factor as f32,
                    ));
                }
                WindowEvent::ReceivedCharacter(chr) => {
                    if utils::is_character_printable(*chr)
                        && !self.modifiers_state.ctrl()
                        && !self.modifiers_state.logo()
                    {
                        self.raw_input
                            .events
                            .push(egui::Event::Text(chr.to_string()));
                    }
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(virtual_keycode) = input.virtual_keycode {
                        let pressed = input.state == winit::event::ElementState::Pressed;
                        let ctrl = self.modifiers_state.ctrl();

                        match (pressed, ctrl, virtual_keycode) {
                            (true, true, VirtualKeyCode::C) => {
                                self.raw_input.events.push(egui::Event::Copy)
                            }
                            (true, true, VirtualKeyCode::X) => {
                                self.raw_input.events.push(egui::Event::Cut)
                            }
                            (true, true, VirtualKeyCode::V) => {
                                if let Ok(contents) = self.clipboard.get_contents() {
                                    self.raw_input.events.push(egui::Event::Paste(contents))
                                }
                            }
                            _ => {
                                if let Some(key) = utils::winit_to_egui_key_code(virtual_keycode) {
                                    self.raw_input.events.push(egui::Event::Key {
                                        key,
                                        pressed,
                                        modifiers: utils::winit_to_egui_modifiers(
                                            self.modifiers_state,
                                        ),
                                    })
                                }
                            }
                        }
                    }
                }
                WindowEvent::ModifiersChanged(input) => {
                    self.modifiers_state = *input;
                    self.raw_input.modifiers = utils::winit_to_egui_modifiers(*input);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let cursor_pos = egui::pos2(
                        position.x as f32 / self.window_size.scale_factor as f32,
                        position.y as f32 / self.window_size.scale_factor as f32,
                    );
                    self.cursor_pos = Some(cursor_pos);
                    self.raw_input
                        .events
                        .push(egui::Event::PointerMoved(cursor_pos));
                }
                WindowEvent::CursorLeft { .. } => {
                    self.cursor_pos = None;
                    self.raw_input.events.push(egui::Event::PointerGone)
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let mut delta = match delta {
                        MouseScrollDelta::LineDelta(x, y) => {
                            let line_height = 8.0; // TODO
                            egui::vec2(*x, *y) * line_height
                        }
                        MouseScrollDelta::PixelDelta(delta) => {
                            egui::vec2(delta.x as f32, delta.y as f32)
                        }
                    };
                    if cfg!(target_os = "macos") {
                        // See https://github.com/rust-windowing/winit/issues/1695 for more info.
                        delta.x *= -1.0;
                    }

                    // The ctrl (cmd on macos) key indicates a zoom is desired
                    if self.raw_input.modifiers.ctrl || self.raw_input.modifiers.command {
                        self.raw_input
                            .events
                            .push(egui::Event::Zoom((delta.y / 200.0).exp()));
                    } else {
                        self.raw_input.events.push(egui::Event::Scroll(delta));
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if let winit::event::MouseButton::Other(..) = button {
                    } else {
                        if let Some(cursor_pos) = self.cursor_pos {
                            self.raw_input.events.push(egui::Event::PointerButton {
                                pos: cursor_pos,
                                button: match button {
                                    MouseButton::Left => egui::PointerButton::Primary,
                                    MouseButton::Right => egui::PointerButton::Secondary,
                                    MouseButton::Middle => egui::PointerButton::Middle,
                                    MouseButton::Other(_) => unreachable!(),
                                },
                                pressed: *state == winit::event::ElementState::Pressed,
                                modifiers: Default::default(),
                            })
                        }
                    }
                }
                WindowEvent::ScaleFactorChanged {
                    scale_factor,
                    new_inner_size,
                } => {
                    self.window_size.physical_width = new_inner_size.width;
                    self.window_size.height = new_inner_size.height;
                    self.window_size.scale_factor = *scale_factor as f32;

                    self.raw_input.pixels_per_point = Some(self.window_size.scale_factor);
                    self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
                        Default::default(),
                        egui::vec2(new_inner_size.width as f32, new_inner_size.height as f32)
                            / self.window_size.scale_factor as f32,
                    ));
                }
                _ => {}
            },
            _ => {}
        }
    }

    /// Updates the internal time for egui animations.
    pub fn update_time(&mut self, elapsed_secs: f64) {
        self.raw_input.time = Some(elapsed_secs);
    }

    /// Returns `true` if egui should handle the event exclusively.
    pub fn capture_event<T>(&self, event: &Event<T>) -> bool {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::ReceivedCharacter(_)
                | WindowEvent::KeyboardInput { .. }
                | WindowEvent::ModifiersChanged(_) => self.raw_context.wants_keyboard_input(),
                WindowEvent::CursorMoved { .. } => self.raw_context.is_using_pointer(),
                WindowEvent::MouseWheel { .. } | WindowEvent::MouseInput { .. } => {
                    self.raw_context.wants_pointer_input()
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Starts a new frame by providing a new `Ui` instance to write into.
    pub fn begin_frame(&mut self) {
        self.raw_context.begin_frame(self.raw_input.take())
    }

    /// Ends the frame. Returns what has happened as `Output` and gives you
    /// the draw instructions as `PaintJobs`. If the optional `window` is set,
    /// it will set the cursor key based on egui's instructions.
    pub fn end_frame(&mut self, window: Option<&winit::window::Window>) -> egui::FullOutput {
        let output = self.raw_context.end_frame();

        // Handle cursor icon
        if let Some(window) = window {
            if let Some(cursor_icon) =
                utils::egui_to_winit_cursor_icon(output.platform_output.cursor_icon)
            {
                window.set_cursor_visible(true);

                if self.cursor_pos.is_some() {
                    window.set_cursor_icon(cursor_icon);
                }
            } else {
                window.set_cursor_visible(false);
            }
        }

        // Handle clipboard
        if !output.platform_output.copied_text.is_empty() {
            if let Err(err) = self
                .clipboard
                .set_contents(output.platform_output.copied_text.clone())
            {
                eprintln!("Copy/Cut error: {}", err);
            }
        }

        output
    }
}

pub struct UiState {
    /// UI context
    pub context: UiContext,
    /// UI rendering pipeline
    pipeline: wgpu::RenderPipeline,
    /// Index buffers for egui meshes
    index_buffers: Vec<SizedBuffer>,
    /// Vertex buffers for egui meshes
    vertex_buffers: Vec<SizedBuffer>,
    /// Uniform buffer used during rendering
    uniform_buffer: SizedBuffer,
    /// Bind group of uniforms
    uniform_bind_group: wgpu::BindGroup,
    /// Bing group layout of textures (texture view + sampler)
    texture_bind_group_layout: wgpu::BindGroupLayout,
    next_user_texture_id: u64,
    /// The mapping from egui texture IDs to the associated bind group (texture
    /// view + sampler)
    textures: HashMap<egui::TextureId, (Option<wgpu::Texture>, wgpu::BindGroup)>,
}

impl UiState {
    /// Creates related resources used for UI rendering.
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        msaa_samples: u32,
        window_size: WindowSize,
        font_config: egui::FontDefinitions,
        style: egui::Style,
    ) -> Self {
        let ui_context = UiContext::new(window_size, font_config, style);

        let shader_module = {
            device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some("ui-shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("assets/shaders/ui.wgsl").into()),
            })
        };
        let uniform_buffer = {
            let raw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ui-uniforms-buffer"),
                contents: bytemuck::cast_slice(&[UiUniform {
                    screen_size: [0.0, 0.0],
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            SizedBuffer {
                raw,
                size: std::mem::size_of::<UiUniform>(),
            }
        };
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui-uniforms-bind-group-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui-uniforms-bind-group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer.raw,
                    offset: 0,
                    size: None,
                }),
            }],
        });
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui-textures-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui-pipeline-layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: if surface_format.describe().srgb {
                    "vs_main"
                } else {
                    "vs_conv_main"
                },
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 20,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Uint32
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::default(),
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::default(),
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

        Self {
            context: ui_context,
            pipeline,
            index_buffers: Vec::with_capacity(64),
            vertex_buffers: Vec::with_capacity(64),
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            next_user_texture_id: 0,
            textures: HashMap::new(),
        }
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_attachment: &wgpu::TextureView,
        meshes: &[egui::epaint::ClippedMesh],
        clear_color: Option<wgpu::Color>,
    ) -> Result<(), Error> {
        let load_op = if let Some(color) = clear_color {
            wgpu::LoadOp::Clear(color)
        } else {
            wgpu::LoadOp::Load
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui-render-pass"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: &color_attachment,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_op,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        // Starts recording commands.
        render_pass.push_debug_group("ui-pass");
        self.render_with_existing_render_pass(&mut render_pass, meshes)?;
        render_pass.pop_debug_group();

        Ok(())
    }

    pub fn render_with_existing_render_pass<'renderpass>(
        &'renderpass self,
        render_pass: &mut wgpu::RenderPass<'renderpass>,
        meshes: &[egui::epaint::ClippedMesh],
    ) -> Result<(), Error> {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

        let WindowSize {
            physical_width: win_width,
            physical_height: win_height,
            scale_factor: win_scale_factor,
        } = self.context.window_size;

        for ((egui::ClippedMesh(rect, mesh), vertex_buffer), index_buffer) in meshes
            .iter()
            .zip(self.vertex_buffers.iter())
            .zip(self.index_buffers.iter())
        {
            let (scissor_x, scissor_y, scissor_width, scissor_height) = {
                // Transform clip rect to physical pixels position.
                let (clip_min_x, clip_min_y, clip_max_x, clip_max_y) = {
                    let clip_min_x = (win_scale_factor * rect.min.x).clamp(0.0, win_width as f32);
                    let clip_min_y = (win_scale_factor * rect.min.y).clamp(0.0, win_height as f32);
                    let clip_max_x =
                        (win_scale_factor * rect.max.x).clamp(clip_min_x, win_width as f32);
                    let clip_max_y =
                        (win_scale_factor * rect.max.y).clamp(clip_min_y, win_height as f32);
                    (
                        clip_min_x.round() as u32,
                        clip_min_y.round() as u32,
                        clip_max_x.round() as u32,
                        clip_max_y.round() as u32,
                    )
                };
                let width = (clip_max_x - clip_min_x).max(1);
                let height = (clip_max_y - clip_min_y).max(1);

                // Clip scissor rectangle to target size.
                let x = clip_min_x.min(win_width);
                let y = clip_min_y.min(win_height);
                let width = width.min(win_width - x);
                let height = height.min(win_height - y);

                (x, y, width, height)
            };

            // Skip rendering with zero-sized clip areas.
            if scissor_width == 0 || scissor_height == 0 {
                continue;
            }

            render_pass.set_scissor_rect(scissor_x, scissor_y, scissor_width, scissor_height);
            let texture_bind_group = self.texture_bind_group(mesh.texture_id)?;
            render_pass.set_bind_group(1, texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.raw.slice(..));
            render_pass.set_index_buffer(index_buffer.raw.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }

        Ok(())
    }

    fn texture_bind_group(&self, texture_id: egui::TextureId) -> Result<&wgpu::BindGroup, Error> {
        self.textures
            .get(&texture_id)
            .ok_or_else(|| Error::Any(format!("[UI] Texture {:?} used but not found", texture_id)))
            .map(|x| &x.1)
    }

    /// Updates the textures used by egui for the fonts etc.
    ///
    /// WARNING: this function should be called before `render()`.
    pub fn update_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &egui::TexturesDelta,
    ) -> Result<(), Error> {
        for (texture_id, image_delta) in textures.set.iter() {
            let image_size = image_delta.image.size();
            let origin = match image_delta.pos {
                Some([x, y]) => wgpu::Origin3d {
                    x: x as u32,
                    y: y as u32,
                    z: 0,
                },
                None => wgpu::Origin3d::ZERO,
            };

            let alpha_srgb_pixels: Option<Vec<_>> = match &image_delta.image {
                egui::ImageData::Color(_) => None,
                egui::ImageData::Alpha(a) => Some(a.srgba_pixels(1.0).collect()),
            };

            let image_data: &[u8] = match &image_delta.image {
                egui::ImageData::Color(c) => bytemuck::cast_slice(c.pixels.as_slice()),
                egui::ImageData::Alpha(_) => bytemuck::cast_slice(
                    alpha_srgb_pixels
                        .as_ref()
                        .expect("[UI] Alpha texture should have been converted already!")
                        .as_slice(),
                ),
            };

            let image_extent = wgpu::Extent3d {
                width: image_size[0] as u32,
                height: image_size[1] as u32,
                depth_or_array_layers: 1,
            };

            let image_data_layout = wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * image_extent.width),
                rows_per_image: None,
            };

            let label_base = match texture_id {
                egui::TextureId::Managed(m) => {
                    format!("egui_image_{}", m)
                }
                egui::TextureId::User(u) => {
                    format!("egui_user_image_{}", u)
                }
            };

            match self.textures.entry(texture_id.clone()) {
                Entry::Occupied(mut o) => match image_delta.pos {
                    None => {
                        let (texture, bind_group) = create_texture_and_bind_group(
                            device,
                            queue,
                            &label_base,
                            origin,
                            image_data,
                            image_data_layout,
                            image_extent,
                            &self.texture_bind_group_layout,
                        );
                        let (texture, _) = o.insert((Some(texture), bind_group));
                        if let Some(texture) = texture {
                            texture.destroy();
                        }
                    }
                    Some(_) => {
                        if let Some(texture) = o.get().0.as_ref() {
                            queue.write_texture(
                                wgpu::ImageCopyTexture {
                                    texture,
                                    mip_level: 0,
                                    origin,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                image_data,
                                image_data_layout,
                                image_extent,
                            );
                        } else {
                            return Err(Error::Any(format!(
                                "Invalid texture ID, update of unmanaged texture {:?}",
                                texture_id
                            )));
                        }
                    }
                },
                Entry::Vacant(v) => {
                    let (texture, bind_group) = create_texture_and_bind_group(
                        device,
                        queue,
                        &label_base,
                        origin,
                        image_data,
                        image_data_layout,
                        image_extent,
                        &self.texture_bind_group_layout,
                    );
                    v.insert((Some(texture), bind_group));
                }
            }
        }

        Ok(())
    }

    /// Remove the textures egui no longer needs.
    ///
    /// WARNING: This function should be called after `render()`.
    pub fn clean_unused_textures(&mut self, textures: egui::TexturesDelta) -> Result<(), Error> {
        for texture_id in textures.free {
            let (texture, _) = self.textures.remove(&texture_id).ok_or_else(|| {
                // This can happen due to a bug in egui, or if the user doesn't call
                // `update_textures` when required.
                Error::Any(format!(
                    "Attempted to remove an unknown texture {:?}",
                    texture_id
                ))
            })?;

            if let Some(texture) = texture {
                texture.destroy();
            }
        }
        Ok(())
    }

    pub fn update_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[egui::epaint::ClippedMesh],
    ) {
        let index_buffer_count = self.index_buffers.len();
        let vertex_buffer_count = self.vertex_buffers.len();
        let WindowSize {
            physical_width: width,
            physical_height: height,
            ..
        } = self.context.window_size;
        // Update uniform buffer
        self.update_buffer(
            device,
            queue,
            BufferType::Uniform,
            0,
            bytemuck::cast_slice(&[UiUniform {
                screen_size: [width as f32, height as f32],
            }]),
        );

        // Update vertex and index buffers
        for (i, egui::ClippedMesh(_, mesh)) in meshes.iter().enumerate() {
            {
                let data: &[u8] = bytemuck::cast_slice(&mesh.vertices);
                if i < vertex_buffer_count {
                    self.update_buffer(device, queue, BufferType::Vertex, i, data)
                } else {
                    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("ui-vertex-buffer"),
                        contents: data,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
                    self.vertex_buffers.push(SizedBuffer {
                        raw: buffer,
                        size: data.len(),
                    });
                }
            }

            {
                let data: &[u8] = bytemuck::cast_slice(&mesh.indices);
                if i < index_buffer_count {
                    self.update_buffer(device, queue, BufferType::Index, i, data)
                } else {
                    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("ui-index-buffer"),
                        contents: data,
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    });
                    self.vertex_buffers.push(SizedBuffer {
                        raw: buffer,
                        size: data.len(),
                    });
                }
            }
        }
    }

    fn update_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer_type: BufferType,
        index: usize,
        data: &[u8],
    ) {
        let (buffer, usage, name) = match buffer_type {
            BufferType::Vertex => (
                &mut self.vertex_buffers[index],
                wgpu::BufferUsages::VERTEX,
                "vertex",
            ),
            BufferType::Index => (
                &mut self.index_buffers[index],
                wgpu::BufferUsages::INDEX,
                "index",
            ),
            BufferType::Uniform => (
                &mut self.uniform_buffer,
                wgpu::BufferUsages::UNIFORM,
                "uniform",
            ),
        };

        if data.len() > buffer.size {
            buffer.size = data.len();
            buffer.raw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("ui-{}-buffer", name).as_str()),
                contents: bytemuck::cast_slice(data),
                usage: usage | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&buffer.raw, 0, data);
        }
    }

    // pub fn begin_frame(&mut self) -> epi::Frame {
    //     self.context.begin_frame();
    //     epi::Frame::new(epi::backend::FrameData {
    //         info: epi::IntegrationInfo {
    //             name: "vgonio-gui",
    //             web_info: None,
    //             prefer_dark_mode: None,
    //             cpu_usage: self.prev_frame_time,
    //             native_pixels_per_point: Some(window.scale_factor() as _),
    //         },
    //         output: Default::default(),
    //         repaint_signal: Arc::new(()),
    //     })
    // }
}

/// Create a texture and bind group from image data.
fn create_texture_and_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label_base: &str,
    origin: wgpu::Origin3d,
    image_data: &[u8],
    image_data_layout: wgpu::ImageDataLayout,
    image_extent: wgpu::Extent3d,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(format!("{}_texture", label_base).as_str()),
        size: image_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        image_data,
        image_data_layout,
        image_extent,
    );

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(format!("{}_sampler", label_base).as_str()),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(format!("{}_texture_bind_group", label_base).as_str()),
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    (texture, bind_group)
}
