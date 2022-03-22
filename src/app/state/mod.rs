mod camera;
mod input;

pub use input::InputState;

use crate::app::gui::{GuiContext, RepaintSignal, UserEvent, VgonioGui, WindowSize};
use crate::gfx::{
    GpuContext, MeshView, RdrPass, ShadowPass, Texture, Vertex, VertexLayout,
    DEFAULT_BIND_GROUP_LAYOUT_DESC,
};
use camera::CameraState;

use crate::error::Error;
use crate::gfx::camera::{Camera, Projection, ProjectionKind};
use crate::htfld::Heightfield;
use crate::math::IDENTITY_MAT4;
use epi::App;
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::default::Default;
use std::io::{BufWriter, Write};
use std::num::NonZeroU32;
use std::time::Instant;
use wgpu::util::DeviceExt;
use wgpu::{VertexFormat, VertexStepMode};
use winit::event::{KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::EventLoopProxy;
use winit::window::Window;

const NUM_INSTANCES_PER_ROW: u32 = 9;
const NUM_INSTANCES_PER_COL: u32 = 9;

// TODO: fix blending.

pub struct InstancingDemo {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_vertices: u32,
    pub num_indices: u32,
    pub current_texture_index: usize,
    pub pass: RdrPass,
    pub object_model_matrix: glam::Mat4,
    pub instancing_buffer: wgpu::Buffer,
    pub instancing_transforms: Vec<Mat4>,
}

pub struct VgonioApp {
    gpu_ctx: GpuContext,
    gui_ctx: GuiContext,
    gui: VgonioGui,
    input: InputState,
    camera: CameraState,
    passes: HashMap<&'static str, RdrPass>,
    depth_attachment: Texture,
    depth_attachment_storage: wgpu::Buffer, // used to store depth attachment
    depth_attachment_image: image::GrayImage,

    shadow_pass: ShadowPass,

    instancing_demo: InstancingDemo,

    heightfield: Option<Box<Heightfield>>,
    heightfield_mesh_view: Option<MeshView>,

    pub start_time: Instant,
    pub prev_frame_time: Option<f32>,

    pub demo_ui: egui_demo_lib::WrapApp,
    pub is_grid_enabled: bool,
}

impl VgonioApp {
    // TODO: broadcast errors; replace unwraps
    pub async fn new(
        window: &Window,
        event_loop: EventLoopProxy<UserEvent>,
    ) -> Result<Self, Error> {
        let gpu_ctx = GpuContext::new(window).await;

        let depth_attachment = Texture::create_depth_texture(
            &gpu_ctx.device,
            gpu_ctx.surface_config.width,
            gpu_ctx.surface_config.height,
            Some("depth-texture"),
        );

        let depth_attachment_storage_size = (std::mem::size_of::<f32>()
            * (gpu_ctx.surface_config.width * gpu_ctx.surface_config.height) as usize)
            as wgpu::BufferAddress;
        let depth_attachment_storage = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: depth_attachment_storage_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let depth_attachment_image =
            image::GrayImage::new(gpu_ctx.surface_config.width, gpu_ctx.surface_config.height);

        // Camera
        let camera = {
            let camera = Camera::new(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
            let projection = Projection::new(
                0.1,
                100.0,
                60.0f32.to_radians(),
                gpu_ctx.surface_config.width,
                gpu_ctx.surface_config.height,
            );
            CameraState::new(camera, projection, ProjectionKind::Perspective)
        };

        let heightfield_pass = create_height_field_pass(&gpu_ctx);
        let visual_grid_pass = create_visual_grid_pass(&gpu_ctx);

        let mut passes = HashMap::new();
        passes.insert("visual_grid", visual_grid_pass);
        passes.insert("heightfield", heightfield_pass);

        let depth_pass = ShadowPass::new(
            &gpu_ctx,
            gpu_ctx.surface_config.width,
            gpu_ctx.surface_config.height,
            true,
        );

        let instancing_demo = create_instancing_demo(&gpu_ctx);

        let gui_ctx = GuiContext::new(window, &gpu_ctx.device, gpu_ctx.surface_config.format, 1);
        let ui = egui_demo_lib::WrapApp::default();
        let gui = VgonioGui::new(event_loop);

        let input = InputState {
            key_map: Default::default(),
            mouse_map: Default::default(),
            scroll_delta: 0.0,
            cursor_delta: [0.0, 0.0],
            cursor_pos: [0.0, 0.0],
        };

        Ok(Self {
            gpu_ctx,
            gui_ctx,
            gui,
            input,
            passes,
            depth_attachment,
            depth_attachment_storage,
            depth_attachment_image,
            shadow_pass: depth_pass,
            instancing_demo,
            heightfield: None,
            heightfield_mesh_view: None,
            camera,
            start_time: Instant::now(),
            prev_frame_time: None,
            demo_ui: ui,
            is_grid_enabled: true,
        })
    }

    #[inline]
    pub fn surface_width(&self) -> u32 {
        self.gpu_ctx.surface_config.width
    }

    #[inline]
    pub fn surface_height(&self) -> u32 {
        self.gpu_ctx.surface_config.height
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.gpu_ctx.surface_config.width = new_size.width;
            self.gpu_ctx.surface_config.height = new_size.height;
            self.gpu_ctx
                .surface
                .configure(&self.gpu_ctx.device, &self.gpu_ctx.surface_config);
            self.depth_attachment = Texture::create_depth_texture(
                &self.gpu_ctx.device,
                self.gpu_ctx.surface_config.width,
                self.gpu_ctx.surface_config.height,
                Some("depth_texture"),
            );

            let depth_map_buffer_size = (std::mem::size_of::<f32>()
                * (self.gpu_ctx.surface_config.width * self.gpu_ctx.surface_config.height) as usize)
                as wgpu::BufferAddress;
            let depth_map_buffer_desc = wgpu::BufferDescriptor {
                label: None,
                size: depth_map_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            };
            self.depth_attachment_storage =
                self.gpu_ctx.device.create_buffer(&depth_map_buffer_desc);

            self.camera
                .projection
                .resize(new_size.width, new_size.height);

            self.shadow_pass
                .resize(&self.gpu_ctx.device, new_size.width, new_size.height);
        }
    }

    pub fn handle_input(&mut self, event: &WindowEvent) -> bool {
        if !self.gui_ctx.handle_event(event) {
            match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                } => {
                    self.input.update_key_map(*keycode, *state);
                    if self.input.is_key_pressed(VirtualKeyCode::Space) {
                        self.instancing_demo.current_texture_index += 1;
                        self.instancing_demo.current_texture_index %= 2;
                    }
                    if self.input.is_key_pressed(VirtualKeyCode::C) {
                        println!("C pressed");
                        self.save_depth_map();
                    }
                    if self.input.is_key_pressed(VirtualKeyCode::B) {
                        println!("L pressed");
                        if let Some(mesh) = &self.heightfield_mesh_view {
                            println!("Bake");
                            self.shadow_pass.bake(
                                &self.gpu_ctx.device,
                                &self.gpu_ctx.queue,
                                &mesh.vertex_buffer,
                                &mesh.index_buffer,
                                mesh.indices_count,
                                mesh.index_format,
                            );
                        }
                    }
                    if self.input.is_key_pressed(VirtualKeyCode::S) {
                        println!("save depth");
                        self.shadow_pass
                            .save_to_image(
                                &self.gpu_ctx.device,
                                0.1,
                                100.0,
                                "depth_pass_depth.png".as_ref(),
                            )
                            .unwrap();
                    }
                    if self.input.is_key_pressed(VirtualKeyCode::K) {
                        println!("measure");
                        let now = std::time::Instant::now();
                        self.measure_microfacet_geometric_term();
                        println!(
                            "elapsed time: {}",
                            (std::time::Instant::now() - now).as_millis()
                        );
                    }
                    true
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    self.input.update_scroll_delta(*delta);
                    true
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    self.input.update_mouse_map(*button, *state);
                    true
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.input.update_cursor_delta((*position).cast::<f32>());
                    true
                }
                _ => false,
            }
        } else {
            true
        }
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        // Update camera uniform.
        self.camera
            .update(&self.input, dt, ProjectionKind::Perspective);
        self.gui.update_gizmo_matrices(
            Mat4::IDENTITY,
            Mat4::look_at_rh(self.camera.camera.eye, Vec3::ZERO, self.camera.camera.up),
            Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0),
        );

        if self.input.is_key_pressed(VirtualKeyCode::Space) {
            self.instancing_demo.current_texture_index += 1;
            self.instancing_demo.current_texture_index %= 2;
        }

        let (view, proj) = (
            self.camera.uniform.view_matrix,
            self.camera.uniform.proj_matrix,
        );

        self.gpu_ctx.queue.write_buffer(
            self.instancing_demo.pass.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[self.instancing_demo.object_model_matrix, view, proj]),
        );

        self.gpu_ctx.queue.write_buffer(
            self.passes
                .get("visual_grid")
                .unwrap()
                .uniform_buffer
                .as_ref()
                .unwrap(),
            0,
            bytemuck::cast_slice(&[
                view,
                proj,
                self.camera.uniform.view_inv_matrix,
                self.camera.uniform.proj_inv_matrix,
            ]),
        );

        self.shadow_pass
            .update_uniforms(&self.gpu_ctx.queue, Mat4::IDENTITY, view, proj);

        if let Some(hf) = &self.heightfield {
            let mut uniform = [0.0f32; 16 * 3 + 4];
            uniform[0..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
            uniform[16..32].copy_from_slice(&self.camera.uniform.view_matrix.to_cols_array());
            uniform[32..48].copy_from_slice(&self.camera.uniform.proj_matrix.to_cols_array());
            uniform[48..52].copy_from_slice(&[hf.min, hf.max, hf.max - hf.min, 0.5]);
            self.gpu_ctx.queue.write_buffer(
                self.passes
                    .get("heightfield")
                    .unwrap()
                    .uniform_buffer
                    .as_ref()
                    .unwrap(),
                0,
                bytemuck::cast_slice(&uniform),
            );
        }

        for m in &mut self.instancing_demo.instancing_transforms {
            *m *= Mat4::from_rotation_x(0.2f32.to_radians());
        }

        self.gpu_ctx.queue.write_buffer(
            &self.instancing_demo.instancing_buffer,
            0,
            bytemuck::cast_slice(&self.instancing_demo.instancing_transforms),
        );

        // Reset mouse movement
        self.input.scroll_delta = 0.0;
        self.input.cursor_delta = [0.0, 0.0];
    }

    pub fn render(
        &mut self,
        window: &winit::window::Window,
        repaint_signal: std::sync::Arc<RepaintSignal>,
    ) -> Result<(), wgpu::SurfaceError> {
        let output_frame = self.gpu_ctx.surface.get_current_texture()?;
        let output_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoders = [
            self.gpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_render_encoder"),
                }),
            self.gpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vgonio_ui_render_encoder"),
                }),
        ];

        if self.gui.current_workspace_name() == "Simulation" {
            let mut render_pass = encoders[0].begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: &output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.046,
                                g: 0.046,
                                b: 0.046,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    },
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_attachment.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            {
                render_pass.set_pipeline(&self.instancing_demo.pass.pipeline);
                render_pass.set_bind_group(
                    0,
                    &self.instancing_demo.pass.bind_groups
                        [self.instancing_demo.current_texture_index + 1],
                    &[],
                );
                render_pass.set_bind_group(1, &self.instancing_demo.pass.bind_groups[0], &[]);
                render_pass.set_vertex_buffer(0, self.instancing_demo.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, self.instancing_demo.instancing_buffer.slice(..));
                render_pass.set_index_buffer(
                    self.instancing_demo.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                render_pass.draw_indexed(
                    0..self.instancing_demo.num_indices,
                    0,
                    0..self.instancing_demo.instancing_transforms.len() as _,
                );
            }

            if self.is_grid_enabled {
                let pass = self.passes.get("visual_grid").unwrap();
                render_pass.set_pipeline(&pass.pipeline);
                render_pass.set_bind_group(0, &pass.bind_groups[0], &[]);
                render_pass.draw(0..6, 0..1);
            }

            if let Some(mesh) = &self.heightfield_mesh_view {
                let pass = self.passes.get("heightfield").unwrap();
                render_pass.set_pipeline(&pass.pipeline);
                // render_pass.set_pipeline(self.shadow_pass.pipeline());
                // render_pass.set_bind_group(0, &self.shadow_pass.bind_groups()[0], &[]);
                render_pass.set_bind_group(0, &pass.bind_groups[0], &[]);
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);
                render_pass.draw_indexed(0..mesh.indices_count, 0, 0..1);
            }
        }

        // {
        //     if let Some(mesh) = &self.heightfield_mesh_view {
        //         let mut pass =
        // encoders[0].begin_render_pass(&wgpu::RenderPassDescriptor {
        //             label: None,
        //             color_attachments: &[
        //                 wgpu::RenderPassColorAttachment {
        //                     view: &self.shadow_pass.color_attachment.view,
        //                     resolve_target: None,
        //                     ops: wgpu::Operations {
        //                         load: wgpu::LoadOp::Clear(wgpu::Color {
        //                             r: 0.1,
        //                             g: 0.2,
        //                             b: 0.3,
        //                             a: 1.0
        //                         }),
        //                         store: true
        //                     }
        //                 }
        //             ],
        //             depth_stencil_attachment:
        // Some(wgpu::RenderPassDepthStencilAttachment {                 view:
        // &self.shadow_pass.depth_attachment.view,                 depth_ops:
        // Some(wgpu::Operations {                     load:
        // wgpu::LoadOp::Clear(1.0),                     store: true
        //                 }),
        //                 stencil_ops: None
        //             })
        //         });
        //         pass.set_pipeline(self.shadow_pass.pipeline());
        //         pass.set_bind_group(0, &self.shadow_pass.bind_groups()[0], &[]);
        //         pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        //         pass.set_index_buffer(mesh.index_buffer.slice(..),
        // mesh.index_format);         pass.draw_indexed(0..mesh.indices_count,
        // 0, 0..1);     }
        // }

        {
            // Record UI
            let ui_start_time = Instant::now();
            let input = self.gui_ctx.egui_state_mut().take_egui_input(window);

            self.gui_ctx.egui_context_mut().begin_frame(input);
            let egui_output = epi::backend::AppOutput::default();
            let ui_frame = epi::Frame::new(epi::backend::FrameData {
                info: epi::IntegrationInfo {
                    name: "vgonio-gui",
                    web_info: None,
                    prefer_dark_mode: None,
                    cpu_usage: self.prev_frame_time,
                    native_pixels_per_point: Some(window.scale_factor() as f32),
                },
                output: egui_output,
                repaint_signal,
            });
            // self.demo_ui.update(self.gui_ctx.egui_context(), &ui_frame);
            self.gui.update(self.gui_ctx.egui_context(), &ui_frame);
            let ui_output_frame = self.gui_ctx.egui_context().end_frame();

            let meshes = self
                .gui_ctx
                .egui_context()
                .tessellate(ui_output_frame.shapes);
            let frame_time = (Instant::now() - ui_start_time).as_secs_f64() as f32;
            self.prev_frame_time = Some(frame_time);

            let win_size = WindowSize {
                physical_width: self.gpu_ctx.surface_config.width,
                physical_height: self.gpu_ctx.surface_config.height,
                scale_factor: window.scale_factor() as f32,
            };

            self.gui_ctx
                .update_textures(
                    &self.gpu_ctx.device,
                    &self.gpu_ctx.queue,
                    ui_output_frame.textures_delta,
                )
                .unwrap();
            self.gui_ctx.update_buffers(
                &self.gpu_ctx.device,
                &self.gpu_ctx.queue,
                &meshes,
                &win_size,
            );
            self.gui_ctx
                .render(&mut encoders[1], &output_view, &meshes, &win_size, None)
                .unwrap();
        }

        self.gpu_ctx.queue.submit(encoders.map(|enc| enc.finish()));

        output_frame.present();

        Ok(())
    }

    pub fn handle_user_event(&mut self, event: UserEvent) {
        match event {
            UserEvent::RequestRedraw => {}
            UserEvent::OpenFile(path) => self.load_height_field(path.as_path()),
            UserEvent::ToggleGrid => {
                self.is_grid_enabled = !self.is_grid_enabled;
            }
        }
    }

    fn load_height_field(&mut self, path: &std::path::Path) {
        match Heightfield::read_from_file(path, None, None) {
            Ok(hf) => {
                let mut hf = hf;
                hf.fill_holes();
                let mesh_view = MeshView::from_height_field(&self.gpu_ctx.device, &hf);
                self.heightfield = Some(Box::new(hf));
                self.heightfield_mesh_view = Some(mesh_view);
            }
            Err(err) => {
                log::error!("HeightField loading error: {}", err);
            }
        }
    }

    fn save_depth_map(&mut self) {
        log::info!("Saving depth map...");
        let mut encoder = self
            .gpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.depth_attachment.raw,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.depth_attachment_storage,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(
                        std::mem::size_of::<f32>() as u32 * self.gpu_ctx.surface_config.width,
                    ),
                    rows_per_image: NonZeroU32::new(self.gpu_ctx.surface_config.height),
                },
            },
            wgpu::Extent3d {
                width: self.gpu_ctx.surface_config.width,
                height: self.gpu_ctx.surface_config.height,
                depth_or_array_layers: 1,
            },
        );
        self.gpu_ctx.queue.submit(Some(encoder.finish()));

        {
            let buffer_slice = self.depth_attachment_storage.slice(..);

            let mapping = buffer_slice.map_async(wgpu::MapMode::Read);
            self.gpu_ctx.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(async {
                mapping.await.unwrap()
            });

            let buffer_view_f32 = buffer_slice.get_mapped_range();
            let data_u8 = unsafe {
                let (_, data, _) = buffer_view_f32.align_to::<f32>();
                data.iter()
                    .map(|d| (remap_depth(*d, 0.1, 100.0) * 255.0) as u8)
                    .collect::<Vec<u8>>()
            };

            self.depth_attachment_image.copy_from_slice(&data_u8);
            self.depth_attachment_image.save("depth_map.png").unwrap();
        }
        self.depth_attachment_storage.unmap();
    }

    fn measure_microfacet_geometric_term(&mut self) {
        if let Some(mesh) = &self.heightfield_mesh_view {
            let radius = (mesh.extent.max - mesh.extent.min).max_element();
            let near = 0.1f32;
            let far = radius * 2.0;

            let proj = {
                let projection = Projection::new(
                    near,
                    far,
                    70.0f32.to_radians(),
                    (radius * 1.414) as u32,
                    (radius * 1.414) as u32,
                );
                projection.matrix(ProjectionKind::Orthographic)
            };

            let mut results = [0.0f32; 360 * 91];

            for i in (0..360).step_by(1) {
                for j in (0..91).step_by(1) {
                    let phi = (i as f32).to_radians(); // azimuth
                    let theta = (j as f32).to_radians(); // inclination
                    let (sin_theta, cos_theta) = theta.sin_cos();
                    let (sin_phi, cos_phi) = phi.sin_cos();
                    let pos = Vec3::new(
                        radius * sin_theta * cos_phi,
                        radius * cos_theta,
                        radius * sin_theta * sin_phi,
                    );
                    let camera = Camera::new(pos, Vec3::ZERO, Vec3::Y);
                    self.shadow_pass.resize(&self.gpu_ctx.device, 512, 512);
                    self.shadow_pass.update_uniforms(
                        &self.gpu_ctx.queue,
                        Mat4::IDENTITY,
                        camera.matrix(),
                        proj,
                    );
                    self.shadow_pass.bake(
                        &self.gpu_ctx.device,
                        &self.gpu_ctx.queue,
                        &mesh.vertex_buffer,
                        &mesh.index_buffer,
                        mesh.indices_count,
                        mesh.index_format,
                    );
                    // self.shadow_pass
                    //     .save_to_image(
                    //         &self.gpu_ctx.device,
                    //         near,
                    //         far,
                    //         format!("shadow_pass_{i}_{j}.png").as_ref(),
                    //     )
                    //     .unwrap();
                    results[i * 91 + j] = self.shadow_pass.compute_pixels_count(&self.gpu_ctx.device) as _;
                }
            }
            let file = std::fs::File::create("measured_geometric_term.txt").unwrap();
            let writer = &mut BufWriter::new(file);
            for i in results.iter() {
                writer.write_all(format!("{} ", i).as_bytes()).unwrap();
            }
        }
    }
}

fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    (2.0 * near * far) / (far + near - depth * (far - near))
}

pub fn remap_depth(depth: f32, near: f32, far: f32) -> f32 {
    linearize_depth(depth, near, far) / (far - near)
}

fn create_height_field_pass(ctx: &GpuContext) -> RdrPass {
    // Load shader
    let shader_module = ctx
        .device
        .create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("height_field_shader_module"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/wgsl/height_field.wgsl").into(),
            ),
        });

    // Create uniform buffer for rendering height field
    let uniform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniform-buffer"),
            contents: bytemuck::cast_slice(&[0.0f32; 16 * 3 + 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("height_field_bind_group_layout"),
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

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("height_field_bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    // Create height field render pipeline
    let render_pipeline_layout =
        ctx.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("height_field_render_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("height_field_render_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Line,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, /* tells when to discard a
                                                             * new pixel */
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: ctx.surface_config.format,
                    // blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

    RdrPass {
        pipeline,
        bind_groups: vec![bind_group],
        uniform_buffer: Some(uniform_buffer),
    }
}

fn create_visual_grid_pass(ctx: &GpuContext) -> RdrPass {
    let grid_vert_shader = ctx.device.create_shader_module(&wgpu::include_spirv!(
        "../assets/shaders/spirv/grid.vert.spv"
    ));
    let grid_frag_shader = ctx.device.create_shader_module(&wgpu::include_spirv!(
        "../assets/shaders/spirv/grid.frag.spv"
    ));
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&DEFAULT_BIND_GROUP_LAYOUT_DESC);
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grid_render_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    let uniform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid_uniform_buffer"),
            contents: bytemuck::bytes_of(&crate::gfx::VisualGridUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grid_bind_group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &grid_vert_shader,
                entry_point: "main",
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, /* tells when to discard a
                                                             * new pixel */
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &grid_frag_shader,
                entry_point: "main",
                targets: &[wgpu::ColorTargetState {
                    format: ctx.surface_config.format,
                    // blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    // blend: Some(wgpu::BlendState {
                    //     color: wgpu::BlendComponent {
                    //         src_factor: wgpu::BlendFactor::One,
                    //         dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    //         operation: wgpu::BlendOperation::Add,
                    //     },
                    //     alpha: wgpu::BlendComponent {
                    //         src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                    //         dst_factor: wgpu::BlendFactor::One,
                    //         operation: wgpu::BlendOperation::Add,
                    //     },
                    // }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

    RdrPass {
        pipeline,
        bind_groups: vec![bind_group],
        uniform_buffer: Some(uniform_buffer),
    }
}

fn create_instancing_demo(ctx: &GpuContext) -> InstancingDemo {
    // Create texture: texture and its sampler
    let sampler = std::sync::Arc::new(ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    }));
    let textures = [
        Texture::create_from_bytes(
            &ctx.device,
            &ctx.queue,
            include_bytes!("../assets/damascus001.jpg"),
            sampler.clone(),
            Some("damascus-texture-001"),
        ),
        Texture::create_from_bytes(
            &ctx.device,
            &ctx.queue,
            include_bytes!("../assets/damascus002.jpg"),
            sampler,
            Some("damascus-texture-002"),
        ),
    ];

    // Descriptor Sets
    // [`BindGroup`] describes a set of resources and how they can be accessed by a
    // shader.
    let texture_bind_group_layout =
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("texture-bind-group-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            // SamplerBindingType::Comparison is only for
                            // TextureSampleType::Depth
                            // SamplerBindingType::Filtering if the sample_type of the texture
                            // is:     TextureSampleType::Float
                            // { filterable: true }
                            // Otherwise you'll get an error.
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

    let texture_bind_group_0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("diffuse-texture-bind-group-001"),
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&textures[0].view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&textures[0].sampler),
            },
        ],
    });

    let texture_bind_group_1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("diffuse-texture-bind-group-002"),
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&textures[1].view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&textures[1].sampler),
            },
        ],
    });

    let uniform_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_uniform_buffer"),
            contents: bytemuck::cast_slice(&[IDENTITY_MAT4, IDENTITY_MAT4, IDENTITY_MAT4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let camera_bind_group_layout = ctx
        .device
        .create_bind_group_layout(&DEFAULT_BIND_GROUP_LAYOUT_DESC);

    let camera_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera-uniform-bind-group"),
        layout: &camera_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let shader = ctx
        .device
        .create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("instancing_vertex_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/wgsl/shader.wgsl").into(),
            ),
        });

    // Pipeline layout
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instancing_pipeline_layout"),
            bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
            push_constant_ranges: &[],
        });

    let instancing_vertex_layout = VertexLayout::new(
        &[
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float32x4,
        ],
        Some(5),
    );

    let instance_displacement = Vec3::new(
        NUM_INSTANCES_PER_ROW as f32 * 0.5,
        NUM_INSTANCES_PER_COL as f32 * 0.5,
        0.0,
    );

    // Instancing
    let instancing_transforms = (0..NUM_INSTANCES_PER_ROW)
        .flat_map(|u| {
            (0..NUM_INSTANCES_PER_COL).map(move |v| {
                let pos = Vec3::new(u as f32, v as f32, 0.0) - instance_displacement;
                let rot = if pos.abs_diff_eq(Vec3::ZERO, 0.0001f32) {
                    Quat::from_axis_angle(Vec3::Z, 0.0)
                } else {
                    Quat::from_axis_angle(pos.normalize(), 45.0f32.to_radians())
                };
                Mat4::from_translation(pos) * Mat4::from_quat(rot)
            })
        })
        .collect::<Vec<_>>();
    let instancing_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("object-instancing-buffer"),
            contents: bytemuck::cast_slice(&instancing_transforms),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::layout().buffer_layout(wgpu::VertexStepMode::Vertex),
                    instancing_vertex_layout.buffer_layout(wgpu::VertexStepMode::Instance),
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less, /* tells when to discard a
                                                             * new pixel */
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: ctx.surface_config.format,
                    // blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

    let vertex_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex-buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

    let index_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index-buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

    let num_indices = INDICES.len() as u32;
    let num_vertices = VERTICES.len() as u32;

    InstancingDemo {
        vertex_buffer,
        index_buffer,
        num_vertices,
        num_indices,
        current_texture_index: 0,
        pass: RdrPass {
            pipeline,
            bind_groups: vec![
                camera_bind_group,
                texture_bind_group_0,
                texture_bind_group_1,
            ],
            uniform_buffer: Some(uniform_buffer),
        },
        object_model_matrix: Default::default(),
        instancing_buffer,
        instancing_transforms,
    }
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.0868241, 0.49240386, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.4131759, 0.99240386],
    }, // A
    Vertex {
        position: [-0.49513406, 0.06958647, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.0048659444, 0.56958647],
    }, // B
    Vertex {
        position: [-0.21918549, -0.44939706, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.28081453, 0.05060294],
    }, // C
    Vertex {
        position: [0.35966998, -0.3473291, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.85967, 0.1526709],
    }, // D
    Vertex {
        position: [0.44147372, 0.2347359, 0.0],
        // color: [0.5, 0.0, 0.5],
        tex_coord: [0.9414737, 0.7347359],
    }, // E
];

const INDICES: &[u16] = &[0, 1, 4, 1, 2, 4, 2, 3, 4];
