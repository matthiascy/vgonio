use crate::{image::TiledImage, RenderParams};
use gxtk::context::{GpuContext, WgpuConfig, WindowSurface};
use std::{borrow::Cow, sync::Arc};
use wgpu::StoreOp;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

const WGSL_DISPLAY: &str = r#"
var<private> coords: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0)
);

@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(coords[index], 0.0, 1.0);
}

@group(0) @binding(0)
var tex: texture_2d<f32>;

@fragment
fn fragment_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(tex, vec2<i32>(pos.xy), 0);
}
"#;

pub struct Display {
    ctx: GpuContext,
    window: Arc<Window>,
    surface: WindowSurface,
    idx_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

impl Display {
    pub async fn new(w: u32, h: u32, evlp: &EventLoop<()>) -> Display {
        let window = Arc::new(
            winit::window::WindowBuilder::new()
                .with_title("Vgonio-Visu")
                .with_inner_size(winit::dpi::PhysicalSize::new(w, h))
                .build(&evlp)
                .unwrap(),
        );
        let mut wgpu_config = WgpuConfig::default();
        wgpu_config.present_mode = wgpu::PresentMode::AutoNoVsync;

        evlp.set_control_flow(ControlFlow::Poll);
        let (ctx, surface) = GpuContext::onscreen(window.clone(), &wgpu_config).await;
        let surface = WindowSurface::new(&ctx, &window, &wgpu_config, surface);

        // Create resources for the pipeline
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("display_shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&WGSL_DISPLAY)),
            });
        let idx_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index_buffer"),
            size: 4 * size_of::<u16>() as u64,
            usage: wgpu::BufferUsages::INDEX,
            mapped_at_creation: true,
        });
        {
            let mut index_data = idx_buf.slice(..).get_mapped_range_mut();
            let u16_view =
                unsafe { std::slice::from_raw_parts_mut(index_data.as_mut_ptr() as *mut u16, 4) };
            u16_view.copy_from_slice(&[0, 1, 2, 3]);
        }
        idx_buf.unmap();

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("display_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }],
                });
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blit_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture.create_view(
                    &wgpu::TextureViewDescriptor {
                        label: Some("blit_texture_view"),
                        format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        aspect: wgpu::TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: None,
                        base_array_layer: 0,
                        array_layer_count: None,
                    },
                )),
            }],
        });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("display_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("display_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vertex_main",
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: "fragment_main",
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface.format(),
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            });

        Display {
            ctx,
            window,
            surface,
            idx_buf,
            pipeline,
            texture,
            bind_group,
        }
    }

    pub fn blit_to_screen(&self, film: &[u8]) {
        let w = self.surface.width();
        let h = self.surface.height();
        self.ctx.queue.write_texture(
            self.texture.as_image_copy(),
            &film,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let frame = self
            .surface
            .get_current_texture()
            .expect("Failed to get current frame.");
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit encoder"),
            });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_index_buffer(self.idx_buf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..4, 0, 0..1);
        }
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

pub fn run<F>(
    evlp: EventLoop<()>,
    display: Display,
    render: F,
    params: &RenderParams,
    film: &mut TiledImage,
) where
    F: Fn(&RenderParams, &mut TiledImage, bool),
{
    let mut img_buf =
        vec![0u8; display.surface.width() as usize * display.surface.height() as usize * 4]
            .into_boxed_slice();

    let mut fps = 0.0f64;
    let mut last_frame_time = std::time::Instant::now();
    evlp.run(move |event, evlp| match event {
        Event::WindowEvent {
            event: win_event, ..
        } => match win_event {
            WindowEvent::CloseRequested => {
                evlp.exit();
            }
            WindowEvent::RedrawRequested => {
                print!("\rFPS: {:.2}", fps);
                film.clear();
                // Render the image
                render(params, film, true);
                // Write the image to the buffer
                film.write_to_flat_buffer(&mut img_buf);
                // Blit the image to the screen
                display.blit_to_screen(&img_buf);

                // Calculate the FPS
                let elapsed = last_frame_time.elapsed().as_secs_f64();
                last_frame_time = std::time::Instant::now();
                fps = 1.0 / elapsed;

                display.window.as_ref().request_redraw();
            }
            _ => {}
        },
        Event::AboutToWait => {
            display.window.request_redraw();
        }
        _ => {}
    })
    .expect("Event loop failed to run.");
}
