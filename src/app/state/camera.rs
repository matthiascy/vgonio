use super::InputState;
use crate::app::gfx::camera::{Camera, CameraController, CameraUniform, OrbitControls, Projection};
use glam::Mat4;

pub struct CameraState {
    pub(crate) camera: Camera,
    pub(crate) uniform: CameraUniform,
    pub(crate) uniform_buffer: wgpu::Buffer,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) controller: OrbitControls,
    pub(crate) projection: Projection,
}

impl CameraState {
    pub fn new(device: &wgpu::Device, camera: Camera, projection: Projection) -> Self {
        use wgpu::util::DeviceExt;
        let uniform = CameraUniform::new(&camera, &projection);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniform-buffer"),
            contents: bytemuck::cast_slice(&[
                Mat4::IDENTITY,
                uniform.view_matrix,
                uniform.proj_matrix,
                uniform.view_inv_matrix,
                uniform.proj_inv_matrix,
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-uniform-bind-group-create_bind_group_layout"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-uniform-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let controller = OrbitControls::new(0.3, f32::INFINITY, 200.0, 400.0, 100.0);

        Self {
            camera,
            uniform,
            uniform_buffer,
            bind_group,
            bind_group_layout,
            controller,
            projection,
        }
    }

    pub fn update(&mut self, input: &InputState, dt: std::time::Duration) {
        self.controller.update(input, &mut self.camera, dt);
        self.uniform.update(&self.camera, &self.projection);
    }
}
