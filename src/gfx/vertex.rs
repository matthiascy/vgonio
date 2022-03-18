pub use wgpu::VertexFormat;
use wgpu::{BufferAddress, VertexAttribute};

#[derive(Debug)]
pub struct VertexLayout {
    /// Vertex attributes (attribute format: size, type, location).
    attributes: Vec<VertexAttribute>,
    /// The stride of the vertex (size in bytes of all vertex attributes).
    stride: BufferAddress,
}

impl VertexLayout {
    pub fn new(attrib_formats: &[VertexFormat], location_offset: Option<u32>) -> Self {
        assert!(
            !attrib_formats.is_empty(),
            "VertexLayout must have at least one attribute!"
        );
        let len = attrib_formats.len();
        let offsets_and_stride = {
            let mut offsets = vec![0; len + 1];
            for i in 1..(len + 1) {
                offsets[i] = offsets[i - 1] + attrib_formats[i - 1].size();
            }
            offsets
        };

        let stride = offsets_and_stride[len];
        let location_offset = location_offset.unwrap_or(0);

        let attributes = attrib_formats
            .iter()
            .zip(offsets_and_stride.iter())
            .enumerate()
            .map(|(i, (format, offset))| VertexAttribute {
                format: *format,
                offset: *offset,
                shader_location: location_offset + i as u32,
            })
            .collect();

        Self { attributes, stride }
    }

    pub fn buffer_layout(&self, step_mode: wgpu::VertexStepMode) -> wgpu::VertexBufferLayout {
        wgpu::VertexBufferLayout {
            array_stride: self.stride,
            step_mode,
            attributes: &self.attributes,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coord: [f32; 2],
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

impl Vertex {
    pub fn layout() -> VertexLayout {
        VertexLayout::new(&[VertexFormat::Float32x3, VertexFormat::Float32x2], None)
    }
}
