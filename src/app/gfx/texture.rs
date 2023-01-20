use image::GenericImageView;
use std::{num::NonZeroU32, sync::Arc};

// TODO: a texture may don't have a sampler (in case used as render pass
// attachments)
// TODO: avoid creating a sampler for each texture (use a cache)
pub struct Texture {
    /// Image (data) allocated on GPU. It holds the pixels and main
    /// memory of the texture, but doesn't contain a lot information
    /// on how to interpret the data.
    pub raw: wgpu::Texture,

    /// Reference to the allocated image data. Besides, it holds information
    /// about how to interpret the data of the texture.
    pub view: wgpu::TextureView,

    /// Holds the data for the specific shader access to the texture, i.e. the
    /// way to blend the pixels, mip-mapping, etc..
    ///
    /// Note: Creation of `wgpu::Sampler` is a completely separate path from
    /// creation of `wgpu::Texture` or `wgpu::TextureView` objects. The same
    /// sampler can be shared between different textures.
    pub sampler: Arc<wgpu::Sampler>,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn new(
        device: &wgpu::Device,
        desc: &wgpu::TextureDescriptor,
        sampler: Option<Arc<wgpu::Sampler>>,
    ) -> Self {
        let texture = device.create_texture(desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = sampler.unwrap_or_else(|| {
            Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                lod_min_clamp: -100.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            }))
        });

        Self {
            raw: texture,
            view,
            sampler,
        }
    }

    pub fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: Option<wgpu::TextureFormat>,
        sampler: Option<Arc<wgpu::Sampler>>,
        label: Option<&str>,
    ) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.unwrap_or(Self::DEPTH_FORMAT),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = sampler.unwrap_or_else(|| {
            Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: Some(wgpu::CompareFunction::LessEqual),
                lod_min_clamp: -100.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            }))
        });
        Self {
            raw: texture,
            view,
            sampler,
        }
    }

    pub fn create_from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        sampler: Arc<wgpu::Sampler>,
        label: Option<&str>,
    ) -> Self {
        let image = image::load_from_memory(bytes).expect("Failed loading image from memory!");
        Self::create_from_dynamic_image(device, queue, &image, sampler, label)
    }

    pub fn create_from_dynamic_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &image::DynamicImage,
        sampler: Arc<wgpu::Sampler>,
        label: Option<&str>,
    ) -> Self {
        let rgba = image.to_rgba8();
        let dims = image.dimensions();
        let size = wgpu::Extent3d {
            width: dims.0,
            height: dims.1,
            depth_or_array_layers: 1,
        };
        let inner = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });
        // Upload texture to device.
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &inner,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * dims.0),
                rows_per_image: NonZeroU32::new(dims.1),
            },
            size,
        );
        let view = inner.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            raw: inner,
            view,
            sampler,
        }
    }
}
