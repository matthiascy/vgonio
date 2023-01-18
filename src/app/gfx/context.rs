use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};
use winit::window::Window;

use crate::app::gui::state::ScreenDescriptor;

/// Surface state for presenting to a window.
pub struct WindowSurface {
    /// Surface (image texture/framebuffer) to draw on.
    pub inner: wgpu::Surface,
    /// Surface configuration (size, format, etc).
    pub config: wgpu::SurfaceConfiguration,
    /// Scale factor of the window.
    pub window_scale_factor: f32,
}

impl Deref for WindowSurface {
    type Target = wgpu::Surface;

    fn deref(&self) -> &Self::Target { &self.inner }
}

impl DerefMut for WindowSurface {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
}

impl WindowSurface {
    /// Gets a descriptor for the surface.
    pub fn screen_descriptor(&self) -> ScreenDescriptor {
        ScreenDescriptor {
            width: self.config.width,
            height: self.config.height,
            scale_factor: self.window_scale_factor,
        }
    }

    /// Resizes the surface to the given size (phiscal pixels).
    ///
    /// # Arguments
    ///
    /// * `width` - New width of the surface.
    /// * `height` - New height of the surface.
    /// * `factor` - New scale factor of the surface.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        factor: Option<f32>,
    ) -> bool {
        if self.config.width != width || self.config.height != height {
            self.config.width = width;
            self.config.height = height;
            self.inner.configure(device, &self.config);
        } else if let Some(factor) = factor {
            if self.window_scale_factor != factor {
                self.window_scale_factor = factor;
            } else {
                return false;
            }
        }
        true
    }

    /// Returns the current width of the window surface in physical pixels.
    pub fn width(&self) -> u32 { self.config.width }

    /// Returns the current height of the window surface in physical pixels.
    pub fn height(&self) -> u32 { self.config.height }

    /// Returns the TextureFormat of the surface.
    pub fn format(&self) -> wgpu::TextureFormat { self.config.format }

    /// Returns the presentiation mode of the window surface.
    pub fn present_mode(&self) -> wgpu::PresentMode { self.config.present_mode }
}

/// Aggregation of necessary resources for using GPU.
pub struct GpuContext {
    /// Context for wgpu objects.
    #[allow(dead_code)]
    instance: wgpu::Instance,

    /// Adapter for wgpu: the physical device + graphics api.
    #[allow(dead_code)]
    adapter: wgpu::Adapter,

    // /// Surface (image texture/framebuffer) to draw on.
    // pub surface: wgpu::Surface,
    /// GPU logical device.
    pub device: Arc<wgpu::Device>,

    /// GPU command queue to execute drawing or computing commands.
    pub queue: Arc<wgpu::Queue>,

    /// Query pool to retrieve information from the GPU.
    pub query_set: Option<wgpu::QuerySet>,

    // /// Information about the window surface (texture format, present mode,
    // /// etc.).
    // pub surface_config: wgpu::SurfaceConfiguration,
    /// Surface state for presenting to a window.
    surface: Option<WindowSurface>,
}

pub struct WgpuConfig {
    /// Device requirements for requesting a device.
    pub device_descriptor: wgpu::DeviceDescriptor<'static>,
    /// Backend API to use.
    pub backends: wgpu::Backends,
    /// Present mode used for the primary swap chain (surface).
    pub present_mode: wgpu::PresentMode,
    /// Power preference for the GPU.
    pub power_preference: wgpu::PowerPreference,
    /// Texture format for the depth buffer.
    pub depth_format: Option<wgpu::TextureFormat>,
}

impl Default for WgpuConfig {
    fn default() -> Self {
        Self {
            device_descriptor: wgpu::DeviceDescriptor {
                label: Some("wgpu-default-device"),
                features: wgpu::Features::default(),
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
            },
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
            present_mode: wgpu::PresentMode::AutoVsync,
            power_preference: wgpu::PowerPreference::HighPerformance,
            depth_format: None,
        }
    }
}

impl WgpuConfig {
    /// Prefers the non-linear color space.
    pub fn preferred_surface_format(&self, formats: &[wgpu::TextureFormat]) -> wgpu::TextureFormat {
        for &format in formats {
            if format == wgpu::TextureFormat::Bgra8UnormSrgb
                || format == wgpu::TextureFormat::Rgba8UnormSrgb
            {
                return format;
            }
        }
        formats[0]
    }
}

impl GpuContext {
    /// Creates a new context and requests necessary resources to use the GPU
    /// for presenting to a window.
    pub async fn new(window: &Window, config: WgpuConfig) -> Self {
        let win_size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap_or_else(|| {
                panic!(
                    "Failed to request physical device! {}",
                    concat!(file!(), ":", line!())
                )
            });

        let features = adapter.features();
        log::trace!("GPU supported features: {:?}", features);

        // Logical device and command queue
        let (device, queue) = adapter
            .request_device(&config.device_descriptor, None)
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "Failed to request logical device! {}",
                    concat!(file!(), ":", line!())
                )
            });

        log::trace!("GPU limits: {:?}", device.limits());

        let query_set = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                count: 2,
                ty: wgpu::QueryType::Timestamp,
                label: None,
            }))
        } else {
            None
        };

        // Swap chain configuration
        let formats = surface.get_supported_formats(&adapter);
        log::info!("Supported surface formats: {:?}", formats);
        let format = config.preferred_surface_format(&formats);
        log::info!("-- choose surface format: {:?}", format);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: win_size.width,
            height: win_size.height,
            present_mode: config.present_mode,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &surface_config);

        Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            query_set,
            surface: Some(WindowSurface {
                inner: surface,
                config: surface_config,
                window_scale_factor: window.scale_factor() as f32,
            }),
        }
    }

    // /// Creates a context for offscreen usage.
    // pub async fn offscreen(config: WgpuConfig) -> Self {
    //     let instance = wgpu::Instance::new(wgpu::Backends::all());
    //     let adapter = instance
    //         .request_adapter(&wgpu::RequestAdapterOptions {
    //             power_preference: config.power_preference,
    //             force_fallback_adapter: false,
    //             compatible_surface: None,
    //         })
    //         .await
    //         .unwrap_or_else(|| {
    //             panic!(
    //                 "Failed to request physical device! {}",
    //                 concat!(file!(), ":", line!())
    //             )
    //         });
    //     let features = adapter.features();
    //     let (device, queue) = adapter
    //         .request_device(&config.device_descriptor, None)
    //         .await
    //         .unwrap_or_else(|_| {
    //             panic!(
    //                 "Failed to request logical device! {}",
    //                 concat!(file!(), ":", line!())
    //             )
    //         });
    //     let query_set = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
    //         Some(device.create_query_set(&wgpu::QuerySetDescriptor {
    //             count: 2,
    //             ty: wgpu::QueryType::Timestamp,
    //             label: None,
    //         }))
    //     } else {
    //         None
    //     };

    // }

    /// Returns the reference to the window surface state.
    pub fn surface(&self) -> Option<&WindowSurface> { self.surface.as_ref() }

    /// Returns the mutable reference to the window surface state.
    pub fn surface_mut(&mut self) -> Option<&mut WindowSurface> { self.surface.as_mut() }

    /// Reconfigures the surface.
    pub fn reconfigure_surface(&mut self) {
        if let Some(surface) = self.surface.as_mut() {
            surface.inner.configure(&self.device, &surface.config);
        }
    }

    /// Returns the aspect ratio of the window surface.
    pub fn aspect_ratio(&self) -> f32 {
        if let Some(surface) = &self.surface {
            surface.config.width as f32 / surface.config.height as f32
        } else {
            1.0
        }
    }

    // /// Returns the [`ScreenDescriptor`] describing the window surface.
    // pub fn screen_desc(&self, window: &Window) -> ScreenDescriptor {
    //     ScreenDescriptor {
    //         width: self.surface_config.width,
    //         height: self.surface_config.height,
    //         scale_factor: window.scale_factor() as _,
    //     }
    // }

    /// Resizes the surface to the new size (physical pixels) then returns if
    /// the surface was resized.
    ///
    /// # Arguments
    ///
    /// * `width` - The new width of the surface.
    /// * `height` - The new height of the surface.
    /// * `factor` - The new scale factor of the surface.
    pub fn resize(&mut self, width: u32, height: u32, factor: Option<f32>) -> bool {
        self.surface.as_mut().map_or(false, |surface| {
            surface.resize(&self.device, width, height, factor)
        })
    }
}
