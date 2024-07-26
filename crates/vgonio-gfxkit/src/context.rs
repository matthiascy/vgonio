use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};
use winit::window::Window;

/// Uniform buffer used when rendering.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct UiUniforms {
    /// Size of the window in logical pixels (points).
    pub logical_size: [f32; 2],
    /// Padding to align the struct to 16 bytes for the reason of the alignment
    /// requirement of the uniform buffer in WebGL.
    pub _padding: [f32; 2],
}

impl Default for UiUniforms {
    fn default() -> Self {
        Self {
            logical_size: [0.0, 0.0],
            _padding: [0.0, 0.0],
        }
    }
}

impl From<&ScreenDescriptor> for UiUniforms {
    fn from(desc: &ScreenDescriptor) -> Self {
        Self {
            logical_size: [desc.width as f32, desc.height as f32],
            _padding: [0.0, 0.0],
        }
    }
}

/// Information about the screen on which the UI is presented.
#[derive(Debug)]
pub struct ScreenDescriptor {
    /// Width of the window in physical pixel.
    pub width: u32,
    /// Height of the window in physical pixel.
    pub height: u32,
    /// HiDPI scale factor; pixels per point.
    pub scale_factor: f32,
}

impl ScreenDescriptor {
    /// Returns the screen width in pixels.
    pub fn physical_size(&self) -> [u32; 2] { [self.width, self.height] }

    /// Returns the screen width in points.
    pub fn logical_size(&self) -> [f32; 2] {
        [
            self.width as f32 / self.scale_factor,
            self.height as f32 / self.scale_factor,
        ]
    }
}

/// Surface state for presenting to a window.
pub struct WindowSurface {
    /// Surface (image texture/framebuffer) to draw on.
    pub inner: wgpu::Surface<'static>,
    /// Surface configuration (size, format, etc).
    pub config: wgpu::SurfaceConfiguration,
    /// Scale factor of the window.
    pub window_scale_factor: f32,
}

impl Deref for WindowSurface {
    type Target = wgpu::Surface<'static>;

    fn deref(&self) -> &Self::Target { &self.inner }
}

impl DerefMut for WindowSurface {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.inner }
}

impl WindowSurface {
    /// Create a new surface for a window.
    pub fn new(
        gpu: &GpuContext,
        window: &Window,
        config: &WgpuConfig,
        surface: wgpu::Surface<'static>,
    ) -> Self {
        let capabilities = surface.get_capabilities(&gpu.adapter);
        log::info!("Supported surface formats: {:?}", &capabilities.formats);
        let format = config
            .target_format
            .unwrap_or_else(|| config.preferred_surface_format(&capabilities.formats));
        log::info!("  - Selected surface format: {:?}", format);
        let win_size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: win_size.width,
            height: win_size.height,
            present_mode: config.present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        let window_scale_factor = window.scale_factor() as f32;
        surface.configure(&gpu.device, &config);

        Self {
            inner: surface,
            config,
            window_scale_factor,
        }
    }

    /// Gets a descriptor for the surface.
    pub fn screen_descriptor(&self) -> ScreenDescriptor {
        ScreenDescriptor {
            width: self.config.width,
            height: self.config.height,
            scale_factor: self.window_scale_factor,
        }
    }

    /// Resizes the surface to the given size (physical pixels).
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

    /// Returns the presentation mode of the window surface.
    pub fn present_mode(&self) -> wgpu::PresentMode { self.config.present_mode }

    /// Returns the aspect ratio of the window surface.
    pub fn aspect_ratio(&self) -> f32 { self.config.width as f32 / self.config.height as f32 }

    /// Reconfigures the surface with its current configuration.
    pub fn reconfigure(&mut self, device: &wgpu::Device) {
        self.inner.configure(device, &self.config);
    }
}

/// Aggregates all the wgpu objects needed to use the GPU.
pub struct GpuContext {
    /// Context for wgpu objects.
    pub instance: wgpu::Instance,

    /// Adapter for wgpu: the physical device + graphics api.
    pub adapter: wgpu::Adapter,

    /// GPU logical device.
    pub device: Arc<wgpu::Device>,

    /// GPU command queue to execute drawing or computing commands.
    pub queue: Arc<wgpu::Queue>,
}

/// Configuration for the [`GpuContext`].
pub struct WgpuConfig {
    /// Device requirements for requesting a device.
    pub device_descriptor: wgpu::DeviceDescriptor<'static>,
    /// Backend API to use.
    pub backends: wgpu::Backends,
    /// Present mode used for the primary swap chain (surface).
    pub present_mode: wgpu::PresentMode,
    /// Power preference for the GPU.
    pub power_preference: wgpu::PowerPreference,
    /// Surface format used for the primary swap chain. If `None`, the surface
    /// will be created with the preferred format returned by
    /// [`WgpuConfig::preferred_surface_format`].
    pub target_format: Option<wgpu::TextureFormat>,
    /// Texture format for the depth buffer.
    pub depth_format: Option<wgpu::TextureFormat>,
}

impl Default for WgpuConfig {
    fn default() -> Self {
        Self {
            device_descriptor: wgpu::DeviceDescriptor {
                label: Some("wgpu-default-device"),
                required_features: wgpu::Features::default(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
            },
            backends: wgpu::Backends::PRIMARY,
            present_mode: wgpu::PresentMode::AutoVsync,
            power_preference: wgpu::PowerPreference::HighPerformance,
            depth_format: None,
            target_format: None,
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
    // returns window surface
    /// Creates a new onscreen rendering context.
    pub async fn onscreen(
        window: Arc<Window>,
        config: &WgpuConfig,
    ) -> (Self, wgpu::Surface<'static>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::VALIDATION,
            dx12_shader_compiler: Default::default(),
            gles_minor_version: Default::default(),
        });
        let surface = instance
            .create_surface(window)
            .expect("Unable to create a window surface!");
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

        let adapter_limits = adapter.limits();

        log::debug!("Adapter limits: {:#?}", adapter_limits);
        log::debug!(
            "Default limits: {:#?}",
            config.device_descriptor.required_limits
        );

        // Logical device and command queue
        let (device, queue) = if !config
            .device_descriptor
            .required_limits
            .check_limits(&adapter_limits)
        {
            log::debug!("Request device with default limits");
            adapter.request_device(&config.device_descriptor, None)
        } else {
            log::debug!("Request device with adapter limits");
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: config.device_descriptor.label,
                    required_features: config.device_descriptor.required_features,
                    required_limits: adapter_limits,
                    memory_hints: Default::default(),
                },
                None,
            )
        }
        .await
        .unwrap_or_else(|_| {
            panic!(
                "Failed to request logical device! {}",
                concat!(file!(), ":", line!())
            )
        });

        log::debug!("Device limits: {:#?}", device.limits());

        (
            Self {
                instance,
                adapter,
                device: Arc::new(device),
                queue: Arc::new(queue),
            },
            surface,
        )
    }

    /// Creates a new offscreen rendering context.
    pub async fn offscreen(config: &WgpuConfig) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: Default::default(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: Default::default(),
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap_or_else(|| {
                panic!(
                    "Failed to request physical device! {}",
                    concat!(file!(), ":", line!())
                )
            });

        let adapter_limits = adapter.limits();

        // Logical device and command queue
        let (device, queue) = if config
            .device_descriptor
            .required_limits
            .check_limits(&adapter_limits)
        {
            adapter.request_device(&config.device_descriptor, None)
        } else {
            adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: config.device_descriptor.label,
                    required_features: config.device_descriptor.required_features,
                    required_limits: adapter_limits,
                    memory_hints: Default::default(),
                },
                None,
            )
        }
        .await
        .unwrap_or_else(|_| {
            panic!(
                "Failed to request logical device! {}",
                concat!(file!(), ":", line!())
            )
        });

        Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        }
    }
}
