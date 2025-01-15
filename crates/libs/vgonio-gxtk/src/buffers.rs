use std::ops::{Range, RangeBounds};

// use egui::NumExt;

/// A contiguous non-growable GPU buffer containing sub-buffer ranges with the
/// same buffer usages.
/// It aims reducing the number of allocated buffers by pre-allocating a large
/// buffer, then sub-allocating from it.
/// Sub-allocation is indicated by a range of `wgpu::BufferAddress` in the
/// buffer.
pub struct SlicedBuffer {
    /// The buffer containing all the sub-buffers.
    buf: wgpu::Buffer,
    /// The capacity of the buffer in bytes.
    cap: wgpu::BufferAddress,
    /// The length of the buffer (currently used) in bytes.
    len: wgpu::BufferAddress,
    /// The usages of the buffer.
    usages: wgpu::BufferUsages,
    /// The sub-buffers of the buffer.
    subslices: Vec<Range<wgpu::BufferAddress>>,
}

impl SlicedBuffer {
    /// Create a new buffer with the given capacity and usages.
    pub fn new(
        device: &wgpu::Device,
        cap: wgpu::BufferAddress,
        usages: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Self {
        Self {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size: cap,
                usage: usages,
                mapped_at_creation: false,
            }),
            cap,
            usages,
            subslices: vec![],
            len: 0,
        }
    }

    /// Create a new buffer with the given capacity, usages and sub-buffer
    /// ranges.
    pub fn new_sliced(
        device: &wgpu::Device,
        cap: wgpu::BufferAddress,
        usages: wgpu::BufferUsages,
        slices: Vec<Range<wgpu::BufferAddress>>,
        label: Option<&str>,
    ) -> Self {
        Self {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size: cap,
                usage: usages,
                mapped_at_creation: false,
            }),
            cap,
            usages,
            subslices: slices,
            len: 0,
        }
    }

    /// Create a new buffer with the initial data, capacity, usages.
    pub fn new_init(
        device: &wgpu::Device,
        usages: wgpu::BufferUsages,
        contents: &[u8],
        label: Option<&str>,
    ) -> Self {
        use wgpu::util::DeviceExt;
        Self {
            buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents,
                usage: usages,
            }),
            cap: contents.len() as wgpu::BufferAddress,
            usages,
            subslices: vec![],
            len: contents.len() as wgpu::BufferAddress,
        }
    }

    /// Create a new buffer with the initial data, capacity, usages and
    /// sub-buffer ranges.
    pub fn new_sliced_init(
        device: &wgpu::Device,
        usages: wgpu::BufferUsages,
        contents: &[u8],
        label: Option<&str>,
        slices: Vec<Range<wgpu::BufferAddress>>,
    ) -> Self {
        use wgpu::util::DeviceExt;
        Self {
            buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents,
                usage: usages,
            }),
            cap: contents.len() as wgpu::BufferAddress,
            usages,
            subslices: slices,
            len: contents.len() as wgpu::BufferAddress,
        }
    }

    /// Get the buffer capacity.
    pub fn capacity(&self) -> wgpu::BufferAddress { self.cap }

    /// Get the number of sub-buffers.
    pub fn num_sub_buffers(&self) -> usize { self.subslices.len() }

    /// Get the length of the buffer in bytes.
    pub fn len(&self) -> wgpu::BufferAddress { self.len }

    /// Get if the buffer is empty.
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Get the buffer usages.
    pub fn usages(&self) -> wgpu::BufferUsages { self.usages }

    /// Get the [`wgpu::Buffer`].
    pub fn buffer(&self) -> &wgpu::Buffer { &self.buf }

    pub fn data_slice<S: RangeBounds<wgpu::BufferAddress>>(&self, bounds: S) -> wgpu::BufferSlice {
        self.buf.slice(bounds)
    }

    /// Get the buffer slice at the given index.
    pub fn subslice(&self, index: usize) -> Range<wgpu::BufferAddress> {
        self.subslices[index].clone()
    }

    /// Get the buffer slices.
    pub fn subslices(&self) -> &Vec<Range<wgpu::BufferAddress>> { &self.subslices }

    /// Get the buffer slice at the given index.
    pub fn subslice_mut(&mut self, index: usize) -> &mut Range<wgpu::BufferAddress> {
        &mut self.subslices[index]
    }

    /// Get the buffer slices.
    pub fn subslices_mut(&mut self) -> &mut Vec<Range<wgpu::BufferAddress>> { &mut self.subslices }

    pub fn grow(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        required_cap: wgpu::BufferAddress,
        copy: bool,
    ) {
        debug_assert!(required_cap > self.cap);
        let new_cap = (self.cap * 2).max(required_cap);
        log::info!("Growing buffer from {} to {}", self.cap, new_cap);
        if copy {
            debug_assert!(
                self.usages.contains(wgpu::BufferUsages::COPY_SRC),
                "Buffer must have COPY_SRC usage to be copied"
            );
            let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: new_cap,
                usage: self.usages,
                mapped_at_creation: false,
            });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            encoder.copy_buffer_to_buffer(&self.buf, 0, &new_buf, 0, self.cap);

            queue.submit(std::iter::once(encoder.finish()));

            self.buf = new_buf;
            self.cap = new_cap;
        } else {
            self.buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: new_cap,
                usage: self.usages,
                mapped_at_creation: false,
            });
            self.cap = new_cap;
        }
    }
}
