use image::RgbaImage;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

/// A tile in a tiled image.
pub struct Tile<'a> {
    /// The index of the tile in the tiled image.
    pub idx: u32,
    /// The x coordinate of the first pixel of the tile in the image.
    pub x: u32,
    /// The y coordinate of the first pixel of the tile in the image.
    pub y: u32,
    /// The width of the tile in pixels.
    pub w: u32,
    /// The height of the tile in pixels.
    pub h: u32,
    /// The underlying pixel buffer, in RGBA format, in row-major order.
    pub pixels: &'a [u32],
}

/// A mutable tile in a tiled image.
pub struct TileMut<'a> {
    /// The index of the tile in the tiled image.
    pub idx: u32,
    /// The x coordinate of the first pixel of the tile in the image.
    pub x: u32,
    /// The y coordinate of the first pixel of the tile in the image.
    pub y: u32,
    /// The width of the tile in pixels.
    pub w: u32,
    /// The height of the tile in pixels.
    pub h: u32,
    /// The underlying pixel buffer, in RGBA format, in row-major order.
    pub pixels: &'a [u32],
}

/// Converts an RGBA pixel to a 32-bit unsigned integer in format 0xAABBGGRR.
#[inline(always)]
pub const fn rgba_to_u32(r: u8, g: u8, b: u8, a: u8) -> u32 {
    (a as u32) << 24 | (b as u32) << 16 | (g as u32) << 8 | r as u32
}

/// An image that is decomposed into smaller tiles for parallel processing.
///
/// Tiles and pixels inside tiles are stored in a flat array in row-major order.
/// The pixel is encoded as a 32-bit RGBA value.
pub struct TiledImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,

    /// Pixel width of a tile.
    pub tile_width: u32,
    /// Pixel height of a tile.
    pub tile_height: u32,
    /// Number of pixels per tile.
    pub pixels_per_tile: u32,

    /// Number of tiles in the x direction.
    pub tiles_per_row: u32,
    /// Number of tiles in the y direction.
    pub tiles_per_col: u32,
    /// Total number of tiles.
    pub tiles_per_image: u32,

    /// Underlying pixel buffer.
    pub pixels: Vec<u32>,
}

impl TiledImage {
    pub fn new(width: u32, height: u32, tile_width: u32, tile_height: u32) -> Self {
        let tiles_per_row = (width + tile_width - 1) / tile_width;
        let tiles_per_col = (height + tile_height - 1) / tile_height;
        let tiles_per_image = tiles_per_row * tiles_per_col;
        let pixels_per_tile = tile_width * tile_height;

        Self {
            width,
            height,
            tile_width,
            tile_height,
            pixels_per_tile,
            tiles_per_row,
            tiles_per_col,
            tiles_per_image,
            pixels: vec![0; (tiles_per_image * pixels_per_tile) as usize],
        }
    }

    pub fn write_to_image(&self, image: &mut RgbaImage) {
        for j in 0..self.height {
            for i in 0..self.width {
                let tile_x = i / self.tile_width;
                let tile_y = j / self.tile_height;
                let tile_idx = tile_y * self.tiles_per_row + tile_x;
                let tile_offset = tile_idx * self.pixels_per_tile;
                let tile_i = i % self.tile_width;
                let tile_j = j % self.tile_height;
                let tile_pixel_idx = tile_offset + tile_j * self.tile_width + tile_i;
                let pixel = self.pixels[tile_pixel_idx as usize];

                #[cfg(debug_assertions)]
                image.put_pixel(i, j, image::Rgba(pixel.to_le_bytes()));

                #[cfg(not(debug_assertions))]
                unsafe {
                    image.unsafe_put_pixel(i, j, image::Rgba(pixel.to_le_bytes()));
                }
            }
        }
    }

    pub fn write_to_flat_buffer(&self, buffer: &mut [u8]) {
        for tile in self.tiles() {
            let base = (tile.y * self.width + tile.x) as usize * 4;
            // Copy tile pixels to buffer
            for i in 0..tile.h {
                let row_offset = i as usize * self.width as usize * 4;
                unsafe {
                    buffer
                        .as_mut_ptr()
                        .add(base + row_offset)
                        .copy_from_nonoverlapping(
                            tile.pixels.as_ptr().add(i as usize * tile.w as usize) as *const u8,
                            tile.w as usize * 4,
                        );
                }
            }
        }
    }

    pub fn tile(&self, index: usize) -> Tile<'_> {
        let idx = index as u32;
        let x = (idx % self.tiles_per_row) * self.tile_width;
        let y = (idx / self.tiles_per_row) * self.tile_height;
        let offset = idx * self.pixels_per_tile;
        Tile {
            idx,
            x,
            y,
            w: self.tile_width,
            h: self.tile_height,
            pixels: &self.pixels[offset as usize..(offset + self.pixels_per_tile) as usize],
        }
    }

    pub fn tile_mut(&mut self, tile_x: u32, tile_y: u32) -> TileMut<'_> {
        let idx = tile_y * self.tiles_per_row + tile_x;
        let x = tile_x * self.tile_width;
        let y = tile_y * self.tile_height;
        let offset = idx * self.pixels_per_tile;
        TileMut {
            idx,
            x,
            y,
            w: self.tile_width,
            h: self.tile_height,
            pixels: &mut self.pixels[offset as usize..(offset + self.pixels_per_tile) as usize],
        }
    }

    pub fn tiles(&self) -> impl Iterator<Item = Tile<'_>> {
        self.pixels
            .chunks(self.pixels_per_tile as usize)
            .enumerate()
            .map(|(idx, pixels)| Tile {
                idx: idx as u32,
                x: (idx as u32 % self.tiles_per_row) * self.tile_width,
                y: (idx as u32 / self.tiles_per_row) * self.tile_height,
                w: self.tile_width,
                h: self.tile_height,
                pixels,
            })
    }

    pub fn tiles_mut(&mut self) -> impl Iterator<Item = TileMut<'_>> {
        self.pixels
            .chunks_mut(self.pixels_per_tile as usize)
            .enumerate()
            .map(|(idx, pixels)| TileMut {
                idx: idx as u32,
                x: (idx as u32 % self.tiles_per_row) * self.tile_width,
                y: (idx as u32 / self.tiles_per_row) * self.tile_height,
                w: self.tile_width,
                h: self.tile_height,
                pixels,
            })
    }

    pub fn par_tiles(&self) -> impl IndexedParallelIterator<Item = Tile<'_>> {
        self.pixels
            .par_chunks(self.pixels_per_tile as usize)
            .enumerate()
            .map(|(idx, pixels)| Tile {
                idx: idx as u32,
                x: (idx as u32 % self.tiles_per_row) * self.tile_width,
                y: (idx as u32 / self.tiles_per_row) * self.tile_height,
                w: self.tile_width,
                h: self.tile_height,
                pixels,
            })
    }

    pub fn par_tiles_mut(&mut self) -> impl IndexedParallelIterator<Item = TileMut<'_>> {
        self.pixels
            .par_chunks_mut(self.pixels_per_tile as usize)
            .enumerate()
            .map(|(idx, pixels)| TileMut {
                idx: idx as u32,
                x: (idx as u32 % self.tiles_per_row) * self.tile_width,
                y: (idx as u32 / self.tiles_per_row) * self.tile_height,
                w: self.tile_width,
                h: self.tile_height,
                pixels,
            })
    }

    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.pixels.as_mut_ptr(), 0, self.pixels.len());
        }
    }
}
