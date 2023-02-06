#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum CacheKind {
    Bvh,
    Mesh,
    HeightField,
    Unknown,
}

#[derive(Debug, Copy, Clone)]
pub struct CacheHeader {
    pub binary: bool,
    pub kind: CacheKind,
    pub size: u32,
}

impl CacheHeader {
    pub fn new(buf: [u8; 6]) -> Self {
        Self {
            binary: buf[0] == b'!',
            kind: match buf[1] {
                0x01 => CacheKind::Bvh,
                0x02 => CacheKind::HeightField,
                0x04 => CacheKind::Mesh,
                _ => CacheKind::Unknown,
            },
            size: u32::from_le_bytes(buf[2..6].try_into().expect("incorrect len")),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum SiUnit {
    Nanometre,
    Micrometre,
}

/// Micro-surface file header
#[derive(Debug)]
pub struct MsHeader {
    pub binary: bool,
    pub unit: SiUnit,
    pub spacing: [f32; 2],
    pub extent: [u32; 2],
    pub size: u32,
}

impl MsHeader {
    /// Read from a buffer without the first 4 bytes: DCMS
    /// Last byte of the buffer is the new line character
    pub fn new(buf: [u8; 23]) -> Self {
        let unit = match buf[1] {
            0x01 => SiUnit::Micrometre,
            0x02 => SiUnit::Nanometre,
            _ => SiUnit::Nanometre,
        };
        let spacing_x = f32::from_le_bytes(buf[2..6].try_into().expect("incorrect len"));
        let spacing_y = f32::from_le_bytes(buf[6..10].try_into().expect("incorrect len"));
        let samples_count_x = u32::from_le_bytes(buf[10..14].try_into().expect("incorrect len"));
        let samples_count_y = u32::from_le_bytes(buf[14..18].try_into().expect("incorrect len"));

        Self {
            binary: buf[0] == b'!',
            unit,
            spacing: [spacing_x, spacing_y],
            extent: [samples_count_x, samples_count_y],
            size: u32::from_le_bytes(buf[18..22].try_into().expect("incorrect len")),
        }
    }

    pub fn write_into<W: std::io::Write>(&self, writer: &mut W) -> Result<(), std::io::Error> {
        let mut buf = [0_u8; 27];

        buf[0..4].clone_from_slice(b"DCMS");

        buf[4] = if self.binary { b'!' } else { b'#' };

        buf[5] = match self.unit {
            SiUnit::Nanometre => 0x01,
            SiUnit::Micrometre => 0x02,
        };

        buf[6..10].clone_from_slice(&f32::to_le_bytes(self.spacing[0]));
        buf[10..14].clone_from_slice(&f32::to_le_bytes(self.spacing[1]));
        buf[14..18].clone_from_slice(&u32::to_le_bytes(self.extent[0]));
        buf[18..22].clone_from_slice(&u32::to_le_bytes(self.extent[1]));
        buf[22..26].clone_from_slice(&u32::to_le_bytes(self.size));
        buf[26] = b'\n';
        writer.write_all(&buf).map(|_| Ok(()))?
    }
}

// /// VGonio measurement output file.
// pub struct Vgmo<T: MicroSurfaceMeasurement> {
//     pub header: VgmoHeader,
//     pub measurement: T,
// }

// pub struct VgmoHeader {
//     pub magic: [u8; 4],
//     pub measurement_type: u8,
//     pub encoding: u8,
// }

// /// Metadata for micro-facet normal distribution function and
// masking/shadowing /// function
// pub struct VgmoHeaderMndfAndMmsf {}

// impl VgmoHeader {
//     /// Size of the header in bytes.
//     pub const SIZE_IN_BYTES: usize = std::mem::size_of::<Self>();
// }
