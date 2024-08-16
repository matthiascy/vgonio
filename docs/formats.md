# VGonio file formats

The file produced or used by Vgonio consists of a header followed by body of data. The header has two parts: the meta
information and data-specific information. The header is always stored in binary format. Depending on the header, the
body may be in ascii (plain text) or binary format. Little-endian format is used for all kinds of values composed of
multiple bytes.

## Header

The header part of the file can be divided into two parts: the meta information and the data-specific information. The
meta information part is common for all kinds of files produced by Vgonio. The data-specific information part is
different
for each kind of file.

### 1. Meta information

The meta information part of the header is 48 bytes in total. It contains the following fields:

| Offset (dec - hex) |   Size   |          Value           | Purpose                                                                                    |
|:------------------:|:--------:|:------------------------:|:-------------------------------------------------------------------------------------------|
|      0 - 0x00      | 4 bytes  | 0x56 0x47 0x4D 0x53/0x4F | ASCII code of "VGMS/VGMO" in hex.                                                          |
|      4 - 0x04      | 4 bytes  |           u32            | Version indicating the version of Vgonio file format.                                      |
|      8 - 0x08      | 4 bytes  |           u32            | Length of the whole file in bytes.                                                         |
|     12 - 0x0C      | 32 bytes |         [u8;32]          | Timestamp in RFC 3339 and ISO 8601 date and time format "yyyy-MM-ddTHH:mm:ss.SSSXXX+HH:MM" |
|     44 - 0x2C      |  1 byte  |       0x00 or 0xFF       | Size of single data sample in bytes. 0x04: 4 bytes(f32), 0xFF: 0x08 bytes(f64)             |
|     45 - 0x2D      |  1 byte  |       0x21 or 0x23       | Data (body) encoding: !(0x21)-binary, #(0x23)-ascii                                        |
|     46 - 0x2E      |  1 byte  |      0x00/0x01/0x02      | Data (body) compression: 0x00: not compressed, 0x01: zlib, 0x02: gzip                      |
|     47 - 0x2F      |  1 byte  |           0x00           | Padding                                                                                    |

### 2. Data-specific information

### VGMS (Micro-surface profile `.vgms`)

| Offset (dec - hex) | Size    | Value | Purpose                                                                     |
|--------------------|---------|-------|-----------------------------------------------------------------------------|
| 48 - 0x30          | 4 byte  | u32   | SI unit. 0x03: micrometre, 0x04: nanometre, units used for spacing, samples |
| 52 - 0x34          | 4 bytes | u32   | Number of samples in horizontal direction                                   |
| 56 - 0x38          | 4 bytes | u32   | Number of samples in vertical direction                                     |
| 60 - 0x3C          | 4 bytes | f32   | Horizontal spacing between two samples                                      |
| 64 - 0x40          | 4 bytes | f32   | Vertical spacing between two samples                                        |

### VGMO (Measurement output `.vgmo`)

For the measurement output file, the data-specific information part of the header is different for each kind of
measurement.
Following the meta information, the first byte of the data-specific information part indicates the type of measurement.

| Offset (dec - hex) | Size    | Value | Purpose                                            |
|--------------------|---------|-------|----------------------------------------------------|
| 48 - 0x30          | 1 bytes | u8    | Type of measurement: bsdf=0x00, adf=0x01, msf=0x02 |

#### Measurement-specific information

##### NDF measurement

| Offset (dec - hex) | Size     | Value | Purpose                                                 |
|--------------------|----------|-------|---------------------------------------------------------|
| 49 - 0x31          | 1 byte   | u8    | Measurement mode: by points=0x00 or by partition=0x01   |
| 50 - 0x32          | 1 byte   | u8    | Crop to disk: true = 0x01, false = 0x00                 |
| 51 - 0x33          | 1 byte   | u8    | Use facet area: true = 0x01, false = 0x00               |
| -----------        | -------- | ----  | ------------------------------------------------------- |

Depending on the measurement mode, the header part of the file is different.

- By points

  The information of the measurement points is stored in the header part of the file, followed by the body part of the
  file.
  The header part of the file is 84 bytes in total with 36 bytes for measurement points information.

  | Offset (dec - hex) | Size    | Value | Purpose                                                         |
                                                                    |-------------------|---------|-------|-----------------------------------------------------------------|
  | 52 - 0x34          | 4 bytes | f32   | Start point of the measurement along azimuthal angle in radians |
  |  56 - 0x38         | 4 bytes | f32   | Stop point of the measurement along azimuthal angle in radians  |
  |  60 - 0x3C         | 4 bytes | f32   | Bin size of the measurement along azimuthal angle in radians    |
  |  64 - 0x40         | 4 bytes | u32   | Bins count of the measurement along azimuthal angle             |
  |  68 - 0x44         | 4 bytes | f32   | Start point of the measurement along polar angle in radians     |
  |  72 - 0x48         | 4 bytes | f32   | Stop point of the measurement along polar angle in radians      |
  |  76 - 0x4C         | 4 bytes | f32   | Bin size of the measurement along polar angle in radians        |
  |  80 - 0x50         | 4 bytes | u32   | Bins count of the measurement along polar angle                 |
  |  84 - 0x54         | 4 bytes | u32   | Sample count of the measurement.                                |

- By partition

  Defaults to Beckers' partition

  | Offset (dec - hex) | Size            | Value | Purpose                                                                                                 |
                                                                  |--------------------|-----------------|-------|---------------------------------------------------------------------------------------------------------|
  | 52 - 0x34          | 4 bytes         | u32   | Receiver domain: upper hemisphere = 0x01, lower = 0x02, whole = 0x00                                    |
  | 56 - 0x38          | 4 bytes         | u32   | Partition scheme: beckers = 0x00, tregenza = 0x01, equal-angle = 0x02                                   |
  | 60 - 0x3C          | 4 bytes         | f32   | Precision of the partition (inclination angle step size) in radians                                     |
  | 64 - 0x40          | 4 bytes         | f32   | Precision of the partition (azimuthal angle step size) in radians, only used when scheme is equal-angle |
  | 68 - 0x44          | 4 bytes         | u32   | Number of rings (`Nr`)                                                                                  |
  | 72 - 0x48          | 4 bytes         | u32   | Number of patches (`Np`)                                                                                |
  | 76 - 0x4C          | 20 * `Nr` bytes |       | Information of each ring                                                                                |

    - Ring information

      | Offset (dec - hex) | Size    | Value | Purpose                                                     |
                                                                                                                                                                                                                                                                                                                                        |--------------------|---------|-------|-------------------------------------------------------------|
      | 0 - 0x00           | 4 bytes | f32   | Minimum colatitude of the annulus.                          |
      | 4 - 0x04           | 4 bytes | f32   | Maximum colatitude of the annulus.                          |
      | 8 - 0x08           | 4 bytes | u32   | Step size of the longitude inside the annulus.              |
      | 12 - 0x0C          | 4 bytes | u32   | Number of patches in the annulus.                           |
      | 16 - 0x10          | 4 bytes | u32   | Base index of the patch of the annulus in the patches data. |

##### Masking/Shadowing measurement

| Offset (dec - hex) | Size    | Value | Purpose                                                      |
|--------------------|---------|-------|--------------------------------------------------------------|
| 49 - 0x31          | 4 bytes | f32   | Start point of measurement's longitude in radians            |
| 53 - 0x35          | 4 bytes | f32   | Stop point of measurement's longitude in radians             |
| 57 - 0x39          | 4 bytes | f32   | Bin size of the measurement along azimuthal angle in radians |
| 61 - 0x3D          | 4 bytes | u32   | Bins count of the measurement along azimuthal angle          |
| 65 - 0x41          | 4 bytes | f32   | Start point of the measurement along polar angle in radians  |
| 69 - 0x45          | 4 bytes | f32   | Stop point of the measurement along polar angle in radians   |
| 73 - 0x49          | 4 bytes | f32   | Bin size of the measurement along polar angle in radians     |
| 77 - 0x4D          | 4 bytes | u32   | Bins count of the measurement along polar angle              |
| 81 - 0x51          | 4 bytes | u32   | Sample count of the measurement.                             |
| 85 - 0x55          | 3 bytes | u8    | Padding. TODO                                                |

##### BSDF measurement

| Offset (dec - hex) | Size    | Value  | Purpose                                                                         |
|--------------------|---------|--------|---------------------------------------------------------------------------------|
| 49 - 0x31          | 1 byte  | u8     | BSDF type: brdf = 0x00, btdf = 0x01, bssdf = 0x02, bssrdf = 0x03, bsstdf = 0x04 |
| 50 - 0x32          | 3 bytes | [u8;3] | Incident medium: vacuum = 'vac', air = 'air', aluminium = 'al', copper = 'cu'   |
| 53 - 0x35          | 3 bytes | [u8;3] | Transmitted medium                                                              |
| 56 - 0x38          | 1 byte  | u8     | Simulation method: grid-rt: 0x00, embree-rt: 0x01, optix-rt: 0x02, wave: 0x03   |
| 57 - 0x39          | 1 byte  | u8     | Is Fresnel enabled during measurement? 0x00: false, 0x01: true                  |
| 58 - 0x3A          | 1 byte  | u8     | Number of receivers. TODO                                                       |
| 59 - 0x3B          | 1 byte  | u8     | Type of number of rays: 0x00: u32, 0xff: u64                                    |
| 60 - 0x3C          | 4 bytes | u32    | Number of emitted rays.                                                         |
|                    | 8 bytes | u64    | iff type of number of rays is u64                                               |
| 64 - 0x40          | 4 bytes | u32    | Max allowed bounces.                                                            |
| 68 - 0x44          | 4 bytes | f32    | Start point of emitter's position along azimuthal angle in radians              |
| 72 - 0x48          | 4 bytes | f32    | Stop point of emitter's position along azimuthal angle in radians               |
| 76 - 0x4C          | 4 bytes | f32    | Step size of emitter's position along azimuthal angle in radians                |
| 80 - 0x50          | 4 bytes | u32    | Number of steps emitter's position along azimuthal angle                        |
| 84 - 0x54          | 4 bytes | f32    | Start point of emitter's colatitude in radians                                  |
| 88 - 0x58          | 4 bytes | f32    | Stop point of emitter's colatitude in radians                                   |
| 92 - 0x5C          | 4 bytes | f32    | Step size of emitter's colatitude in radians                                    |
| 96 - 0x60          | 4 bytes | u32    | Number of steps of emitter's position along polar angle                         |
| 100 - 0x64         | 4 bytes | f32    | Start wavelength of the spectrum.                                               |
| 104 - 0x68         | 4 bytes | f32    | Stop wavelength of the spectrum.                                                |
| 108 - 0x6C         | 4 bytes | f32    | Step size of the spectrum.                                                      |
| 112 - 0x70         | 4 bytes | u32    | Number of steps on the spectrum (Ns).                                           |
| 116 - 0x74         | ...     |        | Information of each receiver                                                    |

- Receiver information

  | Offset (dec - hex) | Size          | Value | Purpose                                                                                                   |
                                                                                      |--------------------|---------------|-------|-----------------------------------------------------------------------------------------------------------|
  | 0 - 0x00           | 4 bytes       | u32   | Receiver domain: upper hemisphere = 0x01, lower = 0x02, whole = 0x00                                      |
  | 4 - 0x04           | 4 bytes       | u32   | Partition scheme: beckers = 0x00, tregenza = 0x01, equal-angle = 0x02                                     |
  | 8 - 0x08           | 4 bytes       | f32   | Precision of the partition (inclination angle step size) in radians                                       |
  | 12 - 0x0C          | 4 bytes       | f32   | Precision of the partition (azimuthal angle step size) in radians, only used when scheme is equal-angle   |
  | 16 - 0x10          | 4 bytes       | u32   | Number of rings (`Nr`)                                                                                    |
  | 20 - 0x14          | 4 bytes       | u32   | Number of patches (`Np`)                                                                                  |
  | 24 - 0x18          | 20 * Nr bytes |       | Information of each ring                                                                                  |

- Ring information

  | Offset (dec - hex) | Size     | Value | Purpose                                                     | 
                                                                            |--------------------|----------|-------|-------------------------------------------------------------|
  | 0 - 0x00           | 4 bytes  | f32   | Minimum colatitude of the annulus.                          |
  | 4 - 0x04           | 4 bytes  | f32   | Maximum colatitude of the annulus.                          |
  | 8 - 0x08           | 4 bytes  | u32   | Step size of the longitude inside the annulus.              |
  | 12 - 0x0C          | 4 bytes  | u32   | Number of patches in the annulus.                           |
  | 16 - 0x10          | 4 bytes  | u32   | Base index of the patch of the annulus in the patches data. |

##### SDF measurement

| Offset (dec - hex) | Size    | Value | Purpose                   |
|--------------------|---------|-------|---------------------------|
| 0 - 0x00           | 4 bytes | u32   | Number of slopes in total |

## Body

### Micro-surface profile (.vgms)

The body of micro-surface profile file contains the actual data samples of the micro-surface. The data can be encoded
in binary or ascii format. It can also be compressed at the same time. The compression format is zlib or gzip.

- Binary format: Sample points of the micro-surface's height field are stored continuously as an 1D array. Each sample
  is stored as a 4 bytes or 8 bytes floating point value.

- Plain text format: The data is stored as 2D matrix in ascii format. Sample values are separated by space character.
  One scanline per text line from left to right and top to bottom. The horizontal dimension increases along the scanline
  and the vertical dimension increases with each successive scanline.

### Measurement output (.vgmo)

### NDF

The measured data contains the measured NDF value at each measurement position, indexed first by azimuthal angle then
inclination angle.

### MSF

To be defined.

### BSDF

The body part of BSDF measurement stores the measured BRDFs and full measurement data including the statistics of the
measurement points and the measured data itself.

#### RawMeasurementData

The raw measurement data contains:

- Array of `BounceAndEnergy` in the order of incident direction, outgoing direction and wavelength.
- Statistics at each measurement point (incident direction).

##### BounceAndEnergy

The `BounceAndEnergy` is a struct that contains the number of rays and the energy of rays hitting the patch per bounce.
Depending on the data type of the number of rays, the size of the struct is different.

| Size         | Value               | Purpose                                      |
|--------------|---------------------|----------------------------------------------|
| 4 bytes      | u32                 | Maximum bounces of rays hitting the patch.   |
| 4 * Nb bytes | [u32; 4 * (Nb + 1)] | Number of rays per bounce                    |
| 4 * Nb bytes | [f32; 4 * (Nb + 1)] | Energy of rays hitting the patch per bounce. |

| Size         | Value               | Purpose                                      |
|--------------|---------------------|----------------------------------------------|
| 4 bytes      | u32                 | Maximum bounces of rays hitting the patch.   |
| 8 * Nb bytes | [u64; 8 * (Nb + 1)] | Number of rays per bounce                    |
| 8 * Nb bytes | [f64; 8 * (Nb + 1)] | Energy of rays hitting the patch per bounce. |

##### BsdfMeasurementStatsPoint

| Size              | Value          | Purpose                                                              |
|-------------------|----------------|----------------------------------------------------------------------|
| 4 bytes           | u32            | Actual maximum bounce at one measurement point. (Nb)                 |
| 4 bytes           | u32            | Number of rays hitting the surface. (n_received)                     |
| 4 bytes           | u32            | Number of rays missed the surface. (n_missed)                        |
| 4 * Ns bytes      | [u32; Ns]      | Number of absorbed rays per wavelength. (n_absorbed)                 |
| 4 * Ns bytes      | [u32; Ns]      | Number of reflected rays per wavelength. (n_reflected)               |
| 4 * Ns bytes      | [u32; Ns]      | Number of rays captured by the receiver per wavelength. (n_captured) |
| 4 * Ns bytes      | [u32; Ns]      | Number of rays escaped from the receiver per wavelength. (n_escaped) |
| 4 * Ns bytes      | [f32; Ns]      | Energy captured by the receiver per wavelength. (e_captured)         |
| 4 * Ns * Nb bytes | [u32; Ns * Nb] | Number of reflected rays per wavelength per bounce [[u32; Nb]; Ns]   |
| 4 * Ns * Nb bytes | [u32; Ns * Nb] | Energy of reflected rays per wavelength per bounce [[u32; Nb]; Ns]   |

In case the number of rays is `u64`:

| Size              | Value          | Purpose                                                              |
|-------------------|----------------|----------------------------------------------------------------------|
| 4 bytes           | u32            | Actual maximum bounce at one measurement point. (Nb)                 |
| 8 bytes           | u64            | Number of rays hitting the surface. (n_received)                     |
| 8 bytes           | u64            | Number of rays missed the surface. (n_missed)                        |
| 8 * Ns bytes      | [u64; Ns]      | Number of absorbed rays per wavelength. (n_absorbed)                 |
| 8 * Ns bytes      | [u64; Ns]      | Number of reflected rays per wavelength. (n_reflected)               |
| 8 * Ns bytes      | [u64; Ns]      | Number of rays captured by the receiver per wavelength. (n_captured) |
| 8 * Ns bytes      | [u64; Ns]      | Number of rays escaped from the receiver per wavelength. (n_escaped) |
| 8 * Ns bytes      | [f64; Ns]      | Energy captured by the receiver per wavelength. (e_captured)         |
| 8 * Ns * Nb bytes | [u64; Ns * Nb] | Number of reflected rays per wavelength per bounce [[u64; Nb]; Ns]   |
| 4 * Ns * Nb bytes | [f64; Ns * Nb] | Energy of reflected rays per wavelength per bounce [[f64; Nb]; Ns]   |

##### VgonioBrdf

The measurement parameters are stored in the header part of the file. The body part contains the measured data. The
measured data is stored as a array with the dimensions of incident direction, outgoing direction and wavelength,
respectively. The data is stored as a 4 bytes floating point value. The number of samples is determined by the number of
incident and outgoing directions and the number of wavelengths specified in the header.

| Size    | Value | Purpose                                                           |
|---------|-------|-------------------------------------------------------------------|
| 4 bytes | u32   | BRDF level (number of bounces), 0 means the sum of all the levels |
| ...     | f32   | BRDF samples                                                      |