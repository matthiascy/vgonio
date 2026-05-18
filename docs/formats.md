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
|     46 - 0x2E      |  1 byte  |    0x00/0x01/0x02/0x03   | Data (body) compression: 0x00: not compressed, 0x01: zlib, 0x02: gzip, 0x03: lz4           |
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

| Offset (dec - hex) | Size   | Value | Purpose                                               |
|--------------------|--------|-------|-------------------------------------------------------|
| 49 - 0x31          | 1 byte | u8    | Measurement mode: by points=0x00 or by partition=0x01 |
| 50 - 0x32          | 1 byte | u8    | Crop to disk: true = 0x01, false = 0x00               |
| 51 - 0x33          | 1 byte | u8    | Use facet area: true = 0x01, false = 0x00             |

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
      | 8 - 0x08           | 4 bytes | f32   | Step size of the longitude inside the annulus (in radians). |
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

  | Offset (dec - hex) | Size          | Value | Purpose                                                                                                 |
  |--------------------|---------------|-------|---------------------------------------------------------------------------------------------------------|
  | 0 - 0x00           | 4 bytes       | u32   | Receiver domain: upper hemisphere = 0x01, lower = 0x02, whole = 0x00                                    |
  | 4 - 0x04           | 4 bytes       | u32   | Partition scheme: beckers = 0x00, tregenza = 0x01, equal-angle = 0x02                                   |
  | 8 - 0x08           | 4 bytes       | f32   | Precision of the partition (inclination angle step size) in radians                                     |
  | 12 - 0x0C          | 4 bytes       | f32   | Precision of the partition (azimuthal angle step size) in radians, only used when scheme is equal-angle |
  | 16 - 0x10          | 4 bytes       | u32   | Number of rings (`Nr`)                                                                                  |
  | 20 - 0x14          | 4 bytes       | u32   | Number of patches (`Np`)                                                                                |
  | 24 - 0x18          | 20 * Nr bytes |       | Information of each ring                                                                                |

- Ring information

  | Offset (dec - hex) | Size    | Value | Purpose                                                     |
  |--------------------|---------|-------|-------------------------------------------------------------|
  | 0 - 0x00           | 4 bytes | f32   | Minimum colatitude of the annulus.                          |
  | 4 - 0x04           | 4 bytes | f32   | Maximum colatitude of the annulus.                          |
  | 8 - 0x08           | 4 bytes | f32   | Step size of the longitude inside the annulus (in radians). |
  | 12 - 0x0C          | 4 bytes | u32   | Number of patches in the annulus.                           |
  | 16 - 0x10          | 4 bytes | u32   | Base index of the patch of the annulus in the patches data. |

##### SDF measurement

| Offset (dec - hex) | Size    | Value | Purpose                   |
|--------------------|---------|-------|---------------------------|
| 0 - 0x00           | 4 bytes | u32   | Number of slopes in total |

## Body

### Micro-surface profile (.vgms)

The body of micro-surface profile file contains the actual data samples of the micro-surface. The data can be encoded
in binary or ascii format. It can also be compressed at the same time. The compression format is zlib, gzip, or lz4.

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

When `nrays64` is false (u32/f32):

| Size               | Value         | Purpose                                        |
|--------------------|---------------|------------------------------------------------|
| 4 bytes            | u32           | Maximum bounces of rays hitting the patch (Nb) |
| 4 * (Nb + 1) bytes | [u32; Nb + 1] | Number of rays per bounce (0 to Nb bounces)    |
| 4 * (Nb + 1) bytes | [f32; Nb + 1] | Energy of rays hitting the patch per bounce    |

When `nrays64` is true (u64/f64):

| Size               | Value         | Purpose                                        |
|--------------------|---------------|------------------------------------------------|
| 4 bytes            | u32           | Maximum bounces of rays hitting the patch (Nb) |
| 8 * (Nb + 1) bytes | [u64; Nb + 1] | Number of rays per bounce (0 to Nb bounces)    |
| 8 * (Nb + 1) bytes | [f64; Nb + 1] | Energy of rays hitting the patch per bounce    |

##### BsdfMeasurementStatsPoint

Statistics collected at a single measurement point (incident direction).

When `nrays64` is false (u32/f32):

| Size              | Value          | Purpose                                                                                      |
|-------------------|----------------|----------------------------------------------------------------------------------------------|
| 4 bytes           | u32            | Actual maximum bounce at one measurement point. (Nb)                                         |
| 4 bytes           | u32            | Number of rays hitting the surface. (n_received)                                             |
| 4 bytes           | u32            | Number of rays missed the surface. (n_missed)                                                |
| 4 * Ns bytes      | [u32; Ns]      | Number of absorbed rays per wavelength. (n_absorbed)                                         |
| 4 * Ns bytes      | [u32; Ns]      | Number of reflected rays per wavelength. (n_reflected)                                       |
| 4 * Ns bytes      | [u32; Ns]      | Number of rays captured by the receiver per wavelength. (n_captured)                         |
| 4 * Ns bytes      | [u32; Ns]      | Number of rays escaped from the receiver per wavelength. (n_escaped)                         |
| 4 * Ns bytes      | [f32; Ns]      | Energy captured by the receiver per wavelength. (e_captured)                                 |
| 4 * Ns * Nb bytes | [u32; Ns * Nb] | Number of reflected rays per wavelength per bounce, stored row-major as [wavelength][bounce] |
| 4 * Ns * Nb bytes | [f32; Ns * Nb] | Energy of reflected rays per wavelength per bounce, stored row-major as [wavelength][bounce] |

In case the number of rays is `u64`:

| Size              | Value          | Purpose                                                                                      |
|-------------------|----------------|----------------------------------------------------------------------------------------------|
| 4 bytes           | u32            | Actual maximum bounce at one measurement point. (Nb)                                         |
| 8 bytes           | u64            | Number of rays hitting the surface. (n_received)                                             |
| 8 bytes           | u64            | Number of rays missed the surface. (n_missed)                                                |
| 8 * Ns bytes      | [u64; Ns]      | Number of absorbed rays per wavelength. (n_absorbed)                                         |
| 8 * Ns bytes      | [u64; Ns]      | Number of reflected rays per wavelength. (n_reflected)                                       |
| 8 * Ns bytes      | [u64; Ns]      | Number of rays captured by the receiver per wavelength. (n_captured)                         |
| 8 * Ns bytes      | [u64; Ns]      | Number of rays escaped from the receiver per wavelength. (n_escaped)                         |
| 8 * Ns bytes      | [f64; Ns]      | Energy captured by the receiver per wavelength. (e_captured)                                 |
| 8 * Ns * Nb bytes | [u64; Ns * Nb] | Number of reflected rays per wavelength per bounce, stored row-major as [wavelength][bounce] |
| 8 * Ns * Nb bytes | [f64; Ns * Nb] | Energy of reflected rays per wavelength per bounce, stored row-major as [wavelength][bounce] |

##### VgonioBrdf

The measurement parameters are stored in the header part of the file. The body part contains the measured data. The
measured data is stored as a array with the dimensions of incident direction, outgoing direction and wavelength,
respectively. The data is stored as a 4 bytes floating point value. The number of samples is determined by the number of
incident and outgoing directions and the number of wavelengths specified in the header.

| Size    | Value | Purpose                                                           |
|---------|-------|-------------------------------------------------------------------|
| 4 bytes | u32   | BRDF level (number of bounces), 0 means the sum of all the levels |
| ...     | f32   | BRDF samples                                                      |

## IOR datasets (`.ior.ron`)

Unlike the binary `.vgms`/`.vgmo` files above, refractive-index (IOR) datasets are stored as
human-readable [RON](https://github.com/ron-rs/ron). Each file holds **one dataset for one
medium**; a directory of them is described by a `sources.toml` manifest.

### Dataset schema

```ron
(
    schema_version: 1,
    medium: "al",
    name: "McPeak2015",
    reference: "",          // optional plain-text citation
    comments: "",           // optional
    data: Tabulated([
        (150.0, 0.0953908, 1.2836663),   // (λ in nm, η, κ)
        // … rows ascending by wavelength …
    ]),
)
```

| Field            | Meaning                                                                          |
|------------------|----------------------------------------------------------------------------------|
| `schema_version` | Must equal the version this build expects (currently `1`); mismatch is rejected. |
| `medium`         | Lowercase medium name (e.g. `al`, `cu`, `air`); must parse to a known `Medium`.  |
| `name`           | Source label (e.g. `McPeak2015`).                                                |
| `reference`      | Optional citation text.                                                          |
| `comments`       | Optional free text.                                                              |
| `data`           | Either `Tabulated([...])` or `Dispersion(...)` (see below).                      |

**`Tabulated`** -- rows of `(λ_nm, η, κ)`, ascending by wavelength. A lookup outside the
covered wavelength range returns *no value* (it does not extrapolate or panic).

**`Dispersion`** -- an analytic η formula plus an optional tabulated κ:

```ron
data: Dispersion(
    formula: Sellmeier(c0: 1.0, terms: [(1.0, 0.01), …]),
    range_nm: (200.0, 2000.0),   // valid wavelength range
    k: Some([(200.0, 0.1), …]),  // optional κ table; absent => κ = 0
),
```

The supported formulas are the nine refractiveindex.info dispersion forms:
`Sellmeier` (1), `Sellmeier2` (2), `Polynomial` (3), `RiiFull` (4), `Cauchy` (5),
`Gases` (6), `Herzberger` (7), `Retro` (8), `Exotic` (9).

### Naming & consistency rule

Files are named `<medium>_<name>.ior.ron` (e.g. `al_McPeak2015.ior.ron`). If the filename
prefix is a recognized medium it **must** match the `medium` field in the body; likewise a
`sources.toml` entry for the file must agree with the body. Any disagreement is a hard error
rather than a silent mismatch. A file whose prefix is not a recognized medium is loaded
purely on its `medium` field.

### `sources.toml` manifest

```toml
upstream = "https://github.com/polyanskiy/refractiveindex.info-database"

[[dataset]]
file    = "al_McPeak2015.ior.ron"
medium  = "al"
default = true                    # the chosen dataset when a medium has several
path    = "main/Al/nk/McPeak.yml" # upstream provenance (informational)
verified = true                   # false ⇒ provenance unconfirmed
```

When a directory contains **more than one** dataset for the same medium, exactly one must
win: either it is the only non-excluded candidate, or it is marked `default = true`. Zero
defaults *and* an ambiguous set is an error; multiple `default = true` for one medium is an
error. Manifest entries pointing at files that do not exist on disk are ignored.

### Data resolution (where datasets come from)

The registry is assembled from up to three layers; later layers override earlier ones
**per medium**:

1. **Embedded baseline**
   The committed `datafiles/ior/` compiled into the binary at build time (the default-on
   `embed-datafiles` Cargo feature). This makes a fresh checkout, a bare `cargo install`,
   and a packaged binary all work with no setup and no source-tree discovery. Building with `--no-default-features` drops this layer so distro packages can ship the data themselves.
2. **System**
   `<sys_data_dir>/ior/` (e.g. `$XDG_DATA_DIRS`, `/usr/share/vgonio/ior/`),
   where a system package places datasets.
3. **User**
   `<user_data_dir>/ior/` (`$XDG_DATA_HOME`), per-user overrides; highest precedence.

To add or replace a dataset at runtime without rebuilding, drop a `.ior.ron` file (and a
matching `sources.toml` entry if the medium would otherwise be ambiguous) into the user
directory. Per-config exclusions are listed by **`.ior.ron` file name** under
`excluded_ior_files` in the vgonio config.
