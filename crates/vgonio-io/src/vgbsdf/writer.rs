//! `VgbsdfWriter` - builds a `.vgbsdf` zip archive on disk.

use std::{
    fs::File,
    io::{BufWriter, Cursor, Seek, Write},
    path::Path,
};
use vgn_core::{
    error::VgonioError,
    utils::partition::{
        write_hemisphere_exr_to, HemisphereEncoding, HemisphereLayer, SphericalPartition,
    },
};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

use crate::vgbsdf::{
    conventions::{attr_key, OutgoingEncoding, OutputKind, CONVENTIONS_V1},
    incident_grid::IncidentGridToml,
    manifest::Manifest,
    partition_toml::PartitionToml,
    spectrum::SpectrumToml,
    INCIDENT_GRID_FILENAME, MANIFEST_FILENAME, PARTITION_FILENAME, SPECTRUM_FILENAME,
};
/// Writer that produces a `.vgbsdf` zip archive over any `Write + Seek` sink.
///
/// Use [`VgbsdfWriter::create`] for the common on-disk case, or
/// [`VgbsdfWriter::new`] to build into an in-memory `Cursor<Vec<u8>>` (the
/// artifact-publish path) or any other seekable writer.
pub struct VgbsdfWriter<W: Write + Seek> {
    zip: ZipWriter<W>,
}

impl VgbsdfWriter<BufWriter<File>> {
    /// Creates a new archive at `path`, overwriting any existing file.
    ///
    /// # Errors
    ///
    /// Returns an error if the target file cannot be created.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self, VgonioError> {
        let file = File::create(path.as_ref()).map_err(|e| {
            VgonioError::new(
                format!(
                    "failed to create VGBSDF archive as {}: {e}",
                    path.as_ref().display()
                ),
                Some(Box::new(e)),
            )
        })?;
        Ok(Self::new(BufWriter::new(file)))
    }
}

impl<W: Write + Seek> VgbsdfWriter<W> {
    /// Wraps `writer` (anything `Write + Seek`) in a fresh archive. Use
    /// `Cursor::new(Vec::new())` for in-memory output.
    pub fn new(writer: W) -> Self {
        VgbsdfWriter {
            zip: ZipWriter::new(writer),
        }
    }

    fn write_member(&mut self, name: &str, bytes: &[u8]) -> Result<(), VgonioError> {
        let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        self.zip.start_file(name, opts).map_err(|e| {
            VgonioError::new(format!("zip start_file {name}: {e}"), Some(Box::new(e)))
        })?;
        self.zip
            .write_all(bytes)
            .map_err(|e| VgonioError::new(format!("zip write {name}: {e}"), Some(Box::new(e))))?;
        Ok(())
    }

    fn write_toml<T: serde::Serialize>(
        &mut self,
        name: &str,
        value: &T,
    ) -> Result<(), VgonioError> {
        let s = toml::to_string_pretty(value)
            .map_err(|e| VgonioError::new(format!("toml encode {name}: {e}"), Some(Box::new(e))))?;
        self.write_member(name, s.as_bytes())
    }

    /// Write the four required metadata members. Call once before any EXRs.
    pub fn write_metadata(
        &mut self,
        manifest: &Manifest,
        partition: &SphericalPartition,
        incident_grid: Option<&IncidentGridToml>, // None for non-BSDF outputs
        spectrum: &SpectrumToml,
    ) -> Result<(), VgonioError> {
        self.write_toml(MANIFEST_FILENAME, manifest)?;
        self.write_toml(
            PARTITION_FILENAME,
            &PartitionToml::from_partition(partition),
        )?;
        if let Some(g) = incident_grid {
            self.write_toml(INCIDENT_GRID_FILENAME, g)?;
        }
        self.write_toml(SPECTRUM_FILENAME, spectrum)?;
        Ok(())
    }

    /// Write the three EXR encodings for one `BrdfLevel` directory (or analog).
    ///
    /// `level_dir` is the in-archive prefix, e.g. "l0", "l1", "l1+".
    /// `partition` defines the patch grid (must match `write_metadata`).
    /// `layers` is one entry per outer index (incident direction for BSDF, one entry for NDF).
    /// `output_kind` controls which encodings are emitted: BSDF/NDF/MSF emit all three,
    ///   SDF skips disc, Heightfield only emits thetaphi.
    pub fn write_level(
        &mut self,
        level_dir: &str,
        partition: &SphericalPartition,
        layers: &[HemisphereLayer<'_>],
        output_kind: OutputKind,
        timestamp: &chrono::DateTime<chrono::Local>,
        disc_res: u32,
        thetaphi: (u32, u32), // (n_phi, n_theta)
    ) -> Result<(), VgonioError> {
        let encodings: &[OutgoingEncoding] = match output_kind {
            OutputKind::Bsdf | OutputKind::Ndf | OutputKind::Msf => &[
                OutgoingEncoding::Disc,
                OutgoingEncoding::Thetaphi,
                OutgoingEncoding::Patches,
            ],
            OutputKind::Sdf => &[OutgoingEncoding::Thetaphi, OutgoingEncoding::Patches],
            OutputKind::Heightfield => &[OutgoingEncoding::Thetaphi],
        };

        for &enc in encodings {
            let hemi_enc = match enc {
                OutgoingEncoding::Disc => HemisphereEncoding::Disc {
                    resolution: disc_res,
                },
                OutgoingEncoding::Thetaphi => HemisphereEncoding::Thetaphi {
                    n_phi: thetaphi.0,
                    n_theta: thetaphi.1,
                },
                OutgoingEncoding::Patches => HemisphereEncoding::Patches,
            };

            // Build attrs.
            let mut attrs = vec![
                (attr_key::CONVENTIONS.into(), CONVENTIONS_V1.into()),
                (attr_key::OUTPUT_KIND.into(), output_kind.as_str().into()),
                (attr_key::ENCODING.into(), enc.as_str().into()),
                (
                    attr_key::CREATED.into(),
                    vgn_core::utils::iso_timestamp_from_datetime(timestamp),
                ),
                (attr_key::PARTITION_REF.into(), PARTITION_FILENAME.into()),
            ];
            match enc {
                OutgoingEncoding::Disc => {
                    attrs.push((
                        attr_key::DISC_PROJECTION.into(),
                        "lambert_equal_area".into(),
                    ));
                    attrs.push((attr_key::DISC_RESOLUTION.into(), disc_res.to_string()));
                },
                OutgoingEncoding::Thetaphi => {
                    attrs.push((attr_key::THETAPHI_N_PHI.into(), thetaphi.0.to_string()));
                    attrs.push((attr_key::THETAPHI_N_THETA.into(), thetaphi.1.to_string()));
                },
                OutgoingEncoding::Patches => {
                    attrs.push((
                        attr_key::PATCHES_N_PATCHES.into(),
                        partition.n_patches().to_string(),
                    ));
                },
            }

            // Stage each EXR encoding into an in-memory buffer, then store its
            // bytes inside the zip. `write_hemisphere_exr_to` writes through any
            // `Write + Seek` sink, so a `Cursor<Vec<u8>>` avoids a tempfile
            // round-trip entirely.
            let mut cursor = Cursor::new(Vec::<u8>::new());
            write_hemisphere_exr_to(partition, hemi_enc, layers, &mut cursor, timestamp, &attrs)?;
            let bytes = cursor.into_inner();

            let name = format!("{level_dir}/{}.exr", enc.as_str());
            self.write_member(&name, &bytes)?;
        }

        Ok(())
    }

    /// Finalises the zip central directory and returns the underlying writer.
    ///
    /// On-disk callers ([`Self::create`]) can ignore the returned
    /// `BufWriter<File>`; in-memory callers recover the `Cursor<Vec<u8>>` to
    /// extract the archive bytes via [`Cursor::into_inner`].
    ///
    /// # Errors
    ///
    /// Returns an error if writing the zip footer fails.
    pub fn finish(self) -> Result<W, VgonioError> {
        self.zip
            .finish()
            .map_err(|e| VgonioError::new(format!("zip finish: {e}"), Some(Box::new(e))))
    }
}
