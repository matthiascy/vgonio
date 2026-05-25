//! `VgbsdfReader` - opens and validates a `.vgbsdf` archive per the spec's
//! reader contract.

use std::{
    fmt::format,
    fs::File,
    io::{BufReader, Cursor, Read},
    path::Path,
};
use vgn_core::error::VgonioError;
use zip::ZipArchive;

use crate::vgbsdf::{
    conventions::{ConventionsVersion, OutgoingEncoding, OutputKind},
    incident_grid::IncidentGridToml,
    manifest::Manifest,
    partition_toml::PartitionToml,
    spectrum::SpectrumToml,
    INCIDENT_GRID_FILENAME, MANIFEST_FILENAME, PARTITION_FILENAME, SPECTRUM_FILENAME,
};

/// A reader for `.vgbsdf` archives. It MUST verify:
///
/// 1. `manifest.toml`, `partition.toml`, `spectrum.toml` are present and well-formed.
/// 2. `manifest.vgonio.conventions` is a known version (only `"v1"` under this spec).
/// 3. `manifest.archive.type` is one of the documented `OutputKind` values.
/// 4. Every level listed in `manifest.bsdf.levels` (BSDF archives) has the EXR members required by
///    `manifest.bsdf.encodings` actually present in the zip.
/// 5. `partition.n_patches` matches the patch dimension of every `patches.exr` member.
/// 6. For BSDF archives, `incident_grid.toml` is present, well-formed, and its position count
///    equals the EXR layer count.
/// 7. For spectral archives (BSDF/MSF), the EXR channel count equals `spectrum.wavelengths.len()`.
pub struct VgbsdfReader {
    archive: ZipArchive<BufReader<File>>,
    pub manifest: Manifest,
    pub partition: PartitionToml,
    pub spectrum: SpectrumToml,
    /// Present only when manifest.archive.type == "bsdf".
    pub incident_grid: Option<IncidentGridToml>,
    /// Parsed `manifest.archive.type` after validation. Always Some after `open()`.
    pub output_kind: OutputKind,
}

impl VgbsdfReader {
    /// Open + eagerly validate (1)-(3): metadata members present and well-formed,
    /// conventions version known, output kind known.
    ///
    /// (4)-(7) are validated lazily by `validate_levels()` because they require
    /// opening every EXR member.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VgonioError> {
        let file = File::open(path.as_ref()).map_err(|e| {
            VgonioError::new(
                format!("open VGBSDF archive {}: {e}", path.as_ref().display()),
                Some(Box::new(e)),
            )
        })?;

        let mut archive = ZipArchive::new(BufReader::new(file)).map_err(|e| {
            VgonioError::new(
                format!(
                    "failed to create zip archive for {}: {e}",
                    path.as_ref().display()
                ),
                Some(Box::new(e)),
            )
        })?;
        let manifest: Manifest = read_toml(&mut archive, MANIFEST_FILENAME)?;
        let partition: PartitionToml = read_toml(&mut archive, PARTITION_FILENAME)?;
        let spectrum: SpectrumToml = read_toml(&mut archive, SPECTRUM_FILENAME)?;

        // (2) conventions version
        if ConventionsVersion::parse(&manifest.vgonio.conventions).is_none() {
            return Err(VgonioError::new(
                format!(
                    "unknown vgonio.conventions = {:?}",
                    manifest.vgonio.conventions
                ),
                None,
            ));
        }

        // (3) output kind
        let output_kind = OutputKind::parse(&manifest.archive.kind).ok_or_else(|| {
            VgonioError::new(
                format!(
                    "unknown manifest.archive.type = {:?}",
                    manifest.archive.kind
                ),
                None,
            )
        })?;

        // incident_grid is required for BSDF, optional otherwise (per Reader contract).
        let incident_grid =
            match read_toml::<IncidentGridToml>(&mut archive, INCIDENT_GRID_FILENAME) {
                Ok(g) => Some(g),
                Err(_) if output_kind != OutputKind::Bsdf => None,
                Err(e) => {
                    return Err(VgonioError::new(
                        format!("BSDF archive missing {INCIDENT_GRID_FILENAME}: {e}"),
                        None,
                    ))
                },
            };

        Ok(Self {
            archive,
            manifest,
            partition,
            spectrum,
            incident_grid,
            output_kind,
        })
    }

    /// Validate (4)-(7): every declared level has the required EXR members, and
    /// each EXR's dimensions match the metadata.
    ///
    /// Cost model:
    /// - **IO:** O(zip central directory + sum of EXR header sizes). We pass the zip member's
    ///   reader directly to `exr::meta::MetaData::read_from_buffered`, which streams just enough
    ///   bytes to parse the magic-bytes + per-part headers
    ///   + channel descriptors (a few hundred bytes per member). Once `read_from_buffered`
    ///   returns we drop the reader, abandoning any unread pixel bytes; for `Store`
    ///   members (our default container compression) those bytes are never even read
    ///   from the underlying file because the zip iterator is sequential.
    /// - **Memory:** O(per-EXR header size), not O(member size). We never allocate a buffer the
    ///   size of the entire pixel payload.
    ///
    /// This requires that the pinned `exr` crate version exposes a `Read`-only (not
    /// `Read + Seek`) metadata reader. As of exr 1.73 this is true:
    /// `MetaData::read_from_buffered<R: Read + Send>(read, pedantic)`. If the pinned
    /// version reverts to a `Read + Seek` bound, the workaround is to wrap the
    /// `ZipFile` in `Cursor::new(read_to_end(...))`, but that gives up the streaming
    /// memory bound. **Verify this when you implement, before claiming the cost model
    /// in this docstring.**
    ///
    /// Earlier drafts of this code (a) called `read_all_flat_layers_from_buffered`,
    /// which fully decoded every pixel, and (b) buffered the entire compressed member
    /// into a `Vec<u8>` before parsing. Both were fixed during plan review.
    pub fn validate_levels(&mut self) -> Result<(), VgonioError> {
        use exr::meta::MetaData;

        let n_patches = self.partition.n_patches as usize;

        let (levels, encodings): (Vec<String>, Vec<OutgoingEncoding>) = match self.output_kind {
            OutputKind::Bsdf => {
                let bsdf = self.manifest.bsdf.as_ref().ok_or_else(|| {
                    VgonioError::new("BSDF archive missing [bsdf] block in manifest", None)
                })?;
                let encs: Vec<OutgoingEncoding> = bsdf
                    .encodings
                    .iter()
                    .map(|s| {
                        OutgoingEncoding::parse(s).ok_or_else(|| {
                            VgonioError::new(
                                format!("unknown encoding {s:?} in manifest.bsdf.encodings"),
                                None,
                            )
                        })
                    })
                    .collect::<Result<_, _>>()?;
                (bsdf.levels.clone(), encs)
            },
            // Non-BSDF archives use a fixed single "l0" level under v1.
            // The encoding set MUST mirror `VgbsdfWriter::write_level`'s match
            // (Task 9 Step 1) — readers reject members the writer never emits,
            // and accept all (and only) those the writer does.
            OutputKind::Ndf | OutputKind::Msf => (
                vec!["l0".into()],
                vec![
                    OutgoingEncoding::Disc,
                    OutgoingEncoding::Thetaphi,
                    OutgoingEncoding::Patches,
                ],
            ),
            OutputKind::Sdf => (
                vec!["l0".into()],
                vec![OutgoingEncoding::Thetaphi, OutgoingEncoding::Patches],
            ),
            OutputKind::Heightfield => (
                vec!["l0".into()],
                vec![OutgoingEncoding::Thetaphi], // ONLY thetaphi — no disc, no patches
            ),
        };

        let n_expected_layers = self
            .incident_grid
            .as_ref()
            .map(|g| g.positions.len())
            .unwrap_or(1);

        for level in &levels {
            for &enc in &encodings {
                let name = format!("{level}/{}.exr", enc.as_str());
                let zf = self.archive.by_name(&name).map_err(|e| {
                    VgonioError::new(
                        format!("declared but missing zip member {name}: {e}"),
                        Some(Box::new(e)),
                    )
                })?;

                // Stream the EXR headers directly out of the zip member without
                // buffering the whole payload. `MetaData::read_from_buffered` reads
                // the magic bytes + per-part headers + channel descriptors, then
                // returns; we drop `zf` and the zip iterator skips the remaining
                // (un-read) member bytes on the next `by_name`. For `Store` members
                // this means the pixel payload is never read off disk.
                //
                // Pre-buffering with BufReader keeps small reads off the underlying
                // zip stream; without it the EXR parser would do many tiny reads.
                let meta = MetaData::read_from_buffered(
                    std::io::BufReader::new(zf),
                    /* pedantic = */ false,
                )
                .map_err(|e| {
                    VgonioError::new(format!("EXR header parse {name}: {e}"), Some(Box::new(e)))
                })?;

                // (4) member exists — already proven by the by_name() above.
                // (6) layer (= EXR "part") count matches incident grid (BSDF) or is 1.
                let n_layers = meta.headers.len();
                if n_layers != n_expected_layers {
                    return Err(VgonioError::new(
                        format!("{name}: layer count {n_layers} != expected {n_expected_layers}"),
                        None,
                    ));
                }

                // (5) patches.exr layer_size == (n_patches, 1).
                // (7) channel count matches spectrum's expected_channel_count().
                let first = &meta.headers[0];
                let w = first.layer_size.width();
                let h = first.layer_size.height();
                let n_chan = first.channels.list.len();

                if enc == OutgoingEncoding::Patches {
                    if w != n_patches || h != 1 {
                        return Err(VgonioError::new(
                            format!("{name}: patches.exr dims {w}x{h}, expected {n_patches}x1"),
                            None,
                        ));
                    }
                }

                let expected = self.spectrum.expected_channel_count();
                if n_chan != expected {
                    let mode = if self.spectrum.is_scalar() {
                        "scalar"
                    } else {
                        "spectral"
                    };
                    return Err(VgonioError::new(
                        format!(
                            "{name}: channel count {n_chan} != expected {expected} ({mode} \
                             archive)"
                        ),
                        None,
                    ));
                }
            }
        }

        Ok(())
    }

    /// Reads the raw bytes of an EXR member, e.g. "l0/disc.exr".
    pub fn read_exr_bytes(
        &mut self,
        level_dir: &str,
        encoding: OutgoingEncoding,
    ) -> Result<Vec<u8>, VgonioError> {
        let name = format!("{level_dir}/{}.exr", encoding.as_str());
        let mut f = self
            .archive
            .by_name(&name)
            .map_err(|e| VgonioError::new(format!("zip member {name}: {e}"), Some(Box::new(e))))?;
        let mut buf = Vec::with_capacity(f.size() as usize);
        f.read_to_end(&mut buf)
            .map_err(|e| VgonioError::new(format!("read member {name}: {e}"), Some(Box::new(e))))?;
        Ok(buf)
    }
}

fn read_toml<T: serde::de::DeserializeOwned>(
    archive: &mut ZipArchive<BufReader<File>>,
    name: &str,
) -> Result<T, VgonioError> {
    let mut s = String::new();
    archive
        .by_name(name)
        .map_err(|e| VgonioError::new(format!("zip member {name}: {e}"), Some(Box::new(e))))?
        .read_to_string(&mut s)
        .map_err(|e| VgonioError::new(format!("read member {name}: {e}"), Some(Box::new(e))))?;
    toml::from_str(&s)
        .map_err(|e| VgonioError::new(format!("toml parse {name}: {e}"), Some(Box::new(e))))
}
