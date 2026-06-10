//! Acquisition related.

use crate::{
    bsdf::BsdfMeasurement,
    io::{
        vgmo::{vgmo_header_ext_from_data, VgmoHeaderExt},
        OutputFileFormatOption,
    },
    mfd::{MeasuredNdfData, MeasuredSdfData},
    params::NdfMeasurementMode,
};
use bytes::Bytes;
use chrono::{DateTime, Local};
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_chacha::ChaCha8Rng;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    prelude::ParallelSliceMut,
};
use std::{
    ffi::OsStr,
    fs::File,
    io::{BufReader, BufWriter, Cursor, Write},
    path::{Path, PathBuf},
};
use vgn_bxdf::brdf::measured::{rgl::RglBrdf, ClausenBrdf, MerlBrdf, VgonioBrdf, Yan18Brdf};
pub use vgn_bxdf::AnyMeasured;
use vgn_core::{
    asset,
    error::VgonioError,
    io::{
        Header, HeaderMeta, ReadFileError, ReadFileErrorKind, WriteFileError, WriteFileErrorKind,
    },
    math::{self, Mat3, Sph2, Sph3, Vec3},
    res::{AssetTypeId, Handle},
    units::{rad, Radians},
    utils::{
        medium::MediumId,
        partition::{SphericalDomain, SphericalPartition},
    },
    MeasurementKind, Version,
};
use vgn_io::MicroSurfaceMesh;
use vgn_job_api::artifact::ArtifactKind;

/// Where the measurement data is loaded from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasurementSource {
    /// Measurement data is loaded from a file.
    Loaded(PathBuf),
    /// Measurement data is generated from a micro-surface.
    /// The handle is the micro-surface handle.
    Measured(Handle),
}

impl MeasurementSource {
    /// Returns the path to the measurement data if it is loaded from a file.
    pub fn path(&self) -> Option<&Path> {
        match self {
            MeasurementSource::Loaded(p) => Some(p.as_path()),
            MeasurementSource::Measured(_) => None,
        }
    }

    /// Returns the micro-surface handle if the measurement data is generated.
    pub fn micro_surface(&self) -> Option<Handle> {
        match self {
            MeasurementSource::Loaded(_) => None,
            MeasurementSource::Measured(ms) => Some(*ms),
        }
    }
}

// TODO: add support for storing data in the memory in a compressed
//       format(maybe LZ4).
/// Structure for storing measurement data in the memory, especially
/// when loading from a file.
#[derive(Debug)]
pub struct Measurement {
    /// Internal tag for displaying the measurement data in the GUI.
    pub name: String,
    /// Origin of the measurement data.
    pub source: MeasurementSource,
    /// Timestamp of the measurement.
    pub timestamp: DateTime<Local>,
    /// Measurement data.
    pub measured: Box<dyn AnyMeasured>,
}

asset!(Measurement, "Measurement");

impl PartialEq for Measurement {
    fn eq(&self, other: &Self) -> bool { self.source == other.source }
}

impl Measurement {
    /// Returns the kind of the measurement data.
    pub fn kind(&self) -> MeasurementKind { self.measured.kind() }

    /// Writes the measurement data to a file in VGMO format.
    pub fn write_to_file(
        &self,
        filepath: &Path,
        format: &OutputFileFormatOption,
    ) -> Result<(), VgonioError> {
        match format {
            OutputFileFormatOption::Vgmo {
                encoding,
                compression,
            } => {
                let timestamp = {
                    let mut timestamp = [0_u8; 32];
                    timestamp.copy_from_slice(
                        vgn_core::utils::iso_timestamp_from_datetime(&self.timestamp).as_bytes(),
                    );
                    timestamp
                };
                let header = Header {
                    meta: HeaderMeta {
                        version: Version::new(0, 1, 0),
                        timestamp,
                        length: 0,
                        sample_size: 4,
                        encoding: *encoding,
                        compression: *compression,
                    },
                    extra: vgmo_header_ext_from_data(&*self.measured),
                };
                let filepath = filepath.with_extension("vgmo");
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&filepath)
                    .map_err(|err| {
                        VgonioError::from_io_error(err, "Failed to open measurement file.")
                    })?;
                let mut writer = BufWriter::new(file);
                crate::io::vgmo::write(&mut writer, header, &*self.measured).map_err(|err| {
                    VgonioError::from_write_file_error(
                        WriteFileError {
                            path: filepath.to_owned().into_boxed_path(),
                            kind: err,
                        },
                        "Failed to write VGMO file.",
                    )
                })?;

                writer.flush().map_err(|err| {
                    VgonioError::from_write_file_error(
                        WriteFileError {
                            path: filepath.to_owned().into_boxed_path(),
                            kind: WriteFileErrorKind::Write(err),
                        },
                        "Failed to flush VGMO file.",
                    )
                })?;
            },
            OutputFileFormatOption::Exr { resolution } => match self.measured.kind() {
                MeasurementKind::Bsdf => {
                    let bsdf = self.measured.downcast_ref::<BsdfMeasurement>().unwrap();
                    bsdf.write_as_exr(filepath, &self.timestamp, *resolution)?
                },
                MeasurementKind::Ndf => {
                    let ndf = self.measured.downcast_ref::<MeasuredNdfData>().unwrap();
                    ndf.write_as_exr(filepath, &self.timestamp, *resolution)?
                },
                MeasurementKind::Gaf => {
                    todo!("Writing MSF to EXR is not supported yet.");
                },
                MeasurementKind::Sdf => {
                    let sdf = self.measured.downcast_ref::<MeasuredSdfData>().unwrap();
                    sdf.write_histogram_as_exr(filepath, &self.timestamp, *resolution)?;
                },
                _ => {},
            },
            OutputFileFormatOption::Vgbsdf { disc_res } => match self.measured.kind() {
                MeasurementKind::Bsdf => {
                    let bsdf = self.measured.downcast_ref::<BsdfMeasurement>().unwrap();
                    let path = filepath.with_extension("vgbsdf");
                    bsdf.write_as_vgbsdf(&path, &self.timestamp, *disc_res)?;
                },
                MeasurementKind::Ndf => {
                    let ndf = self.measured.downcast_ref::<MeasuredNdfData>().unwrap();
                    let path = filepath.with_extension("vgndf");
                    ndf.write_as_vgndf(&path, &self.timestamp, *disc_res)?;
                },
                MeasurementKind::Sdf => {
                    let sdf = self.measured.downcast_ref::<MeasuredSdfData>().unwrap();
                    let path = filepath.with_extension("vgsdf");
                    // Default SDF binning: ~2° zenith × ~5° azimuth, matching the
                    // typical pmf granularity. For finer control, call
                    // write_as_vgsdf directly with a custom precision.
                    let precision = Sph2::new(rad!(0.0349), rad!(0.0873));
                    sdf.write_as_vgsdf(&path, &self.timestamp, precision)?;
                },
                MeasurementKind::Gaf => {
                    return Err(VgonioError::new(
                        "Vgbsdf output for masking-shadowing (GAF/MSF) is not implemented yet — \
                         its 2D (view × incident) shape needs a multi-layer encoding (see Task 15 \
                         plan note)",
                        None,
                    ));
                },
                _ => {},
            },
        }
        Ok(())
    }

    /// Serialises this measurement in `format` to an in-memory byte buffer
    /// suitable for `ctx.artifacts.publish`, returning the [`ArtifactKind`]
    /// that matches the produced bytes.
    ///
    /// This is the artifact-API counterpart of [`Self::write_to_file`]: it
    /// reuses the same per-(kind, format) writers but lands the bytes in
    /// memory instead of at a CLI-chosen path (the CLI now owns `--output`
    /// placement). VGMO goes straight to a `Cursor<Vec<u8>>` (the codec is
    /// `Write + Seek`); the archival (`.vgbsdf` family) and single-file EXR
    /// writers are path-based (they stage EXR layers through the `exr` crate's
    /// path API), so they round-trip through a `NamedTempFile`, mirroring what
    /// `VgbsdfWriter` already does internally for its EXR members.
    ///
    /// Unsupported combinations are rejected with a clear error rather than a
    /// panic, preserving today's [`Self::write_to_file`] behaviour: BSDF + EXR
    /// is multi-file (use `--output-format vgbsdf`); GAF + EXR and GAF +
    /// Vgbsdf are not implemented.
    pub fn write_artifact_bytes(
        &self,
        format: &OutputFileFormatOption,
    ) -> Result<(ArtifactKind, Bytes), VgonioError> {
        match format {
            OutputFileFormatOption::Vgmo {
                encoding,
                compression,
            } => {
                let timestamp = {
                    let mut timestamp = [0_u8; 32];
                    timestamp.copy_from_slice(
                        vgn_core::utils::iso_timestamp_from_datetime(&self.timestamp).as_bytes(),
                    );
                    timestamp
                };
                let header = Header {
                    meta: HeaderMeta {
                        version: Version::new(0, 1, 0),
                        timestamp,
                        length: 0,
                        sample_size: 4,
                        encoding: *encoding,
                        compression: *compression,
                    },
                    extra: vgmo_header_ext_from_data(&*self.measured),
                };
                let mut writer = BufWriter::new(std::io::Cursor::new(Vec::<u8>::new()));
                crate::io::vgmo::write(&mut writer, header, &*self.measured).map_err(|err| {
                    VgonioError::from_write_file_error(
                        WriteFileError {
                            path: PathBuf::from("<vgmo artifact>").into_boxed_path(),
                            kind: err,
                        },
                        "Failed to write VGMO artifact.",
                    )
                })?;
                let cursor = writer.into_inner().map_err(|e| {
                    VgonioError::new(format!("Failed to flush VGMO artifact: {e}"), None)
                })?;
                Ok((ArtifactKind::Vgmo, Bytes::from(cursor.into_inner())))
            },
            OutputFileFormatOption::Exr { resolution } => {
                match self.measured.kind() {
                    MeasurementKind::Bsdf => {
                        return Err(VgonioError::new(
                            "BSDF + EXR output is not supported via the artifact path; use \
                             `--output-format vgbsdf` instead (it packages all per-level EXRs).",
                            None,
                        ))
                    },
                    MeasurementKind::Gaf => {
                        return Err(VgonioError::new(
                            "GAF/MSF EXR output is not implemented yet (deferred ARCH Task 15 MSF \
                             half).",
                            None,
                        ))
                    },
                    _ => {},
                }
                let mut cursor = Cursor::new(Vec::<u8>::new());
                match self.measured.kind() {
                    MeasurementKind::Ndf => {
                        let ndf = self.measured.downcast_ref::<MeasuredNdfData>().unwrap();
                        ndf.write_as_exr_to(&mut cursor, &self.timestamp, *resolution)?;
                    },
                    MeasurementKind::Sdf => {
                        let sdf = self.measured.downcast_ref::<MeasuredSdfData>().unwrap();
                        sdf.write_histogram_as_exr_to(&mut cursor, &self.timestamp, *resolution)?;
                    },
                    _ => unreachable!("BSDF/GAF rejected above"),
                }
                Ok((ArtifactKind::Exr, Bytes::from(cursor.into_inner())))
            },
            OutputFileFormatOption::Vgbsdf { disc_res } => {
                let kind = match self.measured.kind() {
                    MeasurementKind::Bsdf => ArtifactKind::Vgbsdf,
                    MeasurementKind::Ndf => ArtifactKind::Vgndf,
                    MeasurementKind::Sdf => ArtifactKind::Vgsdf,
                    MeasurementKind::Gaf => {
                        return Err(VgonioError::new(
                            "Vgbsdf output for masking-shadowing (GAF/MSF) is not implemented yet \
                             — its 2D (view × incident) shape needs a multi-layer encoding \
                             (deferred ARCH Task 15 MSF half).",
                            None,
                        ))
                    },
                    _ => {
                        return Err(VgonioError::new(
                            "Unsupported measurement kind for vgbsdf output.",
                            None,
                        ))
                    },
                };
                let cursor = match self.measured.kind() {
                    MeasurementKind::Bsdf => {
                        let bsdf = self.measured.downcast_ref::<BsdfMeasurement>().unwrap();
                        bsdf.write_as_vgbsdf_to(
                            Cursor::new(Vec::<u8>::new()),
                            &self.timestamp,
                            *disc_res,
                        )?
                    },
                    MeasurementKind::Ndf => {
                        let ndf = self.measured.downcast_ref::<MeasuredNdfData>().unwrap();
                        ndf.write_as_vgndf_to(
                            Cursor::new(Vec::<u8>::new()),
                            &self.timestamp,
                            *disc_res,
                        )?
                    },
                    MeasurementKind::Sdf => {
                        let sdf = self.measured.downcast_ref::<MeasuredSdfData>().unwrap();
                        // Default SDF binning: ~2° zenith × ~5° azimuth, matching
                        // `write_to_file`'s vgsdf arm.
                        let precision = Sph2::new(rad!(0.0349), rad!(0.0873));
                        sdf.write_as_vgsdf_to(
                            Cursor::new(Vec::<u8>::new()),
                            &self.timestamp,
                            precision,
                        )?
                    },
                    _ => unreachable!("rejected above"),
                };
                Ok((kind, Bytes::from(cursor.into_inner())))
            },
        }
    }

    /// Loads the measurement data from a file.
    pub fn read_from_file(filepath: &Path) -> Result<Self, VgonioError> {
        // The per-format load banners below report through `log::info!`
        // rather than `vgn_core::cli`: this module moves into a capability
        // crate in Phase 2 where `vgn_core::cli` is unreachable, and the sole
        // caller (the surface cache) has no `JobContext` to thread a phase
        // through. Orchestration-level progress is reported by the callers
        // that own a phase; this loader stays reporter-clean.
        // TODO: unified file loading
        if filepath.extension().unwrap() == "bsdf" {
            log::info!("Loading RGL BSDF file: {}", filepath.display());
            let filename = filepath.file_stem().unwrap().to_str().unwrap();

            let medium = filename
                .split('_')
                .find_map(MediumId::try_from_name)
                .unwrap_or_else(|| {
                    log::warn!(
                        "Failed to parse medium from filename: {}. Defaulting to AIR.",
                        filename
                    );
                    MediumId::AIR
                });

            let loaded = RglBrdf::load(filepath, medium);
            log::info!("Loaded RGL BSDF file with medium: {:?}", medium);
            return Ok(Measurement {
                name: format!("bsdf_{}", filepath.file_name().unwrap().to_str().unwrap()),
                source: MeasurementSource::Loaded(filepath.to_path_buf()),
                timestamp: Local::now(),
                measured: Box::new(loaded),
            });
        } else if filepath.extension().unwrap() == "binary" {
            log::info!("Loading MERL BSDF file: {}", filepath.display());
            let loaded = MerlBrdf::load(filepath)?;
            return Ok(Measurement {
                name: format!(
                    "merl_bsdf_{}",
                    filepath.file_name().unwrap().to_str().unwrap()
                ),
                source: MeasurementSource::Loaded(filepath.to_path_buf()),
                timestamp: Local::now(),
                measured: Box::new(loaded),
            });
        }

        let file = File::open(filepath).map_err(|err| {
            VgonioError::from_io_error(
                err,
                format!("Failed to open measurement file: {}", filepath.display()),
            )
        })?;
        let mut reader = BufReader::new(file);

        if let Some(extension) = filepath.extension() {
            if extension == OsStr::new("json") {
                log::info!("Loading Clausen BSDF file: {}", filepath.display());
                return ClausenBrdf::load_from_reader(reader).map(|brdf| Measurement {
                    name: format!(
                        "clausen_{}",
                        filepath.file_name().unwrap().to_str().unwrap()
                    ),
                    source: MeasurementSource::Loaded(filepath.to_path_buf()),
                    timestamp: Local::now(),
                    measured: Box::new(brdf),
                });
            }

            if extension == OsStr::new("exr") {
                log::info!("Loading Yan18 BSDF file: {}", filepath.display());
                return Yan18Brdf::load_from_exr(&filepath, MediumId::AIR, MediumId::AL).map(
                    |brdf| Measurement {
                        name: format!(
                            "yan2018_{}",
                            filepath.file_name().unwrap().to_str().unwrap()
                        ),
                        source: MeasurementSource::Loaded(filepath.to_path_buf()),
                        timestamp: Local::now(),
                        measured: Box::new(brdf),
                    },
                );
            }
        }

        log::info!("Loading VGMO BSDF file: {}", filepath.display());
        let header = Header::<VgmoHeaderExt>::read(&mut reader).map_err(|err| {
            VgonioError::from_read_file_error(
                ReadFileError {
                    path: filepath.to_owned().into_boxed_path(),
                    kind: ReadFileErrorKind::Read(err),
                },
                "Failed to read VGMO file.",
            )
        })?;
        log::debug!("Read VGMO file of length: {}", header.meta.length);
        log::debug!("Header: {:?}", header);

        let path = filepath.to_path_buf();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("invalid file stem")
            .to_string();

        let measured = crate::io::vgmo::read(&mut reader, &header).map_err(|err| {
            VgonioError::from_read_file_error(
                ReadFileError {
                    path: filepath.to_owned().into_boxed_path(),
                    kind: err,
                },
                "Failed to read VGMO file.",
            )
        })?;
        let timestamp = DateTime::from(
            DateTime::parse_from_rfc3339(std::str::from_utf8(&header.meta.timestamp).map_err(
                |err| VgonioError::from_utf8_error(err, "Failed to parse timestamp from file."),
            )?)
            .map_err(|err| {
                VgonioError::new(
                    format!(
                        "Failed to parse timestamp from file: {}",
                        filepath.display()
                    ),
                    Some(Box::new(err)),
                )
            })?,
        );

        Ok(Measurement {
            name,
            source: MeasurementSource::Loaded(path),
            timestamp,
            measured,
        })
    }
}

/// Helper structure dealing with the spherical transform related to the
/// acquisition.
pub struct SphericalTransform;

impl SphericalTransform {
    /// Returns the transformation matrix transforming something from the
    /// local coordinate system to the desired position defined by the
    /// spherical coordinates.
    ///
    /// The local coordinate system is a right-handed system with the z-axis
    /// pointing upwards, the x-axis pointing to the right and the y-axis
    /// pointing forward.
    ///
    /// # Arguments
    ///
    /// * `dest` - The desired position in spherical coordinates; it should always be defined in the
    ///   unit sphere.
    ///
    /// # Returns
    ///
    /// The transformation matrix.
    pub fn transform_to(dest: Sph2) -> Mat3 {
        Mat3::from_axis_angle(Vec3::Z, dest.phi.value())
            * Mat3::from_axis_angle(Vec3::Y, dest.theta.value())
    }

    /// Returns the transformation matrix transforming the spherical cap shape
    /// or samples to the desired position defined by the spherical
    /// coordinates.
    ///
    /// See [`uniform_sampling_on_unit_sphere`] for more information about the
    /// spherical cap samples.
    ///
    /// # Arguments
    ///
    /// * `dest` - The desired position in spherical coordinates; should always be defined in the
    ///   unit sphere.
    /// * `orbit_radius` - The radius of the orbit about which the samples are rotating around.
    pub fn transform_cap(dest: Sph2, orbit_radius: f32) -> Mat3 {
        Self::transform_to(dest) * Mat3::from_diagonal(Vec3::splat(orbit_radius))
    }

    /// Returns the transformation matrix transforming the disk shape or samples
    /// to the desired position defined by the spherical coordinates.
    ///
    /// See [`uniform_sampling_on_unit_disk`] for more information about the
    /// disk samples.
    ///
    /// # Arguments
    ///
    /// * `dest` - The desired position in spherical coordinates; should always be defined in the
    ///   unit sphere.
    /// * `disk_radius` - The radius of the disk on which the samples are distributed.
    /// * `orbit_radius` - The radius of the orbit about which the samples are rotating around.
    pub fn transform_disc(dest: Sph2, disk_radius: f32, orbit_radius: f32) -> Mat3 {
        Self::transform_to(dest)
            * Mat3::from_diagonal(Vec3::new(disk_radius, disk_radius, orbit_radius))
    }
}

/// Generates uniformly distributed samples on the unit disk.
///
/// NOTE: samples are generated on the xy-plane, with the z component set to 1.0
/// in the returned vector. This could simplify the transformation of the
/// samples when rotating the disk.
pub fn uniform_sampling_on_unit_disk(samples: &mut [Vec3], tstart: Radians, tstop: Radians) {
    const SEED: u64 = 0;

    let range: Uniform<f32> = Uniform::new(0.0, 1.0);
    samples
        .par_chunks_mut(8192)
        .enumerate()
        .for_each(|(i, chunks)| {
            let mut rng = ChaCha8Rng::seed_from_u64(SEED);
            rng.set_stream(i as u64);

            chunks.iter_mut().for_each(|v| {
                let r = range.sample(&mut rng).sqrt();
                let a = range.sample(&mut rng) * (tstop - tstart).as_f32() + tstart.as_f32();
                *v = Vec3::new(r * a.cos(), r * a.sin(), 1.0);
            });
        });
}

/// Generates uniformly distributed samples on the unit sphere.
///
/// TODO: sampling with pdf
///
/// The samples are generated on the unit sphere, in the right-handed,
/// Z-up coordinate system.
///
/// x = cos phi * sin theta
/// y = sin phi * sin theta
/// z = cos theta
pub fn uniform_sampling_on_unit_sphere(
    num: usize,
    theta_start: Radians,
    theta_stop: Radians,
    phi_start: Radians,
    phi_stop: Radians,
) -> Box<[Vec3]> {
    const SEED: u64 = 0;
    let range = Uniform::new(0.0, 1.0);
    let mut samples = Box::new_uninit_slice(num);
    log::trace!("  - Generating samples on unit sphere");

    samples
        .par_chunks_mut(8192)
        .enumerate()
        .for_each(|(i, chunks)| {
            let mut rng = ChaCha8Rng::seed_from_u64(SEED);
            rng.set_stream(i as u64);

            let mut j = 0;
            while j < chunks.len() {
                let phi = rad!(range.sample(&mut rng) * std::f32::consts::TAU);
                let theta = rad!((1.0 - 2.0 * range.sample(&mut rng)).acos());
                if (theta_start..theta_stop).contains(&theta)
                    && (phi_start..phi_stop).contains(&phi)
                {
                    chunks[j].write(Sph3::new(1.0, theta, phi).to_cartesian());
                    j += 1;
                }
            }
        });

    unsafe { samples.assume_init() }
}

/// Estimates the radius of the sphere or hemisphere enclosing the specimen.
#[inline(always)]
pub fn estimate_orbit_radius(mesh: &MicroSurfaceMesh) -> f32 {
    mesh.bounds.max_extent() * std::f32::consts::SQRT_2
}

/// Estimates the radius of the disk with the area covering the specimen.
#[inline(always)]
pub fn estimate_disc_radius(mesh: &MicroSurfaceMesh) -> f32 { mesh.bounds.max_extent() * 0.7 }

/// Trait for the data carried on the hemisphere.
pub trait DataCarriedOnHemisphere: Sized {
    /// The extra data that may be used during sampling.
    type Extra: ?Sized;
    /// Checks if the data is carried by the patches on the hemisphere.
    /// The main reason for this is that the `MeasuredNdfData` may not
    /// necessarily be measured by patches on the hemisphere.
    fn is_carried_by_hemisphere(&self) -> bool { true }
    /// Constructs the extra data used during sampling.
    fn new_extra(&self) -> Option<Box<Self::Extra>> { None }
}

impl DataCarriedOnHemisphere for VgonioBrdf {
    type Extra = ();
}

impl DataCarriedOnHemisphere for MeasuredNdfData {
    type Extra = SphericalPartition;

    fn is_carried_by_hemisphere(&self) -> bool { !self.params.mode.is_by_points() }

    fn new_extra(&self) -> Option<Box<SphericalPartition>> {
        match self.params.mode {
            NdfMeasurementMode::ByPoints { .. } => None,
            NdfMeasurementMode::ByPartition { precision } => Some(Box::new(
                SphericalPartition::new_beckers(SphericalDomain::Upper, precision),
            )),
        }
    }
}

/// Structure for sampling the data carried on the hemisphere.
pub struct DataCarriedOnHemisphereSampler<'a, D>
where
    D: DataCarriedOnHemisphere,
{
    /// The data carried on the hemisphere to sample from.
    pub data: &'a D,
    /// Potential extra data used during sampling.
    pub extra: Option<Box<D::Extra>>,
}

impl<'a, D> DataCarriedOnHemisphereSampler<'a, D>
where
    D: DataCarriedOnHemisphere,
{
    /// Creates a new sampler for the data carried on the hemisphere.
    pub fn new(data: &'a D) -> Option<Self> {
        if data.is_carried_by_hemisphere() {
            let extra = data.new_extra();
            Some(Self { data, extra })
        } else {
            None
        }
    }
}

// TODO: extract the common code between the VgonioBrdf and MeasuredNdfData

impl<'a> DataCarriedOnHemisphereSampler<'a, VgonioBrdf> {
    /// Retrieve the BSDF sample data at the given position;
    ///
    /// The position is given in the unit spherical coordinates. The returned
    /// data is the BSDF values for each snapshot and each wavelength at the
    /// given position.
    ///
    /// # Arguments
    ///
    /// * `wi` - The incident direction; it must exist in the BSDF snapshots.
    /// * `wo` - The outgoing direction; if it doesn't exist in the BSDF snapshots, it will be
    ///   interpolated.
    /// * `out` - The interpolated BSDF values at the given position; it should be pre-allocated,
    ///   and the number of elements should NOT exceed the number of wavelengths in the BSDF data;
    ///   panics in case it's empty.
    pub fn sample_point_at(&self, wi: Sph2, wo: Sph2, out: &mut [f32]) {
        log::trace!(
            "Sampling at wi: ({}, {}), wo: ({} {})",
            wi.theta.to_degrees().prettified(),
            wi.phi.to_degrees().prettified(),
            wo.theta.to_degrees().prettified(),
            wo.phi.to_degrees().prettified()
        );
        let n_spectrum = self.data.n_spectrum();
        assert!(
            !out.is_empty() && out.len() <= n_spectrum,
            "The output buffer should be pre-allocated and have at least one wavelength."
        );
        let snapshot_idx = self
            .data
            .params
            .incoming
            .as_slice()
            .iter()
            .position(|snap| wi.approx_eq(snap))
            .expect(
                "The incident direction is not found in the BSDF snapshots. The incident \
                 direction must be one of the directions of the emitter.",
            );
        log::trace!("  - Found snapshot at wi: {}", wi);
        let n_wo = self.data.params.n_wo();
        let snapshot_samples = &self.data.samples.as_slice()
            [snapshot_idx * n_wo * n_spectrum..(snapshot_idx + 1) * n_wo * n_spectrum];
        let partition = &self.data.params.outgoing;
        // 1. Find the upper and lower ring where the position is located.
        // The Upper ring is the ring with the smallest zenith angle.
        let (upper_ring_idx, lower_ring_idx) = partition.find_rings(wo);
        log::trace!(
            "  - Upper ring: {}, Lower ring: {}",
            upper_ring_idx,
            lower_ring_idx
        );
        // 2. Find the patch where the position is located inside the ring.
        if lower_ring_idx == 0 || lower_ring_idx == 1 {
            // Interpolate inside a triangle.
            let lower_ring = partition.rings[1];
            let patch_idx = {
                let patch_idx = lower_ring.find_patch_indices(wo.phi);
                (
                    0,
                    lower_ring.base_index + patch_idx.0,
                    lower_ring.base_index + patch_idx.1,
                )
            };
            let center = (
                partition.patches[patch_idx.0].center(),
                partition.patches[patch_idx.1].center(),
                partition.patches[patch_idx.2].center(),
            );
            let (u, v, w) = math::projected_barycentric_coords(
                wo.to_cartesian(),
                center.0.to_cartesian(),
                center.1.to_cartesian(),
                center.2.to_cartesian(),
            );
            let patch0_samples =
                &snapshot_samples[patch_idx.0 * n_spectrum..(patch_idx.0 + 1) * n_spectrum];
            let patch1_samples =
                &snapshot_samples[patch_idx.1 * n_spectrum..(patch_idx.1 + 1) * n_spectrum];
            let patch2_samples =
                &snapshot_samples[patch_idx.2 * n_spectrum..(patch_idx.2 + 1) * n_spectrum];
            log::trace!(
                "  - Interpolating inside a triangle between patches #{} ({}, {} | {:?}), #{} \
                 ({}, {} | {:?}), and #{} ({}, {} | {:?})",
                patch_idx.0,
                center.0,
                center.0.to_cartesian(),
                patch0_samples,
                patch_idx.1,
                center.1,
                center.1.to_cartesian(),
                patch1_samples,
                patch_idx.2,
                center.2,
                center.2.to_cartesian(),
                patch2_samples
            );
            log::trace!("  - Barycentric coordinates: ({}, {}, {})", u, v, w);
            out.iter_mut().enumerate().for_each(|(i, spl)| {
                *spl = u * patch0_samples[i] + v * patch1_samples[i] + w * patch2_samples[i];
            });
        } else if upper_ring_idx == lower_ring_idx && upper_ring_idx == partition.n_rings() - 1 {
            // This should be the last ring.
            // Interpolate between two patches.
            let ring = partition.rings[upper_ring_idx];
            let patch_idx = ring.find_patch_indices(wo.phi);
            let patch0_idx = ring.base_index + patch_idx.0;
            let patch1_idx = ring.base_index + patch_idx.1;
            let patch0 = partition.patches[patch0_idx];
            let patch1 = partition.patches[patch1_idx];
            let center = (patch0.center(), patch1.center());
            let patch0_samples =
                &snapshot_samples[patch0_idx * n_spectrum..(patch0_idx + 1) * n_spectrum];
            let patch1_samples =
                &snapshot_samples[patch1_idx * n_spectrum..(patch1_idx + 1) * n_spectrum];
            log::trace!(
                "  - Interpolating between two patches: #{} ({}, {} | {:?}) and #{} ({}, {} | \
                 {:?}) at ring #{}",
                patch0_idx,
                center.0,
                center.0.to_cartesian(),
                patch0_samples,
                patch1_idx,
                center.1,
                center.1.to_cartesian(),
                patch1_samples,
                upper_ring_idx
            );
            let t = (math::circular_angle_dist(wo.phi, center.0.phi)
                / math::circular_angle_dist(center.1.phi, center.0.phi))
            .clamp(0.0, 1.0);
            out.iter_mut().enumerate().for_each(|(i, spl)| {
                *spl = (1.0 - t) * patch0_samples[i] + t * patch1_samples[i];
            });
        } else {
            // Interpolate inside a quadrilateral.
            let (upper_t, upper_patch_center, upper_patch_idx) = {
                let upper_ring = partition.rings[upper_ring_idx];
                let upper_patch_idx = {
                    let patches = upper_ring.find_patch_indices(wo.phi);
                    (
                        upper_ring.base_index + patches.0,
                        upper_ring.base_index + patches.1,
                    )
                };
                let upper_patch_center = (
                    partition.patches[upper_patch_idx.0].center(),
                    partition.patches[upper_patch_idx.1].center(),
                );
                log::trace!(
                    "        - upper_#{} center: {}",
                    upper_patch_idx.0,
                    upper_patch_center.0
                );
                log::trace!(
                    "        - upper_#{} center: {}",
                    upper_patch_idx.1,
                    upper_patch_center.1
                );
                let upper_t = (math::circular_angle_dist(wo.phi, upper_patch_center.0.phi)
                    / math::circular_angle_dist(
                        upper_patch_center.1.phi,
                        upper_patch_center.0.phi,
                    ))
                .clamp(0.0, 1.0);
                log::trace!("          - upper_t: {}", upper_t);
                (upper_t, upper_patch_center, upper_patch_idx)
            };

            let (lower_t, lower_patch_center, lower_patch_idx) = {
                let lower_ring = partition.rings[lower_ring_idx];
                let lower_patch_idx = {
                    let patches = lower_ring.find_patch_indices(wo.phi);
                    (
                        lower_ring.base_index + patches.0,
                        lower_ring.base_index + patches.1,
                    )
                };
                let lower_patch_center = (
                    partition.patches[lower_patch_idx.0].center(),
                    partition.patches[lower_patch_idx.1].center(),
                );
                log::trace!(
                    "        - lower_#{} center: {}",
                    lower_patch_idx.0,
                    lower_patch_center.0,
                );
                log::trace!(
                    "        - lower_#{} center: {}",
                    lower_patch_idx.1,
                    lower_patch_center.1
                );
                let lower_t = (math::circular_angle_dist(wo.phi, lower_patch_center.0.phi)
                    / math::circular_angle_dist(
                        lower_patch_center.1.phi,
                        lower_patch_center.0.phi,
                    ))
                .clamp(0.0, 1.0);
                log::trace!("          - lower_t: {}", lower_t);
                (lower_t, lower_patch_center, lower_patch_idx)
            };
            let s = (math::circular_angle_dist(wo.theta, upper_patch_center.0.theta)
                / math::circular_angle_dist(
                    lower_patch_center.0.theta,
                    upper_patch_center.0.theta,
                ))
            .clamp(0.0, 1.0);
            // Bilateral interpolation.
            let upper_patch0_samples = &snapshot_samples
                [upper_patch_idx.0 * n_spectrum..(upper_patch_idx.0 + 1) * n_spectrum];
            let upper_patch1_samples = &snapshot_samples
                [upper_patch_idx.1 * n_spectrum..(upper_patch_idx.1 + 1) * n_spectrum];
            let lower_patch0_samples = &snapshot_samples
                [lower_patch_idx.0 * n_spectrum..(lower_patch_idx.0 + 1) * n_spectrum];
            let lower_patch1_samples = &snapshot_samples
                [lower_patch_idx.1 * n_spectrum..(lower_patch_idx.1 + 1) * n_spectrum];
            log::trace!(
                "  - Interpolating inside a quadrilateral between rings #{} (#{} vals {:?}, #{} \
                 vals {:?} | t = {}), and #{} (#{} vals {:?}, #{} vals {:?} | t = {}), v = {}",
                upper_ring_idx,
                upper_patch_idx.0,
                upper_patch0_samples,
                upper_patch_idx.1,
                upper_patch1_samples,
                upper_t,
                lower_ring_idx,
                lower_patch_idx.0,
                lower_patch0_samples,
                lower_patch_idx.1,
                lower_patch1_samples,
                lower_t,
                s
            );
            out.iter_mut().enumerate().for_each(|(i, spl)| {
                let upper_interp =
                    (1.0 - upper_t) * upper_patch0_samples[i] + upper_t * upper_patch1_samples[i];
                let lower_interp =
                    (1.0 - lower_t) * lower_patch0_samples[i] + lower_t * lower_patch1_samples[i];
                *spl = (1.0 - s) * upper_interp + s * lower_interp;
            });
        }
        log::trace!("  - Sampled: {:?}", &out);
    }

    /// Retrieve the BSDF slice data at the given azimuthal angle in radians.
    ///
    /// # Arguments
    ///
    /// * `wi` - The incident direction; it must exist in the BSDF snapshots.
    /// * `phi` - The azimuthal angle in radians of the outgoing direction.
    ///
    /// # Returns
    ///
    /// The BSDF values for each polar angle and each wavelength at the given
    /// azimuthal angle. The returned data is row-major with shape [n_theta,
    /// n_spectrum].
    pub fn sample_slice_at(&self, wi: Sph2, phi: Radians) -> Option<Box<[f32]>> {
        let n_theta = self.data.params.outgoing.n_rings();
        let n_spectrum = self.data.n_spectrum();
        let mut out = vec![0.0; n_theta * n_spectrum].into_boxed_slice();
        for (i, ring) in self.data.params.outgoing.rings.iter().enumerate() {
            let theta_o = ring.zenith_center();
            self.sample_point_at(
                wi,
                Sph2::new(theta_o, phi),
                &mut out[i * n_spectrum..(i + 1) * n_spectrum],
            );
        }
        Some(out)
    }
}

impl<'a> DataCarriedOnHemisphereSampler<'a, MeasuredNdfData> {
    /// Retrieve the NDF sample data at the given position;
    pub fn sample_point_at(&self, wm: Sph2) -> f32 {
        log::trace!("Sampling NDF at wm: {}", wm,);

        let partition = self.extra.as_ref().unwrap();
        // 1. Find the upper and lower ring where the position is located.
        // The Upper ring is the ring with the smallest zenith angle.
        let (upper_ring_idx, lower_ring_idx) = partition.find_rings(wm);
        log::trace!(
            "  - Upper ring: {}, Lower ring: {}",
            upper_ring_idx,
            lower_ring_idx
        );
        let samples = &self.data.samples;
        // 2. Find the patch where the position is located inside the ring.
        if lower_ring_idx == 0 || lower_ring_idx == 1 {
            // Interpolate inside a triangle.
            let lower_ring = partition.rings[1];
            let patch_idx = {
                let patch_idx = lower_ring.find_patch_indices(wm.phi);
                (
                    0,
                    lower_ring.base_index + patch_idx.0,
                    lower_ring.base_index + patch_idx.1,
                )
            };
            let center = (
                partition.patches[patch_idx.0].center(),
                partition.patches[patch_idx.1].center(),
                partition.patches[patch_idx.2].center(),
            );
            let (u, v, w) = math::projected_barycentric_coords(
                wm.to_cartesian(),
                center.0.to_cartesian(),
                center.1.to_cartesian(),
                center.2.to_cartesian(),
            );
            let patch0_sample = samples[patch_idx.0];
            let patch1_sample = samples[patch_idx.1];
            let patch2_sample = samples[patch_idx.2];
            log::trace!(
                "  - Interpolating inside a triangle between patches #{} ({}, {} | {:?}), #{} \
                 ({}, {} | {:?}), and #{} ({}, {} | {:?})",
                patch_idx.0,
                center.0,
                center.0.to_cartesian(),
                patch0_sample,
                patch_idx.1,
                center.1,
                center.1.to_cartesian(),
                patch1_sample,
                patch_idx.2,
                center.2,
                center.2.to_cartesian(),
                patch2_sample
            );
            log::trace!("  - Barycentric coordinates: ({}, {}, {})", u, v, w);

            u * patch0_sample + v * patch1_sample + w * patch2_sample
        } else if upper_ring_idx == lower_ring_idx && upper_ring_idx == partition.n_rings() - 1 {
            // This should be the last ring.
            // Interpolate between two patches.
            let ring = partition.rings[upper_ring_idx];
            let patch_idx = ring.find_patch_indices(wm.phi);
            let patch0_idx = ring.base_index + patch_idx.0;
            let patch1_idx = ring.base_index + patch_idx.1;
            let patch0 = partition.patches[patch0_idx];
            let patch1 = partition.patches[patch1_idx];
            let center = (patch0.center(), patch1.center());
            let patch0_samples = samples[patch0_idx];
            let patch1_samples = samples[patch1_idx];
            log::trace!(
                "  - Interpolating between two patches: #{} ({}, {} | {:?}) and #{} ({}, {} | \
                 {:?}) at ring #{}",
                patch0_idx,
                center.0,
                center.0.to_cartesian(),
                patch0_samples,
                patch1_idx,
                center.1,
                center.1.to_cartesian(),
                patch1_samples,
                upper_ring_idx
            );
            let t = (math::circular_angle_dist(wm.phi, center.0.phi)
                / math::circular_angle_dist(center.1.phi, center.0.phi))
            .clamp(0.0, 1.0);

            (1.0 - t) * patch0_samples + t * patch1_samples
        } else {
            // Interpolate inside a quadrilateral.
            let (upper_t, upper_patch_center, upper_patch_idx) = {
                let upper_ring = partition.rings[upper_ring_idx];
                let upper_patch_idx = {
                    let patches = upper_ring.find_patch_indices(wm.phi);
                    (
                        upper_ring.base_index + patches.0,
                        upper_ring.base_index + patches.1,
                    )
                };
                let upper_patch_center = (
                    partition.patches[upper_patch_idx.0].center(),
                    partition.patches[upper_patch_idx.1].center(),
                );
                log::trace!(
                    "        - upper_#{} center: {}",
                    upper_patch_idx.0,
                    upper_patch_center.0
                );
                log::trace!(
                    "        - upper_#{} center: {}",
                    upper_patch_idx.1,
                    upper_patch_center.1
                );
                let upper_t = (math::circular_angle_dist(wm.phi, upper_patch_center.0.phi)
                    / math::circular_angle_dist(
                        upper_patch_center.1.phi,
                        upper_patch_center.0.phi,
                    ))
                .clamp(0.0, 1.0);
                log::trace!("          - upper_t: {}", upper_t);
                (upper_t, upper_patch_center, upper_patch_idx)
            };

            let (lower_t, lower_patch_center, lower_patch_idx) = {
                let lower_ring = partition.rings[lower_ring_idx];
                let lower_patch_idx = {
                    let patches = lower_ring.find_patch_indices(wm.phi);
                    (
                        lower_ring.base_index + patches.0,
                        lower_ring.base_index + patches.1,
                    )
                };
                let lower_patch_center = (
                    partition.patches[lower_patch_idx.0].center(),
                    partition.patches[lower_patch_idx.1].center(),
                );
                log::trace!(
                    "        - lower_#{} center: {}",
                    lower_patch_idx.0,
                    lower_patch_center.0,
                );
                log::trace!(
                    "        - lower_#{} center: {}",
                    lower_patch_idx.1,
                    lower_patch_center.1
                );
                let lower_t = (math::circular_angle_dist(wm.phi, lower_patch_center.0.phi)
                    / math::circular_angle_dist(
                        lower_patch_center.1.phi,
                        lower_patch_center.0.phi,
                    ))
                .clamp(0.0, 1.0);
                log::trace!("          - lower_t: {}", lower_t);
                (lower_t, lower_patch_center, lower_patch_idx)
            };
            let s = (math::circular_angle_dist(wm.theta, upper_patch_center.0.theta)
                / math::circular_angle_dist(
                    lower_patch_center.0.theta,
                    upper_patch_center.0.theta,
                ))
            .clamp(0.0, 1.0);
            // Bilateral interpolation.
            let upper_patch0_sample = &samples[upper_patch_idx.0];
            let upper_patch1_samples = samples[upper_patch_idx.1];
            let lower_patch0_samples = samples[lower_patch_idx.0];
            let lower_patch1_samples = samples[lower_patch_idx.1];
            log::trace!(
                "  - Interpolating inside a quadrilateral between rings #{} (#{} vals {:?}, #{} \
                 vals {:?} | t = {}), and #{} (#{} vals {:?}, #{} vals {:?} | t = {}), v = {}",
                upper_ring_idx,
                upper_patch_idx.0,
                upper_patch0_sample,
                upper_patch_idx.1,
                upper_patch1_samples,
                upper_t,
                lower_ring_idx,
                lower_patch_idx.0,
                lower_patch0_samples,
                lower_patch_idx.1,
                lower_patch1_samples,
                lower_t,
                s
            );
            let upper_interp =
                (1.0 - upper_t) * upper_patch0_sample + upper_t * upper_patch1_samples;
            let lower_interp =
                (1.0 - lower_t) * lower_patch0_samples + lower_t * lower_patch1_samples;

            (1.0 - s) * upper_interp + s * lower_interp
        }
    }

    /// Retrieve the NDF slice data at the given azimuthal angle in radians.
    pub fn sample_slice_at(&self, phi: Radians) -> Box<[f32]> {
        let partition = self.extra.as_ref().unwrap();
        let n_theta = partition.n_rings();
        let mut out = vec![0.0; n_theta].into_boxed_slice();
        for (i, ring) in partition.rings.iter().enumerate() {
            let theta_o = ring.zenith_center();
            out[i] = self.sample_point_at(Sph2::new(theta_o, phi));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mfd::MeasuredNdfData,
        params::{NdfMeasurementMode, NdfMeasurementParams},
    };
    use vgn_io::vgbsdf::VgbsdfReader;

    /// CLI-dispatch smoke test: `OutputFileFormatOption::Vgbsdf` on an NDF
    /// measurement must produce a `.vgndf` file (extension chosen per kind by
    /// `Measurement::write_to_file`) that re-opens via `VgbsdfReader`.
    ///
    /// This is the only test that exercises the dispatch arm added in
    /// Task 16; the per-kind writers themselves are covered by their own
    /// smoke tests.
    #[test]
    fn write_to_file_vgbsdf_dispatches_ndf_to_vgndf() {
        let params = NdfMeasurementParams {
            mode: NdfMeasurementMode::ByPartition {
                precision: rad!(0.3),
            },
            crop_to_disk: false,
            use_facet_area: false,
        };
        let partition = SphericalPartition::new(
            params.mode.partition_scheme_for_data_collection(),
            SphericalDomain::Upper,
            params.mode.partition_precision_for_data_collection(),
        );
        let samples: Box<[f32]> = (0..partition.n_patches())
            .map(|i| i as f32)
            .collect::<Vec<_>>()
            .into();
        let measured = MeasuredNdfData { params, samples };

        let dir = tempfile::tempdir().unwrap();
        // No extension — write_to_file selects "vgndf" because measured.kind() is Ndf.
        let stem = dir.path().join("ndf_dispatch_test");

        let m = Measurement {
            name: "dispatch-test".into(),
            source: MeasurementSource::Loaded(stem.clone()),
            timestamp: Local::now(),
            measured: Box::new(measured),
        };
        m.write_to_file(&stem, &OutputFileFormatOption::Vgbsdf { disc_res: 64 })
            .unwrap();

        let expected = stem.with_extension("vgndf");
        assert!(
            expected.is_file(),
            "expected {} to exist",
            expected.display()
        );

        let reader = VgbsdfReader::open(&expected).unwrap();
        assert_eq!(reader.manifest.archive.kind, "ndf");
        assert_eq!(reader.manifest.vgonio.conventions, "v1");
    }
}
