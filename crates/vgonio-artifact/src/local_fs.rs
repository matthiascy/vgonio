//! Local filesystem artifact store.
//!
//! Files are content-addressed: an artifact with sha256 `abcd..ef` lives at
//! `{root}/objects/ab/cdef...`. Publish is atomic via temp-file + rename.

use std::{fs, path::PathBuf, sync::Arc};

use sha2::{Digest, Sha256};
use uuid::Uuid;
use vgn_job_api::{
    artifact::{ArtifactKind, ArtifactOrigin, ArtifactRef, Checksum},
    context::{ArtifactHandle, ArtifactStore},
    error::{JobError, JobErrorCode},
    ids::ArtifactId,
};

#[derive(Debug)]
pub struct LocalFsStore {
    root: PathBuf,
}

impl LocalFsStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Arc<Self>, std::io::Error> {
        let root = root.into();
        fs::create_dir_all(root.join("objects"))?;
        Ok(Arc::new(Self { root }))
    }

    pub fn object_path(&self, checksum: &str) -> PathBuf {
        let (prefix, rest) = checksum.split_at(2);
        self.root.join("objects").join(prefix).join(rest)
    }
}

impl ArtifactStore for LocalFsStore {
    fn resolve(&self, r: &ArtifactRef) -> Result<ArtifactHandle, JobError> {
        let hex_hash = match &r.checksum {
            Checksum::Sha256(h) => h,
            &_ => unimplemented!("unsupported checksum algorithm"),
        };
        let path = self.object_path(hex_hash);
        if !path.exists() {
            return Err(JobError {
                code: JobErrorCode::Io,
                message: format!("artifact not found {hex_hash}"),
                retriable: false,
                details: Some(path.display().to_string()),
            });
        }
        Ok(ArtifactHandle::Path(path))
    }

    fn publish(&self, kind: ArtifactKind, bytes: bytes::Bytes) -> Result<ArtifactRef, JobError> {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hex_hash = hex::encode(hasher.finalize());

        let target = self.object_path(&hex_hash);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(io_to_job_error)?;
        }

        if !target.exists() {
            // Atomic publish via temp file + rename. If the target already exists, we can skip this
            // step since the content is identical.
            // Append uuid to avoid two publishers both write to the same tmp name.
            let uuid = Uuid::new_v4();
            let tmp = target.with_extension(format!("tmp.{}", uuid));
            {
                let mut f = fs::File::create(&tmp).map_err(io_to_job_error)?;
                std::io::Write::write_all(&mut f, &bytes).map_err(io_to_job_error)?;
                f.sync_all().map_err(io_to_job_error)?;
            }
            fs::rename(&tmp, &target).map_err(io_to_job_error)?;
        }

        Ok(ArtifactRef {
            id: ArtifactId::new(),
            kind,
            origin: ArtifactOrigin::LocalPath(target),
            checksum: Checksum::sha256_hex(hex_hash),
            size_bytes: bytes.len() as u64,
            // The store routes by id and has no naming context; producers set
            // a display_name hint on the returned ref if they want one.
            display_name: None,
        })
    }
}

fn io_to_job_error(e: std::io::Error) -> JobError {
    JobError {
        code: JobErrorCode::Io,
        message: e.to_string(),
        retriable: matches!(
            e.kind(),
            std::io::ErrorKind::Interrupted | std::io::ErrorKind::TimedOut
        ),
        details: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use tempfile::tempdir;

    #[test]
    fn roundtrip() {
        let dir = tempdir().unwrap();
        let store = LocalFsStore::new(dir.path()).unwrap();
        let r = store
            .publish(ArtifactKind::Vgbsdf, Bytes::from_static(b"hello"))
            .unwrap();
        let h = store.resolve(&r).unwrap();
        match h {
            ArtifactHandle::Path(p) => {
                let bytes = std::fs::read(p).unwrap();
                assert_eq!(bytes, b"hello");
            },
            _ => panic!("unexpected handle variant"),
        }
    }

    #[test]
    fn dedup_on_same_content() {
        let dir = tempdir().unwrap();
        let store = LocalFsStore::new(dir.path()).unwrap();
        let a = store
            .publish(ArtifactKind::Vgbsdf, Bytes::from_static(b"same"))
            .unwrap();
        let b = store
            .publish(ArtifactKind::Vgbsdf, Bytes::from_static(b"same"))
            .unwrap();
        assert_eq!(a.checksum, b.checksum);
        assert_eq!(a.size_bytes, b.size_bytes);
    }
}
