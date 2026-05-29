//! Wire error DTO and error codes.
//!
//! [`JobError`] is the structured failure record that capability handlers
//! produce when a job cannot complete. It is nested directly into
//! [`crate::progress::Lifecycle::Failed`] (the canonical channel), and is
//! never duplicated on a side result channel; see the progress-module docs for
//! the rationale.
//!
//! # Wire shape
//!
//! `JobError` is a struct with `snake_case` field names (Rust-default; no
//! `rename_all` needed); [`JobErrorCode`] is a `snake_case` external string
//! enum tag, so adding a new variant only requires bumping the crate's
//! [`crate::PROTOCOL_VERSION`] if existing consumers cannot tolerate the
//! unknown discriminant. `#[non_exhaustive]` documents the intent to add
//! variants forward-compatibly.
//!
//! # Composing an error
//!
//! ```
//! use vgn_job_api::error::{JobError, JobErrorCode};
//!
//! let err = JobError {
//!     code: JobErrorCode::HandlerError,
//!     message: "fit did not converge".to_string(),
//!     retriable: false,
//!     details: Some("residual=1.2e-3 after 200 iters".to_string()),
//! };
//! assert!(!err.retriable);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Structured failure record returned by a capability handler.
///
/// `JobError` is `Clone` + `Serialize` + `Deserialize` so it can be (a) nested
/// into [`crate::progress::Lifecycle::Failed`], (b) shipped over the
/// transport, and (c) handed back to the original submitter for inspection.
///
/// It also implements [`std::error::Error`] via `thiserror`, so handlers may
/// return it through `?` chains; the [`std::fmt::Display`] form is
/// `"<Code>: <message>"` (e.g. `"HandlerError: fit did not converge"`).
#[derive(Debug, Clone, Serialize, Deserialize, Error)]
#[error("{code:?}: {message}")]
pub struct JobError {
    /// Machine-readable category. Drives retry decisions and metrics
    /// bucketing; do not encode prose into the code, that goes in `message`.
    pub code: JobErrorCode,
    /// Short, human-readable summary of what went wrong. One line; suitable
    /// for a log message or a UI banner.
    pub message: String,
    /// Whether re-submitting the same logical request (same
    /// [`crate::ids::IdempotencyKey`]) has a non-trivial chance of succeeding.
    ///
    /// The capability handler decides this, not the transport: transient
    /// transport errors set it to `true` automatically, but
    /// [`JobErrorCode::HandlerError`] (math non-convergence, bad input) almost
    /// always sets it to `false` because retrying with the same inputs will
    /// hit the same failure.
    pub retriable: bool,
    /// Optional long-form context (stack snippet, parameter dump, upstream
    /// error chain). `None` when there is nothing useful to add beyond
    /// `message`. Free-form; not parsed.
    pub details: Option<String>,
}

/// Machine-readable failure category.
///
/// Codes are stable wire identifiers; do not rename a variant once it has
/// shipped. New variants can be added (the enum is `#[non_exhaustive]`); when
/// in doubt, prefer reusing an existing code and putting specifics in
/// [`JobError::details`] over minting a new one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum JobErrorCode {
    /// Capability handler returned a logic error: bad input, math failure
    /// (non-convergence, divide-by-zero), assertion violation. Typically
    /// not retriable with the same inputs.
    HandlerError,
    /// I/O error reading inputs or writing outputs (disk full, permission
    /// denied, artifact store unreachable). Retriability depends on the
    /// underlying cause and is set per-error.
    Io,
    /// Job was cancelled by the client (explicit cancel, deadline elapsed
    /// before pickup, transport disconnect during run). Never retriable
    /// automatically; the client decides whether to re-submit.
    Cancelled,
    /// Job exceeded its deadline. Distinct from [`Self::Cancelled`] in that
    /// the executor (not the client) made the decision. Sometimes retriable
    /// with a longer deadline.
    Timeout,
    /// Wire protocol mismatch: envelope `protocol_version` disagrees with the
    /// worker's, or a payload field failed to deserialize. Never retriable
    /// without changing the producer.
    ProtocolMismatch,
    /// Capability not supported by this worker: no
    /// [`crate::handshake::CapabilityDescriptor`] for the envelope's
    /// `capability_id`. Retriable on a different worker.
    UnsupportedCapability,
    /// Required feature not available on this worker: one of
    /// `envelope.resources.required_features` is absent from the worker's
    /// `enabled_features`. Retriable on a different worker.
    MissingFeature,
    /// Transient transport error (connection reset, partial body, retryable
    /// HTTP status). Almost always retriable.
    Transport,
    /// Generic catch-all; check `details`. Prefer a specific code where
    /// possible.
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> JobError {
        JobError {
            code: JobErrorCode::HandlerError,
            message: "fit did not converge".into(),
            retriable: false,
            details: Some("residual=1.2e-3".into()),
        }
    }

    #[test]
    fn job_error_roundtrips_json() {
        let e = sample();
        let json = serde_json::to_string(&e).unwrap();
        let back: JobError = serde_json::from_str(&json).unwrap();
        assert_eq!(e.code, back.code);
        assert_eq!(e.message, back.message);
        assert_eq!(e.retriable, back.retriable);
        assert_eq!(e.details, back.details);
    }

    #[test]
    fn job_error_with_none_details_roundtrips() {
        let e = JobError {
            code: JobErrorCode::Timeout,
            message: "deadline elapsed".into(),
            retriable: true,
            details: None,
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: JobError = serde_json::from_str(&json).unwrap();
        assert!(back.details.is_none());
        assert!(back.retriable);
    }

    #[test]
    fn job_error_display_uses_code_debug_and_message() {
        // The `#[error("{code:?}: {message}")]` attribute renders the variant
        // name (Debug), *not* the snake_case wire form. Lock that into a test
        // so a future `rename` of the attribute doesn't silently change the
        // log format.
        let e = sample();
        assert_eq!(e.to_string(), "HandlerError: fit did not converge");
    }

    #[test]
    fn job_error_implements_std_error() {
        // Compile-time check that `JobError: std::error::Error` so handlers
        // can use it through `?`.
        fn assert_is_error<E: std::error::Error>(_: &E) {}
        assert_is_error(&sample());
    }

    #[test]
    fn every_job_error_code_roundtrips() {
        // Walk every variant by hand: an exhaustive `match` would defeat the
        // `#[non_exhaustive]` linting on the *crate* boundary, but we own this
        // enum inside the crate so listing them is fine.
        for code in [
            JobErrorCode::HandlerError,
            JobErrorCode::Io,
            JobErrorCode::Cancelled,
            JobErrorCode::Timeout,
            JobErrorCode::ProtocolMismatch,
            JobErrorCode::UnsupportedCapability,
            JobErrorCode::MissingFeature,
            JobErrorCode::Transport,
            JobErrorCode::Other,
        ] {
            let json = serde_json::to_string(&code).unwrap();
            let back: JobErrorCode = serde_json::from_str(&json).unwrap();
            assert_eq!(code, back, "round-trip changed value for {code:?}");
        }
    }

    #[test]
    fn job_error_code_wire_is_snake_case() {
        // Multi-word variants are the easy ones to fat-finger.
        assert_eq!(
            serde_json::to_string(&JobErrorCode::HandlerError).unwrap(),
            "\"handler_error\""
        );
        assert_eq!(
            serde_json::to_string(&JobErrorCode::ProtocolMismatch).unwrap(),
            "\"protocol_mismatch\""
        );
        assert_eq!(
            serde_json::to_string(&JobErrorCode::UnsupportedCapability).unwrap(),
            "\"unsupported_capability\""
        );
        assert_eq!(
            serde_json::to_string(&JobErrorCode::MissingFeature).unwrap(),
            "\"missing_feature\""
        );
    }

    #[test]
    fn job_error_wire_field_names_are_snake_case() {
        // Sanity: serde uses Rust field names by default, and they are already
        // snake_case here. Lock it in so a future `rename_all = "camelCase"`
        // edit on the struct gets caught.
        let v = serde_json::to_value(sample()).unwrap();
        for key in ["code", "message", "retriable", "details"] {
            assert!(v.get(key).is_some(), "expected field `{key}` on the wire");
        }
    }
}
