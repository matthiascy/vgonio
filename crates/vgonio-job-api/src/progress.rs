//! Structured progress events emitted by capability handlers.
//!
//! [`ProgressEvent`] is the canonical observability channel for a job: every
//! status update, every partial-progress tick, every warning, and the terminal
//! outcome all travel as variants of this one enum. The local executor surfaces
//! events on an in-process channel; the remote executor relays them over the
//! transport as [`ProgressEnvelope`]s (which add the [`JobId`] back, since the
//! event itself carries no identity).
//!
//! # Lifecycle ordering
//!
//! A well-behaved capability handler emits events in this order:
//!
//! ```text
//! Queued     -- enqueued by the executor (optional; some executors skip this)
//!   -> Started  -- worker picked up the job and began execution
//!     -> Progress | Note | Warning  (zero or more, interleaved freely)
//!     -> one terminal event:
//!          Completed   -- normal success
//!          Failed      -- carries a JobError; this is the *canonical* error record
//!          Cancelled   -- client-initiated abort
//! ```
//!
//! Exactly one terminal event is emitted per job. Receivers must tolerate
//! `Started` arriving without a preceding `Queued` (e.g. when the local
//! executor runs synchronously), and must treat any event after the terminal as
//! a protocol violation.
//!
//! # Why `Failed` carries the error
//!
//! Earlier drafts considered emitting `Failed` *and* returning a separate
//! `Result` on a result channel. The plan rejects that split: storing the same
//! error in two places invites disagreement between them on the wire. The
//! progress stream is the single source of truth, so [`JobError`] is nested
//! directly into [`ProgressEvent::Failed`].

use crate::{error::JobError, ids::JobId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// One structured observation about a job's progress.
///
/// Wire shape is internally tagged on the `type` discriminator:
/// `{"type":"queued","at":"..."}`, `{"type":"progress","fraction":0.5,...}`,
/// etc. The discriminator is named `type` rather than `kind` because
/// [`Self::Progress`] already exposes a `kind` field of its own
/// ([`ProgressKind`]); colliding the two would force renaming a documented
/// field. Variant discriminants are `snake_case`. `#[non_exhaustive]` so new
/// event kinds can land without a protocol bump; receivers should accept
/// unknown variants tolerantly (today that means a deserialization error,
/// which Phase 3 will soften via a catch-all wrapper at the transport layer).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProgressEvent {
    /// The executor accepted the envelope and is holding it in the run queue;
    /// no worker has picked it up yet. Optional: synchronous executors skip
    /// straight to [`Self::Started`].
    Queued {
        /// UTC instant the job was enqueued.
        at: DateTime<Utc>,
    },
    /// A worker began executing this job. From here on, [`Self::Progress`] /
    /// [`Self::Note`] / [`Self::Warning`] may arrive freely until a terminal
    /// event.
    Started {
        /// UTC instant execution began.
        at: DateTime<Utc>,
    },
    /// Partial progress update. The receiver decides how to render it (progress
    /// bar, log line, ignored); the capability handler's job is just to emit
    /// these often enough to keep long-running work observable.
    Progress {
        /// Completion ratio in `[0.0, 1.0]`. Values are clamped to that range
        /// by the receiver; handlers should not emit `NaN` (it survives JSON
        /// as `null` and breaks downstream math).
        fraction: f32,
        /// Optional human-readable note attached to this tick (e.g.
        /// `"tracing rays on patch 12/48"`).
        message: Option<String>,
        /// What the `fraction` is measuring against. See [`ProgressKind`].
        kind: ProgressKind,
    },
    /// Free-form informational message with a severity level. Distinct from
    /// [`Self::Warning`] in that no abnormal condition is implied; this is the
    /// channel for things the user wants to see in a log but which do not
    /// signal trouble.
    Note {
        /// Severity hint, conventionally borrowed from syslog: `0` = emerg,
        /// `4` = warning, `6` = info, `7` = debug. Receivers may filter on it
        /// but should not assume any specific scale beyond "higher = noisier".
        level: u8,
        /// The note text.
        message: String,
    },
    /// Non-fatal abnormal condition: the job will continue (or already has)
    /// but the result may need scrutiny. Use [`ProgressEvent::Failed`] for
    /// conditions that abort the job.
    Warning {
        /// What went wrong but was recovered from.
        message: String,
    },
    /// Terminal: job ran to completion successfully. The capability's result
    /// artifacts are referenced separately (see the executor return value /
    /// [`crate::artifact::ArtifactRef`]).
    Completed {
        /// UTC instant execution finished.
        at: DateTime<Utc>,
    },
    /// Terminal: job aborted with an error. The nested [`JobError`] is the
    /// canonical record; do not duplicate it on a side channel.
    Failed {
        /// UTC instant the failure was recorded.
        at: DateTime<Utc>,
        /// Structured error describing what went wrong.
        error: JobError,
    },
    /// Terminal: job was cancelled by client request (or by the executor in
    /// response to a deadline / shutdown).
    Cancelled {
        /// UTC instant the cancellation took effect.
        at: DateTime<Utc>,
    },
}

/// What the [`ProgressEvent::Progress::fraction`] is measuring.
///
/// All three are valid simultaneously: a long-running measurement may emit
/// `Overall` ticks for the whole job and `PerStep` ticks for the current phase.
/// Receivers route on `kind` to pick which progress bar to update.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProgressKind {
    /// `fraction` ranges over the whole job from submission to terminal event.
    /// At most one `Overall` tick per logical step; receivers expect it to be
    /// monotonically non-decreasing.
    Overall,
    /// `fraction` ranges over the current internal step (e.g. one of N
    /// measurement passes). Resets to 0 when the next step begins.
    PerStep,
    /// `fraction` is the ratio of items processed so far in the current step
    /// (e.g. patches measured, residuals fitted). Useful when step boundaries
    /// are coarse and a finer granularity is wanted.
    PerItem,
}

/// Wire wrapper that pairs a [`ProgressEvent`] with the [`JobId`] it belongs to.
///
/// The event itself is identity-less so it can be re-used across an in-process
/// channel where the receiver already knows which job it is listening to.
/// Once events leave the process (remote executor relaying over a transport),
/// they are wrapped in this envelope so the receiver can demux multiple jobs
/// over one connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressEnvelope {
    /// The job this event describes.
    pub job_id: JobId,
    /// The observation itself.
    pub event: ProgressEvent,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{JobError, JobErrorCode};

    fn ts() -> DateTime<Utc> {
        // Fixed instant so JSON expectations are stable across runs.
        DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn assert_roundtrips(event: &ProgressEvent) {
        let json = serde_json::to_string(event).unwrap();
        let back: ProgressEvent = serde_json::from_str(&json).unwrap();
        // `ProgressEvent` is not `PartialEq` (it nests `f32`), so compare the
        // re-serialized form: if both serializations match, the value survived
        // intact at the wire level, which is what the contract promises.
        let json_back = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json_back, "round-trip changed wire bytes");
    }

    #[test]
    fn queued_roundtrips_json() { assert_roundtrips(&ProgressEvent::Queued { at: ts() }); }

    #[test]
    fn started_roundtrips_json() { assert_roundtrips(&ProgressEvent::Started { at: ts() }); }

    #[test]
    fn progress_roundtrips_json() {
        assert_roundtrips(&ProgressEvent::Progress {
            fraction: 0.5,
            message: Some("tracing rays".into()),
            kind: ProgressKind::Overall,
        });
    }

    #[test]
    fn progress_without_message_roundtrips() {
        assert_roundtrips(&ProgressEvent::Progress {
            fraction: 0.25,
            message: None,
            kind: ProgressKind::PerItem,
        });
    }

    #[test]
    fn note_roundtrips_json() {
        assert_roundtrips(&ProgressEvent::Note {
            level: 6,
            message: "loaded heightfield".into(),
        });
    }

    #[test]
    fn warning_roundtrips_json() {
        assert_roundtrips(&ProgressEvent::Warning {
            message: "clamped negative reflectance".into(),
        });
    }

    #[test]
    fn completed_roundtrips_json() { assert_roundtrips(&ProgressEvent::Completed { at: ts() }); }

    #[test]
    fn failed_roundtrips_json() {
        assert_roundtrips(&ProgressEvent::Failed {
            at: ts(),
            error: JobError {
                code: JobErrorCode::HandlerError,
                message: "fit did not converge".into(),
                retriable: false,
                details: Some("residual=1.2e-3".into()),
            },
        });
    }

    #[test]
    fn cancelled_roundtrips_json() { assert_roundtrips(&ProgressEvent::Cancelled { at: ts() }); }

    #[test]
    fn variants_wire_as_snake_case() {
        // Spot-check the internally-tagged + snake_case wire shape so a stray
        // edit to the `#[serde(...)]` attribute on `ProgressEvent` is caught.
        let queued = serde_json::to_value(ProgressEvent::Queued { at: ts() }).unwrap();
        assert_eq!(queued.get("type").and_then(|v| v.as_str()), Some("queued"));
        assert!(queued.get("at").is_some(), "internal tag must keep `at` as a sibling");

        let progress = serde_json::to_value(ProgressEvent::Progress {
            fraction: 0.5,
            message: None,
            kind: ProgressKind::PerStep,
        })
        .unwrap();
        assert_eq!(progress.get("type").and_then(|v| v.as_str()), Some("progress"));
        // The variant's own `kind` field must stay reachable under its own
        // name; this is why the discriminator is `type`, not `kind`.
        assert_eq!(progress.get("kind").and_then(|v| v.as_str()), Some("per_step"));
        assert_eq!(progress.get("fraction").and_then(serde_json::Value::as_f64), Some(0.5));
    }

    #[test]
    fn progress_kind_each_variant_roundtrips() {
        for k in [ProgressKind::Overall, ProgressKind::PerStep, ProgressKind::PerItem] {
            let json = serde_json::to_string(&k).unwrap();
            let back: ProgressKind = serde_json::from_str(&json).unwrap();
            assert_eq!(k, back);
        }
    }

    #[test]
    fn progress_kind_wire_is_snake_case() {
        assert_eq!(serde_json::to_string(&ProgressKind::Overall).unwrap(), "\"overall\"");
        assert_eq!(serde_json::to_string(&ProgressKind::PerStep).unwrap(), "\"per_step\"");
        assert_eq!(serde_json::to_string(&ProgressKind::PerItem).unwrap(), "\"per_item\"");
    }

    #[test]
    fn envelope_roundtrips_json() {
        let env = ProgressEnvelope {
            job_id: JobId::new(),
            event: ProgressEvent::Started { at: ts() },
        };
        let json = serde_json::to_string(&env).unwrap();
        let back: ProgressEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(env.job_id, back.job_id);
        // Compare the re-serialized event (see `assert_roundtrips` for why).
        assert_eq!(
            serde_json::to_string(&env.event).unwrap(),
            serde_json::to_string(&back.event).unwrap(),
        );
    }

    #[test]
    fn envelope_carries_job_id_alongside_event() {
        // The envelope's reason for existing is to add identity that the bare
        // event doesn't carry; lock that into the wire shape.
        let env = ProgressEnvelope {
            job_id: JobId::new(),
            event: ProgressEvent::Completed { at: ts() },
        };
        let v = serde_json::to_value(&env).unwrap();
        assert!(v.get("job_id").is_some(), "envelope must expose job_id");
        assert!(v.get("event").is_some(), "envelope must expose event");
    }
}
