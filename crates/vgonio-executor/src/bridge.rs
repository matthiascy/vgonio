//! Renders a job's [`JobEvent`] stream into the existing CLI reporter.
//!
//! Capability handlers emit structured [`Activity`] events through
//! `ctx.progress` instead of calling the `cli_*!` macros directly;
//! [`LocalStatusBridge`] turns those events (plus the executor-owned
//! [`Lifecycle`] bracket) back into CLI lines, so end-user-visible output
//! stays close to what users see today once the executor seam is in
//! place. The same bridge type will eventually back the remote-executor
//! path too; only the event source changes, not the rendering.
//!
//! # Layering
//!
//! The bridge is a *consumer* of the executor's event channel; it owns no
//! threads of its own. Typical usage:
//!
//! ```no_run
//! use vgn_executor::{CapabilityRegistry, Executor, LocalExecutor, LocalStatusBridge};
//! # use vgn_job_api::envelope::JobEnvelope;
//! # fn build_envelope() -> JobEnvelope { unimplemented!() }
//!
//! let executor = LocalExecutor::from_cache_dir(CapabilityRegistry::new(), "/tmp/cache").unwrap();
//! let handle = executor.submit(build_envelope()).unwrap();
//!
//! // Block this thread until the job's terminal event arrives.
//! let bridge = LocalStatusBridge::new();
//! bridge.run(handle.events);
//!
//! // After the event stream closes, fetch the typed result.
//! let _outcome = handle.result.recv().unwrap();
//! ```
//!
//! # Phase indent
//!
//! Phases nest, and the bridge tracks open phases so each
//! [`Activity::PhaseBegin`]'s indent reflects its depth in the tree:
//! a top-level phase renders at `indent_base`, a phase nested inside it
//! renders at `indent_base + 2`, and so on. [`Activity::Message`] and
//! [`Activity::Progress`] within a phase render at one extra level of
//! indent so they sit visually under the phase heading.
//!
//! # Pitfalls noted in the plan
//!
//! - **Default-verbosity noise.** Lifecycle `Started` / `Queued` are surfaced only at verbosity ≥ 1
//!   so the default-verbose CLI output stays close to what users see today.
//! - **Indent inheritance.** Capability handlers don't know which indent level the caller wants;
//!   the bridge fixes a base indent at construction and uses it consistently. Multi-level
//!   orchestrations construct a bridge with the right base before submitting nested jobs.

use std::{collections::HashMap, sync::mpsc, time::Duration};

use vgn_core::cli::{self, format_duration, Indent};
use vgn_job_api::progress::{
    Activity, JobEvent, Lifecycle, PhaseInstanceId, PhaseOutcome, ProgressUnits,
};

/// Consumes a [`JobEvent`] receiver and renders each event to the
/// process-wide CLI reporter ([`vgn_core::cli`]).
///
/// Stateless apart from the configured `indent_base`. The same bridge
/// value can render any number of jobs sequentially, but its
/// [`Self::run`] method takes the receiver by value so each call drains
/// exactly one stream. Per-stream state (open-phase indent tracking)
/// lives in a local map inside `run`, so two bridges driven concurrently
/// from different threads cannot trample each other's state.
pub struct LocalStatusBridge {
    /// Spaces of indentation prepended to top-level rendered lines.
    /// Nested phases add `2 * depth` extra columns; sub-lines under a
    /// phase add another two.
    indent_base: u32,
}

impl Default for LocalStatusBridge {
    fn default() -> Self { Self::new() }
}

impl LocalStatusBridge {
    /// Builds a bridge rendering at [`Indent::SECTION`] (two spaces). This is
    /// the right level for top-level job output under the CLI's section header.
    pub fn new() -> Self {
        Self {
            indent_base: Indent::SECTION.as_u32(),
        }
    }

    /// Builds a bridge at a caller-chosen base indent.
    ///
    /// Use for nested orchestrations: the parent job runs at indent `N`, each
    /// child sub-job's bridge should sit at `N + 2` (one [`Indent::nest`]
    /// step) so the visual hierarchy mirrors the call graph.
    pub fn with_indent(indent: Indent) -> Self {
        Self {
            indent_base: indent.as_u32(),
        }
    }

    /// Drains the receiver, rendering each event, and returns once the
    /// channel closes (the executor drops its send side after the terminal
    /// event).
    ///
    /// Blocks the calling thread. If the caller wants the bridge to run
    /// alongside other work, spawn it: the receiver is [`Send`], and
    /// [`LocalStatusBridge`] holds no thread-bound state.
    pub fn run(self, events: mpsc::Receiver<JobEvent>) {
        let mut open: HashMap<PhaseInstanceId, u32> = HashMap::new();
        for ev in events.iter() {
            self.render(&mut open, &ev);
        }
    }

    /// Maps one event to the equivalent CLI emission. Split out so tests can
    /// drive it directly without standing up a channel.
    ///
    /// `open` tracks the depth (0 = top-level) of every phase currently
    /// open in this stream; `PhaseBegin` inserts, `PhaseEnd` removes.
    fn render(&self, open: &mut HashMap<PhaseInstanceId, u32>, ev: &JobEvent) {
        match ev {
            JobEvent::Lifecycle(ev) => self.render_lifecycle(open, ev),
            JobEvent::Activity(ev) => self.render_activity(open, ev),
            // `JobEvent` is `#[non_exhaustive]`; surface unknowns to the
            // diag log rather than panic.
            other => log::debug!("LocalStatusBridge: unknown job event: {other:?}"),
        }
    }

    fn render_lifecycle(&self, open: &mut HashMap<PhaseInstanceId, u32>, ev: &Lifecycle) {
        let base = self.indent_base;
        match ev {
            Lifecycle::Queued { .. } => {
                cli::step_v(1, base, format_args!("queued"));
            },
            Lifecycle::Started { .. } => {
                cli::step_v(1, base, format_args!("started"));
            },
            Lifecycle::Completed { .. } => {
                cli::success(base, format_args!("completed"));
            },
            Lifecycle::Failed { error, .. } => {
                // The progress protocol requires receivers to reconcile any
                // phase still open at a terminal lifecycle event. A handler
                // that bubbles an error via `Result::Err` does not emit a
                // closing `PhaseEnd`, so close the dangling phases here
                // (deepest first) before the job-level error line.
                self.reconcile_open_phases(open, true);
                cli::error(base, format_args!("{}", error.message));
            },
            Lifecycle::Cancelled { .. } => {
                self.reconcile_open_phases(open, false);
                cli::warning(base, format_args!("cancelled"));
            },
            // `Lifecycle` is `#[non_exhaustive]`. A new variant from a
            // newer producer must not silently disappear; surface it
            // through the diagnostic log channel.
            ev => log::debug!("LocalStatusBridge: unhandled lifecycle event: {ev:?}"),
        }
    }

    /// Closes every phase left open at a terminal lifecycle event, rendering
    /// a short marker per phase so the user sees they did not complete. The
    /// terminal error / cancellation message is rendered separately by the
    /// caller, so the per-phase markers stay terse to avoid repeating it.
    /// Phases are closed innermost-first (deepest indent first) for readable
    /// nesting, and `open` is emptied so the renderer's state is consistent.
    fn reconcile_open_phases(&self, open: &mut HashMap<PhaseInstanceId, u32>, failed: bool) {
        if open.is_empty() {
            return;
        }
        let mut dangling: Vec<(PhaseInstanceId, u32)> =
            open.drain().map(|(id, depth)| (id, depth)).collect();
        // Deepest first; ties broken by instance id for deterministic output.
        dangling.sort_by(|a, b| b.1.cmp(&a.1).then(a.0 .0.cmp(&b.0 .0)));
        for (_, depth) in dangling {
            let indent = self.indent_base + 2 * depth;
            if failed {
                cli::error(indent, format_args!("interrupted"));
            } else {
                cli::warning(indent, format_args!("cancelled"));
            }
        }
    }

    fn render_activity(&self, open: &mut HashMap<PhaseInstanceId, u32>, ev: &Activity) {
        match ev {
            Activity::PhaseBegin {
                instance,
                parent,
                label,
                ..
            } => {
                // Depth inherits from parent (parent.depth + 1); a top-level
                // phase sits at depth 0 / base indent. An unknown parent
                // (out-of-order event) is treated as depth 0 rather than
                // panicking; the diag log records it.
                let depth = parent
                    .and_then(|p| {
                        open.get(&p).copied().or_else(|| {
                            log::debug!(
                                "LocalStatusBridge: PhaseBegin references unknown parent {p:?}"
                            );
                            None
                        })
                    })
                    .map(|d| d + 1)
                    .unwrap_or(0);
                open.insert(*instance, depth);
                let indent = self.indent_base + 2 * depth;
                cli::step(indent, format_args!("{label}"));
            },
            Activity::PhaseEnd {
                instance,
                outcome,
                duration_micros,
                summary,
                ..
            } => {
                let depth = open.remove(instance).unwrap_or_else(|| {
                    log::debug!("LocalStatusBridge: PhaseEnd for unknown instance {instance:?}");
                    0
                });
                let indent = self.indent_base + 2 * depth;
                let body = format_phase_end_body(summary.as_deref(), *duration_micros);
                match outcome {
                    PhaseOutcome::Ok => cli::success(indent, format_args!("{body}")),
                    PhaseOutcome::Skipped { reason } => {
                        cli::note(indent, format_args!("skipped: {reason}"))
                    },
                    PhaseOutcome::PartialFailure { detail } => {
                        cli::warning(indent, format_args!("{body} ({detail})"))
                    },
                    PhaseOutcome::Failed { error } => {
                        cli::error(indent, format_args!("{}", error.message))
                    },
                    // `PhaseOutcome` is `#[non_exhaustive]`.
                    other => log::debug!("LocalStatusBridge: unhandled phase outcome: {other:?}"),
                }
            },
            Activity::Message { phase, level, text } => {
                // Messages inside a phase sit one indent deeper than the
                // phase's heading line so the hierarchy reads correctly.
                let indent = self.indent_base + 2 * open.get(phase).copied().unwrap_or(0) + 2;
                cli::note_v(*level, indent, format_args!("{text}"));
            },
            Activity::Progress {
                phase,
                fraction,
                units,
                label,
            } => {
                let indent = self.indent_base + 2 * open.get(phase).copied().unwrap_or(0) + 2;
                let pct = (fraction.clamp(0.0, 1.0) * 100.0).round() as u32;
                let suffix = label.as_deref().unwrap_or("");
                let counter = format_progress_counter(*fraction, units);
                cli::note(indent, format_args!("{pct:>3}% {counter}{suffix}"));
            },
            Activity::Warning { phase, text } => {
                let indent = self.indent_base + 2 * open.get(phase).copied().unwrap_or(0);
                cli::warning(indent, format_args!("{text}"));
            },
            // `Activity` is `#[non_exhaustive]`.
            other => log::debug!("LocalStatusBridge: unhandled activity event: {other:?}"),
        }
    }
}

fn format_phase_end_body(summary: Option<&str>, duration_micros: Option<u64>) -> String {
    match summary {
        Some(s) => {
            if duration_micros.is_some() {
                let dur = format_duration(Duration::from_micros(duration_micros.unwrap()));
                format!("{s} ({dur})")
            } else {
                s.to_string()
            }
        },
        None => {
            if duration_micros.is_some() {
                let dur = format_duration(Duration::from_micros(duration_micros.unwrap()));
                format!("done ({dur})")
            } else {
                "done".into()
            }
        },
    }
}

fn format_progress_counter(fraction: f32, units: &ProgressUnits) -> String {
    match units {
        ProgressUnits::Items { total } => {
            let done = (fraction.clamp(0.0, 1.0) * *total as f32).round() as u64;
            format!("({done}/{total}) ")
        },
        ProgressUnits::Bytes { total } => {
            let done = (fraction.clamp(0.0, 1.0) * *total as f32).round() as u64;
            format!("({done}/{total} B) ")
        },
        ProgressUnits::Steps { total } => {
            let done = (fraction.clamp(0.0, 1.0) * *total as f32).round() as u32;
            format!("(step {done}/{total}) ")
        },
        ProgressUnits::Ratio => String::new(),
        // `ProgressUnits` is `#[non_exhaustive]`.
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{mpsc, Mutex, OnceLock};

    use chrono::{DateTime, Utc};
    use vgn_core::cli::{set_status_sink, SilentSink};
    use vgn_job_api::{
        error::{JobError, JobErrorCode},
        progress::{
            Activity, JobEvent, Lifecycle, PhaseInstanceId, PhaseKind, PhaseOutcome, ProgressUnits,
        },
    };

    use super::*;

    /// Process-wide mutex for tests that touch the global CLI sink. The
    /// CLI sink is a singleton in `vgn_core`, so any two tests that swap it
    /// concurrently would race; this serializes them within the crate
    /// without pulling in a test-ordering dependency.
    fn sink_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Installs the silent sink for the duration of a test. Holding the
    /// returned guard guarantees no other test is mutating the sink.
    fn silence_cli() -> std::sync::MutexGuard<'static, ()> {
        let guard = sink_lock().lock().unwrap();
        set_status_sink(SilentSink);
        guard
    }

    fn ts() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn err() -> JobError {
        JobError {
            code: JobErrorCode::HandlerError,
            message: "fit did not converge".into(),
            retriable: false,
            details: None,
        }
    }

    fn all_event_variants() -> Vec<JobEvent> {
        let phase = PhaseInstanceId(1);
        let nested = PhaseInstanceId(2);
        vec![
            JobEvent::Lifecycle(Lifecycle::Queued { at: ts() }),
            JobEvent::Lifecycle(Lifecycle::Started { at: ts() }),
            JobEvent::Activity(Activity::PhaseBegin {
                instance: phase,
                parent: None,
                kind: PhaseKind::measure_load_surfaces(),
                label: "Loading 12 surfaces".into(),
                at: ts(),
            }),
            JobEvent::Activity(Activity::PhaseBegin {
                instance: nested,
                parent: Some(phase),
                kind: PhaseKind::measure_load_iors(),
                label: "IOR database".into(),
                at: ts(),
            }),
            JobEvent::Activity(Activity::Message {
                phase: nested,
                level: 4,
                text: "loaded heightfield".into(),
            }),
            JobEvent::Activity(Activity::Progress {
                phase: nested,
                fraction: 0.5,
                units: ProgressUnits::Items { total: 132 },
                label: Some("patch 66/132".into()),
            }),
            JobEvent::Activity(Activity::Progress {
                fraction: 1.0,
                phase: nested,
                units: ProgressUnits::Bytes { total: 1024 },
                label: None,
            }),
            JobEvent::Activity(Activity::Warning {
                phase: nested,
                text: "clamped negative reflectance".into(),
            }),
            JobEvent::Activity(Activity::PhaseEnd {
                instance: nested,
                outcome: PhaseOutcome::Ok,
                duration_micros: Some(340_000),
                summary: Some("4 files".into()),
                at: ts(),
            }),
            JobEvent::Activity(Activity::PhaseEnd {
                instance: phase,
                outcome: PhaseOutcome::PartialFailure {
                    detail: "1 unreadable".into(),
                },
                duration_micros: Some(1_200_000),
                summary: Some("11 of 12 surfaces".into()),
                at: ts(),
            }),
            JobEvent::Lifecycle(Lifecycle::Completed { at: ts() }),
            JobEvent::Lifecycle(Lifecycle::Failed {
                at: ts(),
                error: err(),
            }),
            JobEvent::Lifecycle(Lifecycle::Cancelled { at: ts() }),
        ]
    }

    #[test]
    fn render_handles_every_job_event_variant() {
        // Smoke test: pushing every variant through `render` must not panic.
        let _g = silence_cli();
        let bridge = LocalStatusBridge::new();
        let mut open = HashMap::new();
        for ev in all_event_variants() {
            bridge.render(&mut open, &ev);
        }
    }

    #[test]
    fn run_returns_when_sender_drops() {
        // The executor closes the channel when the job finishes. The bridge
        // must observe that and stop waiting, otherwise CLI adapters would
        // hang forever after the job's terminal event.
        let _g = silence_cli();
        let (tx, rx) = mpsc::channel::<JobEvent>();
        tx.send(JobEvent::Lifecycle(Lifecycle::Started { at: ts() }))
            .unwrap();
        tx.send(JobEvent::Lifecycle(Lifecycle::Completed { at: ts() }))
            .unwrap();
        drop(tx);
        LocalStatusBridge::new().run(rx); // returns
    }

    #[test]
    fn run_returns_immediately_on_empty_closed_channel() {
        // Degenerate but real: the executor could close the channel without
        // sending anything (e.g. an error before the worker thread even
        // starts). `run` must still return cleanly.
        let _g = silence_cli();
        let (tx, rx) = mpsc::channel::<JobEvent>();
        drop(tx);
        LocalStatusBridge::new().run(rx);
    }

    #[test]
    fn run_drains_all_events_in_order() {
        let _g = silence_cli();
        let (tx, rx) = mpsc::channel::<JobEvent>();
        let mut sent = 0usize;
        for ev in all_event_variants() {
            tx.send(ev).unwrap();
            sent += 1;
        }
        drop(tx);

        let bridge = LocalStatusBridge::new();
        let mut open = HashMap::new();
        let mut received = 0usize;
        for ev in &rx {
            bridge.render(&mut open, &ev);
            received += 1;
        }
        assert_eq!(received, sent);
    }

    #[test]
    fn with_indent_overrides_default() {
        let default = LocalStatusBridge::new();
        let nested = LocalStatusBridge::with_indent(Indent::SUBSECTION);
        assert_eq!(default.indent_base, Indent::SECTION.as_u32());
        assert_eq!(nested.indent_base, Indent::SUBSECTION.as_u32());
    }

    #[test]
    fn render_clamps_out_of_range_fraction() {
        // The producer is supposed to clamp, but the bridge must clamp
        // defensively so a stray NaN/negative/over-1 from a buggy capability
        // doesn't panic during the cast to u32.
        let _g = silence_cli();
        let bridge = LocalStatusBridge::new();
        let mut open = HashMap::new();
        let phase = PhaseInstanceId(0);
        open.insert(phase, 0);
        for fraction in [-1.0_f32, 0.0, 0.5, 1.0, 2.0, f32::NAN, f32::INFINITY] {
            bridge.render(
                &mut open,
                &JobEvent::Activity(Activity::Progress {
                    phase,
                    fraction,
                    units: ProgressUnits::Ratio,
                    label: None,
                }),
            );
        }
    }

    #[test]
    fn phase_depth_tracks_parent_child_chain() {
        // Nested PhaseBegin -> child sits one indent below parent. Verify
        // by inspecting the open-phase map after the events.
        let _g = silence_cli();
        let bridge = LocalStatusBridge::new();
        let mut open = HashMap::new();
        let p = PhaseInstanceId(1);
        let c = PhaseInstanceId(2);
        bridge.render(
            &mut open,
            &JobEvent::Activity(Activity::PhaseBegin {
                instance: p,
                parent: None,
                kind: PhaseKind::measure_load_surfaces(),
                label: "Outer".into(),
                at: ts(),
            }),
        );
        bridge.render(
            &mut open,
            &JobEvent::Activity(Activity::PhaseBegin {
                instance: c,
                parent: Some(p),
                kind: PhaseKind::measure_load_iors(),
                label: "Inner".into(),
                at: ts(),
            }),
        );
        assert_eq!(open.get(&p), Some(&0));
        assert_eq!(open.get(&c), Some(&1));
    }

    #[test]
    fn phase_end_removes_open_entry() {
        let _g = silence_cli();
        let bridge = LocalStatusBridge::new();
        let mut open = HashMap::new();
        let p = PhaseInstanceId(7);
        bridge.render(
            &mut open,
            &JobEvent::Activity(Activity::PhaseBegin {
                instance: p,
                parent: None,
                kind: PhaseKind::measure_bsdf(),
                label: "x".into(),
                at: ts(),
            }),
        );
        assert!(open.contains_key(&p));
        bridge.render(
            &mut open,
            &JobEvent::Activity(Activity::PhaseEnd {
                instance: p,
                outcome: PhaseOutcome::Ok,
                duration_micros: Some(0),
                summary: None,
                at: ts(),
            }),
        );
        assert!(!open.contains_key(&p));
    }

    #[test]
    fn phase_end_for_unknown_instance_does_not_panic() {
        // Receivers may see a `PhaseEnd` for a phase whose `PhaseBegin` was
        // dropped (e.g. partial replay, executor restart). The bridge must
        // not panic; the diag log records it.
        let _g = silence_cli();
        let bridge = LocalStatusBridge::new();
        let mut open = HashMap::new();
        bridge.render(
            &mut open,
            &JobEvent::Activity(Activity::PhaseEnd {
                instance: PhaseInstanceId(99),
                outcome: PhaseOutcome::Ok,
                duration_micros: Some(0),
                summary: None,
                at: ts(),
            }),
        );
    }

    /// Helper: open `n` nested phases (each the child of the previous) without
    /// closing them, then return the populated `open` map.
    fn open_nested_phases(bridge: &LocalStatusBridge, n: u32) -> HashMap<PhaseInstanceId, u32> {
        let mut open = HashMap::new();
        let mut parent = None;
        for i in 0..n {
            let id = PhaseInstanceId(i);
            bridge.render(
                &mut open,
                &JobEvent::Activity(Activity::PhaseBegin {
                    instance: id,
                    parent,
                    kind: PhaseKind::measure_bsdf(),
                    label: format!("phase {i}"),
                    at: ts(),
                }),
            );
            parent = Some(id);
        }
        open
    }

    #[test]
    fn failed_lifecycle_reconciles_open_phases() {
        // The progress protocol requires receivers to treat phases still open
        // at a terminal `Failed` as ended. A handler that returns `Err`
        // emits no closing `PhaseEnd`, so the bridge must empty `open` itself.
        let _g = silence_cli();
        let bridge = LocalStatusBridge::new();
        let mut open = open_nested_phases(&bridge, 3);
        assert_eq!(open.len(), 3);
        bridge.render(
            &mut open,
            &JobEvent::Lifecycle(Lifecycle::Failed {
                at: ts(),
                error: JobError {
                    code: JobErrorCode::HandlerError,
                    message: "boom".into(),
                    retriable: false,
                    details: None,
                },
            }),
        );
        assert!(
            open.is_empty(),
            "open phases must be reconciled on Lifecycle::Failed"
        );
    }

    #[test]
    fn cancelled_lifecycle_reconciles_open_phases() {
        let _g = silence_cli();
        let bridge = LocalStatusBridge::new();
        let mut open = open_nested_phases(&bridge, 2);
        assert_eq!(open.len(), 2);
        bridge.render(
            &mut open,
            &JobEvent::Lifecycle(Lifecycle::Cancelled { at: ts() }),
        );
        assert!(
            open.is_empty(),
            "open phases must be reconciled on Lifecycle::Cancelled"
        );
    }

    #[test]
    fn completed_lifecycle_leaves_open_map_untouched() {
        // `Completed` is the normal-success terminal; a well-behaved handler
        // closed its phases already. The bridge does not force-close on
        // success (a stray open phase there is a handler bug, surfaced by the
        // missing `✓`, not something the renderer should paper over).
        let _g = silence_cli();
        let bridge = LocalStatusBridge::new();
        let mut open = open_nested_phases(&bridge, 1);
        bridge.render(
            &mut open,
            &JobEvent::Lifecycle(Lifecycle::Completed { at: ts() }),
        );
        assert_eq!(open.len(), 1, "Completed must not reconcile open phases");
    }
}
