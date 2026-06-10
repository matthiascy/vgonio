//! In-process capability handler registry used by [`crate::LocalExecutor`].
//!
//! A [`CapabilityRegistry`] is a `CapabilityId -> Handler` map. The
//! [`Handler`] closure is what actually runs a job: the executor passes it the
//! envelope and a freshly minted [`JobContext`], and the closure returns the
//! encoded result bytes (or a [`JobError`]).
//!
//! Adapters build a registry at startup, register one closure per capability
//! they care about, and hand the registry to [`crate::LocalExecutor::new`] (or
//! [`crate::LocalExecutor::from_cache_dir`]). After submission, the registry
//! is internally wrapped in an [`std::sync::RwLock`] so handlers can be added
//! at runtime if needed; today only the startup path uses it.

use std::{collections::HashMap, sync::Arc};

use vgn_job_api::{context::JobContext, envelope::JobEnvelope, error::JobError, ids::CapabilityId};

use crate::JobOutcome;

/// An in-process capability handler: a closure that runs one job.
///
/// The handler receives the raw [`JobEnvelope`] and the executor-built
/// [`JobContext`], and returns the encoded result bytes. Capability-specific
/// request decoding happens **inside** the handler, since only the handler
/// knows the request type for its capability.
///
/// Boxed behind an [`Arc`] so the registry can hand cheap clones to every
/// dispatched job and so handlers can carry their own shared state
/// (`Arc<MyService>` captured by the closure is the canonical pattern). The
/// `Send + Sync` bound is what lets [`crate::LocalExecutor`] dispatch jobs
/// onto background threads.
///
/// Phase 1 deliberately uses a closure type rather than a trait object: a
/// closure is the simplest thing that lets handler implementations carry
/// state via captures, and trait objects can be re-introduced later without a
/// wire-format change.
pub type Handler =
    Arc<dyn Fn(&JobEnvelope, JobContext) -> Result<JobOutcome, JobError> + Send + Sync>;

/// Maps [`CapabilityId`] to its in-process [`Handler`].
///
/// Construct with [`CapabilityRegistry::new`] (or [`Default`]), populate via
/// [`Self::register`], and hand off to [`crate::LocalExecutor`]. The registry
/// itself is not `Send + Sync`-aware on its own; the executor wraps it.
///
/// Registration is *last-write-wins*: calling [`Self::register`] twice with
/// the same [`CapabilityId`] replaces the previous handler silently. This is
/// deliberate (tests and override hooks both want it), so callers that need
/// "fail if already present" semantics should check with [`Self::get`] first.
#[derive(Default)]
pub struct CapabilityRegistry {
    handlers: HashMap<CapabilityId, Handler>,
}

impl CapabilityRegistry {
    /// Returns an empty registry. Identical to [`Default::default`].
    pub fn new() -> Self { Self::default() }

    /// Registers `handler` for `capability_id`, replacing any previous
    /// handler for the same id (see the type-level docs for the rationale).
    pub fn register(&mut self, capability_id: CapabilityId, handler: Handler) {
        self.handlers.insert(capability_id, handler);
    }

    /// Returns the handler for `id`, or [`None`] if no handler has been
    /// registered.
    ///
    /// The returned reference borrows from the registry; clone it (the
    /// `Handler` type is an [`Arc`], so the clone is one atomic increment)
    /// before doing anything that requires releasing the registry lock.
    pub fn get(&self, id: &CapabilityId) -> Option<&Handler> { self.handlers.get(id) }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    /// A handler that does nothing (returns empty bytes). We never invoke
    /// the closure in these tests (registry tests check the registry's
    /// own contract, not handler behaviour), so the body doesn't matter.
    fn dummy_handler() -> Handler {
        Arc::new(|_env, _ctx| {
            Ok(JobOutcome {
                payload: Bytes::new(),
                artifacts: vec![],
            })
        })
    }

    #[test]
    fn new_starts_empty() {
        let reg = CapabilityRegistry::new();
        assert!(reg.get(&CapabilityId("fit".into())).is_none());
    }

    #[test]
    fn default_is_equivalent_to_new() {
        let reg = CapabilityRegistry::default();
        assert!(reg.get(&CapabilityId("anything".into())).is_none());
    }

    #[test]
    fn register_then_get_returns_some() {
        let mut reg = CapabilityRegistry::new();
        let id = CapabilityId("echo".into());
        reg.register(id.clone(), dummy_handler());
        assert!(reg.get(&id).is_some());
    }

    #[test]
    fn get_unknown_id_returns_none() {
        let mut reg = CapabilityRegistry::new();
        reg.register(CapabilityId("known".into()), dummy_handler());
        assert!(reg.get(&CapabilityId("unknown".into())).is_none());
    }

    #[test]
    fn register_replaces_existing_handler_silently() {
        // "Last write wins" is part of the contract; lock it in with a test
        // so a future change (e.g. returning Result<(), AlreadyRegistered>)
        // trips here first. We assert by Arc pointer identity rather than by
        // calling the closure, so the test stays decoupled from how
        // JobContext is constructed.
        let mut reg = CapabilityRegistry::new();
        let id = CapabilityId("dup".into());

        let first = dummy_handler();
        let second = dummy_handler();
        let second_ptr = Arc::as_ptr(&second);

        reg.register(id.clone(), first);
        reg.register(id.clone(), second);

        let stored = reg.get(&id).expect("must still be registered");
        assert!(
            std::ptr::eq(Arc::as_ptr(stored), second_ptr),
            "second register call must replace the first",
        );
    }

    #[test]
    fn distinct_ids_keep_independent_handlers() {
        let mut reg = CapabilityRegistry::new();
        let a = CapabilityId("a".into());
        let b = CapabilityId("b".into());

        let h_a = dummy_handler();
        let h_b = dummy_handler();
        let a_ptr = Arc::as_ptr(&h_a);
        let b_ptr = Arc::as_ptr(&h_b);

        reg.register(a.clone(), h_a);
        reg.register(b.clone(), h_b);

        assert!(std::ptr::eq(Arc::as_ptr(reg.get(&a).unwrap()), a_ptr));
        assert!(std::ptr::eq(Arc::as_ptr(reg.get(&b).unwrap()), b_ptr));
    }

    #[test]
    fn handler_type_is_send_sync() {
        // Compile-time check: the executor dispatches handlers onto
        // background threads, so Handler must be Send + Sync. If a future
        // edit to the type alias regresses this, the test stops compiling.
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let h = dummy_handler();
        assert_send_sync(&h);
    }
}
