//! Process-wide string interner. Leaks `Box<str>` into a `'static` pool so a `&str`
//! handed in at registry-load time can be stored as `&'static str` for the process
//! lifetime.

use std::{collections::HashSet, sync::Mutex};

static POOL: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);

/// Intern a string. If the same content has been interned before, returns the
/// existing `&'static str`; otherwise allocates, leaks, and inserts.
pub fn intern(s: &str) -> &'static str {
    let mut guard = POOL.lock().expect("medium intern pool poisoned");
    let pool = guard.get_or_insert_with(HashSet::new);
    if let Some(existing) = pool.get(s) {
        return *existing;
    }
    // Leak a Box<str>. Lives for process lifetime.
    let leaked: &'static str = Box::leak(s.to_owned().into_boxed_str());
    pool.insert(leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_returns_same_pointer_for_same_content() {
        let a = intern("al");
        let b = intern("al");
        assert_eq!(
            a.as_ptr(),
            b.as_ptr(),
            "expected same pointer after intern of same content"
        );
    }

    #[test]
    fn intern_distinct_content_distinct_pointers() {
        let a = intern("cu");
        let b = intern("ni");
        assert_ne!(a.as_ptr(), b.as_ptr());
        assert_eq!(a, "cu");
        assert_eq!(b, "ni");
    }
}
