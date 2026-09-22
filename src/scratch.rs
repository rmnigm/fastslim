//! Scratch space allocated once per worker thread of the current rayon pool.

use std::sync::{Mutex, PoisonError};

/// One `T` per thread of the current rayon pool, handed to whichever item that
/// thread is working on. The lock is uncontended because each worker only
/// ever takes its own slot.
pub struct PerThread<T> {
    slots: Vec<Mutex<T>>,
}

impl<T> PerThread<T> {
    /// Allocate one slot per thread of the pool this is called from.
    pub fn new(init: impl Fn() -> T) -> Self {
        let slots = (0..rayon::current_num_threads())
            .map(|_| Mutex::new(init()))
            .collect();
        PerThread { slots }
    }

    /// Run `f` with the calling worker's slot.
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        // Rayon may run small inputs on the calling thread, which has no worker index.
        let slot = rayon::current_thread_index().unwrap_or(0) % self.slots.len();
        let mut guard = self.slots[slot]
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        f(&mut guard)
    }
}
