//! Model namespace.
//!
//! Deliberately not a universal model trait: the foundation's `StateModel`
//! already expresses the stateful contract the GRU needs
//! (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §16).

pub mod recurrent;
