//! Model namespace.
//!
//! Deliberately not a universal model trait: the foundation's `StateModel`
//! already expresses the stateful contract each model needs
//! (`docs/KILO_CONNECTOME_SPRINT_REVISED.md` §16).
//!
//! `recurrent` holds the stateful GRU slice. `classic` holds the classic ML
//! slices, starting with stateless linear-regression prediction; `online` holds
//! ordered online adaptation, starting with recursive least squares. See
//! `docs/ML_VERTICAL_SLICES.md`.

pub mod classic;
pub mod online;
pub mod recurrent;
