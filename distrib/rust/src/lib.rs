//! Randomized linear-algebra routines (experimental Rust port of librla).
//!
//! Randomized algorithms for low-rank matrix approximations:
//!
//! * [`orth_sketch`] - approximate orthonormal basis for the column space
//! * [`qr_sketch`]   - truncated QR factorization with column pivoting
//! * [`svd_sketch`]  - truncated singular value decomposition
//! * [`id_sketch`]   - interpolative decomposition (ID)
//!
//! Deterministic:
//!
//! * [`id_qrpiv`]    - interpolative decomposition via QR with column pivoting
//!
//! Every routine takes `rtol`: tolerance mode for `rtol < 1` (adaptive rank),
//! rank mode for `rtol >= 1` (target rank `floor(rtol)`). Matrices are `f64`
//! or [`Complex64`], stored column-major in [`Mat`];
//! all dense factorizations go through LAPACK. Any [`LinearOperator`]
//! (a dense `Mat`, or a matrix-free operator providing block `A X` and
//! `A^H W`) is accepted; in tolerance mode the deterministic fallback may
//! materialize a matrix-free operator at the cost of O(n) matvecs.
//!
//! `extra_samples` (default 12) is the buffer beyond the target rank in both
//! modes: rank mode samples `floor(rtol) + extra_samples` columns; tolerance
//! mode accepts a sketch only when at least `extra_samples + 1` pivoted column
//! norms fall at or below `rtol` times the largest.
//!
//! ```
//! use librla::{svd_sketch, Mat, Options};
//! let a = Mat::<f64>::from_fn(200, 100, |i, j| 1.0 / (i + j + 1) as f64);
//! let svd = svd_sketch(&a, 1e-8, Options::default()).unwrap();
//! let err = (&a - &svd.u.scale_columns(&svd.s).matmul(&svd.vh)).norm_fro() / a.norm_fro();
//! assert!(err < 1e-7);
//! ```
//!
//! Author: Adrianna Gillman, Zydrunas Gimbutas
//! SPDX-License-Identifier: MIT
//! Assisted by: Claude Code (Anthropic)

#![warn(missing_docs)]

mod error;
mod ffi;
mod lapack;
mod mat;
mod operator;
mod rng;
mod scalar;
mod sketch;

pub use error::Error;
pub use lapack::dense;
pub use mat::Mat;
pub use num_complex::Complex64;
pub use operator::{FnOperator, LinearOperator};
pub use rng::Rng;
pub use scalar::Scalar;
pub use sketch::{
    id_qrpiv, id_sketch, orth_sketch, qr_sketch, svd_sketch, IdDecomp, Method, Options, OrthSketch,
    QrSketch, SvdSketch,
};
