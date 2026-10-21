// Error type.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use std::fmt;

/// Errors returned by librla.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// An ID method name other than `"fast"`, `"svd"` or `"lstsq"`.
    InvalidMethod(String),
    /// RNG seed entries outside `[0, 4095]` or an even last entry.
    InvalidSeed([i32; 4]),
    /// A non-finite `rtol`, or a size that does not fit LAPACK's 32-bit integers.
    InvalidArgument(String),
    /// A matrix-free operator returned a block of the wrong shape.
    ShapeMismatch {
        /// Which operator call produced the block (`"apply"` / `"apply_adjoint"`).
        call: &'static str,
        /// Shape librla expected.
        expected: (usize, usize),
        /// Shape the operator returned.
        got: (usize, usize),
    },
    /// A LAPACK routine reported `info != 0`.
    Lapack {
        /// Routine name (e.g. `"geqp3"`).
        routine: &'static str,
        /// The `info` value it returned.
        info: i32,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidMethod(m) => {
                write!(f, "method must be one of {{'fast', 'svd', 'lstsq'}}, got '{m}'")
            }
            Error::InvalidSeed(s) => write!(
                f,
                "invalid LAPACK seed {s:?}: entries must lie in [0, 4095] and the last must be odd"
            ),
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Error::ShapeMismatch { call, expected, got } => write!(
                f,
                "operator {call} returned a {}x{} block, expected {}x{}",
                got.0, got.1, expected.0, expected.1
            ),
            Error::Lapack { routine, info } => {
                write!(f, "LAPACK {routine} failed with info = {info}")
            }
        }
    }
}

impl std::error::Error for Error {}
