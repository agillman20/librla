// Random number generator: the LAPACK larnv seed shared by all librla ports.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use crate::error::Error;
use crate::ffi::Int;
use crate::mat::Mat;
use crate::scalar::Scalar;

/// Random number generator for the test matrices: the four-integer seed of
/// LAPACK `dlarnv` / `zlarnv`, advanced in place by every draw.
///
/// Draws are uniform on (-1, 1) (`idist = 2`); for complex matrices the real
/// and imaginary parts are each uniform on (-1, 1). An `n x l` matrix is
/// filled column-major in one call, so the same seed gives the same sketch
/// in every librla port that uses this generator.
///
/// `Rng::default()` has seed `[0, 0, 0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rng {
    iseed: [i32; 4],
}

impl Default for Rng {
    fn default() -> Self {
        Rng { iseed: [0, 0, 0, 1] }
    }
}

impl Rng {
    /// Seed from four integers in `[0, 4095]`, the last one odd.
    pub fn new(i0: i32, i1: i32, i2: i32, i3: i32) -> Result<Self, Error> {
        let iseed = [i0, i1, i2, i3];
        if iseed.iter().any(|&v| !(0..=4095).contains(&v)) || i3 % 2 == 0 {
            return Err(Error::InvalidSeed(iseed));
        }
        Ok(Rng { iseed })
    }

    /// Seed from one 64-bit integer:
    /// `[(s>>36)&4095, (s>>24)&4095, (s>>12)&4095, (s&4095)|1]`.
    pub fn from_seed(s: u64) -> Self {
        let f = |shift: u32| ((s >> shift) & 4095) as i32;
        Rng { iseed: [f(36), f(24), f(12), f(0) | 1] }
    }

    /// Current LAPACK seed.
    pub fn iseed(&self) -> [i32; 4] {
        self.iseed
    }

    /// Fill `x` with uniform(-1, 1) entries (one larnv call per 2^31 - 64 entries).
    pub fn fill<T: Scalar>(&mut self, x: &mut [T]) {
        // larnv consumes the seed in chunks of 64 entries, so splitting the
        // call at a multiple of 64 gives the same stream as one call.
        const CHUNK: usize = (i32::MAX as usize / 64) * 64;
        for part in x.chunks_mut(CHUNK) {
            // SAFETY: part has part.len() <= i32::MAX writable entries; iseed has 4.
            unsafe { T::larnv(2, self.iseed.as_mut_ptr(), part.len() as Int, part.as_mut_ptr()) };
        }
    }

    /// An `m x n` matrix of uniform(-1, 1) entries, filled column-major.
    pub fn uniform<T: Scalar>(&mut self, m: usize, n: usize) -> Mat<T> {
        let mut a = Mat::zeros(m, n);
        self.fill(a.as_mut_slice());
        a
    }
}
