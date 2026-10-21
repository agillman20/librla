// Minimal column-major dense matrix.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use crate::lapack;
use crate::scalar::Scalar;
use std::ops::{Index, IndexMut, Mul, Range, Sub};

/// Dense `nrows x ncols` matrix stored column-major (LAPACK layout,
/// leading dimension `nrows`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mat<T> {
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
}

impl<T: Scalar> Mat<T> {
    /// The `m x n` zero matrix.
    pub fn zeros(m: usize, n: usize) -> Self {
        Mat { nrows: m, ncols: n, data: vec![T::zero(); m * n] }
    }

    /// The `n x n` identity.
    pub fn identity(n: usize) -> Self {
        let mut a = Self::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = T::one();
        }
        a
    }

    /// `m x n` matrix with entries `f(i, j)` (0-based).
    pub fn from_fn(m: usize, n: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                data.push(f(i, j));
            }
        }
        Mat { nrows: m, ncols: n, data }
    }

    /// Wrap column-major data. Panics if `data.len() != m * n`.
    pub fn from_col_major(m: usize, n: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), m * n, "Mat::from_col_major: data length must equal m * n");
        Mat { nrows: m, ncols: n, data }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// `(nrows, ncols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Column-major entries.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable column-major entries.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume into the column-major entries.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Column `j`.
    pub fn col(&self, j: usize) -> &[T] {
        &self.data[j * self.nrows..(j + 1) * self.nrows]
    }

    /// Mutable column `j`.
    pub fn col_mut(&mut self, j: usize) -> &mut [T] {
        &mut self.data[j * self.nrows..(j + 1) * self.nrows]
    }

    /// Main diagonal.
    pub fn diag(&self) -> Vec<T> {
        (0..self.nrows.min(self.ncols)).map(|i| self[(i, i)]).collect()
    }

    /// Conjugate transpose `A^H`.
    pub fn adjoint(&self) -> Self {
        Self::from_fn(self.ncols, self.nrows, |i, j| self[(j, i)].conj())
    }

    /// Product `A B` (BLAS gemm).
    pub fn matmul(&self, b: &Mat<T>) -> Self {
        lapack::gemm(b'N', self, b'N', b)
    }

    /// Product `A^H B` (BLAS gemm with conjugate transpose).
    pub fn adjoint_matmul(&self, b: &Mat<T>) -> Self {
        lapack::gemm(b'C', self, b'N', b)
    }

    /// Columns `idx` in the given order (e.g. `A[:, piv[..k]]`).
    pub fn select_columns(&self, idx: &[usize]) -> Self {
        let mut data = Vec::with_capacity(self.nrows * idx.len());
        for &j in idx {
            data.extend_from_slice(self.col(j));
        }
        Mat { nrows: self.nrows, ncols: idx.len(), data }
    }

    /// Contiguous column block `A[:, range]`.
    pub fn columns(&self, range: Range<usize>) -> Self {
        let data = self.data[range.start * self.nrows..range.end * self.nrows].to_vec();
        Mat { nrows: self.nrows, ncols: range.len(), data }
    }

    /// Contiguous row block `A[range, :]`.
    pub fn rows(&self, range: Range<usize>) -> Self {
        Self::from_fn(range.len(), self.ncols, |i, j| self[(range.start + i, j)])
    }

    /// `A diag(s)`: column `j` scaled by `s[j]`.
    pub fn scale_columns(&self, s: &[f64]) -> Self {
        assert_eq!(s.len(), self.ncols, "Mat::scale_columns: need one factor per column");
        let mut out = self.clone();
        for (j, &sj) in s.iter().enumerate() {
            let f = T::from_real(sj);
            out.col_mut(j).iter_mut().for_each(|x| *x *= f);
        }
        out
    }

    /// `diag(s) A`: row `i` scaled by `s[i]`.
    pub fn scale_rows(&self, s: &[f64]) -> Self {
        assert_eq!(s.len(), self.nrows, "Mat::scale_rows: need one factor per row");
        let mut out = self.clone();
        for j in 0..self.ncols {
            for (x, &si) in out.col_mut(j).iter_mut().zip(s) {
                *x *= T::from_real(si);
            }
        }
        out
    }

    /// Frobenius norm (scaled to avoid overflow and underflow).
    pub fn norm_fro(&self) -> f64 {
        let scale = self.data.iter().map(|x| x.abs()).fold(0.0, f64::max);
        if scale == 0.0 || !scale.is_finite() {
            return scale;
        }
        let ssq: f64 = self.data.iter().map(|x| (x.abs() / scale).powi(2)).sum();
        scale * ssq.sqrt()
    }

    /// Largest entry modulus (0 for an empty matrix).
    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }

    /// `true` when every entry is finite.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }
}

impl<T> Index<(usize, usize)> for Mat<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(i < self.nrows && j < self.ncols, "Mat index out of bounds");
        &self.data[i + j * self.nrows]
    }
}

impl<T> IndexMut<(usize, usize)> for Mat<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(i < self.nrows && j < self.ncols, "Mat index out of bounds");
        &mut self.data[i + j * self.nrows]
    }
}

impl<T: Scalar> Mul for &Mat<T> {
    type Output = Mat<T>;
    fn mul(self, b: &Mat<T>) -> Mat<T> {
        self.matmul(b)
    }
}

impl<T: Scalar> Sub for &Mat<T> {
    type Output = Mat<T>;
    fn sub(self, b: &Mat<T>) -> Mat<T> {
        assert_eq!(self.shape(), b.shape(), "Mat subtraction: shape mismatch");
        let data = self.data.iter().zip(&b.data).map(|(&x, &y)| x - y).collect();
        Mat { nrows: self.nrows, ncols: self.ncols, data }
    }
}
