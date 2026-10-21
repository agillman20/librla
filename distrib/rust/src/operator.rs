// Linear operators: dense matrices and matrix-free operators share one interface.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use crate::mat::Mat;
use crate::scalar::Scalar;
use std::fmt;

/// An `m x n` linear operator applied to blocks of vectors.
///
/// Implement `apply` (`Y = A X`, `X` is `n x k`) and `apply_adjoint`
/// (`Z = A^H W`, `W` is `m x k`) for a matrix-free operator. A dense
/// matrix ([`Mat`]) implements this trait and returns itself from
/// `as_dense`, which lets librla skip materialization and form `Q^H A`
/// directly; matrix-free operators keep the default `None`.
pub trait LinearOperator<T: Scalar> {
    /// `(m, n)`.
    fn shape(&self) -> (usize, usize);
    /// `Y = A X` for an `n x k` block `X`; must return `m x k`.
    fn apply(&self, x: &Mat<T>) -> Mat<T>;
    /// `Z = A^H W` for an `m x k` block `W`; must return `n x k`.
    fn apply_adjoint(&self, w: &Mat<T>) -> Mat<T>;
    /// The explicit matrix, if the operator has one.
    fn as_dense(&self) -> Option<&Mat<T>> {
        None
    }
}

impl<T: Scalar> LinearOperator<T> for Mat<T> {
    fn shape(&self) -> (usize, usize) {
        Mat::shape(self)
    }
    fn apply(&self, x: &Mat<T>) -> Mat<T> {
        self.matmul(x)
    }
    fn apply_adjoint(&self, w: &Mat<T>) -> Mat<T> {
        self.adjoint_matmul(w)
    }
    fn as_dense(&self) -> Option<&Mat<T>> {
        Some(self)
    }
}

/// A matrix-free operator built from two closures.
///
/// ```
/// use librla::{FnOperator, Mat};
/// let b = Mat::<f64>::from_fn(6, 2, |i, j| (i + j) as f64);
/// let op = FnOperator::new(6, 6, |x: &Mat<f64>| b.matmul(&b.adjoint_matmul(x)),
///                                |w: &Mat<f64>| b.matmul(&b.adjoint_matmul(w)));
/// ```
pub struct FnOperator<F, G> {
    m: usize,
    n: usize,
    apply: F,
    apply_adjoint: G,
}

impl<F, G> FnOperator<F, G> {
    /// `m x n` operator with `apply(X) = A X` and `apply_adjoint(W) = A^H W`.
    pub fn new(m: usize, n: usize, apply: F, apply_adjoint: G) -> Self {
        FnOperator { m, n, apply, apply_adjoint }
    }
}

impl<F, G> fmt::Debug for FnOperator<F, G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FnOperator({}x{})", self.m, self.n)
    }
}

impl<T, F, G> LinearOperator<T> for FnOperator<F, G>
where
    T: Scalar,
    F: Fn(&Mat<T>) -> Mat<T>,
    G: Fn(&Mat<T>) -> Mat<T>,
{
    fn shape(&self) -> (usize, usize) {
        (self.m, self.n)
    }
    fn apply(&self, x: &Mat<T>) -> Mat<T> {
        (self.apply)(x)
    }
    fn apply_adjoint(&self, w: &Mat<T>) -> Mat<T> {
        (self.apply_adjoint)(w)
    }
}

/// Sized handle to any operator, so `&A` with `A: ?Sized` becomes `&dyn`.
pub(crate) struct ByRef<'a, A: ?Sized>(pub(crate) &'a A);

impl<T: Scalar, A: LinearOperator<T> + ?Sized> LinearOperator<T> for ByRef<'_, A> {
    fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }
    fn apply(&self, x: &Mat<T>) -> Mat<T> {
        self.0.apply(x)
    }
    fn apply_adjoint(&self, w: &Mat<T>) -> Mat<T> {
        self.0.apply_adjoint(w)
    }
    fn as_dense(&self) -> Option<&Mat<T>> {
        self.0.as_dense()
    }
}

/// Adjoint `A^H` of a matrix-free operator (apply and apply_adjoint swapped).
pub(crate) struct Adjoint<'a, T: Scalar>(pub(crate) &'a dyn LinearOperator<T>);

impl<T: Scalar> LinearOperator<T> for Adjoint<'_, T> {
    fn shape(&self) -> (usize, usize) {
        let (m, n) = self.0.shape();
        (n, m)
    }
    fn apply(&self, x: &Mat<T>) -> Mat<T> {
        self.0.apply_adjoint(x)
    }
    fn apply_adjoint(&self, w: &Mat<T>) -> Mat<T> {
        self.0.apply(w)
    }
}
