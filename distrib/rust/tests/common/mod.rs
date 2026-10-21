// Shared helpers for the librla integration tests.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#![allow(dead_code)]

use librla::{dense, IdDecomp, LinearOperator, Mat, Rng, Scalar, SvdSketch};
use std::fmt::Display;

/// Print one PASS/FAIL line and fail the test on FAIL.
pub fn check(name: &str, ok: bool, detail: impl Display) {
    println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
    assert!(ok, "FAIL {name}: {detail}");
}

/// Generate `mod $name { fn real(); fn complex(); }` calling `$f::<T>`.
#[allow(unused_macros)]
macro_rules! both {
    ($name:ident, $f:ident) => {
        mod $name {
            #[test]
            fn real() {
                super::$f::<f64>();
            }
            #[test]
            fn complex() {
                super::$f::<librla::Complex64>();
            }
        }
    };
}

pub fn kind<T: Scalar>() -> &'static str {
    if T::IS_COMPLEX {
        "complex"
    } else {
        "real"
    }
}

pub fn seed() -> Rng {
    Rng::new(1, 2, 3, 5).unwrap()
}

/// `m x k` matrix with orthonormal columns (Q of a LAPACK pivoted QR).
pub fn orthonormal<T: Scalar>(rng: &mut Rng, m: usize, k: usize) -> Mat<T> {
    dense::qr_pivoted(&rng.uniform::<T>(m, k)).unwrap().0
}

/// `U diag(s) V^H` with random orthonormal `U` (`m x r`), `V` (`n x r`), `r = s.len()`.
pub fn with_svals<T: Scalar>(rng: &mut Rng, m: usize, n: usize, s: &[f64]) -> Mat<T> {
    let u = orthonormal::<T>(rng, m, s.len());
    let v = orthonormal::<T>(rng, n, s.len());
    u.scale_columns(s).matmul(&v.adjoint())
}

/// Singular values `10^(-j/4)`, `j = 0..min(m,n)`.
pub fn decaying<T: Scalar>(rng: &mut Rng, m: usize, n: usize) -> Mat<T> {
    let s: Vec<f64> = (0..m.min(n)).map(|j| 10f64.powf(-(j as f64) / 4.0)).collect();
    with_svals(rng, m, n, &s)
}

/// `||Q^H Q - I||_F`.
pub fn ortho_err<T: Scalar>(q: &Mat<T>) -> f64 {
    (&q.adjoint_matmul(q) - &Mat::identity(q.ncols())).norm_fro()
}

/// `||Vh Vh^H - I||_F`.
pub fn ortho_err_rows<T: Scalar>(vh: &Mat<T>) -> f64 {
    ortho_err(&vh.adjoint())
}

/// `||A - B||_F / ||A||_F`.
pub fn rel<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> f64 {
    (a - b).norm_fro() / a.norm_fro()
}

/// `||A - Q Q^H A||_F / ||A||_F`.
pub fn span_err<T: Scalar>(a: &Mat<T>, q: &Mat<T>) -> f64 {
    rel(a, &q.matmul(&q.adjoint_matmul(a)))
}

/// `||A - U diag(s) Vh||_F / ||A||_F`.
pub fn svd_err<T: Scalar>(a: &Mat<T>, r: &SvdSketch<T>) -> f64 {
    rel(a, &r.u.scale_columns(&r.s).matmul(&r.vh))
}

/// `||A[:, piv[k..]] - A[:, piv[..k]] T||_F / ||A||_F`.
pub fn id_err<T: Scalar>(a: &Mat<T>, r: &IdDecomp<T>) -> f64 {
    let rest = a.select_columns(&r.piv[r.k..]);
    let skel = a.select_columns(&r.piv[..r.k]);
    (&rest - &skel.matmul(&r.t)).norm_fro() / a.norm_fro()
}

/// Optimal rank-k relative Frobenius error from the full SVD.
pub fn optimal_err<T: Scalar>(a: &Mat<T>, k: usize) -> f64 {
    let s = dense::svd(a).unwrap().1;
    let tail: f64 = s.iter().skip(k).map(|v| v * v).sum();
    let all: f64 = s.iter().map(|v| v * v).sum();
    (tail / all).sqrt()
}

/// `true` if `p` is a permutation of `0..n`.
pub fn is_perm(p: &[usize], n: usize) -> bool {
    let mut seen = vec![false; n];
    p.len() == n && p.iter().all(|&j| j < n && !std::mem::replace(&mut seen[j], true))
}

/// Largest modulus strictly below the diagonal.
pub fn lower_max<T: Scalar>(r: &Mat<T>) -> f64 {
    let mut mx: f64 = 0.0;
    for j in 0..r.ncols() {
        for i in (j + 1)..r.nrows() {
            mx = mx.max(r[(i, j)].abs());
        }
    }
    mx
}

/// Max relative difference of two equal-length real vectors (scaled by `a[0]`).
pub fn vec_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let scale = a.first().copied().unwrap_or(1.0).abs().max(f64::MIN_POSITIVE);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / scale
}

/// A dense matrix behind the operator interface that does NOT expose it
/// (`as_dense` stays `None`), so librla takes the matrix-free code paths.
pub struct Hidden<'a, T>(pub &'a Mat<T>);

impl<T: Scalar> LinearOperator<T> for Hidden<'_, T> {
    fn shape(&self) -> (usize, usize) {
        self.0.shape()
    }
    fn apply(&self, x: &Mat<T>) -> Mat<T> {
        self.0.matmul(x)
    }
    fn apply_adjoint(&self, w: &Mat<T>) -> Mat<T> {
        self.0.adjoint_matmul(w)
    }
}

/// Genuinely matrix-free low-rank product `A = B C`, applied as `B (C x)`.
pub struct LowRank<T> {
    pub b: Mat<T>,
    pub c: Mat<T>,
}

impl<T: Scalar> LinearOperator<T> for LowRank<T> {
    fn shape(&self) -> (usize, usize) {
        (self.b.nrows(), self.c.ncols())
    }
    fn apply(&self, x: &Mat<T>) -> Mat<T> {
        self.b.matmul(&self.c.matmul(x))
    }
    fn apply_adjoint(&self, w: &Mat<T>) -> Mat<T> {
        self.c.adjoint_matmul(&self.b.adjoint_matmul(w))
    }
}
