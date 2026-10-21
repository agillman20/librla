// Scalar types (f64 and Complex64) and their dispatch to d/z LAPACK routines.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

use crate::ffi::{self, Int};
use num_complex::Complex64;
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use std::os::raw::c_char;

/// Element type of librla matrices: implemented for `f64` and
/// [`Complex64`](num_complex::Complex64) only (sealed).
pub trait Scalar:
    sealed::Lapack
    + Copy
    + Debug
    + Display
    + Default
    + PartialEq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// `true` for `Complex64`.
    const IS_COMPLEX: bool;
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Embed a real number.
    fn from_real(x: f64) -> Self;
    /// Complex conjugate (identity for `f64`).
    fn conj(self) -> Self;
    /// Absolute value (modulus).
    fn abs(self) -> f64;
    /// Real part.
    fn re(self) -> f64;
    /// Imaginary part (0 for `f64`).
    fn im(self) -> f64;
    /// `true` when all parts are finite.
    fn is_finite(self) -> bool;
}

impl Scalar for f64 {
    const IS_COMPLEX: bool = false;
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_real(x: f64) -> Self {
        x
    }
    fn conj(self) -> Self {
        self
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn re(self) -> f64 {
        self
    }
    fn im(self) -> f64 {
        0.0
    }
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

impl Scalar for Complex64 {
    const IS_COMPLEX: bool = true;
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
    fn from_real(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }
    fn abs(self) -> f64 {
        self.norm()
    }
    fn re(self) -> f64 {
        self.re
    }
    fn im(self) -> f64 {
        self.im
    }
    fn is_finite(self) -> bool {
        Complex64::is_finite(self)
    }
}

/// Raw LAPACK/BLAS entry points with one signature per routine for both
/// precisions; `rwork` is ignored by the real (d) routines. The trait lives
/// in a private module, so these methods are not callable outside the crate.
pub(crate) mod sealed {
    use super::*;

    #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
    pub trait Lapack: Sized {
        unsafe fn larnv(idist: Int, iseed: *mut Int, n: Int, x: *mut Self);
        unsafe fn geqp3(
            m: Int, n: Int, a: *mut Self, lda: Int, jpvt: *mut Int, tau: *mut Self,
            work: *mut Self, lwork: Int, rwork: *mut f64, info: &mut Int,
        );
        unsafe fn orgqr(
            m: Int, n: Int, k: Int, a: *mut Self, lda: Int, tau: *const Self, work: *mut Self,
            lwork: Int, info: &mut Int,
        );
        unsafe fn gesdd(
            jobz: u8, m: Int, n: Int, a: *mut Self, lda: Int, s: *mut f64, u: *mut Self, ldu: Int,
            vt: *mut Self, ldvt: Int, work: *mut Self, lwork: Int, rwork: *mut f64,
            iwork: *mut Int, info: &mut Int,
        );
        unsafe fn gesvd(
            jobu: u8, jobvt: u8, m: Int, n: Int, a: *mut Self, lda: Int, s: *mut f64, u: *mut Self,
            ldu: Int, vt: *mut Self, ldvt: Int, work: *mut Self, lwork: Int, rwork: *mut f64,
            info: &mut Int,
        );
        unsafe fn gelsd(
            m: Int, n: Int, nrhs: Int, a: *mut Self, lda: Int, b: *mut Self, ldb: Int, s: *mut f64,
            rcond: f64, rank: &mut Int, work: *mut Self, lwork: Int, rwork: *mut f64,
            iwork: *mut Int, info: &mut Int,
        );
        unsafe fn trsm(
            side: u8, uplo: u8, transa: u8, diag: u8, m: Int, n: Int, alpha: Self, a: *const Self,
            lda: Int, b: *mut Self, ldb: Int,
        );
        unsafe fn gemm(
            transa: u8, transb: u8, m: Int, n: Int, k: Int, alpha: Self, a: *const Self, lda: Int,
            b: *const Self, ldb: Int, beta: Self, c: *mut Self, ldc: Int,
        );
        /// Real part of a LAPACK workspace-size answer.
        fn work_size(self) -> f64;
    }

    fn ch(c: &u8) -> *const c_char {
        c as *const u8 as *const c_char
    }

    impl Lapack for f64 {
        unsafe fn larnv(idist: Int, iseed: *mut Int, n: Int, x: *mut Self) {
            ffi::dlarnv_(&idist, iseed, &n, x)
        }
        unsafe fn geqp3(
            m: Int, n: Int, a: *mut Self, lda: Int, jpvt: *mut Int, tau: *mut Self,
            work: *mut Self, lwork: Int, _rwork: *mut f64, info: &mut Int,
        ) {
            ffi::dgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, info)
        }
        unsafe fn orgqr(
            m: Int, n: Int, k: Int, a: *mut Self, lda: Int, tau: *const Self, work: *mut Self,
            lwork: Int, info: &mut Int,
        ) {
            ffi::dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info)
        }
        unsafe fn gesdd(
            jobz: u8, m: Int, n: Int, a: *mut Self, lda: Int, s: *mut f64, u: *mut Self, ldu: Int,
            vt: *mut Self, ldvt: Int, work: *mut Self, lwork: Int, _rwork: *mut f64,
            iwork: *mut Int, info: &mut Int,
        ) {
            ffi::dgesdd_(
                ch(&jobz),
                &m,
                &n,
                a,
                &lda,
                s,
                u,
                &ldu,
                vt,
                &ldvt,
                work,
                &lwork,
                iwork,
                info,
                1,
            )
        }
        unsafe fn gesvd(
            jobu: u8, jobvt: u8, m: Int, n: Int, a: *mut Self, lda: Int, s: *mut f64, u: *mut Self,
            ldu: Int, vt: *mut Self, ldvt: Int, work: *mut Self, lwork: Int, _rwork: *mut f64,
            info: &mut Int,
        ) {
            ffi::dgesvd_(
                ch(&jobu),
                ch(&jobvt),
                &m,
                &n,
                a,
                &lda,
                s,
                u,
                &ldu,
                vt,
                &ldvt,
                work,
                &lwork,
                info,
                1,
                1,
            )
        }
        unsafe fn gelsd(
            m: Int, n: Int, nrhs: Int, a: *mut Self, lda: Int, b: *mut Self, ldb: Int, s: *mut f64,
            rcond: f64, rank: &mut Int, work: *mut Self, lwork: Int, _rwork: *mut f64,
            iwork: *mut Int, info: &mut Int,
        ) {
            ffi::dgelsd_(
                &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork, iwork, info,
            )
        }
        unsafe fn trsm(
            side: u8, uplo: u8, transa: u8, diag: u8, m: Int, n: Int, alpha: Self, a: *const Self,
            lda: Int, b: *mut Self, ldb: Int,
        ) {
            ffi::dtrsm_(
                ch(&side),
                ch(&uplo),
                ch(&transa),
                ch(&diag),
                &m,
                &n,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
                1,
                1,
                1,
                1,
            )
        }
        unsafe fn gemm(
            transa: u8, transb: u8, m: Int, n: Int, k: Int, alpha: Self, a: *const Self, lda: Int,
            b: *const Self, ldb: Int, beta: Self, c: *mut Self, ldc: Int,
        ) {
            ffi::dgemm_(
                ch(&transa),
                ch(&transb),
                &m,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
                &beta,
                c,
                &ldc,
                1,
                1,
            )
        }
        fn work_size(self) -> f64 {
            self
        }
    }

    impl Lapack for Complex64 {
        unsafe fn larnv(idist: Int, iseed: *mut Int, n: Int, x: *mut Self) {
            ffi::zlarnv_(&idist, iseed, &n, x)
        }
        unsafe fn geqp3(
            m: Int, n: Int, a: *mut Self, lda: Int, jpvt: *mut Int, tau: *mut Self,
            work: *mut Self, lwork: Int, rwork: *mut f64, info: &mut Int,
        ) {
            ffi::zgeqp3_(&m, &n, a, &lda, jpvt, tau, work, &lwork, rwork, info)
        }
        unsafe fn orgqr(
            m: Int, n: Int, k: Int, a: *mut Self, lda: Int, tau: *const Self, work: *mut Self,
            lwork: Int, info: &mut Int,
        ) {
            ffi::zungqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info)
        }
        unsafe fn gesdd(
            jobz: u8, m: Int, n: Int, a: *mut Self, lda: Int, s: *mut f64, u: *mut Self, ldu: Int,
            vt: *mut Self, ldvt: Int, work: *mut Self, lwork: Int, rwork: *mut f64,
            iwork: *mut Int, info: &mut Int,
        ) {
            ffi::zgesdd_(
                ch(&jobz),
                &m,
                &n,
                a,
                &lda,
                s,
                u,
                &ldu,
                vt,
                &ldvt,
                work,
                &lwork,
                rwork,
                iwork,
                info,
                1,
            )
        }
        unsafe fn gesvd(
            jobu: u8, jobvt: u8, m: Int, n: Int, a: *mut Self, lda: Int, s: *mut f64, u: *mut Self,
            ldu: Int, vt: *mut Self, ldvt: Int, work: *mut Self, lwork: Int, rwork: *mut f64,
            info: &mut Int,
        ) {
            ffi::zgesvd_(
                ch(&jobu),
                ch(&jobvt),
                &m,
                &n,
                a,
                &lda,
                s,
                u,
                &ldu,
                vt,
                &ldvt,
                work,
                &lwork,
                rwork,
                info,
                1,
                1,
            )
        }
        unsafe fn gelsd(
            m: Int, n: Int, nrhs: Int, a: *mut Self, lda: Int, b: *mut Self, ldb: Int, s: *mut f64,
            rcond: f64, rank: &mut Int, work: *mut Self, lwork: Int, rwork: *mut f64,
            iwork: *mut Int, info: &mut Int,
        ) {
            ffi::zgelsd_(
                &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork, rwork, iwork, info,
            )
        }
        unsafe fn trsm(
            side: u8, uplo: u8, transa: u8, diag: u8, m: Int, n: Int, alpha: Self, a: *const Self,
            lda: Int, b: *mut Self, ldb: Int,
        ) {
            ffi::ztrsm_(
                ch(&side),
                ch(&uplo),
                ch(&transa),
                ch(&diag),
                &m,
                &n,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
                1,
                1,
                1,
                1,
            )
        }
        unsafe fn gemm(
            transa: u8, transb: u8, m: Int, n: Int, k: Int, alpha: Self, a: *const Self, lda: Int,
            b: *const Self, ldb: Int, beta: Self, c: *mut Self, ldc: Int,
        ) {
            ffi::zgemm_(
                ch(&transa),
                ch(&transb),
                &m,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
                &beta,
                c,
                &ldc,
                1,
                1,
            )
        }
        fn work_size(self) -> f64 {
            self.re
        }
    }
}
