// Hand-written declarations of the Fortran LAPACK/BLAS symbols used by librla.
//
// LP64 (32-bit integer) interface. Routines with CHARACTER arguments take the
// trailing hidden string lengths (size_t, value 1) expected by gfortran-built
// libraries such as OpenBLAS; Accelerate's legacy interface ignores them, so
// one declaration serves both. No BLAS function returning a complex value is
// declared (the return ABI differs between backends).
//
// With the Accelerate backend (unless `accelerate-legacy` is selected),
// build.rs sets `librla_newlapack` and every routine binds to Accelerate's
// new LAPACK interface (`<name>$NEWLAPACK`, LP64, LAPACK 3.11, macOS 13.3+),
// which is markedly faster for the complex routines than the legacy
// `<name>_` symbols. Signatures are the same; the hidden lengths are ignored.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#![allow(clippy::too_many_arguments)]

use num_complex::Complex64 as C64;
use std::os::raw::c_char;

pub(crate) type Int = i32;

extern "C" {
    #[cfg_attr(librla_newlapack, link_name = "dlarnv$NEWLAPACK")]
    pub(crate) fn dlarnv_(idist: *const Int, iseed: *mut Int, n: *const Int, x: *mut f64);
    #[cfg_attr(librla_newlapack, link_name = "zlarnv$NEWLAPACK")]
    pub(crate) fn zlarnv_(idist: *const Int, iseed: *mut Int, n: *const Int, x: *mut C64);

    #[cfg_attr(librla_newlapack, link_name = "dgeqp3$NEWLAPACK")]
    pub(crate) fn dgeqp3_(
        m: *const Int, n: *const Int, a: *mut f64, lda: *const Int, jpvt: *mut Int, tau: *mut f64,
        work: *mut f64, lwork: *const Int, info: *mut Int,
    );
    #[cfg_attr(librla_newlapack, link_name = "zgeqp3$NEWLAPACK")]
    pub(crate) fn zgeqp3_(
        m: *const Int, n: *const Int, a: *mut C64, lda: *const Int, jpvt: *mut Int, tau: *mut C64,
        work: *mut C64, lwork: *const Int, rwork: *mut f64, info: *mut Int,
    );

    #[cfg_attr(librla_newlapack, link_name = "dorgqr$NEWLAPACK")]
    pub(crate) fn dorgqr_(
        m: *const Int, n: *const Int, k: *const Int, a: *mut f64, lda: *const Int, tau: *const f64,
        work: *mut f64, lwork: *const Int, info: *mut Int,
    );
    #[cfg_attr(librla_newlapack, link_name = "zungqr$NEWLAPACK")]
    pub(crate) fn zungqr_(
        m: *const Int, n: *const Int, k: *const Int, a: *mut C64, lda: *const Int, tau: *const C64,
        work: *mut C64, lwork: *const Int, info: *mut Int,
    );

    #[cfg_attr(librla_newlapack, link_name = "dgesdd$NEWLAPACK")]
    pub(crate) fn dgesdd_(
        jobz: *const c_char, m: *const Int, n: *const Int, a: *mut f64, lda: *const Int,
        s: *mut f64, u: *mut f64, ldu: *const Int, vt: *mut f64, ldvt: *const Int, work: *mut f64,
        lwork: *const Int, iwork: *mut Int, info: *mut Int, jobz_len: usize,
    );
    #[cfg_attr(librla_newlapack, link_name = "zgesdd$NEWLAPACK")]
    pub(crate) fn zgesdd_(
        jobz: *const c_char, m: *const Int, n: *const Int, a: *mut C64, lda: *const Int,
        s: *mut f64, u: *mut C64, ldu: *const Int, vt: *mut C64, ldvt: *const Int, work: *mut C64,
        lwork: *const Int, rwork: *mut f64, iwork: *mut Int, info: *mut Int, jobz_len: usize,
    );

    #[cfg_attr(librla_newlapack, link_name = "dgesvd$NEWLAPACK")]
    pub(crate) fn dgesvd_(
        jobu: *const c_char, jobvt: *const c_char, m: *const Int, n: *const Int, a: *mut f64,
        lda: *const Int, s: *mut f64, u: *mut f64, ldu: *const Int, vt: *mut f64, ldvt: *const Int,
        work: *mut f64, lwork: *const Int, info: *mut Int, jobu_len: usize, jobvt_len: usize,
    );
    #[cfg_attr(librla_newlapack, link_name = "zgesvd$NEWLAPACK")]
    pub(crate) fn zgesvd_(
        jobu: *const c_char, jobvt: *const c_char, m: *const Int, n: *const Int, a: *mut C64,
        lda: *const Int, s: *mut f64, u: *mut C64, ldu: *const Int, vt: *mut C64, ldvt: *const Int,
        work: *mut C64, lwork: *const Int, rwork: *mut f64, info: *mut Int, jobu_len: usize,
        jobvt_len: usize,
    );

    #[cfg_attr(librla_newlapack, link_name = "dgelsd$NEWLAPACK")]
    pub(crate) fn dgelsd_(
        m: *const Int, n: *const Int, nrhs: *const Int, a: *mut f64, lda: *const Int, b: *mut f64,
        ldb: *const Int, s: *mut f64, rcond: *const f64, rank: *mut Int, work: *mut f64,
        lwork: *const Int, iwork: *mut Int, info: *mut Int,
    );
    #[cfg_attr(librla_newlapack, link_name = "zgelsd$NEWLAPACK")]
    pub(crate) fn zgelsd_(
        m: *const Int, n: *const Int, nrhs: *const Int, a: *mut C64, lda: *const Int, b: *mut C64,
        ldb: *const Int, s: *mut f64, rcond: *const f64, rank: *mut Int, work: *mut C64,
        lwork: *const Int, rwork: *mut f64, iwork: *mut Int, info: *mut Int,
    );

    #[cfg_attr(librla_newlapack, link_name = "dtrsm$NEWLAPACK")]
    pub(crate) fn dtrsm_(
        side: *const c_char, uplo: *const c_char, transa: *const c_char, diag: *const c_char,
        m: *const Int, n: *const Int, alpha: *const f64, a: *const f64, lda: *const Int,
        b: *mut f64, ldb: *const Int, side_len: usize, uplo_len: usize, transa_len: usize,
        diag_len: usize,
    );
    #[cfg_attr(librla_newlapack, link_name = "ztrsm$NEWLAPACK")]
    pub(crate) fn ztrsm_(
        side: *const c_char, uplo: *const c_char, transa: *const c_char, diag: *const c_char,
        m: *const Int, n: *const Int, alpha: *const C64, a: *const C64, lda: *const Int,
        b: *mut C64, ldb: *const Int, side_len: usize, uplo_len: usize, transa_len: usize,
        diag_len: usize,
    );

    #[cfg_attr(librla_newlapack, link_name = "dgemm$NEWLAPACK")]
    pub(crate) fn dgemm_(
        transa: *const c_char, transb: *const c_char, m: *const Int, n: *const Int, k: *const Int,
        alpha: *const f64, a: *const f64, lda: *const Int, b: *const f64, ldb: *const Int,
        beta: *const f64, c: *mut f64, ldc: *const Int, transa_len: usize, transb_len: usize,
    );
    #[cfg_attr(librla_newlapack, link_name = "zgemm$NEWLAPACK")]
    pub(crate) fn zgemm_(
        transa: *const c_char, transb: *const c_char, m: *const Int, n: *const Int, k: *const Int,
        alpha: *const C64, a: *const C64, lda: *const Int, b: *const C64, ldb: *const Int,
        beta: *const C64, c: *mut C64, ldc: *const Int, transa_len: usize, transb_len: usize,
    );
}
