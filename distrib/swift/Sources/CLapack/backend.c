/*
 * backend.c - binds the rla_* entry points to the selected BLAS/LAPACK
 *
 * Default on Apple platforms: Accelerate's LAPACK 3.11 interface, reached
 * by giving each Fortran prototype the asm label `<name>$NEWLAPACK` (LP64,
 * macOS 13.3+). The legacy unsuffixed Accelerate symbols are LAPACK 3.2.1
 * and much slower for complex gemm/gesdd and for geqp3. With
 * LIBRLA_OPENBLAS defined (Package.swift sets it for OpenBLAS and Linux)
 * the plain Fortran names `<name>_` are used.
 *
 * Routines with CHARACTER arguments get one trailing hidden string length
 * (size_t, value 1) per CHARACTER argument: gfortran-built libraries expect
 * them and Accelerate ignores them.
 *
 * Author: Adrianna Gillman, Zydrunas Gimbutas
 * SPDX-License-Identifier: MIT
 * Assisted by: Claude Code (Anthropic)
 */

#include <stddef.h>
#include "librla_lapack.h"

#if defined(__APPLE__) && !defined(LIBRLA_OPENBLAS)
#define SYMBOL(name) __asm__("_" #name "$NEWLAPACK")
#else
#define SYMBOL(name)
#endif

/* Fortran prototypes. */
void dlarnv_(const int *, int *, const int *, void *) SYMBOL(dlarnv);
void zlarnv_(const int *, int *, const int *, void *) SYMBOL(zlarnv);
void dgeqp3_(const int *, const int *, void *, const int *, int *, void *, void *,
             const int *, int *) SYMBOL(dgeqp3);
void zgeqp3_(const int *, const int *, void *, const int *, int *, void *, void *,
             const int *, double *, int *) SYMBOL(zgeqp3);
void dorgqr_(const int *, const int *, const int *, void *, const int *, const void *,
             void *, const int *, int *) SYMBOL(dorgqr);
void zungqr_(const int *, const int *, const int *, void *, const int *, const void *,
             void *, const int *, int *) SYMBOL(zungqr);
void dgesdd_(const char *, const int *, const int *, void *, const int *, double *,
             void *, const int *, void *, const int *, void *, const int *, int *,
             int *, size_t) SYMBOL(dgesdd);
void zgesdd_(const char *, const int *, const int *, void *, const int *, double *,
             void *, const int *, void *, const int *, void *, const int *, double *,
             int *, int *, size_t) SYMBOL(zgesdd);
void dgesvd_(const char *, const char *, const int *, const int *, void *, const int *,
             double *, void *, const int *, void *, const int *, void *, const int *,
             int *, size_t, size_t) SYMBOL(dgesvd);
void zgesvd_(const char *, const char *, const int *, const int *, void *, const int *,
             double *, void *, const int *, void *, const int *, void *, const int *,
             double *, int *, size_t, size_t) SYMBOL(zgesvd);
void dgelsd_(const int *, const int *, const int *, void *, const int *, void *,
             const int *, double *, const double *, int *, void *, const int *, int *,
             int *) SYMBOL(dgelsd);
void zgelsd_(const int *, const int *, const int *, void *, const int *, void *,
             const int *, double *, const double *, int *, void *, const int *,
             double *, int *, int *) SYMBOL(zgelsd);
void dtrsm_(const char *, const char *, const char *, const char *, const int *,
            const int *, const void *, const void *, const int *, void *, const int *,
            size_t, size_t, size_t, size_t) SYMBOL(dtrsm);
void ztrsm_(const char *, const char *, const char *, const char *, const int *,
            const int *, const void *, const void *, const int *, void *, const int *,
            size_t, size_t, size_t, size_t) SYMBOL(ztrsm);
void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const void *, const void *, const int *, const void *, const int *,
            const void *, void *, const int *, size_t, size_t) SYMBOL(dgemm);
void zgemm_(const char *, const char *, const int *, const int *, const int *,
            const void *, const void *, const int *, const void *, const int *,
            const void *, void *, const int *, size_t, size_t) SYMBOL(zgemm);

/* Forwarding entry points. */
void rla_dlarnv(const int *idist, int *iseed, const int *n, void *x) {
    dlarnv_(idist, iseed, n, x);
}
void rla_zlarnv(const int *idist, int *iseed, const int *n, void *x) {
    zlarnv_(idist, iseed, n, x);
}
void rla_dgeqp3(const int *m, const int *n, void *a, const int *lda, int *jpvt,
                void *tau, void *work, const int *lwork, int *info) {
    dgeqp3_(m, n, a, lda, jpvt, tau, work, lwork, info);
}
void rla_zgeqp3(const int *m, const int *n, void *a, const int *lda, int *jpvt,
                void *tau, void *work, const int *lwork, double *rwork, int *info) {
    zgeqp3_(m, n, a, lda, jpvt, tau, work, lwork, rwork, info);
}
void rla_dorgqr(const int *m, const int *n, const int *k, void *a, const int *lda,
                const void *tau, void *work, const int *lwork, int *info) {
    dorgqr_(m, n, k, a, lda, tau, work, lwork, info);
}
void rla_zungqr(const int *m, const int *n, const int *k, void *a, const int *lda,
                const void *tau, void *work, const int *lwork, int *info) {
    zungqr_(m, n, k, a, lda, tau, work, lwork, info);
}
void rla_dgesdd(const char *jobz, const int *m, const int *n, void *a, const int *lda,
                double *s, void *u, const int *ldu, void *vt, const int *ldvt,
                void *work, const int *lwork, int *iwork, int *info) {
    dgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info, 1);
}
void rla_zgesdd(const char *jobz, const int *m, const int *n, void *a, const int *lda,
                double *s, void *u, const int *ldu, void *vt, const int *ldvt,
                void *work, const int *lwork, double *rwork, int *iwork, int *info) {
    zgesdd_(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info, 1);
}
void rla_dgesvd(const char *jobu, const char *jobvt, const int *m, const int *n,
                void *a, const int *lda, double *s, void *u, const int *ldu,
                void *vt, const int *ldvt, void *work, const int *lwork, int *info) {
    dgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info, 1, 1);
}
void rla_zgesvd(const char *jobu, const char *jobvt, const int *m, const int *n,
                void *a, const int *lda, double *s, void *u, const int *ldu,
                void *vt, const int *ldvt, void *work, const int *lwork,
                double *rwork, int *info) {
    zgesvd_(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info, 1, 1);
}
void rla_dgelsd(const int *m, const int *n, const int *nrhs, void *a, const int *lda,
                void *b, const int *ldb, double *s, const double *rcond, int *rank,
                void *work, const int *lwork, int *iwork, int *info) {
    dgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info);
}
void rla_zgelsd(const int *m, const int *n, const int *nrhs, void *a, const int *lda,
                void *b, const int *ldb, double *s, const double *rcond, int *rank,
                void *work, const int *lwork, double *rwork, int *iwork, int *info) {
    zgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, rwork, iwork, info);
}
void rla_dtrsm(const char *side, const char *uplo, const char *transa, const char *diag,
               const int *m, const int *n, const void *alpha, const void *a,
               const int *lda, void *b, const int *ldb) {
    dtrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, 1, 1, 1, 1);
}
void rla_ztrsm(const char *side, const char *uplo, const char *transa, const char *diag,
               const int *m, const int *n, const void *alpha, const void *a,
               const int *lda, void *b, const int *ldb) {
    ztrsm_(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, 1, 1, 1, 1);
}
void rla_dgemm(const char *transa, const char *transb, const int *m, const int *n,
               const int *k, const void *alpha, const void *a, const int *lda,
               const void *b, const int *ldb, const void *beta, void *c, const int *ldc) {
    dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 1, 1);
}
void rla_zgemm(const char *transa, const char *transb, const int *m, const int *n,
               const int *k, const void *alpha, const void *a, const int *lda,
               const void *b, const int *ldb, const void *beta, void *c, const int *ldc) {
    zgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 1, 1);
}
