/*
 * librla_lapack.h - BLAS/LAPACK entry points used by the Swift port
 *
 * Each rla_<name> forwards to the Fortran routine <name>_ (see backend.c,
 * which picks the symbol for the selected backend: Accelerate's LAPACK 3.11
 * `<name>$NEWLAPACK` interface by default on Apple platforms, plain
 * `<name>_` for OpenBLAS and reference LAPACK). The choice lives in the C
 * file because a C target's defines do not reach the Swift importer.
 *
 * Conventions: 32-bit (LP64) integers; arrays and scalars of the working
 * type (real*8 or complex*16, interleaved re, im) are passed as void * so
 * generic Swift code can hand over [T] buffers directly, while real-only
 * arguments (singular values, rwork, rcond) stay double *. CHARACTER
 * arguments are single characters; backend.c supplies the hidden string
 * lengths. No routine returning a complex value is used.
 *
 * Author: Adrianna Gillman, Zydrunas Gimbutas
 * SPDX-License-Identifier: MIT
 * Assisted by: Claude Code (Anthropic)
 */

#ifndef LIBRLA_LAPACK_H
#define LIBRLA_LAPACK_H

/* Random numbers: idist = 2 gives uniform(-1, 1). */
void rla_dlarnv(const int *idist, int *iseed, const int *n, void *x);
void rla_zlarnv(const int *idist, int *iseed, const int *n, void *x);

/* QR with column pivoting. */
void rla_dgeqp3(const int *m, const int *n, void *a, const int *lda, int *jpvt,
                void *tau, void *work, const int *lwork, int *info);
void rla_zgeqp3(const int *m, const int *n, void *a, const int *lda, int *jpvt,
                void *tau, void *work, const int *lwork, double *rwork, int *info);

/* Form Q from Householder reflectors. */
void rla_dorgqr(const int *m, const int *n, const int *k, void *a, const int *lda,
                const void *tau, void *work, const int *lwork, int *info);
void rla_zungqr(const int *m, const int *n, const int *k, void *a, const int *lda,
                const void *tau, void *work, const int *lwork, int *info);

/* SVD, divide and conquer. */
void rla_dgesdd(const char *jobz, const int *m, const int *n, void *a, const int *lda,
                double *s, void *u, const int *ldu, void *vt, const int *ldvt,
                void *work, const int *lwork, int *iwork, int *info);
void rla_zgesdd(const char *jobz, const int *m, const int *n, void *a, const int *lda,
                double *s, void *u, const int *ldu, void *vt, const int *ldvt,
                void *work, const int *lwork, double *rwork, int *iwork, int *info);

/* SVD, QR iteration (fallback when gesdd fails to converge). */
void rla_dgesvd(const char *jobu, const char *jobvt, const int *m, const int *n,
                void *a, const int *lda, double *s, void *u, const int *ldu,
                void *vt, const int *ldvt, void *work, const int *lwork, int *info);
void rla_zgesvd(const char *jobu, const char *jobvt, const int *m, const int *n,
                void *a, const int *lda, double *s, void *u, const int *ldu,
                void *vt, const int *ldvt, void *work, const int *lwork,
                double *rwork, int *info);

/* Minimum-norm least squares via SVD. */
void rla_dgelsd(const int *m, const int *n, const int *nrhs, void *a, const int *lda,
                void *b, const int *ldb, double *s, const double *rcond, int *rank,
                void *work, const int *lwork, int *iwork, int *info);
void rla_zgelsd(const int *m, const int *n, const int *nrhs, void *a, const int *lda,
                void *b, const int *ldb, double *s, const double *rcond, int *rank,
                void *work, const int *lwork, double *rwork, int *iwork, int *info);

/* Triangular solve with multiple right-hand sides. */
void rla_dtrsm(const char *side, const char *uplo, const char *transa, const char *diag,
               const int *m, const int *n, const void *alpha, const void *a,
               const int *lda, void *b, const int *ldb);
void rla_ztrsm(const char *side, const char *uplo, const char *transa, const char *diag,
               const int *m, const int *n, const void *alpha, const void *a,
               const int *lda, void *b, const int *ldb);

/* General matrix product. */
void rla_dgemm(const char *transa, const char *transb, const int *m, const int *n,
               const int *k, const void *alpha, const void *a, const int *lda,
               const void *b, const int *ldb, const void *beta, void *c, const int *ldc);
void rla_zgemm(const char *transa, const char *transb, const int *m, const int *n,
               const int *k, const void *alpha, const void *a, const int *lda,
               const void *b, const int *ldb, const void *beta, void *c, const int *ldc);

#endif /* LIBRLA_LAPACK_H */
