// librla_lapack.hpp - Fortran LAPACK/BLAS prototypes used by librla.hpp
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)
//
// The symbols are declared here directly (no vendor header), so the same
// declarations link against Apple Accelerate, OpenBLAS or reference
// LAPACK/BLAS.
//
// Conventions:
//   - LP64: every INTEGER argument is a 32-bit int.
//   - std::complex<double> is layout-compatible with COMPLEX*16.
//   - Routines with CHARACTER arguments take the trailing hidden string
//     lengths (one std::size_t per CHARACTER argument, value 1).
//     gfortran-built OpenBLAS reads them; Accelerate's f2c-style legacy
//     interface ignores them, so one declaration serves both.
//   - No routine returning a complex value (zdotc, ...) is declared: the
//     return ABI differs between Accelerate and gfortran.
//
// Symbol binding:
//   - Default: the plain Fortran symbols `name_`, which every backend
//     exports (OpenBLAS, reference LAPACK, and Accelerate's legacy
//     LAPACK 3.2.1 interface).
//   - With LIBRLA_ACCELERATE_NEWLAPACK defined (macOS only), each
//     declaration is bound by an asm label to Accelerate's new LP64
//     interface `name$NEWLAPACK` (LAPACK 3.11, macOS 13.3 or later), which
//     is markedly faster for complex gemm, gesdd and geqp3. The argument
//     lists are the same; the trailing string lengths are ignored. The
//     CMake Accelerate backend defines it unless
//     -DLIBRLA_ACCELERATE_LEGACY=ON. The resulting binary links only
//     against Accelerate.

#ifndef LIBRLA_LAPACK_HPP
#define LIBRLA_LAPACK_HPP

#include <complex>
#include <cstddef>

#if defined(LIBRLA_ACCELERATE_NEWLAPACK)
#  if !defined(__APPLE__)
#    error "LIBRLA_ACCELERATE_NEWLAPACK requires Apple Accelerate (macOS)"
#  endif
#  define LIBRLA_LAPACK_SYMBOL(name) __asm__("_" #name "$NEWLAPACK")
#else
#  define LIBRLA_LAPACK_SYMBOL(name)
#endif

extern "C" {

// Random numbers: idist = 2 gives uniform(-1, 1) (complex: re and im each)
void dlarnv_(const int* idist, int* iseed, const int* n, double* x)
    LIBRLA_LAPACK_SYMBOL(dlarnv);
void zlarnv_(const int* idist, int* iseed, const int* n, std::complex<double>* x)
    LIBRLA_LAPACK_SYMBOL(zlarnv);

// QR with column pivoting
void dgeqp3_(const int* m, const int* n, double* a, const int* lda, int* jpvt,
             double* tau, double* work, const int* lwork, int* info)
    LIBRLA_LAPACK_SYMBOL(dgeqp3);
void zgeqp3_(const int* m, const int* n, std::complex<double>* a, const int* lda,
             int* jpvt, std::complex<double>* tau, std::complex<double>* work,
             const int* lwork, double* rwork, int* info)
    LIBRLA_LAPACK_SYMBOL(zgeqp3);

// Form Q from the Householder reflectors
void dorgqr_(const int* m, const int* n, const int* k, double* a, const int* lda,
             const double* tau, double* work, const int* lwork, int* info)
    LIBRLA_LAPACK_SYMBOL(dorgqr);
void zungqr_(const int* m, const int* n, const int* k, std::complex<double>* a,
             const int* lda, const std::complex<double>* tau,
             std::complex<double>* work, const int* lwork, int* info)
    LIBRLA_LAPACK_SYMBOL(zungqr);

// SVD, divide and conquer
void dgesdd_(const char* jobz, const int* m, const int* n, double* a, const int* lda,
             double* s, double* u, const int* ldu, double* vt, const int* ldvt,
             double* work, const int* lwork, int* iwork, int* info,
             std::size_t jobz_len)
    LIBRLA_LAPACK_SYMBOL(dgesdd);
void zgesdd_(const char* jobz, const int* m, const int* n, std::complex<double>* a,
             const int* lda, double* s, std::complex<double>* u, const int* ldu,
             std::complex<double>* vt, const int* ldvt, std::complex<double>* work,
             const int* lwork, double* rwork, int* iwork, int* info,
             std::size_t jobz_len)
    LIBRLA_LAPACK_SYMBOL(zgesdd);

// SVD, QR iteration (fallback when gesdd fails to converge)
void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n,
             double* a, const int* lda, double* s, double* u, const int* ldu,
             double* vt, const int* ldvt, double* work, const int* lwork, int* info,
             std::size_t jobu_len, std::size_t jobvt_len)
    LIBRLA_LAPACK_SYMBOL(dgesvd);
void zgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n,
             std::complex<double>* a, const int* lda, double* s,
             std::complex<double>* u, const int* ldu, std::complex<double>* vt,
             const int* ldvt, std::complex<double>* work, const int* lwork,
             double* rwork, int* info, std::size_t jobu_len, std::size_t jobvt_len)
    LIBRLA_LAPACK_SYMBOL(zgesvd);

// Minimum-norm least squares via SVD
void dgelsd_(const int* m, const int* n, const int* nrhs, double* a, const int* lda,
             double* b, const int* ldb, double* s, const double* rcond, int* rank,
             double* work, const int* lwork, int* iwork, int* info)
    LIBRLA_LAPACK_SYMBOL(dgelsd);
void zgelsd_(const int* m, const int* n, const int* nrhs, std::complex<double>* a,
             const int* lda, std::complex<double>* b, const int* ldb, double* s,
             const double* rcond, int* rank, std::complex<double>* work,
             const int* lwork, double* rwork, int* iwork, int* info)
    LIBRLA_LAPACK_SYMBOL(zgelsd);

// BLAS 3
void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
            const int* m, const int* n, const double* alpha, const double* a,
            const int* lda, double* b, const int* ldb, std::size_t side_len,
            std::size_t uplo_len, std::size_t transa_len, std::size_t diag_len)
    LIBRLA_LAPACK_SYMBOL(dtrsm);
void ztrsm_(const char* side, const char* uplo, const char* transa, const char* diag,
            const int* m, const int* n, const std::complex<double>* alpha,
            const std::complex<double>* a, const int* lda, std::complex<double>* b,
            const int* ldb, std::size_t side_len, std::size_t uplo_len,
            std::size_t transa_len, std::size_t diag_len)
    LIBRLA_LAPACK_SYMBOL(ztrsm);
void dgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const double* alpha, const double* a, const int* lda,
            const double* b, const int* ldb, const double* beta, double* c,
            const int* ldc, std::size_t transa_len, std::size_t transb_len)
    LIBRLA_LAPACK_SYMBOL(dgemm);
void zgemm_(const char* transa, const char* transb, const int* m, const int* n,
            const int* k, const std::complex<double>* alpha,
            const std::complex<double>* a, const int* lda,
            const std::complex<double>* b, const int* ldb,
            const std::complex<double>* beta, std::complex<double>* c, const int* ldc,
            std::size_t transa_len, std::size_t transb_len)
    LIBRLA_LAPACK_SYMBOL(zgemm);

}  // extern "C"

#endif  // LIBRLA_LAPACK_HPP
