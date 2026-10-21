// librla.hpp - Randomized linear-algebra routines (C++17, header-only)
//
// Randomized algorithms for low-rank matrix approximations:
//     orth_sketch  - Approximate orthonormal basis for column space
//     qr_sketch    - Truncated QR factorization with column pivoting
//     svd_sketch   - Truncated singular value decomposition (SVD)
//     id_sketch    - Interpolative decomposition (ID)
//
// Deterministic:
//     id_qrpiv     - Interpolative decomposition via QR with pivoting
//
// Usage:
//
//     auto [Q, flag, diagR] = librla::orth_sketch(A, rtol);
//     auto [Q, R, p]        = librla::qr_sketch(A, rtol);
//     auto [U, s, Vh]       = librla::svd_sketch(A, rtol);
//     auto [k, piv, T]      = librla::id_sketch(A, rtol);
//     // tolerance mode: rtol < 1, rank mode: rtol >= 1
//
// A is a librla::Matrix<S> (column-major, dense) or a
// librla::LinearOperator<S> (matrix-free: block applies of A and A^H),
// with S = double or std::complex<double>. All dense linear algebra goes
// through LAPACK/BLAS (see librla_lapack.hpp). Indices are 0-based.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#ifndef LIBRLA_HPP
#define LIBRLA_HPP

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "librla_lapack.hpp"

namespace librla {

// The one version string of the C++ port.
inline constexpr const char* version = "0.1.0";

using cdouble = std::complex<double>;

// ==============================================================
// Matrix: minimal column-major dense matrix
// ==============================================================

// Dimensions are int, the LAPACK (LP64) integer type: every dimension that
// reaches librla is eventually passed to LAPACK.
template <class S>
class Matrix {
public:
    using value_type = S;

    Matrix() = default;

    // rows x cols matrix of zeros
    Matrix(int rows, int cols)
        : rows_(check_dim(rows)), cols_(check_dim(cols)),
          data_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) {}

    // rows x cols matrix from column-major data
    Matrix(int rows, int cols, std::vector<S> data)
        : rows_(check_dim(rows)), cols_(check_dim(cols)), data_(std::move(data)) {
        if (data_.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols))
            throw std::invalid_argument("librla::Matrix: data size != rows*cols");
    }

    static Matrix identity(int n) {
        Matrix I(n, n);
        for (int i = 0; i < n; ++i) I(i, i) = S(1);
        return I;
    }

    int rows() const noexcept { return rows_; }
    int cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    S* data() noexcept { return data_.data(); }
    const S* data() const noexcept { return data_.data(); }
    S* col(int j) noexcept { return data_.data() + offset(0, j); }
    const S* col(int j) const noexcept { return data_.data() + offset(0, j); }
    const std::vector<S>& storage() const noexcept { return data_; }

    S& operator()(int i, int j) noexcept { return data_[offset(i, j)]; }
    const S& operator()(int i, int j) const noexcept { return data_[offset(i, j)]; }

private:
    static int check_dim(int d) {
        if (d < 0) throw std::invalid_argument("librla::Matrix: negative dimension");
        return d;
    }
    std::size_t offset(int i, int j) const noexcept {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(j) * static_cast<std::size_t>(rows_);
    }

    int rows_ = 0;
    int cols_ = 0;
    std::vector<S> data_;
};

// ==============================================================
// Rng: LAPACK dlarnv/zlarnv generator state (iseed[4])
// ==============================================================

// Test matrices are drawn with dlarnv/zlarnv (idist = 2: uniform(-1,1);
// complex: real and imaginary parts each uniform(-1,1)), filling the
// n x block_size matrix column-major. The state is the LAPACK iseed array
// (entries in [0, 4095], iseed[3] odd), advanced in place by every draw.
// The same seed gives the same sketch in the C++, Rust, Swift and Fortran
// ports.
class Rng {
public:
    // Default state {0, 0, 0, 1}: deterministic results.
    Rng() noexcept : iseed_{{0, 0, 0, 1}} {}

    Rng(int i0, int i1, int i2, int i3) : iseed_{{i0, i1, i2, i3}} { validate(); }

    explicit Rng(const std::array<int, 4>& iseed) : iseed_(iseed) { validate(); }

    // From a 64-bit seed s:
    // iseed = {(s>>36)&4095, (s>>24)&4095, (s>>12)&4095, (s&4095)|1}
    explicit Rng(std::uint64_t seed) noexcept
        : iseed_{{static_cast<int>((seed >> 36) & 4095u),
                  static_cast<int>((seed >> 24) & 4095u),
                  static_cast<int>((seed >> 12) & 4095u),
                  static_cast<int>((seed & 4095u) | 1u)}} {}

    const std::array<int, 4>& state() const noexcept { return iseed_; }

    // Fill x[0..n) with uniform(-1,1) values (complex: re and im each).
    void fill(double* x, std::size_t n) { fill_impl(x, n); }
    void fill(cdouble* x, std::size_t n) { fill_impl(x, n); }

    // rows x cols uniform(-1,1) matrix, column-major.
    template <class S>
    Matrix<S> uniform(int rows, int cols) {
        Matrix<S> X(rows, cols);
        fill(X.data(), X.size());
        return X;
    }

private:
    void validate() const {
        for (int v : iseed_)
            if (v < 0 || v > 4095)
                throw std::invalid_argument("librla::Rng: iseed entries must be in [0, 4095]");
        if (iseed_[3] % 2 == 0)
            throw std::invalid_argument("librla::Rng: iseed[3] must be odd");
    }

    // One LAPACK call for any n < 2^31. Larger requests are split at
    // multiples of 64 values, the chunk size dlarnv/zlarnv use internally,
    // so the stream is the same as a single call.
    template <class S>
    void fill_impl(S* x, std::size_t n) {
        const std::size_t chunk = static_cast<std::size_t>(INT_MAX / 64) * 64;
        const int idist = 2;
        while (n > 0) {
            const int len = static_cast<int>(std::min(n, chunk));
            if constexpr (std::is_same_v<S, double>)
                dlarnv_(&idist, iseed_.data(), &len, x);
            else
                zlarnv_(&idist, iseed_.data(), &len, x);
            x += len;
            n -= static_cast<std::size_t>(len);
        }
    }

    std::array<int, 4> iseed_;
};

// ==============================================================
// LinearOperator: matrix-free operator
// ==============================================================

// rows x cols operator given by block applies:
//   apply(X)         = A   X   (X: cols x k  ->  rows x k)
//   apply_adjoint(W) = A^H W   (W: rows x k  ->  cols x k)
// If `dense` is set, the operator IS that matrix: librla then uses it
// directly (no materialization, BLAS products), as for a Matrix argument.
template <class S>
struct LinearOperator {
    using Apply = std::function<Matrix<S>(const Matrix<S>&)>;

    int rows = 0;
    int cols = 0;
    Apply apply;
    Apply apply_adjoint;
    const Matrix<S>* dense = nullptr;

    LinearOperator() = default;
    LinearOperator(int rows_, int cols_, Apply apply_, Apply apply_adjoint_,
                   const Matrix<S>* dense_ = nullptr)
        : rows(rows_), cols(cols_), apply(std::move(apply_)),
          apply_adjoint(std::move(apply_adjoint_)), dense(dense_) {}

    // A^H as an operator (applies swapped; `dense` is not carried over).
    LinearOperator adjoint() const {
        return LinearOperator(cols, rows, apply_adjoint, apply, nullptr);
    }

    // Operator view of a dense matrix (keeps a reference to A, which must
    // outlive the operator). With expose_dense = false librla treats it as
    // matrix-free.
    static LinearOperator from_matrix(const Matrix<S>& A, bool expose_dense = true);
};

// ==============================================================
// Options and results
// ==============================================================

enum class IdMethod {
    fast,   // triangular solve R11 \ R12 (fastest, default)
    svd,    // SVD-based pseudoinverse of R11
    lstsq   // least squares from the original A columns (most accurate)
};

// "fast" | "svd" | "lstsq"; anything else throws std::invalid_argument.
inline IdMethod parse_id_method(const std::string& name) {
    if (name == "fast") return IdMethod::fast;
    if (name == "svd") return IdMethod::svd;
    if (name == "lstsq") return IdMethod::lstsq;
    throw std::invalid_argument("librla: method must be one of {fast, svd, lstsq}, got '" +
                                name + "'");
}

inline const char* to_string(IdMethod method) {
    switch (method) {
        case IdMethod::fast: return "fast";
        case IdMethod::svd: return "svd";
        case IdMethod::lstsq: return "lstsq";
    }
    throw std::invalid_argument("librla: method must be one of {fast, svd, lstsq}");
}

struct Options {
    int block_size = 42;      // initial sketch size, tolerance mode (ignored in rank mode)
    int power_iter = 0;       // power iterations; 1 or 2 help slowly decaying spectra
    int extra_samples = 12;   // buffer columns beyond the target rank (both modes)
    Rng* rng = nullptr;       // advanced in place; nullptr: fresh Rng{0,0,0,1}
    IdMethod method = IdMethod::fast;  // id_sketch only
};

template <class S>
struct OrthResult {
    Matrix<S> Q;                 // m x k orthonormal (k includes the buffer)
    int flag = 0;                // 0 success; 1 early termination (Q is m x 0)
    std::vector<double> diagR;   // |diag(R)| of the pivoted QR of the sketch
};

template <class S>
struct QRResult {
    Matrix<S> Q;          // m x k orthonormal
    Matrix<S> R;          // k x n upper triangular
    std::vector<int> p;   // column permutation (len n): A[:, p] ~= Q R
};

template <class S>
struct SVDResult {
    Matrix<S> U;               // m x k orthonormal columns
    std::vector<double> s;     // k singular values, descending
    Matrix<S> Vh;              // k x n orthonormal rows: A ~= U diag(s) Vh
};

template <class S>
struct IDResult {
    int k = 0;              // number of skeleton columns
    std::vector<int> piv;   // column permutation (len n)
    Matrix<S> T;            // k x (n-k): A[:, piv[k:]] ~= A[:, piv[:k]] T
};

// ==============================================================
// Dense helpers (LAPACK/BLAS wrappers)
// ==============================================================

namespace detail {

template <class S>
inline constexpr bool is_complex = std::is_same_v<S, cdouble>;

template <class S>
inline constexpr void check_scalar() {
    static_assert(std::is_same_v<S, double> || std::is_same_v<S, cdouble>,
                  "librla supports double and std::complex<double>");
}

inline constexpr double eps = std::numeric_limits<double>::epsilon();

inline double cj(double x) { return x; }
inline cdouble cj(cdouble x) { return std::conj(x); }

inline int ld(int rows) { return rows > 1 ? rows : 1; }

inline int lwork_of(double w) { return std::max(1, static_cast<int>(w)); }
inline int lwork_of(cdouble w) { return std::max(1, static_cast<int>(w.real())); }

inline int to_int(std::size_t v, const char* what) {
    if (v > static_cast<std::size_t>(INT_MAX))
        throw std::length_error(std::string("librla: ") + what + " exceeds LAPACK int range");
    return static_cast<int>(v);
}

inline void check_info(int info, const char* routine) {
    if (info < 0)
        throw std::logic_error(std::string("librla: illegal argument ") +
                               std::to_string(-info) + " to " + routine);
    if (info > 0)
        throw std::runtime_error(std::string("librla: ") + routine +
                                 " failed, info = " + std::to_string(info));
}

// First c columns of A (contiguous prefix in column-major storage).
template <class S>
Matrix<S> left_cols(const Matrix<S>& A, int c) {
    const std::size_t len = static_cast<std::size_t>(A.rows()) * static_cast<std::size_t>(c);
    return Matrix<S>(A.rows(), c, std::vector<S>(A.data(), A.data() + len));
}

// First r rows of A.
template <class S>
Matrix<S> top_rows(const Matrix<S>& A, int r) {
    Matrix<S> B(r, A.cols());
    for (int j = 0; j < A.cols(); ++j)
        std::copy(A.col(j), A.col(j) + r, B.col(j));
    return B;
}

// Block A[r0:r0+nr, c0:c0+nc].
template <class S>
Matrix<S> block(const Matrix<S>& A, int r0, int c0, int nr, int nc) {
    Matrix<S> B(nr, nc);
    for (int j = 0; j < nc; ++j)
        std::copy(A.col(c0 + j) + r0, A.col(c0 + j) + r0 + nr, B.col(j));
    return B;
}

template <class S>
Matrix<S> select_columns(const Matrix<S>& A, const int* idx, int count) {
    Matrix<S> B(A.rows(), count);
    for (int j = 0; j < count; ++j) {
        if (idx[j] < 0 || idx[j] >= A.cols())
            throw std::out_of_range("librla: column index out of range");
        std::copy(A.col(idx[j]), A.col(idx[j]) + A.rows(), B.col(j));
    }
    return B;
}

template <class S>
std::vector<double> abs_diag(const Matrix<S>& R) {
    const int d = std::min(R.rows(), R.cols());
    std::vector<double> out(static_cast<std::size_t>(d));
    for (int i = 0; i < d; ++i) out[static_cast<std::size_t>(i)] = std::abs(R(i, i));
    return out;
}

// C = op(A) op(B), op in {'N', 'T', 'C'} (gemm)
template <class S>
Matrix<S> gemm(char ta, const Matrix<S>& A, char tb, const Matrix<S>& B) {
    check_scalar<S>();
    const bool na = (ta == 'N' || ta == 'n'), nb = (tb == 'N' || tb == 'n');
    const int m = na ? A.rows() : A.cols();
    const int k = na ? A.cols() : A.rows();
    const int kb = nb ? B.rows() : B.cols();
    const int n = nb ? B.cols() : B.rows();
    if (k != kb) throw std::invalid_argument("librla::matmul: inner dimensions differ");
    Matrix<S> C(m, n);
    if (m == 0 || n == 0 || k == 0) return C;
    const S one(1), zero(0);
    const int lda = ld(A.rows()), ldb = ld(B.rows()), ldc = ld(m);
    if constexpr (is_complex<S>)
        zgemm_(&ta, &tb, &m, &n, &k, &one, A.data(), &lda, B.data(), &ldb, &zero,
               C.data(), &ldc, 1, 1);
    else
        dgemm_(&ta, &tb, &m, &n, &k, &one, A.data(), &lda, B.data(), &ldb, &zero,
               C.data(), &ldc, 1, 1);
    return C;
}

template <class S>
struct PivotedQR {
    Matrix<S> Q;          // m x min(m,n)
    Matrix<S> R;          // min(m,n) x n, upper triangular
    std::vector<int> p;   // 0-based: A[:, p] = Q R
};

// Economic QR with column pivoting: geqp3 (all columns free) + orgqr/ungqr.
template <class S>
PivotedQR<S> qr_piv(Matrix<S> A, bool want_q = true) {
    check_scalar<S>();
    const int m = A.rows(), n = A.cols(), mn = std::min(m, n);
    PivotedQR<S> out;
    out.p.resize(static_cast<std::size_t>(n));
    if (mn == 0) {
        std::iota(out.p.begin(), out.p.end(), 0);
        out.Q = Matrix<S>(m, 0);
        out.R = Matrix<S>(0, n);
        return out;
    }
    std::vector<int> jpvt(static_cast<std::size_t>(n), 0);
    std::vector<S> tau(static_cast<std::size_t>(mn));
    std::vector<double> rwork(is_complex<S> ? 2 * static_cast<std::size_t>(n) : 0);
    const int lda = ld(m);
    int info = 0;
    auto geqp3 = [&](S* work, int lwork) {
        if constexpr (is_complex<S>)
            zgeqp3_(&m, &n, A.data(), &lda, jpvt.data(), tau.data(), work, &lwork,
                    rwork.data(), &info);
        else
            dgeqp3_(&m, &n, A.data(), &lda, jpvt.data(), tau.data(), work, &lwork, &info);
    };
    S wq(0);
    geqp3(&wq, -1);
    check_info(info, "geqp3 (workspace query)");
    std::vector<S> work(static_cast<std::size_t>(lwork_of(wq)));
    geqp3(work.data(), static_cast<int>(work.size()));
    check_info(info, "geqp3");
    for (int j = 0; j < n; ++j) out.p[static_cast<std::size_t>(j)] = jpvt[static_cast<std::size_t>(j)] - 1;

    // R: upper triangle (geqp3 stores Householder vectors below the diagonal)
    out.R = Matrix<S>(mn, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= std::min(j, mn - 1); ++i) out.R(i, j) = A(i, j);
    if (!want_q) return out;

    auto orgqr = [&](S* w, int lwork) {
        if constexpr (is_complex<S>)
            zungqr_(&m, &mn, &mn, A.data(), &lda, tau.data(), w, &lwork, &info);
        else
            dorgqr_(&m, &mn, &mn, A.data(), &lda, tau.data(), w, &lwork, &info);
    };
    wq = S(0);
    orgqr(&wq, -1);
    check_info(info, "orgqr (workspace query)");
    work.assign(static_cast<std::size_t>(lwork_of(wq)), S(0));
    orgqr(work.data(), static_cast<int>(work.size()));
    check_info(info, "orgqr");
    out.Q = left_cols(A, mn);
    return out;
}

template <class S>
struct EconSVD {
    Matrix<S> U;             // m x min(m,n)
    std::vector<double> s;   // min(m,n), descending
    Matrix<S> Vh;            // min(m,n) x n
};

// Economic SVD: gesdd (jobz = 'S'), falling back to gesvd if gesdd fails
// to converge.
template <class S>
EconSVD<S> svd_econ(const Matrix<S>& A0) {
    check_scalar<S>();
    const int m = A0.rows(), n = A0.cols(), mn = std::min(m, n), mx = std::max(m, n);
    EconSVD<S> out;
    out.U = Matrix<S>(m, mn);
    out.Vh = Matrix<S>(mn, n);
    out.s.assign(static_cast<std::size_t>(mn), 0.0);
    if (mn == 0) return out;

    const int lda = ld(m), ldu = ld(m), ldvt = ld(mn);
    const std::size_t smn = static_cast<std::size_t>(mn), smx = static_cast<std::size_t>(mx);
    int info = 0;
    {
        Matrix<S> A = A0;
        std::vector<int> iwork(8 * smn);
        std::vector<double> rwork(is_complex<S>
            ? std::max(5 * smn * smn + 7 * smn, 2 * smx * smn + 2 * smn * smn + smn) : 0);
        const char jobz = 'S';
        auto gesdd = [&](S* work, int lwork) {
            if constexpr (is_complex<S>)
                zgesdd_(&jobz, &m, &n, A.data(), &lda, out.s.data(), out.U.data(), &ldu,
                        out.Vh.data(), &ldvt, work, &lwork, rwork.data(), iwork.data(),
                        &info, 1);
            else
                dgesdd_(&jobz, &m, &n, A.data(), &lda, out.s.data(), out.U.data(), &ldu,
                        out.Vh.data(), &ldvt, work, &lwork, iwork.data(), &info, 1);
        };
        S wq(0);
        gesdd(&wq, -1);
        check_info(info, "gesdd (workspace query)");
        std::vector<S> work(static_cast<std::size_t>(lwork_of(wq)));
        gesdd(work.data(), static_cast<int>(work.size()));
        if (info < 0) check_info(info, "gesdd");
        if (info == 0) return out;
    }
    // gesdd did not converge: QR-iteration SVD on a fresh copy
    Matrix<S> A = A0;
    std::vector<double> rwork(is_complex<S> ? 5 * smn : 0);
    const char job = 'S';
    auto gesvd = [&](S* work, int lwork) {
        if constexpr (is_complex<S>)
            zgesvd_(&job, &job, &m, &n, A.data(), &lda, out.s.data(), out.U.data(), &ldu,
                    out.Vh.data(), &ldvt, work, &lwork, rwork.data(), &info, 1, 1);
        else
            dgesvd_(&job, &job, &m, &n, A.data(), &lda, out.s.data(), out.U.data(), &ldu,
                    out.Vh.data(), &ldvt, work, &lwork, &info, 1, 1);
    };
    S wq(0);
    gesvd(&wq, -1);
    check_info(info, "gesvd (workspace query)");
    std::vector<S> work(static_cast<std::size_t>(lwork_of(wq)));
    gesvd(work.data(), static_cast<int>(work.size()));
    check_info(info, "gesvd");
    return out;
}

// Minimum-norm least squares min ||A X - B|| via gelsd, rcond = eps
// (scipy.linalg.lstsq default). A: m x k, B: m x nrhs -> X: k x nrhs.
template <class S>
Matrix<S> lstsq(const Matrix<S>& A0, const Matrix<S>& B0) {
    check_scalar<S>();
    const int m = A0.rows(), k = A0.cols(), nrhs = B0.cols(), mn = std::min(m, k);
    Matrix<S> X(k, nrhs);
    if (mn == 0 || nrhs == 0) return X;
    Matrix<S> A = A0;
    const int ldb = std::max({1, m, k});
    Matrix<S> B(ldb, nrhs);
    for (int j = 0; j < nrhs; ++j) std::copy(B0.col(j), B0.col(j) + m, B.col(j));

    // Workspace sizes from the LAPACK documentation (SMLSIZ = 25), raised to
    // the query results where the query reports them.
    const std::size_t smn = static_cast<std::size_t>(mn), snr = static_cast<std::size_t>(nrhs);
    const std::size_t smlsiz = 25;
    std::size_t nlvl = 1;
    while ((smn >> nlvl) >= smlsiz + 1) ++nlvl;  // >= floor(log2(mn/(smlsiz+1))) + 1
    std::size_t liwork = std::max<std::size_t>(1, 3 * smn * nlvl + 11 * smn);
    std::size_t lrwork = 10 * smn + 2 * smn * smlsiz + 8 * smn * nlvl + 3 * smlsiz * snr +
                         std::max((smlsiz + 1) * (smlsiz + 1), smn * (1 + snr) + 2 * snr);

    std::vector<double> sv(smn);
    const double rcond = eps;
    const int lda = ld(m);
    int rank = 0, info = 0;
    std::vector<int> iwork;
    std::vector<double> rwork;
    auto gelsd = [&](S* work, int lwork) {
        if constexpr (is_complex<S>)
            zgelsd_(&m, &k, &nrhs, A.data(), &lda, B.data(), &ldb, sv.data(), &rcond, &rank,
                    work, &lwork, rwork.data(), iwork.data(), &info);
        else
            dgelsd_(&m, &k, &nrhs, A.data(), &lda, B.data(), &ldb, sv.data(), &rcond, &rank,
                    work, &lwork, iwork.data(), &info);
    };
    iwork.assign(liwork, 0);
    rwork.assign(is_complex<S> ? lrwork : 1, 0.0);
    S wq(0);
    gelsd(&wq, -1);
    check_info(info, "gelsd (workspace query)");
    liwork = std::max(liwork, static_cast<std::size_t>(std::max(1, iwork[0])));
    if constexpr (is_complex<S>)
        lrwork = std::max(lrwork, static_cast<std::size_t>(std::max(1.0, rwork[0])));
    iwork.assign(liwork, 0);
    rwork.assign(is_complex<S> ? lrwork : 1, 0.0);
    std::vector<S> work(static_cast<std::size_t>(lwork_of(wq)));
    gelsd(work.data(), static_cast<int>(work.size()));
    check_info(info, "gelsd");
    return top_rows(B, k);
}

// B := R^{-1} B, R upper triangular k x k (trsm reads the upper triangle only).
template <class S>
void trsm_upper(const Matrix<S>& R, Matrix<S>& B) {
    check_scalar<S>();
    const int k = R.rows(), nrhs = B.cols();
    if (k == 0 || nrhs == 0) return;
    const S one(1);
    const char side = 'L', uplo = 'U', trans = 'N', diag = 'N';
    const int lda = ld(k), ldb = ld(B.rows());
    if constexpr (is_complex<S>)
        ztrsm_(&side, &uplo, &trans, &diag, &k, &nrhs, &one, R.data(), &lda, B.data(), &ldb,
               1, 1, 1, 1);
    else
        dtrsm_(&side, &uplo, &trans, &diag, &k, &nrhs, &one, R.data(), &lda, B.data(), &ldb,
               1, 1, 1, 1);
}

}  // namespace detail

// ==============================================================
// Public dense utilities
// ==============================================================

// op(A) op(B) with op in {'N', 'T', 'C'} (BLAS gemm).
template <class S>
Matrix<S> matmul(const Matrix<S>& A, const Matrix<S>& B, char op_a = 'N', char op_b = 'N') {
    return detail::gemm(op_a, A, op_b, B);
}

// Conjugate transpose A^H.
template <class S>
Matrix<S> adjoint(const Matrix<S>& A) {
    Matrix<S> B(A.cols(), A.rows());
    for (int j = 0; j < A.cols(); ++j)
        for (int i = 0; i < A.rows(); ++i) B(j, i) = detail::cj(A(i, j));
    return B;
}

// A[:, idx[begin:end]]  (end = -1: to the end of idx)
template <class S>
Matrix<S> take_columns(const Matrix<S>& A, const std::vector<int>& idx, int begin = 0,
                       int end = -1) {
    const int sz = detail::to_int(idx.size(), "index vector");
    if (end < 0) end = sz;
    if (begin < 0 || begin > end || end > sz)
        throw std::out_of_range("librla::take_columns: bad range");
    return detail::select_columns(A, idx.data() + begin, end - begin);
}

// Frobenius norm (scaled to avoid overflow).
template <class S>
double norm_fro(const Matrix<S>& A) {
    double scale = 0.0;
    for (const S& a : A.storage()) scale = std::max(scale, std::abs(a));
    if (scale == 0.0 || !std::isfinite(scale)) return scale;
    double ssq = 0.0;
    for (const S& a : A.storage()) {
        const double v = std::abs(a) / scale;
        ssq += v * v;
    }
    return scale * std::sqrt(ssq);
}

template <class S>
LinearOperator<S> LinearOperator<S>::from_matrix(const Matrix<S>& A, bool expose_dense) {
    const Matrix<S>* Ap = &A;
    return LinearOperator<S>(
        A.rows(), A.cols(),
        [Ap](const Matrix<S>& X) { return detail::gemm('N', *Ap, 'N', X); },
        [Ap](const Matrix<S>& W) { return detail::gemm('C', *Ap, 'N', W); },
        expose_dense ? Ap : nullptr);
}

// ==============================================================
// Algorithms (internal, on a dense-or-operator view)
// ==============================================================

namespace detail {

// Either a dense matrix (BLAS products, used directly by the deterministic
// paths) or a matrix-free operator.
template <class S>
struct Op {
    int m = 0;
    int n = 0;
    const Matrix<S>* dense = nullptr;
    const LinearOperator<S>* lin = nullptr;

    // A X
    Matrix<S> apply(const Matrix<S>& X) const {
        if (dense) return gemm('N', *dense, 'N', X);
        return checked(lin->apply(X), m, X.cols(), "apply");
    }
    // A^H W
    Matrix<S> apply_adjoint(const Matrix<S>& W) const {
        if (dense) return gemm('C', *dense, 'N', W);
        return checked(lin->apply_adjoint(W), n, W.cols(), "apply_adjoint");
    }
    // Q^H A  (operator: (A^H Q)^H)
    Matrix<S> left_adjoint(const Matrix<S>& Q) const {
        if (dense) return gemm('C', Q, 'N', *dense);
        return librla::adjoint(apply_adjoint(Q));
    }
    // A as a dense matrix: the matrix itself, or the operator materialized
    // into `storage` by applying it to identity columns (n applies in
    // blocks of 64 columns).
    const Matrix<S>& matrix(Matrix<S>& storage) const {
        if (dense) return *dense;
        storage = Matrix<S>(m, n);
        const int nb = 64;
        for (int j0 = 0; j0 < n; j0 += nb) {
            const int b = std::min(nb, n - j0);
            Matrix<S> E(n, b);
            for (int j = 0; j < b; ++j) E(j0 + j, j) = S(1);
            Matrix<S> Y = apply(E);
            for (int j = 0; j < b; ++j) std::copy(Y.col(j), Y.col(j) + m, storage.col(j0 + j));
        }
        return storage;
    }

private:
    static Matrix<S> checked(Matrix<S> Y, int rows, int cols, const char* what) {
        if (Y.rows() != rows || Y.cols() != cols)
            throw std::runtime_error(std::string("librla::LinearOperator::") + what +
                                     " returned a matrix of the wrong shape");
        return Y;
    }
};

template <class S>
Op<S> make_op(const Matrix<S>& A) {
    check_scalar<S>();
    return Op<S>{A.rows(), A.cols(), &A, nullptr};
}

template <class S>
Op<S> make_op(const LinearOperator<S>& L) {
    check_scalar<S>();
    if (L.rows < 0 || L.cols < 0)
        throw std::invalid_argument("librla::LinearOperator: negative dimension");
    if (L.dense) {
        if (L.dense->rows() != L.rows || L.dense->cols() != L.cols)
            throw std::invalid_argument("librla::LinearOperator: dense matrix shape mismatch");
        return Op<S>{L.rows, L.cols, L.dense, nullptr};
    }
    if (!L.apply || !L.apply_adjoint)
        throw std::invalid_argument("librla::LinearOperator: apply and apply_adjoint required");
    return Op<S>{L.rows, L.cols, nullptr, &L};
}

inline void check_options(const Options& o) {
    if (o.block_size < 1) throw std::invalid_argument("librla: block_size must be >= 1");
    if (o.power_iter < 0) throw std::invalid_argument("librla: power_iter must be >= 0");
    if (o.extra_samples < 0) throw std::invalid_argument("librla: extra_samples must be >= 0");
}

inline void check_method(IdMethod method) {
    if (method != IdMethod::fast && method != IdMethod::svd && method != IdMethod::lstsq)
        throw std::invalid_argument("librla: method must be one of {fast, svd, lstsq}");
}

// floor(rtol) for rank mode (rtol >= 1), saturated at INT_MAX
inline int target_rank(double rtol) {
    if (rtol >= static_cast<double>(INT_MAX)) return INT_MAX;
    return static_cast<int>(std::floor(rtol));
}

// Number of entries >= rtol * d[0] (d: |diag(R)| or singular values)
inline int rank_from_diag(const std::vector<double>& d, double rtol) {
    if (d.empty() || d[0] <= 0) return 0;
    const double thr = rtol * d[0];
    return static_cast<int>(std::count_if(d.begin(), d.end(), [thr](double v) { return v >= thr; }));
}

inline int rank_from_svals(const std::vector<double>& s, double rtol) {
    return rank_from_diag(s, rtol);
}

// x <- Q factor of pivoted QR of A^H (A x), power_iter times
template <class S>
Matrix<S> power_iteration(const Op<S>& A, Matrix<S> x, int power_iter) {
    for (int i = 0; i < power_iter; ++i)
        x = qr_piv(A.apply_adjoint(A.apply(x))).Q;
    return x;
}

template <class S>
OrthResult<S> orth_sketch(const Op<S>& A, double rtol, const Options& o, Rng& rng) {
    check_options(o);
    const int m = A.m, n = A.n, mn = std::min(m, n);

    // Rank mode (rtol >= 1): single oversampled sketch, buffer included
    if (rtol >= 1) {
        const long long bsl = static_cast<long long>(target_rank(rtol)) + o.extra_samples;
        if (bsl > INT_MAX) throw std::length_error("librla: rank + extra_samples too large");
        const int bs = static_cast<int>(bsl);
        Matrix<S> x = rng.uniform<S>(n, bs);
        x = power_iteration(A, std::move(x), o.power_iter);
        PivotedQR<S> f = qr_piv(A.apply(x));
        std::vector<double> diagR = abs_diag(f.R);
        return {std::move(f.Q), 0, std::move(diagR)};
    }

    // Tolerance mode (rtol < 1): geometric growth with tolerance checking
    if (rtol < eps || o.block_size >= mn) return {Matrix<S>(m, 0), 1, {}};

    int bs = o.block_size;
    const std::size_t extra = static_cast<std::size_t>(o.extra_samples);
    for (;;) {
        Matrix<S> x = rng.uniform<S>(n, bs);
        x = power_iteration(A, std::move(x), o.power_iter);
        PivotedQR<S> f = qr_piv(A.apply(x));

        // Buffered tolerance test (diagR is sorted decreasing; the
        // multiplied form stays valid when diagR[0] == 0)
        std::vector<double> diagR = abs_diag(f.R);
        const std::size_t len = diagR.size();
        if (len == 0 || (len > extra && diagR[len - 1 - extra] <= rtol * diagR[0]))
            return {std::move(f.Q), 0, std::move(diagR)};

        bs = (bs > mn / 4) ? mn : std::min(4 * bs, mn);
        if (bs >= mn) return {Matrix<S>(m, 0), 1, {}};
    }
}

template <class S>
QRResult<S> qr_sketch(const Op<S>& A, double rtol, const Options& o, Rng& rng) {
    const int mn = std::min(A.m, A.n);
    const bool rank_mode = rtol >= 1;
    const int kmax = rank_mode ? target_rank(rtol) : 0;

    // Sketch; the basis includes the extra_samples buffer columns
    OrthResult<S> sk = orth_sketch(A, rtol, o, rng);
    const int k = sk.flag == 0 ? sk.Q.cols() : mn;

    // Fallback to full QR if needed
    const bool needs_fallback = (sk.flag != 0 || k >= mn) && !rank_mode;
    if (needs_fallback) {
        Matrix<S> storage;
        PivotedQR<S> f = qr_piv(A.matrix(storage));
        const int rank = rank_mode ? std::min(kmax, f.Q.cols()) : rank_from_diag(abs_diag(f.R), rtol);
        return {left_cols(f.Q, rank), top_rows(f.R, rank), std::move(f.p)};
    }

    // Project and compute QR
    PivotedQR<S> f = qr_piv(A.left_adjoint(sk.Q));
    Matrix<S> Q = gemm('N', sk.Q, 'N', f.Q);
    const int rank = rank_mode ? std::min(kmax, Q.cols()) : rank_from_diag(abs_diag(f.R), rtol);
    return {left_cols(Q, rank), top_rows(f.R, rank), std::move(f.p)};
}

template <class S>
SVDResult<S> svd_sketch(const Op<S>& A, double rtol, const Options& o, Rng& rng) {
    const int m = A.m, n = A.n, mn = std::min(m, n);

    // Wide: factor the adjoint and swap/adjoint the factors
    if (m < n) {
        SVDResult<S> r;
        if (A.dense) {
            const Matrix<S> Ah = librla::adjoint(*A.dense);
            r = svd_sketch(make_op(Ah), rtol, o, rng);
        } else {
            const LinearOperator<S> Lh = A.lin->adjoint();
            r = svd_sketch(make_op(Lh), rtol, o, rng);
        }
        return {librla::adjoint(r.Vh), std::move(r.s), librla::adjoint(r.U)};
    }

    const bool rank_mode = rtol >= 1;
    const int kmax = rank_mode ? target_rank(rtol) : 0;

    OrthResult<S> sk = orth_sketch(A, rtol, o, rng);
    const int k = sk.flag == 0 ? sk.Q.cols() : mn;

    // Fallback to full SVD if needed
    const bool needs_fallback = (sk.flag != 0 || k >= mn) && !rank_mode;
    EconSVD<S> f;
    Matrix<S> U;
    if (needs_fallback) {
        Matrix<S> storage;
        f = svd_econ(A.matrix(storage));
        U = std::move(f.U);
    } else {
        // Project and compute SVD
        f = svd_econ(A.left_adjoint(sk.Q));
        U = gemm('N', sk.Q, 'N', f.U);
    }
    const int len = static_cast<int>(f.s.size());
    const int rank = rank_mode ? std::min(kmax, len) : rank_from_svals(f.s, rtol);
    f.s.resize(static_cast<std::size_t>(rank));
    return {left_cols(U, rank), std::move(f.s), top_rows(f.Vh, rank)};
}

// ---- T matrix computation for ID (R: at least k rows, n columns) ----

template <class S>
Matrix<S> compute_T_lstsq(const Op<S>& A, const std::vector<int>& piv, int k) {
    const int n = A.n;
    if (k == 0 || k >= n) return Matrix<S>(k, n - k);
    Matrix<S> storage;
    const Matrix<S>& Am = A.matrix(storage);
    return lstsq(select_columns(Am, piv.data(), k), select_columns(Am, piv.data() + k, n - k));
}

template <class S>
Matrix<S> compute_T_svd(const Matrix<S>& R, int k, double rtol_for_svd) {
    const int n = R.cols();
    if (k == 0) return Matrix<S>(k, n - k);
    const Matrix<S> R11 = block(R, 0, 0, k, k);
    const Matrix<S> R12 = block(R, 0, k, k, n - k);

    EconSVD<S> f = svd_econ(R11);

    // Floor the relative threshold at machine precision so a rank-deficient
    // R11 drops near-zero singular values instead of inverting them; an
    // all-zero R11 yields T = 0.
    const double smax = f.s.empty() ? 0.0 : *std::max_element(f.s.begin(), f.s.end());
    if (smax == 0) return Matrix<S>(k, n - k);
    const double tol = std::max(rtol_for_svd, eps);
    std::vector<int> keep;
    for (int i = 0; i < static_cast<int>(f.s.size()); ++i)
        if (f.s[static_cast<std::size_t>(i)] >= tol * smax) keep.push_back(i);
    if (keep.empty()) return Matrix<S>(k, n - k);

    // T = Vh[keep,:]^H diag(1/s[keep]) U[:,keep]^H R12
    const int r = static_cast<int>(keep.size());
    const Matrix<S> Uk = select_columns(f.U, keep.data(), r);
    Matrix<S> Vk(r, k);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < r; ++i) Vk(i, j) = f.Vh(keep[static_cast<std::size_t>(i)], j);
    Matrix<S> W = gemm('C', Uk, 'N', R12);
    for (int i = 0; i < r; ++i) {
        const double inv_s = 1.0 / f.s[static_cast<std::size_t>(keep[static_cast<std::size_t>(i)])];
        for (int j = 0; j < W.cols(); ++j) W(i, j) *= inv_s;
    }
    return gemm('C', Vk, 'N', W);
}

template <class S>
Matrix<S> compute_T_fast(const Matrix<S>& R, int k, double rtol_for_svd) {
    const int n = R.cols();
    if (k == 0) return Matrix<S>(k, n - k);

    // R11 (near-)singular: fall back to the SVD-based minimum-norm solve,
    // which stays finite and still reconstructs A.
    double dmin = std::numeric_limits<double>::infinity(), dmax = 0.0;
    for (int i = 0; i < k; ++i) {
        const double d = std::abs(R(i, i));
        dmin = std::min(dmin, d);
        dmax = std::max(dmax, d);
    }
    if (dmin <= eps * dmax) return compute_T_svd(R, k, rtol_for_svd);

    const Matrix<S> R11 = block(R, 0, 0, k, k);
    Matrix<S> T = block(R, 0, k, k, n - k);
    trsm_upper(R11, T);
    return T;
}

template <class S>
Matrix<S> compute_T(IdMethod method, const Op<S>& A, const Matrix<S>& R,
                    const std::vector<int>& piv, int k, double rtol_for_svd) {
    switch (method) {
        case IdMethod::lstsq: return compute_T_lstsq(A, piv, k);
        case IdMethod::svd: return compute_T_svd(R, k, rtol_for_svd);
        case IdMethod::fast: return compute_T_fast(R, k, rtol_for_svd);
    }
    throw std::invalid_argument("librla: method must be one of {fast, svd, lstsq}");
}

template <class S>
IDResult<S> id_sketch(const Op<S>& A, double rtol, const Options& o, Rng& rng) {
    check_method(o.method);
    QRResult<S> f = qr_sketch(A, rtol, o, rng);
    const int k = f.R.rows();
    // Rank mode: minimal filtering (only exact zeros); tolerance mode: rtol
    const double rtol_for_svd = rtol >= 1 ? 0.0 : rtol;
    Matrix<S> T = compute_T(o.method, A, f.R, f.p, k, rtol_for_svd);
    return {k, std::move(f.p), std::move(T)};
}

template <class S>
IDResult<S> id_qrpiv(const Op<S>& A, double rtol, IdMethod method) {
    check_method(method);
    const int m = A.m, n = A.n;
    const bool rank_mode = rtol >= 1;

    // Full QR with pivoting (deterministic)
    Matrix<S> storage;
    const Matrix<S>& Am = A.matrix(storage);
    PivotedQR<S> f = qr_piv(Am, false);

    const int k = rank_mode ? std::min(target_rank(rtol), std::min(m, n))
                            : rank_from_diag(abs_diag(f.R), rtol);
    const double rtol_for_svd = rank_mode ? 0.0 : rtol;

    if (k == 0) return {0, std::move(f.p), Matrix<S>(0, n)};
    if (k == n) return {k, std::move(f.p), Matrix<S>(k, 0)};

    // lstsq reuses the already materialized matrix
    Matrix<S> T = compute_T(method, make_op(Am), f.R, f.p, k, rtol_for_svd);
    return {k, std::move(f.p), std::move(T)};
}

}  // namespace detail

// ==============================================================
// Public API
// ==============================================================

// orth_sketch: approximate orthonormal basis for the column space of A.
//
// Tolerance mode (rtol < 1): adaptively grows the sketch (block_size,
//   then x4) until diagR[len-1-extra_samples] <= rtol * diagR[0], i.e. the
//   sketch covers the rtol-rank with extra_samples columns to spare. A fresh
//   test matrix is drawn on every pass.
// Rank mode (rtol >= 1): one sketch of floor(rtol) + extra_samples columns.
//
// Returns {Q, flag, diagR}: Q is m x k with orthonormal columns (including
// the buffer columns; rank mode: k = min(m, floor(rtol) + extra_samples));
// diagR = |diag(R)| of the pivoted QR of the sketch (decreasing).
// flag = 1 (tolerance mode only) when rtol < machine epsilon or the sketch
// would reach min(m,n); then Q is m x 0 and diagR is empty. qr_sketch,
// svd_sketch and id_sketch handle flag = 1 with a deterministic fallback.
template <class S>
OrthResult<S> orth_sketch(const Matrix<S>& A, double rtol, const Options& opts = {}) {
    Rng local;
    return detail::orth_sketch(detail::make_op(A), rtol, opts, opts.rng ? *opts.rng : local);
}
template <class S>
OrthResult<S> orth_sketch(const LinearOperator<S>& A, double rtol, const Options& opts = {}) {
    Rng local;
    return detail::orth_sketch(detail::make_op(A), rtol, opts, opts.rng ? *opts.rng : local);
}

// qr_sketch: truncated QR with column pivoting, A[:, p] ~= Q R.
//
// Sketches a basis Qs, takes the pivoted QR of Qs^H A and expands back.
// Tolerance mode keeps the rows of R with |R(i,i)| >= rtol * |R(0,0)|;
// rank mode keeps floor(rtol) (capped at the available columns). If the
// sketch terminates early (flag 1, tolerance mode) a deterministic pivoted
// QR of A is computed instead (materializing a matrix-free operator).
//
// Returns {Q (m x k), R (k x n, upper triangular), p (len n, 0-based)}.
template <class S>
QRResult<S> qr_sketch(const Matrix<S>& A, double rtol, const Options& opts = {}) {
    Rng local;
    return detail::qr_sketch(detail::make_op(A), rtol, opts, opts.rng ? *opts.rng : local);
}
template <class S>
QRResult<S> qr_sketch(const LinearOperator<S>& A, double rtol, const Options& opts = {}) {
    Rng local;
    return detail::qr_sketch(detail::make_op(A), rtol, opts, opts.rng ? *opts.rng : local);
}

// svd_sketch: truncated SVD, A ~= U diag(s) Vh.
//
// Sketches a basis Qs, takes the SVD of Qs^H A and expands back. Tolerance
// mode keeps s >= rtol * s[0]; rank mode keeps floor(rtol) (capped).
// Wide A (m < n) is handled through its adjoint. If the sketch terminates
// early a deterministic SVD of A is computed instead.
//
// Returns {U (m x k), s (len k, descending), Vh (k x n)}; Vh is V^H.
template <class S>
SVDResult<S> svd_sketch(const Matrix<S>& A, double rtol, const Options& opts = {}) {
    Rng local;
    return detail::svd_sketch(detail::make_op(A), rtol, opts, opts.rng ? *opts.rng : local);
}
template <class S>
SVDResult<S> svd_sketch(const LinearOperator<S>& A, double rtol, const Options& opts = {}) {
    Rng local;
    return detail::svd_sketch(detail::make_op(A), rtol, opts, opts.rng ? *opts.rng : local);
}

// id_sketch: interpolative decomposition via randomized sketching,
// A[:, piv[k:]] ~= A[:, piv[:k]] T.
//
// The permutation comes from qr_sketch; T (k x (n-k)) from opts.method:
//   fast  - triangular solve R11 \ R12 (SVD solve if R11 is near-singular)
//   svd   - SVD pseudoinverse of R11 (threshold max(rtol, eps) in tolerance
//           mode, eps in rank mode)
//   lstsq - least squares on the columns of A (materializes an operator)
// An invalid method throws std::invalid_argument.
//
// Returns {k, piv (len n, 0-based), T (k x (n-k))}.
template <class S>
IDResult<S> id_sketch(const Matrix<S>& A, double rtol, const Options& opts = {}) {
    Rng local;
    return detail::id_sketch(detail::make_op(A), rtol, opts, opts.rng ? *opts.rng : local);
}
template <class S>
IDResult<S> id_sketch(const LinearOperator<S>& A, double rtol, const Options& opts = {}) {
    Rng local;
    return detail::id_sketch(detail::make_op(A), rtol, opts, opts.rng ? *opts.rng : local);
}

// id_qrpiv: interpolative decomposition via deterministic QR with column
// pivoting (geqp3) of A (a matrix-free operator is materialized).
// Same return convention and methods as id_sketch.
template <class S>
IDResult<S> id_qrpiv(const Matrix<S>& A, double rtol, IdMethod method = IdMethod::fast) {
    return detail::id_qrpiv(detail::make_op(A), rtol, method);
}
template <class S>
IDResult<S> id_qrpiv(const LinearOperator<S>& A, double rtol, IdMethod method = IdMethod::fast) {
    return detail::id_qrpiv(detail::make_op(A), rtol, method);
}
template <class S>
IDResult<S> id_qrpiv(const Matrix<S>& A, double rtol, const std::string& method) {
    return id_qrpiv(A, rtol, parse_id_method(method));
}
template <class S>
IDResult<S> id_qrpiv(const LinearOperator<S>& A, double rtol, const std::string& method) {
    return id_qrpiv(A, rtol, parse_id_method(method));
}

}  // namespace librla

#endif  // LIBRLA_HPP
