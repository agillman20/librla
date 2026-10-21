// test_all.cpp - test suite for the librla C++ port
//
// Prints one PASS/FAIL line per check, then the CROSSPORT lines shared
// with the Rust, Swift and Fortran ports; exits nonzero on any failure.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "librla.hpp"

using namespace librla;

namespace {

int n_pass = 0;
int n_fail = 0;

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 2, 3)))
#endif
void check(bool ok, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", buf);
    (ok ? n_pass : n_fail)++;
}

template <class S> const char* tname();
template <> const char* tname<double>() { return "real"; }
template <> const char* tname<cdouble>() { return "cplx"; }

// ---------------- dense helpers ----------------

template <class S>
Matrix<S> sub(const Matrix<S>& A, const Matrix<S>& B) {
    Matrix<S> C = A;
    for (int j = 0; j < A.cols(); ++j)
        for (int i = 0; i < A.rows(); ++i) C(i, j) -= B(i, j);
    return C;
}

template <class S>
double rel_diff(const Matrix<S>& A, const Matrix<S>& B) {
    const double nA = norm_fro(A);
    return norm_fro(sub(A, B)) / (nA > 0 ? nA : 1.0);
}

template <class S>
double ortho_err_cols(const Matrix<S>& Q) {
    return norm_fro(sub(matmul(Q, Q, 'C', 'N'), Matrix<S>::identity(Q.cols())));
}

template <class S>
double ortho_err_rows(const Matrix<S>& V) {
    return norm_fro(sub(matmul(V, V, 'N', 'C'), Matrix<S>::identity(V.rows())));
}

template <class S>
Matrix<S> scale_cols(Matrix<S> U, const std::vector<double>& s) {
    for (int j = 0; j < U.cols(); ++j)
        for (int i = 0; i < U.rows(); ++i) U(i, j) *= s[static_cast<std::size_t>(j)];
    return U;
}

template <class S>
bool all_finite(const Matrix<S>& A) {
    for (const S& a : A.storage())
        if (!std::isfinite(std::abs(a))) return false;
    return true;
}

template <class S>
bool strictly_upper_zero(const Matrix<S>& R) {
    for (int j = 0; j < R.cols(); ++j)
        for (int i = j + 1; i < R.rows(); ++i)
            if (R(i, j) != S(0)) return false;
    return true;
}

bool is_perm(const std::vector<int>& p, int n) {
    if (static_cast<int>(p.size()) != n) return false;
    std::vector<int> seen(static_cast<std::size_t>(n), 0);
    for (int v : p) {
        if (v < 0 || v >= n || seen[static_cast<std::size_t>(v)]) return false;
        seen[static_cast<std::size_t>(v)] = 1;
    }
    return true;
}

template <class S>
bool same(const Matrix<S>& A, const Matrix<S>& B) {
    return A.rows() == B.rows() && A.cols() == B.cols() && A.storage() == B.storage();
}

// Random m x k matrix with orthonormal columns (pivoted QR of a uniform matrix)
template <class S>
Matrix<S> rand_orth(Rng& rng, int m, int k) {
    return detail::qr_piv(rng.uniform<S>(m, k)).Q;
}

// U diag(s) V^H with random orthonormal U (m x r), V (n x r), r = len(s)
template <class S>
Matrix<S> with_svals(Rng& rng, int m, int n, const std::vector<double>& s) {
    const int r = static_cast<int>(s.size());
    Matrix<S> U = rand_orth<S>(rng, m, r), V = rand_orth<S>(rng, n, r);
    return matmul(scale_cols(U, s), V, 'N', 'C');
}

// sigma_j = 10^(-j/4), j = 0..min(m,n)-1
template <class S>
Matrix<S> decay_matrix(Rng& rng, int m, int n) {
    std::vector<double> s(static_cast<std::size_t>(std::min(m, n)));
    for (std::size_t j = 0; j < s.size(); ++j) s[j] = std::pow(10.0, -static_cast<double>(j) / 4);
    return with_svals<S>(rng, m, n, s);
}

// Frobenius norm of the singular values beyond k, relative to ||A||_F
template <class S>
double opt_err(const Matrix<S>& A, int k) {
    const std::vector<double> s = detail::svd_econ(A).s;
    double t = 0, a = 0;
    for (std::size_t j = 0; j < s.size(); ++j) {
        a += s[j] * s[j];
        if (static_cast<int>(j) >= k) t += s[j] * s[j];
    }
    return std::sqrt(t / a);
}

template <class S>
double qr_err(const Matrix<S>& A, const QRResult<S>& f) {
    return rel_diff(take_columns(A, f.p), matmul(f.Q, f.R));
}

template <class S>
double svd_err(const Matrix<S>& A, const SVDResult<S>& f) {
    return rel_diff(A, matmul(scale_cols(f.U, f.s), f.Vh));
}

template <class S>
double id_err(const Matrix<S>& A, const IDResult<S>& f) {
    const Matrix<S> rest = take_columns(A, f.piv, f.k);
    const Matrix<S> approx = matmul(take_columns(A, f.piv, 0, f.k), f.T);
    const double nA = norm_fro(A);
    return norm_fro(sub(rest, approx)) / (nA > 1.0 ? nA : 1.0);
}

double max_sval_diff(const std::vector<double>& s, const std::vector<double>& ref) {
    double d = 0;
    for (std::size_t i = 0; i < s.size() && i < ref.size(); ++i) d = std::max(d, std::abs(s[i] - ref[i]));
    return d / ref[0];
}

// Operator wrapping a dense matrix, NOT exposing it as dense
template <class S>
LinearOperator<S> matfree(const Matrix<S>& A) {
    return LinearOperator<S>::from_matrix(A, false);
}

template <class S>
bool rng_equal(const Rng& r, std::array<int, 4> v) {
    return r.state() == v;
}

// ---------------- 1. RNG ----------------

void test_rng() {
    std::printf("\n--- RNG (dlarnv/zlarnv reference values) ---\n");
    const double ref[6] = {3.7327920546847082E-01, 8.2093410748050388E-01, 5.5866811353917711E-01,
                           6.4291221902741569E-01, 6.8760847451716955E-01, 1.6449965895444763E-01};
    const std::array<int, 4> after = {2384, 3667, 635, 1229};

    Rng r(1, 2, 3, 5);
    double x[6];
    r.fill(x, 6);
    double d = 0;
    for (int i = 0; i < 6; ++i) d = std::max(d, std::abs(x[i] - ref[i]));
    check(d <= 1e-15, "dlarnv(2, {1,2,3,5}, 6) matches reference (max diff %.1e)", d);
    check(r.state() == after, "dlarnv iseed after = {2384,3667,635,1229}");

    Rng rz(1, 2, 3, 5);
    cdouble z[3];
    rz.fill(z, 3);
    d = 0;
    for (int i = 0; i < 3; ++i)
        d = std::max({d, std::abs(z[i].real() - ref[2 * i]), std::abs(z[i].imag() - ref[2 * i + 1])});
    check(d <= 1e-15, "zlarnv(2, {1,2,3,5}, 3) matches reference (max diff %.1e)", d);
    check(rz.state() == after, "zlarnv iseed after = {2384,3667,635,1229}");

    Rng r64(static_cast<std::uint64_t>(0x0123456789ABCDEFull));
    const std::uint64_t s = 0x0123456789ABCDEFull;
    const std::array<int, 4> want = {static_cast<int>((s >> 36) & 4095), static_cast<int>((s >> 24) & 4095),
                                     static_cast<int>((s >> 12) & 4095), static_cast<int>((s & 4095) | 1)};
    check(r64.state() == want, "Rng(uint64) seed mapping");
    check(Rng().state() == (std::array<int, 4>{0, 0, 0, 1}), "default Rng state {0,0,0,1}");

    bool threw_even = false, threw_range = false;
    try { Rng bad(1, 2, 3, 4); } catch (const std::invalid_argument&) { threw_even = true; }
    try { Rng bad(1, 4096, 3, 5); } catch (const std::invalid_argument&) { threw_range = true; }
    check(threw_even && threw_range, "invalid iseed (even last entry, entry > 4095) throws");
}

// ---------------- 2. orth_sketch ----------------

template <class S>
void test_orth() {
    std::printf("\n--- orth_sketch (%s) ---\n", tname<S>());
    Rng gen(11, 22, 33, 45);
    const Matrix<S> A = decay_matrix<S>(gen, 400, 300);
    const double nA = norm_fro(A);
    const double rtol = 1e-6;

    {
        auto [Q, flag, diagR] = orth_sketch(A, rtol);
        const double span = norm_fro(sub(A, matmul(Q, matmul(Q, A, 'C', 'N')))) / nA;
        const std::size_t L = diagR.size();
        const bool test_ok = L > 12 && diagR[L - 13] <= rtol * diagR[0];
        check(flag == 0 && test_ok && span < 10 * rtol,
              "tol 1e-6 400x300: flag=%d k=%d span err=%.1e, buffered test holds", flag, Q.cols(), span);
        const double oe = ortho_err_cols(Q);
        check(oe < 1e-13, "tol mode Q orthonormal (%.1e)", oe);
    }
    {
        auto [Q, flag, diagR] = orth_sketch(A, 20.0);
        check(flag == 0 && Q.cols() == 32 && diagR.size() == 32 && ortho_err_cols(Q) < 1e-13,
              "rank 20: Q has 20+12 = %d columns, flag=%d", Q.cols(), flag);
        Options o;
        o.extra_samples = 3;
        auto r2 = orth_sketch(A, 7.9, o);
        check(r2.Q.cols() == 10, "rank floor(7.9)=7 with extra_samples=3: %d columns", r2.Q.cols());
    }
    {
        Rng g2(5, 6, 7, 9);
        const Matrix<S> B = g2.uniform<S>(60, 40);
        auto [Q, flag, diagR] = orth_sketch(B, 50.0);
        check(flag == 0 && Q.rows() == 60 && Q.cols() == 60 && ortho_err_cols(Q) < 1e-13,
              "rank 50 on 60x40: Q has min(m, 50+12) = %d columns", Q.cols());
    }
    {
        Options o;
        o.extra_samples = 0;
        auto [Q, flag, diagR] = orth_sketch(A, rtol, o);
        check(flag == 0 && Q.cols() == 42 && diagR.back() <= rtol * diagR[0],
              "extra_samples=0 legacy last-column check: k=%d", Q.cols());
    }
    {
        auto [Q, flag, diagR] = orth_sketch(A, 1e-17);
        check(flag == 1 && Q.rows() == 400 && Q.cols() == 0 && diagR.empty(),
              "rtol < eps: flag=%d, Q 400x%d", flag, Q.cols());
    }
    {
        Rng g2(5, 6, 7, 11);
        const Matrix<S> B = g2.uniform<S>(100, 80);
        auto r1 = orth_sketch(B, rtol);
        check(r1.flag == 1 && r1.Q.cols() == 0 && r1.diagR.empty(),
              "full-rank 100x80: sketch reaches min(m,n), flag=%d", r1.flag);
        Options o;
        o.block_size = 300;
        auto r2 = orth_sketch(A, rtol, o);
        check(r2.flag == 1 && r2.Q.cols() == 0, "block_size >= min(m,n): flag=%d", r2.flag);
    }
}

// ---------------- 3. qr_sketch ----------------

template <class S>
void test_qr() {
    std::printf("\n--- qr_sketch (%s) ---\n", tname<S>());
    Rng gen(11, 22, 33, 47);
    const Matrix<S> At = decay_matrix<S>(gen, 400, 300);
    const Matrix<S> Aw = adjoint(At);
    const double rtol = 1e-6;

    for (const Matrix<S>* Ap : {&At, &Aw}) {
        const Matrix<S>& A = *Ap;
        const char* shape = A.rows() >= A.cols() ? "tall 400x300" : "wide 300x400";
        auto f = qr_sketch(A, rtol);
        const double e = qr_err(A, f);
        check(e < 10 * rtol && strictly_upper_zero(f.R) && is_perm(f.p, A.cols()) &&
                  ortho_err_cols(f.Q) < 1e-13 && f.R.rows() == f.Q.cols(),
              "%s tol 1e-6: k=%d err=%.1e, R upper, Q orthonormal", shape, f.Q.cols(), e);

        auto g = qr_sketch(A, 20.0);
        auto det = detail::qr_piv(A);
        const double e_det = rel_diff(take_columns(A, det.p),
                                      matmul(detail::left_cols(det.Q, 20), detail::top_rows(det.R, 20)));
        const double eg = qr_err(A, g);
        check(g.Q.cols() == 20 && g.R.rows() == 20 && g.R.cols() == A.cols() && strictly_upper_zero(g.R) &&
                  eg < 4 * e_det,
              "%s rank 20: err=%.2e vs deterministic pivoted QR %.2e", shape, eg, e_det);
    }

    // Fallback: full-rank matrix, the sketch terminates early (flag 1)
    Rng g2(3, 1, 4, 1);
    const Matrix<S> F = g2.uniform<S>(60, 50);
    auto f = qr_sketch(F, 1e-10);
    auto det = detail::qr_piv(F);
    const int r = detail::rank_from_diag(detail::abs_diag(det.R), 1e-10);
    check(same(f.Q, detail::left_cols(det.Q, r)) && same(f.R, detail::top_rows(det.R, r)) && f.p == det.p,
          "fallback (60x50 full rank, rtol 1e-10) equals deterministic pivoted QR, k=%d", f.Q.cols());
}

// ---------------- 4. svd_sketch ----------------

template <class S>
void test_svd() {
    std::printf("\n--- svd_sketch (%s) ---\n", tname<S>());
    Rng gen(11, 22, 33, 49);
    const Matrix<S> At = decay_matrix<S>(gen, 400, 300);
    const Matrix<S> Aw = adjoint(At);
    const double rtol = 3e-7;  // sigma_26 = 3.16e-7 is the last one kept: k = 27

    for (const Matrix<S>* Ap : {&At, &Aw}) {
        const Matrix<S>& A = *Ap;
        const char* shape = A.rows() >= A.cols() ? "tall 400x300" : "wide 300x400";
        const std::vector<double> sref = detail::svd_econ(A).s;

        auto f = svd_sketch(A, rtol);
        const double e = svd_err(A, f), ds = max_sval_diff(f.s, sref);
        check(f.s.size() == 27 && e < 10 * rtol && ds < 1e-12,
              "%s tol 3e-7: k=%zu err=%.1e, s vs LAPACK SVD %.1e", shape, f.s.size(), e, ds);
        check(f.U.rows() == A.rows() && f.Vh.cols() == A.cols() && ortho_err_cols(f.U) < 1e-13 &&
                  ortho_err_rows(f.Vh) < 1e-13,
              "%s tol mode: U, Vh orthonormal (%.1e, %.1e)", shape, ortho_err_cols(f.U), ortho_err_rows(f.Vh));

        auto g = svd_sketch(A, 20.0);
        const double eg = svd_err(A, g), opt = opt_err(A, 20), dg = max_sval_diff(g.s, sref);
        check(g.s.size() == 20 && eg < 4 * opt && dg < 1e-10 && ortho_err_cols(g.U) < 1e-13 &&
                  ortho_err_rows(g.Vh) < 1e-13,
              "%s rank 20: err=%.2e (optimal %.2e), s vs LAPACK SVD %.1e", shape, eg, opt, dg);
    }

    // Fallback (full rank): deterministic SVD
    Rng g2(3, 1, 4, 3);
    const Matrix<S> F = g2.uniform<S>(60, 50);
    auto f = svd_sketch(F, 1e-10);
    auto det = detail::svd_econ(F);
    check(same(f.U, det.U) && f.s == det.s && same(f.Vh, det.Vh),
          "fallback (60x50 full rank, rtol 1e-10) equals LAPACK SVD, k=%zu", f.s.size());
}

// ---------------- 5. ID ----------------

template <class S>
void test_id() {
    std::printf("\n--- id_sketch / id_qrpiv (%s) ---\n", tname<S>());
    Rng gen(11, 22, 33, 51);
    const Matrix<S> A = decay_matrix<S>(gen, 400, 300);
    const double rtol = 1e-6;
    const double opt20 = opt_err(A, 20);

    for (IdMethod meth : {IdMethod::fast, IdMethod::svd, IdMethod::lstsq}) {
        Options o;
        o.method = meth;
        auto a = id_sketch(A, rtol, o);
        auto b = id_qrpiv(A, rtol, meth);
        const double ea = id_err(A, a), eb = id_err(A, b);
        check(a.T.rows() == a.k && a.T.cols() == 300 - a.k && is_perm(a.piv, 300) && ea < 100 * rtol,
              "id_sketch %-5s tol 1e-6: k=%d err=%.1e", to_string(meth), a.k, ea);
        check(b.T.rows() == b.k && b.T.cols() == 300 - b.k && is_perm(b.piv, 300) && eb < 100 * rtol,
              "id_qrpiv  %-5s tol 1e-6: k=%d err=%.1e", to_string(meth), b.k, eb);

        auto c = id_sketch(A, 20.0, o);
        auto d = id_qrpiv(A, 20.0, meth);
        const double ec = id_err(A, c), ed = id_err(A, d);
        check(c.k == 20 && c.T.cols() == 280 && ec < 100 * opt20,
              "id_sketch %-5s rank 20: err=%.2e (optimal %.2e)", to_string(meth), ec, opt20);
        check(d.k == 20 && d.T.cols() == 280 && ed < 100 * opt20,
              "id_qrpiv  %-5s rank 20: err=%.2e (optimal %.2e)", to_string(meth), ed, opt20);
    }

    // Requested rank > true rank (rank 3, ask 8): T finite, still reconstructs
    Rng g2(2, 7, 1, 9);
    const Matrix<S> L = matmul(g2.uniform<S>(40, 3), g2.uniform<S>(3, 20));
    const Matrix<S> Z(40, 20);
    for (IdMethod meth : {IdMethod::fast, IdMethod::svd, IdMethod::lstsq}) {
        Options o;
        o.method = meth;
        for (int which = 0; which < 2; ++which) {
            const Matrix<S>& M = which == 0 ? L : Z;
            const int r = which == 0 ? 8 : 5;
            auto a = id_sketch(M, static_cast<double>(r), o);
            auto b = id_qrpiv(M, static_cast<double>(r), meth);
            const double ea = id_err(M, a), eb = id_err(M, b);
            check(a.k == r && a.T.cols() == 20 - r && all_finite(a.T) && ea < 1e-8 && b.k == r &&
                      b.T.cols() == 20 - r && all_finite(b.T) && eb < 1e-8,
                  "over-rank %s ask %d %-5s: T finite, err sketch=%.1e qrpiv=%.1e",
                  which == 0 ? "rank-3" : "zero  ", r, to_string(meth), ea, eb);
        }
    }

    // Zero matrix in tolerance mode: k = 0 everywhere
    const Matrix<S> Z0(100, 60);
    {
        auto a = id_sketch(Z0, 1e-6);
        auto b = id_qrpiv(Z0, 1e-6);
        auto q = qr_sketch(Z0, 1e-6);
        auto s = svd_sketch(Z0, 1e-6);
        check(a.k == 0 && a.T.rows() == 0 && a.T.cols() == 60 && b.k == 0 && b.T.cols() == 60 &&
                  q.Q.cols() == 0 && s.s.empty(),
              "zero 100x60 tol 1e-6: k=0 (id_sketch, id_qrpiv, qr_sketch, svd_sketch)");
    }

    // k == n
    Rng g3(8, 8, 8, 9);
    const Matrix<S> W = g3.uniform<S>(50, 20);
    for (IdMethod meth : {IdMethod::fast, IdMethod::svd, IdMethod::lstsq}) {
        Options o;
        o.method = meth;
        auto a = id_sketch(W, 25.0, o);
        auto b = id_qrpiv(W, 20.0, meth);
        auto c = id_qrpiv(W, 1e-12, meth);
        check(a.k == 20 && a.T.rows() == 20 && a.T.cols() == 0 && b.k == 20 && b.T.cols() == 0 &&
                  c.k == 20 && c.T.cols() == 0 && is_perm(a.piv, 20) && is_perm(b.piv, 20),
              "k == n (50x20) %-5s: T is 20x0", to_string(meth));
    }
}

// ---------------- 6. matrix-free ----------------

template <class S>
void test_linop() {
    std::printf("\n--- matrix-free operators (%s) ---\n", tname<S>());
    Rng gen(11, 22, 33, 53);
    const Matrix<S> At = decay_matrix<S>(gen, 400, 300);
    const Matrix<S> Aw = adjoint(At);
    const double tiny = 1e-12;

    for (const Matrix<S>* Ap : {&At, &Aw}) {
        const Matrix<S>& A = *Ap;
        const char* shape = A.rows() >= A.cols() ? "tall" : "wide";
        const LinearOperator<S> op = matfree(A);
        for (double rtol : {20.0, 3e-7}) {
            Rng r1(1, 2, 3, 5), r2(1, 2, 3, 5);
            Options o1, o2;
            o1.rng = &r1;
            o2.rng = &r2;
            auto qd = qr_sketch(A, rtol, o1);
            auto qo = qr_sketch(op, rtol, o2);
            // Q^H A (dense) and (A^H Q)^H (operator) differ in roundoff; columns
            // of Q near the threshold amplify it, so compare the product Q R
            const double dqr = rel_diff(matmul(qd.Q, qd.R), matmul(qo.Q, qo.R));
            check(qd.p == qo.p && qd.Q.cols() == qo.Q.cols() && dqr < tiny,
                  "%s qr_sketch rtol=%g: operator == dense (k=%d, QR diff %.1e)", shape, rtol, qo.Q.cols(), dqr);
            auto sd = svd_sketch(A, rtol, o1);
            auto so = svd_sketch(op, rtol, o2);
            check(sd.s.size() == so.s.size() && max_sval_diff(so.s, sd.s) < tiny &&
                      rel_diff(matmul(scale_cols(sd.U, sd.s), sd.Vh), matmul(scale_cols(so.U, so.s), so.Vh)) < tiny,
                  "%s svd_sketch rtol=%g: operator == dense (k=%zu)", shape, rtol, so.s.size());
            for (IdMethod meth : {IdMethod::fast, IdMethod::lstsq}) {
                o1.method = o2.method = meth;
                auto id = id_sketch(A, rtol, o1);
                auto io = id_sketch(op, rtol, o2);
                // T = R11 \ R12 amplifies roundoff by cond(R11): compare A_skel T
                bool ok = id.k == io.k && id.piv == io.piv;
                const double dT = ok ? rel_diff(matmul(take_columns(A, id.piv, 0, id.k), id.T),
                                                matmul(take_columns(A, io.piv, 0, io.k), io.T))
                                     : 1.0;
                check(ok && dT < tiny, "%s id_sketch %-5s rtol=%g: operator == dense (k=%d, A_skel T diff %.1e)",
                      shape, to_string(meth), rtol, io.k, dT);
            }
            check(r1.state() == r2.state(), "%s rtol=%g: same rng advance", shape, rtol);
        }
    }

    // Tolerance mode with the materializing fallback: bitwise equal
    Rng g2(3, 1, 4, 5);
    const Matrix<S> F = g2.uniform<S>(60, 50);
    const LinearOperator<S> fop = matfree(F);
    {
        auto qd = qr_sketch(F, 1e-10);
        auto qo = qr_sketch(fop, 1e-10);
        auto sd = svd_sketch(F, 1e-10);
        auto so = svd_sketch(fop, 1e-10);
        check(same(qd.Q, qo.Q) && same(qd.R, qo.R) && qd.p == qo.p && same(sd.U, so.U) && sd.s == so.s,
              "fallback (60x50 full rank) via materialized operator == dense, bitwise");
        auto id = id_qrpiv(F, 1e-10, IdMethod::lstsq);
        auto io = id_qrpiv(fop, 1e-10, IdMethod::lstsq);
        check(id.k == io.k && id.piv == io.piv && same(id.T, io.T), "id_qrpiv on operator == dense, bitwise");
    }
    {
        // Operator that exposes its dense matrix takes the dense path
        const LinearOperator<S> dop = LinearOperator<S>::from_matrix(At);
        auto a = svd_sketch(At, 1e-6);
        auto b = svd_sketch(dop, 1e-6);
        auto c = svd_sketch(LinearOperator<S>::from_matrix(Aw), 1e-6);
        auto d = svd_sketch(Aw, 1e-6);
        check(same(a.U, b.U) && a.s == b.s && same(a.Vh, b.Vh) && same(c.U, d.U) && c.s == d.s,
              "operator exposing dense matrix == dense path, bitwise (tall and wide)");
    }

    // Genuinely matrix-free low-rank product A = B C, applied as B (C x)
    Rng g3(9, 9, 9, 9);
    const int m = 500, n = 400, r = 10;
    const Matrix<S> B = g3.uniform<S>(m, r), C = g3.uniform<S>(r, n);
    const Matrix<S> BC = matmul(B, C);
    long long applied = 0;
    const LinearOperator<S> lr(
        m, n,
        [&](const Matrix<S>& X) { applied += X.cols(); return matmul(B, matmul(C, X)); },
        [&](const Matrix<S>& Y) { applied += Y.cols(); return matmul(C, matmul(B, Y, 'C', 'N'), 'C', 'N'); });
    {
        auto [Q, flag, diagR] = orth_sketch(lr, static_cast<double>(r));
        const double span = rel_diff(BC, matmul(Q, matmul(Q, BC, 'C', 'N')));
        check(flag == 0 && Q.cols() == r + 12 && span < 1e-13, "low-rank B*C orth_sketch rank 10: span err %.1e",
              span);
        applied = 0;
        auto s = svd_sketch(lr, static_cast<double>(r));
        const double e = svd_err(BC, s);
        check(s.s.size() == 10 && e < 1e-13 && applied < n,
              "low-rank B*C svd_sketch rank 10: err %.1e, %lld columns applied (< n)", e, applied);
        auto q = qr_sketch(lr, static_cast<double>(r));
        check(q.Q.cols() == 10 && qr_err(BC, q) < 1e-13, "low-rank B*C qr_sketch rank 10: err %.1e", qr_err(BC, q));
        Options o;
        o.method = IdMethod::svd;
        auto id = id_sketch(lr, static_cast<double>(r), o);
        check(id.k == 10 && id_err(BC, id) < 1e-12, "low-rank B*C id_sketch(svd) rank 10: err %.1e", id_err(BC, id));
        auto w = svd_sketch(lr.adjoint(), static_cast<double>(r));
        check(w.s.size() == 10 && svd_err(adjoint(BC), w) < 1e-13,
              "low-rank (B*C)^H (wide) svd_sketch rank 10: err %.1e", svd_err(adjoint(BC), w));
        auto t = svd_sketch(lr, 1e-8);
        check(t.s.size() == 10 && svd_err(BC, t) < 1e-12,
              "low-rank B*C svd_sketch tol 1e-8: k=%zu err %.1e", t.s.size(), svd_err(BC, t));
    }

    // Wrong output shape from apply is reported
    bool threw = false;
    try {
        const LinearOperator<S> bad(10, 8, [](const Matrix<S>& X) { return Matrix<S>(9, X.cols()); },
                                    [](const Matrix<S>& Y) { return Matrix<S>(8, Y.cols()); });
        (void)svd_sketch(bad, 2.0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "operator returning a wrongly shaped block throws");
}

// ---------------- 7. power iteration ----------------

template <class S>
void test_power() {
    std::printf("\n--- power iteration (%s) ---\n", tname<S>());
    Rng gen(4, 4, 4, 5);
    std::vector<double> s(200);
    for (std::size_t j = 0; j < s.size(); ++j) s[j] = 1.0 / static_cast<double>(j + 1);
    const Matrix<S> A = with_svals<S>(gen, 300, 200, s);
    const double opt = opt_err(A, 20);
    double err[3];
    for (int q = 0; q < 3; ++q) {
        Rng r(1, 2, 3, 5);
        Options o;
        o.power_iter = q;
        o.rng = &r;
        err[q] = svd_err(A, svd_sketch(A, 20.0, o));
    }
    check(err[1] <= err[0] && err[2] <= err[1] * (1 + 1e-10),
          "slow decay 300x200 rank 20: err q=0 %.4e, q=1 %.4e, q=2 %.4e (optimal %.4e)", err[0], err[1],
          err[2], opt);
    Rng r(1, 2, 3, 5);
    Options o;
    o.power_iter = 2;
    o.rng = &r;
    auto f = qr_sketch(A, 1e-2, o);
    check(qr_err(A, f) < 0.1 && ortho_err_cols(f.Q) < 1e-13, "power_iter=2 tolerance mode qr_sketch: k=%d err %.2e",
          f.Q.cols(), qr_err(A, f));
}

// ---------------- 8. reproducibility ----------------

template <class S>
void test_seed() {
    std::printf("\n--- seed reproducibility (%s) ---\n", tname<S>());
    Rng gen(6, 6, 6, 7);
    const Matrix<S> A = decay_matrix<S>(gen, 200, 150);
    Rng r1(1, 2, 3, 5), r2(1, 2, 3, 5);
    Options o1, o2;
    o1.rng = &r1;
    o2.rng = &r2;
    auto a = svd_sketch(A, 1e-6, o1);
    auto b = svd_sketch(A, 1e-6, o2);
    check(same(a.U, b.U) && a.s == b.s && same(a.Vh, b.Vh) && r1.state() == r2.state(),
          "same seed -> identical svd_sketch output and rng state");
    check(!rng_equal<S>(r1, {1, 2, 3, 5}), "rng advanced to {%d,%d,%d,%d}", r1.state()[0], r1.state()[1],
          r1.state()[2], r1.state()[3]);
    auto c = svd_sketch(A, 1e-6, o1);
    check(!same(a.U, c.U), "second call with the advanced rng draws a new sketch");
    auto d1 = id_sketch(A, 10.0);
    auto d2 = id_sketch(A, 10.0);
    Rng def;
    Options od;
    od.rng = &def;
    auto d3 = id_sketch(A, 10.0, od);
    check(d1.piv == d2.piv && same(d1.T, d2.T) && same(d1.T, d3.T),
          "default rng (nullptr) is deterministic and equals Rng{0,0,0,1}");
}

// ---------------- 9. errors ----------------

template <class S>
void test_errors() {
    std::printf("\n--- invalid arguments (%s) ---\n", tname<S>());
    Rng gen(1, 1, 1, 1);
    const Matrix<S> A = gen.uniform<S>(30, 20);
    auto throws = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check(throws([] { (void)parse_id_method("qr"); }), "parse_id_method(\"qr\") throws invalid_argument");
    check(throws([&] { (void)id_qrpiv(A, 5.0, "bogus"); }), "id_qrpiv(method \"bogus\") throws invalid_argument");
    check(throws([&] {
              Options o;
              o.method = static_cast<IdMethod>(7);
              (void)id_sketch(A, 5.0, o);
          }),
          "id_sketch(invalid method enum) throws invalid_argument");
    check(throws([&] { (void)id_qrpiv(matfree(A), 5.0, static_cast<IdMethod>(-1)); }),
          "id_qrpiv(operator, invalid method enum) throws invalid_argument");
    check(throws([&] {
              Options o;
              o.block_size = 0;
              (void)orth_sketch(A, 0.1, o);
          }),
          "block_size = 0 throws invalid_argument");
}

// ---------------- cross-port check ----------------

template <class S>
Matrix<S> crossport_matrix(int m, int n) {
    Matrix<S> A(m, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            if constexpr (std::is_same_v<S, double>)
                A(i, j) = 1.0 / (i + j + 1);
            else
                A(i, j) = cdouble(1.0, (i - j) / 80.0) / static_cast<double>(i + j + 1);
        }
    return A;
}

void print_line(const char* name, const std::vector<double>& v) {
    std::printf("CROSSPORT %-10s", name);
    for (double x : v) std::printf(" %.15e", x);
    std::printf("\n");
}

void crossport() {
    std::printf("\n--- cross-port check ---\n");
    Options o;
    o.extra_samples = 3;
    Rng r(1, 2, 3, 5);
    o.rng = &r;
    auto orr = orth_sketch(crossport_matrix<double>(50, 30), 5.0, o);
    const std::array<int, 4> after = r.state();
    Rng rc(1, 2, 3, 5);
    o.rng = &rc;
    auto orc = orth_sketch(crossport_matrix<cdouble>(50, 30), 5.0, o);

    Rng ri(1, 2, 3, 5);
    Options oi;
    oi.rng = &ri;
    auto id = id_sketch(crossport_matrix<double>(300, 200), 1e-10, oi);

    Rng rs(1, 2, 3, 5);
    Options os;
    os.power_iter = 1;
    os.rng = &rs;
    auto sv = svd_sketch(crossport_matrix<cdouble>(120, 80), 10.0, os);

    print_line("orth_real", orr.diagR);
    print_line("orth_cplx", orc.diagR);
    std::printf("CROSSPORT %-10s %d %d %d %d\n", "seed_after", after[0], after[1], after[2], after[3]);
    std::printf("CROSSPORT %-10s %d", "id_real", id.k);
    for (int i = 0; i < 10 && i < static_cast<int>(id.piv.size()); ++i) std::printf(" %d", id.piv[static_cast<std::size_t>(i)]);
    std::printf("\n");
    std::printf("CROSSPORT %-10s %zu", "svd_cplx", sv.s.size());
    for (std::size_t i = 0; i < 5 && i < sv.s.size(); ++i) std::printf(" %.15e", sv.s[i]);
    std::printf("\n");
}

template <class S>
void run_all() {
    test_orth<S>();
    test_qr<S>();
    test_svd<S>();
    test_id<S>();
    test_linop<S>();
    test_power<S>();
    test_seed<S>();
    test_errors<S>();
}

}  // namespace

int main() {
    std::printf("librla C++ %s test suite\n", librla::version);
    try {
        test_rng();
        run_all<double>();
        run_all<cdouble>();
        crossport();
    } catch (const std::exception& e) {
        std::printf("  [FAIL] unexpected exception: %s\n", e.what());
        ++n_fail;
    }
    std::printf("\n%d checks: %d passed, %d failed\n", n_pass + n_fail, n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
