// demo_basic.cpp - basic librla C++ demo: dense and matrix-free use
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

#include <cstdio>

#include "librla.hpp"

using librla::Matrix;

int main() {
    // Hilbert matrix 1000 x 500 (numerically low rank)
    const int m = 1000, n = 500;
    Matrix<double> A(m, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) A(i, j) = 1.0 / (i + j + 1);
    const double nA = librla::norm_fro(A);

    // Truncated SVD to relative tolerance 1e-10
    auto [U, s, Vh] = librla::svd_sketch(A, 1e-10);
    Matrix<double> Us = U;
    for (int j = 0; j < Us.cols(); ++j)
        for (int i = 0; i < m; ++i) Us(i, j) *= s[static_cast<std::size_t>(j)];
    Matrix<double> E = librla::matmul(Us, Vh);
    for (std::size_t t = 0; t < E.size(); ++t) E.data()[t] -= A.data()[t];
    std::printf("svd_sketch  (dense, rtol=1e-10): rank %zu, s[0] = %.6f, rel. error %.2e\n", s.size(),
                s[0], librla::norm_fro(E) / nA);

    // Interpolative decomposition, rank 20, least-squares T
    librla::Options opts;
    opts.method = librla::IdMethod::lstsq;
    auto [k, piv, T] = librla::id_sketch(A, 20, opts);
    Matrix<double> R = librla::matmul(librla::take_columns(A, piv, 0, k), T);
    Matrix<double> C = librla::take_columns(A, piv, k);
    for (std::size_t t = 0; t < R.size(); ++t) R.data()[t] -= C.data()[t];
    std::printf("id_sketch   (dense, rank 20, lstsq): k = %d, skeleton starts %d %d %d, rel. error %.2e\n",
                k, piv[0], piv[1], piv[2], librla::norm_fro(R) / nA);

    // Matrix-free: A = B C (rank 5) applied as B (C X), never formed
    librla::Rng rng(2026);
    const Matrix<double> B = rng.uniform<double>(m, 5), Cf = rng.uniform<double>(5, n);
    librla::LinearOperator<double> op(
        m, n,
        [&](const Matrix<double>& X) { return librla::matmul(B, librla::matmul(Cf, X)); },
        [&](const Matrix<double>& W) {
            return librla::matmul(Cf, librla::matmul(B, W, 'T', 'N'), 'T', 'N');
        });
    opts = librla::Options{};
    opts.rng = &rng;
    auto [Q, Rq, p] = librla::qr_sketch(op, 5, opts);
    std::printf("qr_sketch   (matrix-free B*C, rank 5): Q is %dx%d, R is %dx%d, |R(0,0)| = %.4f\n",
                Q.rows(), Q.cols(), Rq.rows(), Rq.cols(), std::abs(Rq(0, 0)));
    auto [Qo, flag, diagR] = librla::orth_sketch(op, 5, opts);
    std::printf("orth_sketch (matrix-free B*C, rank 5): flag %d, %d columns, diagR[5]/diagR[0] = %.1e\n",
                flag, Qo.cols(), diagR[5] / diagR[0]);
    return 0;
}
