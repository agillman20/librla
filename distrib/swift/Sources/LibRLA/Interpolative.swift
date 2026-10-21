// Interpolative.swift - interpolative decompositions idSketch and idQRPiv
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

// MARK: - Randomized ID

/// Interpolative decomposition by randomized sketching:
/// `A[:, piv[k...]] ≈ A[:, piv[..<k]] T`.
///
/// The skeleton columns `piv[..<k]` and the triangular factor come from
/// `qrSketch`; `T` (`k x (n-k)`) is computed by `method`.
///
/// - Parameters: as for `orthSketch`, plus `method` (`.fast` by default).
/// - Returns: `k` (rank), `piv` (0-based column permutation of length `n`),
///   `t` (`k x (n-k)` interpolation matrix).
public func idSketch<Op: LinearOperator>(
    _ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0, extraSamples: Int = 12,
    method: IDMethod = .fast, rng: inout Rng
) -> (k: Int, piv: [Int], t: Matrix<Op.Element>) {
    let (_, r, piv) = qrSketch(A, rtol: rtol, blockSize: blockSize, powerIter: powerIter,
                               extraSamples: extraSamples, rng: &rng)
    let k = r.rowCount
    // Rank mode filters only exact zeros in the SVD-based solve.
    let rtolForSVD = rtol >= 1 ? 0 : rtol
    let t: Matrix<Op.Element>
    switch method {
    case .lstsq: t = interpolationLeastSquares(A, piv: piv, k: k)
    case .svd: t = interpolationSVD(r, k: k, rtolForSVD: rtolForSVD)
    case .fast: t = interpolationFast(r, k: k, rtolForSVD: rtolForSVD)
    }
    return (k, piv, t)
}

/// `idSketch` with a fresh default `Rng()` (deterministic).
public func idSketch<Op: LinearOperator>(
    _ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0, extraSamples: Int = 12,
    method: IDMethod = .fast
) -> (k: Int, piv: [Int], t: Matrix<Op.Element>) {
    var rng = Rng()
    return idSketch(A, rtol: rtol, blockSize: blockSize, powerIter: powerIter,
                    extraSamples: extraSamples, method: method, rng: &rng)
}

// MARK: - Deterministic ID

/// Interpolative decomposition by deterministic QR with column pivoting
/// (LAPACK `geqp3` on all of `A`): `A[:, piv[k...]] ≈ A[:, piv[..<k]] T`.
///
/// A matrix-free operator is materialized (O(n) applications).
///
/// - Parameters:
///   - A: an `m x n` matrix or operator.
///   - rtol: relative tolerance (< 1) or target rank (>= 1).
///   - method: how `T` is computed (`.fast` by default).
/// - Returns: `k`, `piv` (0-based), `t` (`k x (n-k)`).
public func idQRPiv<Op: LinearOperator>(
    _ A: Op, rtol: Double, method: IDMethod = .fast
) -> (k: Int, piv: [Int], t: Matrix<Op.Element>) {
    precondition(rtol.isFinite, "rtol must be finite, got \(rtol)")
    let (m, n) = A.shape
    let a = A.materialized()
    let (_, r, piv) = LAPACK.pivotedQR(a, formQ: false)

    let k: Int, rtolForSVD: Double
    if rtol >= 1 {
        k = min(Int(rtol.rounded(.down)), min(m, n))
        rtolForSVD = 0  // minimal filtering in rank mode
    } else {
        k = rankFromDiagonal(r.diagonal.map(\.modulus), rtol: rtol)
        rtolForSVD = rtol
    }

    if k == 0 { return (0, piv, Matrix(rows: 0, columns: n)) }
    if k == n { return (k, piv, Matrix(rows: k, columns: 0)) }

    let t: Matrix<Op.Element>
    switch method {
    case .lstsq: t = interpolationLeastSquares(a, piv: piv, k: k)
    case .svd: t = interpolationSVD(r, k: k, rtolForSVD: rtolForSVD)
    case .fast: t = interpolationFast(r, k: k, rtolForSVD: rtolForSVD)
    }
    return (k, piv, t)
}

// MARK: - Interpolation matrix T

/// `T` by least squares on the columns of `A`: `A[:, piv[..<k]] T = A[:, piv[k...]]`
/// (`gelsd` with `rcond = eps`, as scipy.linalg.lstsq). A matrix-free
/// operator is materialized only when there is something to solve.
func interpolationLeastSquares<Op: LinearOperator>(_ A: Op, piv: [Int], k: Int) -> Matrix<Op.Element> {
    let n = A.shape.columns
    if k == 0 || k >= n { return Matrix(rows: k, columns: max(n - k, 0)) }
    let a = A.materialized()
    return LAPACK.leastSquares(a.columns(piv[..<k]), a.columns(piv[k...]), rcond: Double.ulpOfOne)
}

/// `T = pinv(R11) R12` via the SVD of `R11`, dropping singular values below
/// `max(rtolForSVD, eps) * s_max`; an all-zero `R11` gives `T = 0`.
func interpolationSVD<T: Scalar>(_ r: Matrix<T>, k: Int, rtolForSVD: Double) -> Matrix<T> {
    let n = r.columnCount
    if k == 0 { return Matrix(rows: 0, columns: n) }
    let r11 = r.submatrix(rows: 0..<k, columns: 0..<k)
    let r12 = r.submatrix(rows: 0..<k, columns: k..<n)

    let (u, s, vh) = LAPACK.svd(r11)
    // Flooring the threshold at eps keeps a rank-deficient R11 (requested
    // rank above the true rank) from inverting near-zero singular values.
    let smax = s.max() ?? 0
    if smax == 0 { return Matrix(rows: k, columns: n - k) }
    let tol = max(rtolForSVD, Double.ulpOfOne)
    let keep = s.indices.filter { s[$0] >= tol * smax }
    if keep.isEmpty { return Matrix(rows: k, columns: n - k) }

    let inverse = keep.map { 1.0 / s[$0] }
    let w = Matrix.adjointProduct(u.columns(keep), r12).scalingRows(by: inverse)
    return Matrix.adjointProduct(vh.rows(keep), w)
}

/// `T = R11 \ R12` by triangular solve, falling back to the SVD-based
/// minimum-norm solve when `min|diag R11| <= eps * max|diag R11|`.
func interpolationFast<T: Scalar>(_ r: Matrix<T>, k: Int, rtolForSVD: Double) -> Matrix<T> {
    let n = r.columnCount
    if k == 0 { return Matrix(rows: 0, columns: n) }
    let r11 = r.submatrix(rows: 0..<k, columns: 0..<k)
    let r12 = r.submatrix(rows: 0..<k, columns: k..<n)

    let d = r11.diagonal.map(\.modulus)
    if d.min()! <= Double.ulpOfOne * d.max()! {
        return interpolationSVD(r, k: k, rtolForSVD: rtolForSVD)
    }
    return BLAS.solveUpperTriangular(r11, r12)
}
