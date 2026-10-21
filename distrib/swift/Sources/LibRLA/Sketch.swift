// Sketch.swift - orthSketch, qrSketch and svdSketch
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

// MARK: - Orthogonal sketch

/// Approximate orthonormal basis for the column space of `A` by randomized
/// sketching: `Q` from the pivoted QR of `A Ω`, with `Ω` uniform(-1, 1).
///
/// - Tolerance mode (`rtol < 1`): starting from `blockSize` columns, draw a
///   fresh `Ω` and grow the sketch fourfold until it covers the numerical
///   rank at `rtol` with `extraSamples` columns to spare, i.e.
///   `diagR[count - 1 - extraSamples] <= rtol * diagR[0]`.
/// - Rank mode (`rtol >= 1`): one sketch of `floor(rtol) + extraSamples` columns.
///
/// In both modes `Q` includes the `extraSamples` buffer columns.
///
/// - Parameters:
///   - A: an `m x n` matrix or operator.
///   - rtol: relative tolerance (< 1) or target rank (>= 1).
///   - blockSize: initial sketch size in tolerance mode (ignored in rank mode).
///   - powerIter: number of power iterations `x = Aᴴ(A x)` (1 or 2 help for
///     slowly decaying spectra).
///   - extraSamples: buffer columns beyond the target rank; 0 gives the
///     legacy last-column tolerance check.
///   - rng: random state, advanced by every test matrix drawn.
/// - Returns: `q` (`m x k` orthonormal), `flag` (0 success; 1 when
///   `rtol < eps` or the sketch reached `min(m, n)` in tolerance mode, with
///   `q` then `m x 0`), and `diagR` = |diag(R)| of the sketch's pivoted QR.
public func orthSketch<Op: LinearOperator>(
    _ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0, extraSamples: Int = 12,
    rng: inout Rng
) -> (q: Matrix<Op.Element>, flag: Int, diagR: [Double]) {
    checkOptions(rtol: rtol, blockSize: blockSize, powerIter: powerIter, extraSamples: extraSamples)
    let (m, n) = A.shape

    // Rank mode: single oversampled sketch, buffer included.
    if rtol >= 1 {
        let kmax = Int(rtol.rounded(.down))
        let (q, diagR) = sketchBasis(A, columns: kmax + extraSamples, powerIter: powerIter, rng: &rng)
        return (q, 0, diagR)
    }

    // Tolerance mode: geometric growth with the buffered tolerance check.
    let earlyExit = (q: Matrix<Op.Element>(rows: m, columns: 0), flag: 1, diagR: [Double]())
    if rtol < Double.ulpOfOne { return earlyExit }
    var blockSize = blockSize
    if blockSize >= min(m, n) { return earlyExit }

    while true {
        let (q, diagR) = sketchBasis(A, columns: blockSize, powerIter: powerIter, rng: &rng)
        // diagR is sorted decreasing; the multiplied form stays valid when diagR[0] == 0.
        if diagR.isEmpty
            || (diagR.count > extraSamples && diagR[diagR.count - 1 - extraSamples] <= rtol * diagR[0]) {
            return (q, 0, diagR)
        }
        blockSize = min(blockSize * 4, min(m, n))
        if blockSize >= min(m, n) { return earlyExit }
    }
}

/// `orthSketch` with a fresh default `Rng()` (deterministic).
public func orthSketch<Op: LinearOperator>(
    _ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0, extraSamples: Int = 12
) -> (q: Matrix<Op.Element>, flag: Int, diagR: [Double]) {
    var rng = Rng()
    return orthSketch(A, rtol: rtol, blockSize: blockSize, powerIter: powerIter,
                      extraSamples: extraSamples, rng: &rng)
}

/// One pass: Ω (n x columns), power iteration, Y = A Ω, pivoted QR of Y.
private func sketchBasis<Op: LinearOperator>(
    _ A: Op, columns: Int, powerIter: Int, rng: inout Rng
) -> (q: Matrix<Op.Element>, diagR: [Double]) {
    var x = Matrix<Op.Element>.random(rows: A.shape.columns, columns: columns, using: &rng)
    for _ in 0..<powerIter {
        x = LAPACK.pivotedQRBasis(A.adjointTimes(A.times(x)))
    }
    let (q, r, _) = LAPACK.pivotedQR(A.times(x))
    return (q, r.diagonal.map(\.modulus))
}

func checkOptions(rtol: Double, blockSize: Int, powerIter: Int, extraSamples: Int) {
    precondition(rtol.isFinite, "rtol must be finite, got \(rtol)")
    precondition(blockSize >= 0, "blockSize must be nonnegative, got \(blockSize)")
    precondition(powerIter >= 0, "powerIter must be nonnegative, got \(powerIter)")
    precondition(extraSamples >= 0, "extraSamples must be nonnegative, got \(extraSamples)")
}

// MARK: - Truncated QR with column pivoting

/// Truncated QR with column pivoting by randomized sketching:
/// `A[:, p] ≈ Q R`.
///
/// Sketches a basis `Qs` with `orthSketch`, factors `Qsᴴ A` by pivoted QR
/// and maps back. When the matrix is effectively full rank at `rtol`
/// (tolerance mode), a deterministic pivoted QR of `A` is used instead.
///
/// - Parameters: as for `orthSketch`. In tolerance mode the rank is the
///   number of `|R_ii| >= rtol |R_00|`; in rank mode it is `floor(rtol)`
///   (capped by the sketch size).
/// - Returns: `q` (`m x k`, orthonormal), `r` (`k x n`, upper triangular),
///   `p` (0-based column permutation of length `n`).
public func qrSketch<Op: LinearOperator>(
    _ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0, extraSamples: Int = 12,
    rng: inout Rng
) -> (q: Matrix<Op.Element>, r: Matrix<Op.Element>, p: [Int]) {
    let (m, n) = A.shape
    let rankMode = rtol >= 1

    // The basis includes the extraSamples buffer columns (truncated after the QR).
    let (qs, flag, _) = orthSketch(A, rtol: rtol, blockSize: blockSize, powerIter: powerIter,
                                   extraSamples: extraSamples, rng: &rng)
    let k = flag == 0 ? qs.columnCount : min(m, n)
    let needsFallback = (flag != 0 || k >= min(m, n)) && !rankMode

    let (q, r, p): (Matrix<Op.Element>, Matrix<Op.Element>, [Int])
    if needsFallback {
        (q, r, p) = LAPACK.pivotedQR(A.materialized())
    } else {
        let (qProj, rProj, pProj) = LAPACK.pivotedQR(A.projected(onto: qs))
        (q, r, p) = (qs * qProj, rProj, pProj)
    }
    let rank = rankMode
        ? min(Int(rtol.rounded(.down)), q.columnCount)
        : rankFromDiagonal(r.diagonal.map(\.modulus), rtol: rtol)
    return (q.columns(0..<rank), r.rows(0..<rank), p)
}

/// `qrSketch` with a fresh default `Rng()` (deterministic).
public func qrSketch<Op: LinearOperator>(
    _ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0, extraSamples: Int = 12
) -> (q: Matrix<Op.Element>, r: Matrix<Op.Element>, p: [Int]) {
    var rng = Rng()
    return qrSketch(A, rtol: rtol, blockSize: blockSize, powerIter: powerIter,
                    extraSamples: extraSamples, rng: &rng)
}

// MARK: - Truncated SVD

/// Truncated SVD by randomized sketching: `A ≈ U diag(s) Vh`.
///
/// Sketches a basis `Qs` with `orthSketch`, takes the SVD of `Qsᴴ A` and
/// maps back. When the matrix is effectively full rank at `rtol`
/// (tolerance mode), a deterministic SVD of `A` is used instead. A wide
/// `A` (`m < n`) is handled through its adjoint.
///
/// - Parameters: as for `orthSketch`. In tolerance mode the rank is the
///   number of `s_i >= rtol s_0`; in rank mode it is `floor(rtol)`
///   (capped by the sketch size).
/// - Returns: `u` (`m x k`, orthonormal columns), `s` (descending),
///   `vh` (`k x n`, orthonormal rows; V conjugate-transposed as in
///   Python and Julia).
public func svdSketch<Op: LinearOperator>(
    _ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0, extraSamples: Int = 12,
    rng: inout Rng
) -> (u: Matrix<Op.Element>, s: [Double], vh: Matrix<Op.Element>) {
    if A.shape.rows < A.shape.columns {
        // Aᴴ ≈ V diag(s) Uᴴ
        let (v, s, uh) = tallSVDSketch(AdjointOperator(A), rtol: rtol, blockSize: blockSize,
                                       powerIter: powerIter, extraSamples: extraSamples, rng: &rng)
        return (uh.adjoint, s, v.adjoint)
    }
    return tallSVDSketch(A, rtol: rtol, blockSize: blockSize, powerIter: powerIter,
                         extraSamples: extraSamples, rng: &rng)
}

/// `svdSketch` with a fresh default `Rng()` (deterministic).
public func svdSketch<Op: LinearOperator>(
    _ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0, extraSamples: Int = 12
) -> (u: Matrix<Op.Element>, s: [Double], vh: Matrix<Op.Element>) {
    var rng = Rng()
    return svdSketch(A, rtol: rtol, blockSize: blockSize, powerIter: powerIter,
                     extraSamples: extraSamples, rng: &rng)
}

private func tallSVDSketch<Op: LinearOperator>(
    _ A: Op, rtol: Double, blockSize: Int, powerIter: Int, extraSamples: Int, rng: inout Rng
) -> (u: Matrix<Op.Element>, s: [Double], vh: Matrix<Op.Element>) {
    let (m, n) = A.shape
    let rankMode = rtol >= 1

    // The basis includes the extraSamples buffer columns for more accurate
    // singular values (truncated after the SVD).
    let (qs, flag, _) = orthSketch(A, rtol: rtol, blockSize: blockSize, powerIter: powerIter,
                                   extraSamples: extraSamples, rng: &rng)
    let k = flag == 0 ? qs.columnCount : min(m, n)
    let needsFallback = (flag != 0 || k >= min(m, n)) && !rankMode

    let (u, s, vh): (Matrix<Op.Element>, [Double], Matrix<Op.Element>)
    if needsFallback {
        (u, s, vh) = LAPACK.svd(A.materialized())
    } else {
        let (uProj, sProj, vhProj) = LAPACK.svd(A.projected(onto: qs))
        (u, s, vh) = (qs * uProj, sProj, vhProj)
    }
    let rank = rankMode ? min(Int(rtol.rounded(.down)), s.count) : rankFromSingularValues(s, rtol: rtol)
    return (u.columns(0..<rank), Array(s[0..<rank]), vh.rows(0..<rank))
}

// MARK: - Rank determination

/// Numerical rank from descending singular values: the number of `s_i >= rtol * s_0`.
func rankFromSingularValues(_ s: [Double], rtol: Double) -> Int {
    guard let s0 = s.first, s0 > 0 else { return 0 }
    return s.count { $0 >= rtol * s0 }
}

/// Numerical rank from |diag(R)| of a pivoted QR: the number of `d_i >= rtol * d_0`.
func rankFromDiagonal(_ d: [Double], rtol: Double) -> Int {
    guard let d0 = d.first, d0 > 0 else { return 0 }
    return d.count { $0 >= rtol * d0 }
}
