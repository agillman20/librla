// LAPACK.swift - thin generic wrappers over the BLAS/LAPACK routines librla uses
//
// Every wrapper dispatches on the element type to the d* (Double) or z*
// (Complex<Double>) routine, queries the workspace size (lwork = -1), and
// treats info < 0 as a programming error. Dimensions are LP64 Int32.
// Matrix storage is handed to LAPACK through explicit raw-byte views;
// workspaces and tau are plain [Double] buffers holding `width` doubles
// per element (1 for real, 2 for complex).
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import CLapack
import RealModule

/// Converts a dimension to a 32-bit LAPACK integer, trapping on overflow.
@inline(__always)
func lapackInt(_ x: Int) -> Int32 {
    guard let value = Int32(exactly: x) else {
        preconditionFailure("dimension \(x) exceeds the 32-bit LAPACK integer range")
    }
    return value
}

@inline(__always)
private func checkArguments(_ info: Int32, _ routine: StaticString) {
    precondition(info >= 0, "\(routine): argument \(-info) had an illegal value")
}

extension Scalar {
    /// Doubles per element: 1 for Double, 2 for Complex<Double>.
    static var width: Int { kind == .real ? 1 : 2 }

    /// The real number `x` as LAPACK expects it in a `const void *` scalar argument.
    static func lapackScalar(_ x: Double) -> [Double] { kind == .real ? [x] : [x, 0] }

    /// The adjoint transpose flag ('T' for real data, 'C' for complex).
    static var adjointFlag: String { kind == .real ? "T" : "C" }
}

extension Array where Element: Scalar {
    /// Raw view of the storage for a LAPACK call (the array must be nonempty).
    mutating func withMutableRaw<R>(_ body: (UnsafeMutableRawPointer) -> R) -> R {
        withUnsafeMutableBytes { body($0.baseAddress!) }
    }
}

/// A zeroed buffer of `count` elements of `T`, as raw doubles.
private func workspace<T: Scalar>(_: T.Type, _ count: Int) -> [Double] {
    [Double](repeating: 0, count: max(1, count) * T.width)
}

/// The optimal workspace length reported by a query (real part of work[0]).
private func optimalLength(_ query: [Double], atLeast minimum: Int = 1) -> Int32 {
    lapackInt(max(minimum, Int(query[0])))
}

enum BLAS {
    /// `op(a) * op(b)` with `op` the identity or the adjoint (`gemm`).
    static func gemm<T: Scalar>(_ a: Matrix<T>, _ b: Matrix<T>,
                                adjointA: Bool = false, adjointB: Bool = false) -> Matrix<T> {
        let m = adjointA ? a.columnCount : a.rowCount
        let k = adjointA ? a.rowCount : a.columnCount
        let kb = adjointB ? b.columnCount : b.rowCount
        let n = adjointB ? b.rowCount : b.columnCount
        precondition(k == kb, "matrix product: inner dimensions \(k) and \(kb) differ")
        guard m > 0, n > 0, k > 0 else { return Matrix(rows: m, columns: n) }
        var M = lapackInt(m), N = lapackInt(n), K = lapackInt(k)
        var lda = lapackInt(a.rowCount), ldb = lapackInt(b.rowCount), ldc = M
        let alpha = T.lapackScalar(1), beta = T.lapackScalar(0)
        let ta = adjointA ? T.adjointFlag : "N", tb = adjointB ? T.adjointFlag : "N"
        // beta = 0: gemm writes every entry of C without reading it.
        return Matrix(rows: m, columns: n) { pc in
            a.withRawElements { pa in
                b.withRawElements { pb in
                    switch T.kind {
                    case .real:
                        rla_dgemm(ta, tb, &M, &N, &K, alpha, pa, &lda, pb, &ldb, beta, pc, &ldc)
                    case .complex:
                        rla_zgemm(ta, tb, &M, &N, &K, alpha, pa, &lda, pb, &ldb, beta, pc, &ldc)
                    }
                }
            }
        }
    }

    /// Solves `R X = B` for upper-triangular `R` (`trsm`; the strictly lower
    /// part of `R` is not referenced).
    static func solveUpperTriangular<T: Scalar>(_ r: Matrix<T>, _ b: consuming Matrix<T>) -> Matrix<T> {
        precondition(r.rowCount == r.columnCount && r.rowCount == b.rowCount, "triangular solve: shape mismatch")
        var x = consume b
        guard x.rowCount > 0, x.columnCount > 0 else { return x }
        var M = lapackInt(x.rowCount), N = lapackInt(x.columnCount), lda = M, ldb = M
        let alpha = T.lapackScalar(1)
        r.withRawElements { pr in
            x.elements.withMutableRaw { px in
                switch T.kind {
                case .real: rla_dtrsm("L", "U", "N", "N", &M, &N, alpha, pr, &lda, px, &ldb)
                case .complex: rla_ztrsm("L", "U", "N", "N", &M, &N, alpha, pr, &lda, px, &ldb)
                }
            }
        }
        return x
    }
}

enum LAPACK {
    /// Economic QR with column pivoting, `A[:, p] = Q R` (`geqp3` + `orgqr`).
    /// `Q` is `m x min(m,n)`, `R` is `min(m,n) x n` upper trapezoidal with
    /// its strictly lower part zeroed, `p` is 0-based. With `formQ: false`
    /// the returned `Q` is empty (`m x 0`). `A` is consumed: a temporary
    /// argument is factored in place without a copy.
    static func pivotedQR<T: Scalar>(_ A: consuming Matrix<T>, formQ: Bool = true)
        -> (q: Matrix<T>, r: Matrix<T>, p: [Int])
    {
        var a = consume A
        let (tau, p) = geqp3(&a)
        let r = a.upperTrapezoid()
        let q = formQ ? orgqr(a, tau: tau) : Matrix<T>(rows: a.rowCount, columns: 0)
        return (q, r, p)
    }

    /// The orthonormal factor `Q` (`m x min(m,n)`) of the pivoted QR of `A`.
    static func pivotedQRBasis<T: Scalar>(_ A: consuming Matrix<T>) -> Matrix<T> {
        var a = consume A
        let (tau, _) = geqp3(&a)
        return orgqr(a, tau: tau)
    }

    /// `geqp3` in place with all columns free. Returns tau (raw doubles,
    /// `min(m,n)` elements) and 0-based pivots.
    static func geqp3<T: Scalar>(_ a: inout Matrix<T>) -> (tau: [Double], pivots: [Int]) {
        let m = a.rowCount, n = a.columnCount
        guard m > 0, n > 0 else { return ([], Array(0..<n)) }
        var M = lapackInt(m), N = lapackInt(n), lda = M, info: Int32 = 0
        var jpvt = [Int32](repeating: 0, count: n)
        var tau = workspace(T.self, min(m, n))
        var rwork = [Double](repeating: 0, count: 2 * n)
        func run(_ work: inout [Double], _ lwork: inout Int32) {
            a.elements.withMutableRaw { pa in
                switch T.kind {
                case .real: rla_dgeqp3(&M, &N, pa, &lda, &jpvt, &tau, &work, &lwork, &info)
                case .complex: rla_zgeqp3(&M, &N, pa, &lda, &jpvt, &tau, &work, &lwork, &rwork, &info)
                }
            }
        }
        var query = workspace(T.self, 1)
        var lwork: Int32 = -1
        run(&query, &lwork)
        checkArguments(info, "geqp3")
        lwork = optimalLength(query, atLeast: 3 * n + 1)
        var work = workspace(T.self, Int(lwork))
        run(&work, &lwork)
        checkArguments(info, "geqp3")
        return (tau, jpvt.map { Int($0) - 1 })
    }

    /// Forms the first `min(m,n)` columns of Q from `geqp3` output (`orgqr`/`ungqr`).
    static func orgqr<T: Scalar>(_ a: consuming Matrix<T>, tau: [Double]) -> Matrix<T> {
        let m = a.rowCount, k = min(m, a.columnCount)
        // Reuses a's storage when it is uniquely referenced and already m x k.
        var q = a.columns(0..<k)
        _ = consume a
        guard m > 0, k > 0 else { return q }
        var M = lapackInt(m), K = lapackInt(k), lda = M, info: Int32 = 0
        func run(_ work: inout [Double], _ lwork: inout Int32) {
            q.elements.withMutableRaw { pq in
                switch T.kind {
                case .real: rla_dorgqr(&M, &K, &K, pq, &lda, tau, &work, &lwork, &info)
                case .complex: rla_zungqr(&M, &K, &K, pq, &lda, tau, &work, &lwork, &info)
                }
            }
        }
        var query = workspace(T.self, 1)
        var lwork: Int32 = -1
        run(&query, &lwork)
        checkArguments(info, "orgqr")
        lwork = optimalLength(query, atLeast: k)
        var work = workspace(T.self, Int(lwork))
        run(&work, &lwork)
        checkArguments(info, "orgqr")
        return q
    }

    /// Economic SVD `A = U diag(s) Vh` (`gesdd`, falling back to `gesvd`
    /// if divide and conquer fails to converge).
    static func svd<T: Scalar>(_ A: Matrix<T>) -> (u: Matrix<T>, s: [Double], vh: Matrix<T>) {
        let m = A.rowCount, n = A.columnCount, mn = min(m, n)
        guard mn > 0 else { return (Matrix(rows: m, columns: 0), [], Matrix(rows: 0, columns: n)) }
        if let result = gesdd(A) { return result }
        return gesvd(A)
    }

    private static func gesdd<T: Scalar>(_ A: Matrix<T>) -> (u: Matrix<T>, s: [Double], vh: Matrix<T>)? {
        let m = A.rowCount, n = A.columnCount, mn = min(m, n), mx = max(m, n)
        var a = A
        var s = [Double](repeating: 0, count: mn)
        var u = Matrix<T>(rows: m, columns: mn), vt = Matrix<T>(rows: mn, columns: n)
        var M = lapackInt(m), N = lapackInt(n), lda = M, ldu = M, ldvt = lapackInt(mn), info: Int32 = 0
        var iwork = [Int32](repeating: 0, count: 8 * mn)
        // rwork bound for jobz = 'S': LAPACK >= 3.7 and the older (3.2.1) formula.
        let lrwork = max(mn * max(5 * mn + 7, 2 * mx + 2 * mn + 1), 5 * mn * mn + 7 * mn)
        var rwork = [Double](repeating: 0, count: T.kind == .complex ? lrwork : 1)
        func run(_ work: inout [Double], _ lwork: inout Int32) {
            a.elements.withMutableRaw { pa in
                u.elements.withMutableRaw { pu in
                    vt.elements.withMutableRaw { pvt in
                        switch T.kind {
                        case .real:
                            rla_dgesdd("S", &M, &N, pa, &lda, &s, pu, &ldu, pvt, &ldvt,
                                    &work, &lwork, &iwork, &info)
                        case .complex:
                            rla_zgesdd("S", &M, &N, pa, &lda, &s, pu, &ldu, pvt, &ldvt,
                                    &work, &lwork, &rwork, &iwork, &info)
                        }
                    }
                }
            }
        }
        var query = workspace(T.self, 1)
        var lwork: Int32 = -1
        run(&query, &lwork)
        checkArguments(info, "gesdd")
        lwork = optimalLength(query)
        var work = workspace(T.self, Int(lwork))
        run(&work, &lwork)
        checkArguments(info, "gesdd")
        return info == 0 ? (u, s, vt) : nil
    }

    private static func gesvd<T: Scalar>(_ A: Matrix<T>) -> (u: Matrix<T>, s: [Double], vh: Matrix<T>) {
        let m = A.rowCount, n = A.columnCount, mn = min(m, n)
        var a = A
        var s = [Double](repeating: 0, count: mn)
        var u = Matrix<T>(rows: m, columns: mn), vt = Matrix<T>(rows: mn, columns: n)
        var M = lapackInt(m), N = lapackInt(n), lda = M, ldu = M, ldvt = lapackInt(mn), info: Int32 = 0
        var rwork = [Double](repeating: 0, count: 5 * mn)
        func run(_ work: inout [Double], _ lwork: inout Int32) {
            a.elements.withMutableRaw { pa in
                u.elements.withMutableRaw { pu in
                    vt.elements.withMutableRaw { pvt in
                        switch T.kind {
                        case .real:
                            rla_dgesvd("S", "S", &M, &N, pa, &lda, &s, pu, &ldu, pvt, &ldvt,
                                    &work, &lwork, &info)
                        case .complex:
                            rla_zgesvd("S", "S", &M, &N, pa, &lda, &s, pu, &ldu, pvt, &ldvt,
                                    &work, &lwork, &rwork, &info)
                        }
                    }
                }
            }
        }
        var query = workspace(T.self, 1)
        var lwork: Int32 = -1
        run(&query, &lwork)
        checkArguments(info, "gesvd")
        lwork = optimalLength(query)
        var work = workspace(T.self, Int(lwork))
        run(&work, &lwork)
        checkArguments(info, "gesvd")
        precondition(info == 0, "SVD did not converge (gesdd and gesvd both failed, info = \(info))")
        return (u, s, vt)
    }

    /// Minimum-norm least-squares solution of `A X = B` (`gelsd`), with
    /// singular values below `rcond * s_max` treated as zero.
    static func leastSquares<T: Scalar>(_ A: Matrix<T>, _ B: Matrix<T>, rcond: Double) -> Matrix<T> {
        precondition(A.rowCount == B.rowCount, "least squares: row counts differ")
        let m = A.rowCount, n = A.columnCount, nrhs = B.columnCount, mn = min(m, n)
        guard m > 0, n > 0, nrhs > 0 else { return Matrix(rows: n, columns: nrhs) }
        let ldbCount = max(m, n)
        let stride = MemoryLayout<T>.stride
        var b = Matrix<T>(rows: ldbCount, columns: nrhs) { dst in
            B.withRawElements { src in
                for j in 0..<nrhs {
                    let column = dst + j * ldbCount * stride
                    column.copyMemory(from: src + j * m * stride, byteCount: m * stride)
                    RawStorage.zero(column + m * stride, count: (ldbCount - m) * T.width)
                }
            }
        }
        var a = A
        var s = [Double](repeating: 0, count: mn)
        var M = lapackInt(m), N = lapackInt(n), NRHS = lapackInt(nrhs)
        var lda = M, ldb = lapackInt(ldbCount), rank: Int32 = 0, info: Int32 = 0, rcond = rcond
        // Minimum iwork/rwork sizes from the gelsd documentation (SMLSIZ = 25);
        // older LAPACK versions do not report them in the workspace query.
        let smlsiz = 25
        let nlvl = max(0, Int(Double.log2(Double(mn) / Double(smlsiz + 1))) + 1)
        let liworkMin = max(1, 3 * mn * nlvl + 11 * mn)
        let lrworkMin = 10 * mn + 2 * mn * smlsiz + 8 * mn * nlvl + 3 * smlsiz * nrhs
            + max((smlsiz + 1) * (smlsiz + 1), mn * (1 + nrhs) + 2 * nrhs)
        var iwork = [Int32](repeating: 0, count: 1)
        var rwork = [Double](repeating: 0, count: 1)
        func run(_ work: inout [Double], _ lwork: inout Int32) {
            a.elements.withMutableRaw { pa in
                b.elements.withMutableRaw { pb in
                    switch T.kind {
                    case .real:
                        rla_dgelsd(&M, &N, &NRHS, pa, &lda, pb, &ldb, &s, &rcond, &rank,
                                &work, &lwork, &iwork, &info)
                    case .complex:
                        rla_zgelsd(&M, &N, &NRHS, pa, &lda, pb, &ldb, &s, &rcond, &rank,
                                &work, &lwork, &rwork, &iwork, &info)
                    }
                }
            }
        }
        var query = workspace(T.self, 1)
        var lwork: Int32 = -1
        run(&query, &lwork)
        checkArguments(info, "gelsd")
        lwork = optimalLength(query)
        iwork = [Int32](repeating: 0, count: max(liworkMin, Int(iwork[0])))
        rwork = [Double](repeating: 0, count: max(lrworkMin, Int(rwork[0])))
        var work = workspace(T.self, Int(lwork))
        run(&work, &lwork)
        checkArguments(info, "gelsd")
        precondition(info == 0, "least squares: SVD did not converge (gelsd info = \(info))")
        return b.rows(0..<n)
    }
}
