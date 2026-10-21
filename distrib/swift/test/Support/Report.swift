// Report.swift - PASS/FAIL reporting, test scalars, fixtures and error metrics
//
// The checks live in this ordinary library target so that both the
// swift-testing suite (`swift test`) and the `librla-test` executable
// (`swift run librla-test`, one PASS/FAIL line per check) run them.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import ComplexModule
import Foundation
import LibRLA
import RealModule

/// Collects check results, printing one PASS/FAIL line per check when `echo` is on.
public final class CheckReport {
    public struct Result: Sendable {
        public let name: String
        public let passed: Bool
        public let detail: String
    }

    public private(set) var results: [Result] = []
    private let echo: Bool

    public init(echo: Bool = true) {
        self.echo = echo
    }

    public func check(_ name: String, _ passed: Bool, _ detail: String = "") {
        results.append(Result(name: name, passed: passed, detail: detail))
        if echo {
            print("\(passed ? "PASS" : "FAIL")  \(name)\(detail.isEmpty ? "" : "  [\(detail)]")")
        }
    }

    public var failures: [Result] { results.filter { !$0.passed } }
}

/// Scalars the suite runs over: Double and Complex<Double>.
public protocol TestScalar: Scalar {
    static var tag: String { get }
    /// `re + i im` (the imaginary part is dropped for Double).
    static func make(_ re: Double, _ im: Double) -> Self
}

extension Double: TestScalar {
    public static var tag: String { "real" }
    public static func make(_ re: Double, _ im: Double) -> Double { re }
}

extension Complex: TestScalar where RealType == Double {
    public static var tag: String { "cplx" }
    public static func make(_ re: Double, _ im: Double) -> Complex<Double> { Complex(re, im) }
}

// MARK: - Formatting

func sci(_ x: Double) -> String { String(format: "%.2e", x) }

// MARK: - Fixtures

/// `m x r` matrix with orthonormal columns (Q of a random matrix's pivoted QR).
func randomOrthonormal<T: TestScalar>(_ m: Int, _ r: Int, rng: inout Rng) -> Matrix<T> {
    Matrix<T>.random(rows: m, columns: r, using: &rng).pivotedQR().q
}

/// `U diag(sigma) Vᴴ` with random orthonormal `U` (`m x r`), `V` (`n x r`),
/// `r = min(m, n)`. Returns the matrix and its singular values.
func spectrumMatrix<T: TestScalar>(_ m: Int, _ n: Int, seed: UInt64,
                                   sigma: (Int) -> Double) -> (a: Matrix<T>, s: [Double]) {
    var rng = Rng(seed: seed)
    let r = min(m, n)
    let u: Matrix<T> = randomOrthonormal(m, r, rng: &rng)
    let v: Matrix<T> = randomOrthonormal(n, r, rng: &rng)
    let s = (0..<r).map(sigma)
    return (u.scalingColumns(by: s) * v.adjoint, s)
}

/// 400 x 300 with singular values 10^(-j/4): numerical rank ~4 log10(1/rtol).
func fastDecayMatrix<T: TestScalar>(_: T.Type, seed: UInt64 = 1) -> (a: Matrix<T>, s: [Double]) {
    spectrumMatrix(400, 300, seed: seed) { pow(10.0, -Double($0) / 4) }
}

/// A random (numerically full-rank) matrix.
func randomMatrix<T: TestScalar>(_ m: Int, _ n: Int, seed: UInt64) -> Matrix<T> {
    var rng = Rng(seed: seed)
    return Matrix<T>.random(rows: m, columns: n, using: &rng)
}

/// The cross-port matrices: Ar[i,j] = 1/(i+j+1), Ac[i,j] = (1 + 1i*(i-j)/80)/(i+j+1).
func crossportMatrix<T: TestScalar>(_ m: Int, _ n: Int) -> Matrix<T> {
    Matrix(rows: m, columns: n) { i, j in
        let d = Double(i + j + 1)
        return T.make(1 / d, (Double(i - j) / 80) / d)
    }
}

/// A matrix-free operator that wraps `m` without exposing it as dense.
func hiddenDense<T: TestScalar>(_ m: Matrix<T>) -> FunctionOperator<T> {
    FunctionOperator(rows: m.rowCount, columns: m.columnCount,
                     apply: { m * $0 }, applyAdjoint: { Matrix.adjointProduct(m, $0) })
}

// MARK: - Metrics

func relError<T: TestScalar>(_ a: Matrix<T>, _ b: Matrix<T>) -> Double {
    let scale = a.frobeniusNorm
    return (a - b).frobeniusNorm / (scale > 0 ? scale : 1)
}

/// ||Qᴴ Q - I||_F.
func orthonormalityError<T: TestScalar>(_ q: Matrix<T>) -> Double {
    (Matrix.adjointProduct(q, q) - Matrix<T>.identity(q.columnCount)).frobeniusNorm
}

/// ||Q Qᴴ - I||_F for a matrix with orthonormal rows.
func rowOrthonormalityError<T: TestScalar>(_ vh: Matrix<T>) -> Double {
    orthonormalityError(vh.adjoint)
}

/// ||A - Q Qᴴ A||_F / ||A||_F.
func spanError<T: TestScalar>(_ a: Matrix<T>, _ q: Matrix<T>) -> Double {
    relError(a, q * Matrix.adjointProduct(q, a))
}

func svdReconstruction<T: TestScalar>(_ u: Matrix<T>, _ s: [Double], _ vh: Matrix<T>) -> Matrix<T> {
    u.scalingColumns(by: s) * vh
}

/// ||A[:, piv[k...]] - A[:, piv[..<k]] T||_F / ||A||_F.
func idError<T: TestScalar>(_ a: Matrix<T>, k: Int, piv: [Int], t: Matrix<T>) -> Double {
    let scale = a.frobeniusNorm
    let residual = a.columns(piv[k...]) - a.columns(piv[..<k]) * t
    return residual.frobeniusNorm / (scale > 0 ? scale : 1)
}

/// sqrt(sum_{j >= k} s_j^2) / sqrt(sum s_j^2): the optimal rank-k Frobenius error.
func optimalTail(_ s: [Double], _ k: Int) -> Double {
    let total = s.reduce(0) { $0 + $1 * $1 }
    let tail = s.dropFirst(k).reduce(0) { $0 + $1 * $1 }
    return (tail / total).squareRoot()
}

func maxAbsDiff(_ x: [Double], _ y: [Double]) -> Double {
    guard x.count == y.count else { return .infinity }
    return zip(x, y).reduce(0) { max($0, abs($1.0 - $1.1)) }
}

func isPermutation(_ p: [Int], _ n: Int) -> Bool {
    p.count == n && Set(p) == Set(0..<n)
}

func isUpperTriangular<T: TestScalar>(_ r: Matrix<T>) -> Bool {
    for j in 0..<r.columnCount {
        for i in (j + 1)..<max(j + 1, r.rowCount) where r[i, j] != .zero { return false }
    }
    return true
}

func isFinite<T: TestScalar>(_ m: Matrix<T>) -> Bool {
    m.elements.allSatisfy { $0.modulus.isFinite }
}
