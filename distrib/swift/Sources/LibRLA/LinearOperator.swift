// LinearOperator.swift - dense matrices and matrix-free operators behind one protocol
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

/// An `m x n` linear map that LibRLA can sketch.
///
/// Conform a type to factor an implicit matrix (a fast transform, a
/// product of factors, a PDE solution operator, ...) without storing it.
/// `Matrix` conforms, and `FunctionOperator` wraps a pair of closures.
///
/// Operators are applied to blocks of vectors. A matrix-free operator in
/// tolerance mode may be materialized column by column (O(n) applications)
/// when the sketch falls back to a deterministic factorization, as in
/// librla.py; prefer rank mode when that cost is unacceptable.
public protocol LinearOperator<Element> {
    associatedtype Element: Scalar

    /// `(m, n)`: the operator maps vectors of length `n` to length `m`.
    var shape: (rows: Int, columns: Int) { get }

    /// `A X` for an `n x k` block `X`; returns an `m x k` matrix.
    func apply(_ x: Matrix<Element>) -> Matrix<Element>

    /// `Aᴴ W` (conjugate transpose) for an `m x k` block `W`; returns an `n x k` matrix.
    func applyAdjoint(_ w: Matrix<Element>) -> Matrix<Element>

    /// The operator as a dense matrix when it already is one, so LibRLA
    /// can skip materialization and use BLAS directly. Defaults to `nil`.
    var denseMatrix: Matrix<Element>? { get }
}

extension LinearOperator {
    public var denseMatrix: Matrix<Element>? { nil }
}

extension Matrix: LinearOperator {
    public func apply(_ x: Matrix<Element>) -> Matrix<Element> { self * x }
    public func applyAdjoint(_ w: Matrix<Element>) -> Matrix<Element> { Matrix.adjointProduct(self, w) }
    public var denseMatrix: Matrix<Element>? { self }
}

/// A matrix-free operator given by closures for `A X` and `Aᴴ W`.
///
///     let A = FunctionOperator<Double>(rows: m, columns: n,
///                                      apply: { X in B * (C * X) },
///                                      applyAdjoint: { W in C.adjoint * (B.adjoint * W) })
public struct FunctionOperator<Element: Scalar>: LinearOperator {
    public let shape: (rows: Int, columns: Int)
    private let applyBlock: (Matrix<Element>) -> Matrix<Element>
    private let applyAdjointBlock: (Matrix<Element>) -> Matrix<Element>

    /// - Parameters:
    ///   - apply: maps an `n x k` block `X` to `A X` (`m x k`).
    ///   - applyAdjoint: maps an `m x k` block `W` to `Aᴴ W` (`n x k`).
    public init(rows: Int, columns: Int,
                apply: @escaping (Matrix<Element>) -> Matrix<Element>,
                applyAdjoint: @escaping (Matrix<Element>) -> Matrix<Element>) {
        precondition(rows >= 0 && columns >= 0, "operator dimensions must be nonnegative")
        shape = (rows, columns)
        applyBlock = apply
        applyAdjointBlock = applyAdjoint
    }

    public func apply(_ x: Matrix<Element>) -> Matrix<Element> { applyBlock(x) }
    public func applyAdjoint(_ w: Matrix<Element>) -> Matrix<Element> { applyAdjointBlock(w) }
}

/// The adjoint `Aᴴ` of an operator (used by `svdSketch` for wide input).
/// A dense base is adjointed once up front so BLAS sees a plain matrix.
struct AdjointOperator<Base: LinearOperator>: LinearOperator {
    typealias Element = Base.Element
    let base: Base
    let denseMatrix: Matrix<Element>?

    init(_ base: Base) {
        self.base = base
        denseMatrix = base.denseMatrix?.adjoint
    }

    var shape: (rows: Int, columns: Int) { (base.shape.columns, base.shape.rows) }
    func apply(_ x: Matrix<Element>) -> Matrix<Element> { base.applyAdjoint(x) }
    func applyAdjoint(_ w: Matrix<Element>) -> Matrix<Element> { base.apply(w) }
}

// Products used by the algorithms (librla.py _matvec, _rmatvec, _matmat_left,
// _get_matrix). Dense operators go straight to BLAS.
extension LinearOperator {
    /// `A X`.
    func times(_ x: Matrix<Element>) -> Matrix<Element> {
        precondition(x.rowCount == shape.columns, "A * X: X has \(x.rowCount) rows, expected \(shape.columns)")
        if let a = denseMatrix { return a * x }
        let y = apply(x)
        precondition(y.shape == (shape.rows, x.columnCount),
                     "apply returned \(y.shape), expected \((shape.rows, x.columnCount))")
        return y
    }

    /// `Aᴴ W`.
    func adjointTimes(_ w: Matrix<Element>) -> Matrix<Element> {
        precondition(w.rowCount == shape.rows, "Aᴴ * W: W has \(w.rowCount) rows, expected \(shape.rows)")
        if let a = denseMatrix { return Matrix.adjointProduct(a, w) }
        let z = applyAdjoint(w)
        precondition(z.shape == (shape.columns, w.columnCount),
                     "applyAdjoint returned \(z.shape), expected \((shape.columns, w.columnCount))")
        return z
    }

    /// `Qᴴ A`, computed as `(Aᴴ Q)ᴴ` for a matrix-free operator.
    func projected(onto q: Matrix<Element>) -> Matrix<Element> {
        if let a = denseMatrix { return Matrix.adjointProduct(q, a) }
        return adjointTimes(q).adjoint
    }

    /// The operator as a dense matrix, applying it to blocks of identity
    /// columns when it is matrix-free (O(n) applications in total).
    func materialized() -> Matrix<Element> {
        if let a = denseMatrix { return a }
        let (m, n) = shape
        let blockColumns = 64
        var elements: [Element] = []
        elements.reserveCapacity(m * n)
        var start = 0
        while start < n {
            let count = min(blockColumns, n - start)
            var e = Matrix<Element>(rows: n, columns: count)
            for j in 0..<count { e[start + j, j] = 1 }
            elements.append(contentsOf: times(e).elements)
            start += count
        }
        return Matrix(rows: m, columns: n, elements: elements)
    }
}
