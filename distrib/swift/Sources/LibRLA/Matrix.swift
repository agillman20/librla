// Matrix.swift - column-major dense matrix with value semantics
//
// Bulk operations (copies of row/column blocks, transposes, scaling, sums,
// norms) work on the raw storage as Doubles (one per real element, two per
// complex element) rather than element by element through the generic
// Scalar protocol: the library's generic code is not specialized for the
// caller's element type, and per-element generic calls would dominate.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import RealModule

/// A dense column-major matrix of `Double` or `Complex<Double>`.
///
/// `Matrix` has value semantics (copy-on-write storage). Entry `(i, j)`
/// (0-based) is `elements[i + j * rowCount]`, the LAPACK layout with
/// leading dimension `rowCount`.
public struct Matrix<Element: Scalar>: Hashable, Sendable {
    /// Number of rows.
    public let rowCount: Int
    /// Number of columns.
    public let columnCount: Int
    /// Entries in column-major order.
    public internal(set) var elements: [Element]

    /// `(rows, columns)`.
    public var shape: (rows: Int, columns: Int) { (rowCount, columnCount) }

    /// A `rows x columns` matrix filled with `value` (zero by default).
    public init(rows: Int, columns: Int, repeating value: Element = .zero) {
        precondition(rows >= 0 && columns >= 0, "matrix dimensions must be nonnegative")
        if value == .zero {
            self.init(rows: rows, columns: columns) { RawStorage.zero($0, count: rows * columns * Element.width) }
        } else {
            self.init(rows: rows, columns: columns, elements: Array(repeating: value, count: rows * columns))
        }
    }

    /// A `rows x columns` matrix from column-major `elements`.
    public init(rows: Int, columns: Int, elements: [Element]) {
        precondition(rows >= 0 && columns >= 0, "matrix dimensions must be nonnegative")
        precondition(elements.count == rows * columns,
                     "expected \(rows * columns) elements, got \(elements.count)")
        rowCount = rows
        columnCount = columns
        self.elements = elements
    }

    /// A `rows x columns` matrix with entry `(i, j)` equal to `entry(i, j)`.
    @inlinable
    public init(rows: Int, columns: Int, _ entry: (_ i: Int, _ j: Int) throws -> Element) rethrows {
        precondition(rows >= 0 && columns >= 0, "matrix dimensions must be nonnegative")
        var elements: [Element] = []
        elements.reserveCapacity(rows * columns)
        for j in 0..<columns {
            for i in 0..<rows {
                elements.append(try entry(i, j))
            }
        }
        self.init(rows: rows, columns: columns, elements: elements)
    }

    /// A matrix whose storage `fill` initializes completely, as raw bytes
    /// (`rows * columns * width` Doubles).
    init(rows: Int, columns: Int, rawFill fill: (UnsafeMutableRawPointer) -> Void) {
        let count = rows * columns
        rowCount = rows
        columnCount = columns
        elements = [Element](unsafeUninitializedCapacity: count) { buffer, initialized in
            if count > 0 { fill(UnsafeMutableRawPointer(buffer.baseAddress!)) }
            initialized = count
        }
    }

    /// The `n x n` identity matrix.
    public static func identity(_ n: Int) -> Matrix {
        var eye = Matrix(rows: n, columns: n)
        for i in 0..<n { eye[i, i] = 1 }
        return eye
    }

    /// Entry `(i, j)`, 0-based.
    public subscript(i: Int, j: Int) -> Element {
        get {
            precondition(i >= 0 && i < rowCount && j >= 0 && j < columnCount, "index out of range")
            return elements[i + j * rowCount]
        }
        set {
            precondition(i >= 0 && i < rowCount && j >= 0 && j < columnCount, "index out of range")
            elements[i + j * rowCount] = newValue
        }
    }

    /// Raw read access to the storage; the matrix must be nonempty.
    func withRawElements<R>(_ body: (UnsafeRawPointer) -> R) -> R {
        elements.withUnsafeBytes { body($0.baseAddress!) }
    }

    /// The conjugate transpose `Aᴴ` (the transpose for real matrices).
    public var adjoint: Matrix {
        let m = rowCount, n = columnCount, w = Element.width
        guard m > 0, n > 0 else { return Matrix(rows: n, columns: m) }
        return Matrix(rows: n, columns: m) { dst in
            withRawElements { src in RawStorage.adjoint(src, m, n, into: dst, width: w) }
        }
    }

    /// The main diagonal.
    public var diagonal: [Element] {
        (0..<min(rowCount, columnCount)).map { self[$0, $0] }
    }

    /// The Frobenius norm `sqrt(sum |a_ij|^2)`, scaled to avoid overflow.
    public var frobeniusNorm: Double {
        let count = elements.count * Element.width
        guard count > 0 else { return 0 }
        return withRawElements { RawStorage.norm($0, count: count) }
    }

    /// The contiguous block of columns `range`.
    public func columns(_ range: Range<Int>) -> Matrix {
        precondition(range.lowerBound >= 0 && range.upperBound <= columnCount, "column range out of bounds")
        if range == 0..<columnCount { return self }
        let bytes = rowCount * MemoryLayout<Element>.stride
        guard bytes > 0, !range.isEmpty else { return Matrix(rows: rowCount, columns: range.count) }
        return Matrix(rows: rowCount, columns: range.count) { dst in
            withRawElements { src in
                dst.copyMemory(from: src + range.lowerBound * bytes, byteCount: range.count * bytes)
            }
        }
    }

    /// The columns `indices`, in that order (e.g. `A.columns(piv)` for `A[:, piv]`).
    public func columns<C: Collection<Int>>(_ indices: C) -> Matrix {
        let bytes = rowCount * MemoryLayout<Element>.stride
        for j in indices { precondition(j >= 0 && j < columnCount, "column index out of bounds") }
        guard bytes > 0, !indices.isEmpty else { return Matrix(rows: rowCount, columns: indices.count) }
        return Matrix(rows: rowCount, columns: indices.count) { dst in
            withRawElements { src in
                for (position, j) in indices.enumerated() {
                    (dst + position * bytes).copyMemory(from: src + j * bytes, byteCount: bytes)
                }
            }
        }
    }

    /// The contiguous block of rows `range`.
    public func rows(_ range: Range<Int>) -> Matrix {
        precondition(range.lowerBound >= 0 && range.upperBound <= rowCount, "row range out of bounds")
        return submatrix(rows: range, columns: 0..<columnCount)
    }

    /// The rows `indices`, in that order.
    public func rows<C: Collection<Int>>(_ indices: C) -> Matrix {
        let rowIndex = Array(indices)
        for i in rowIndex { precondition(i >= 0 && i < rowCount, "row index out of bounds") }
        let stride = MemoryLayout<Element>.stride, m = rowCount, k = rowIndex.count
        guard k > 0, columnCount > 0 else { return Matrix(rows: k, columns: columnCount) }
        return Matrix(rows: k, columns: columnCount) { dst in
            withRawElements { src in
                for j in 0..<columnCount {
                    for (position, i) in rowIndex.enumerated() {
                        (dst + (position + j * k) * stride).copyMemory(from: src + (i + j * m) * stride,
                                                                        byteCount: stride)
                    }
                }
            }
        }
    }

    /// The block `A[rows, columns]`, copied column by column.
    func submatrix(rows: Range<Int>, columns: Range<Int>) -> Matrix {
        precondition(rows.lowerBound >= 0 && rows.upperBound <= rowCount
                     && columns.lowerBound >= 0 && columns.upperBound <= columnCount, "block out of bounds")
        if rows == 0..<rowCount { return self.columns(columns) }
        let stride = MemoryLayout<Element>.stride, m = rowCount
        guard !rows.isEmpty, !columns.isEmpty else { return Matrix(rows: rows.count, columns: columns.count) }
        return Matrix(rows: rows.count, columns: columns.count) { dst in
            withRawElements { src in
                for (position, j) in columns.enumerated() {
                    (dst + position * rows.count * stride).copyMemory(
                        from: src + (rows.lowerBound + j * m) * stride, byteCount: rows.count * stride)
                }
            }
        }
    }

    /// The leading `min(m,n) x n` upper trapezoid, with zeros below the diagonal.
    func upperTrapezoid() -> Matrix {
        let m = rowCount, n = columnCount, k = min(m, n), stride = MemoryLayout<Element>.stride
        guard k > 0 else { return Matrix(rows: k, columns: n) }
        return Matrix(rows: k, columns: n) { dst in
            withRawElements { src in
                for j in 0..<n {
                    let kept = min(j + 1, k)
                    let column = dst + j * k * stride
                    column.copyMemory(from: src + j * m * stride, byteCount: kept * stride)
                    RawStorage.zero(column + kept * stride, count: (k - kept) * Element.width)
                }
            }
        }
    }

    /// `A * diag(d)`: column `j` multiplied by the real number `d[j]`.
    public func scalingColumns(by d: [Double]) -> Matrix {
        precondition(d.count == columnCount, "expected \(columnCount) scale factors")
        let perColumn = rowCount * Element.width
        return mapRaw { src, dst in
            for j in 0..<columnCount {
                RawStorage.scale(src + j * perColumn * 8, into: dst + j * perColumn * 8, count: perColumn, by: d[j])
            }
        }
    }

    /// `diag(d) * A`: row `i` multiplied by the real number `d[i]`.
    public func scalingRows(by d: [Double]) -> Matrix {
        precondition(d.count == rowCount, "expected \(rowCount) scale factors")
        let w = Element.width
        return mapRaw { src, dst in
            for j in 0..<columnCount {
                for i in 0..<rowCount {
                    let offset = (i + j * rowCount) * w * 8
                    RawStorage.scale(src + offset, into: dst + offset, count: w, by: d[i])
                }
            }
        }
    }

    /// A same-shape matrix computed from this one's raw storage.
    private func mapRaw(_ body: (UnsafeRawPointer, UnsafeMutableRawPointer) -> Void) -> Matrix {
        guard !elements.isEmpty else { return self }
        return Matrix(rows: rowCount, columns: columnCount) { dst in withRawElements { body($0, dst) } }
    }

    /// Matrix product through BLAS `gemm`.
    public static func * (lhs: Matrix, rhs: Matrix) -> Matrix {
        BLAS.gemm(lhs, rhs)
    }

    /// `lhsᴴ * rhs` through BLAS `gemm`, without forming the adjoint.
    public static func adjointProduct(_ lhs: Matrix, _ rhs: Matrix) -> Matrix {
        BLAS.gemm(lhs, rhs, adjointA: true)
    }

    public static func + (lhs: Matrix, rhs: Matrix) -> Matrix {
        combine(lhs, rhs, sign: 1)
    }

    public static func - (lhs: Matrix, rhs: Matrix) -> Matrix {
        combine(lhs, rhs, sign: -1)
    }

    private static func combine(_ lhs: Matrix, _ rhs: Matrix, sign: Double) -> Matrix {
        precondition(lhs.shape == rhs.shape, "shape mismatch \(lhs.shape) vs \(rhs.shape)")
        let count = lhs.elements.count * Element.width
        return lhs.mapRaw { a, dst in
            rhs.withRawElements { b in
                for i in 0..<count {
                    let x = a.load(fromByteOffset: 8 * i, as: Double.self)
                    let y = b.load(fromByteOffset: 8 * i, as: Double.self)
                    dst.storeBytes(of: sign > 0 ? x + y : x - y, toByteOffset: 8 * i, as: Double.self)
                }
            }
        }
    }
}

/// Loops over raw storage viewed as Doubles (byte offsets, so no memory
/// rebinding is involved; loads and stores compile to plain accesses).
enum RawStorage {
    static func zero(_ p: UnsafeMutableRawPointer, count: Int) {
        for i in 0..<count { p.storeBytes(of: 0.0, toByteOffset: 8 * i, as: Double.self) }
    }

    static func scale(_ src: UnsafeRawPointer, into dst: UnsafeMutableRawPointer, count: Int, by factor: Double) {
        for i in 0..<count {
            dst.storeBytes(of: src.load(fromByteOffset: 8 * i, as: Double.self) * factor,
                           toByteOffset: 8 * i, as: Double.self)
        }
    }

    /// sqrt(sum x_i^2) over `count` Doubles, scaled by the largest |x_i|.
    static func norm(_ p: UnsafeRawPointer, count: Int) -> Double {
        var scale = 0.0
        for i in 0..<count { scale = max(scale, abs(p.load(fromByteOffset: 8 * i, as: Double.self))) }
        guard scale > 0, scale.isFinite else { return scale }
        var sum = 0.0
        for i in 0..<count {
            let r = p.load(fromByteOffset: 8 * i, as: Double.self) / scale
            sum += r * r
        }
        return scale * sum.squareRoot()
    }

    /// Writes the conjugate transpose of the `m x n` matrix at `src` (elements
    /// of `width` Doubles) to `dst` (`n x m`), in cache-sized tiles.
    static func adjoint(_ src: UnsafeRawPointer, _ m: Int, _ n: Int, into dst: UnsafeMutableRawPointer, width: Int) {
        let tile = 32, bytes = 8 * width
        for j0 in stride(from: 0, to: n, by: tile) {
            for i0 in stride(from: 0, to: m, by: tile) {
                for j in j0..<min(j0 + tile, n) {
                    for i in i0..<min(i0 + tile, m) {
                        let from = (i + j * m) * bytes, to = (j + i * n) * bytes
                        dst.storeBytes(of: src.load(fromByteOffset: from, as: Double.self), toByteOffset: to, as: Double.self)
                        if width == 2 {
                            dst.storeBytes(of: -src.load(fromByteOffset: from + 8, as: Double.self),
                                           toByteOffset: to + 8, as: Double.self)
                        }
                    }
                }
            }
        }
    }
}

extension Matrix: CustomStringConvertible {
    public var description: String {
        var lines = ["\(rowCount)x\(columnCount) Matrix<\(Element.self)>"]
        let shownRows = min(rowCount, 8), shownCols = min(columnCount, 6)
        for i in 0..<shownRows {
            var row = (0..<shownCols).map { "\(self[i, $0])" }.joined(separator: "  ")
            if shownCols < columnCount { row += "  ..." }
            lines.append("  " + row)
        }
        if shownRows < rowCount { lines.append("  ...") }
        return lines.joined(separator: "\n")
    }
}

// Deterministic dense factorizations (LAPACK), handy for checking sketches.
extension Matrix {
    /// Economic QR with column pivoting (`geqp3`): `A[:, p] = Q R`, with
    /// `Q` `m x min(m,n)` orthonormal, `R` `min(m,n) x n` upper triangular
    /// and `p` 0-based.
    public func pivotedQR() -> (q: Matrix, r: Matrix, p: [Int]) {
        LAPACK.pivotedQR(self)
    }

    /// Economic SVD (`gesdd`, `gesvd` fallback): `A = U diag(s) Vh`, with
    /// `s` descending and `Vh` the conjugate transpose of `V`.
    public func svd() -> (u: Matrix, s: [Double], vh: Matrix) {
        LAPACK.svd(self)
    }
}
