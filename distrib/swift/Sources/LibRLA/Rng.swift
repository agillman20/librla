// Rng.swift - LAPACK larnv random number state shared by all librla ports
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import CLapack

/// Random number state for the test matrices: the LAPACK `iseed[4]` array.
///
/// Test matrices are drawn with `dlarnv`/`zlarnv` (`idist = 2`, uniform
/// on (-1, 1); complex entries have independent real and imaginary parts),
/// one call per matrix, column-major. Every librla port (C++, Rust, Swift,
/// Fortran) uses the same generator, so the same seed gives the same
/// sketch in each of them. Each draw advances the state, so pass the same
/// `Rng` (as `inout`) to successive calls to get fresh test matrices.
public struct Rng: Hashable, Sendable, CustomStringConvertible {
    private var state: [Int32]

    /// The current LAPACK seed: four integers in 0...4095, the last one odd.
    public var iseed: [Int] { state.map { Int($0) } }

    /// The default deterministic seed `[0, 0, 0, 1]`.
    public init() {
        state = [0, 0, 0, 1]
    }

    /// A seed given as the four LAPACK `iseed` entries.
    ///
    /// - Throws: `LibRLAError.invalidSeed` unless there are four entries in
    ///   0...4095 and the last one is odd.
    public init(iseed: [Int]) throws {
        guard iseed.count == 4, iseed.allSatisfy({ (0...4095).contains($0) }), iseed[3] % 2 == 1 else {
            throw LibRLAError.invalidSeed(iseed)
        }
        state = iseed.map { Int32($0) }
    }

    /// A seed derived from a 64-bit integer:
    /// `iseed = [(s >> 36) & 4095, (s >> 24) & 4095, (s >> 12) & 4095, (s & 4095) | 1]`.
    public init(seed s: UInt64) {
        state = [Int32((s >> 36) & 4095), Int32((s >> 24) & 4095),
                 Int32((s >> 12) & 4095), Int32((s & 4095) | 1)]
    }

    public var description: String { "Rng(iseed: \(iseed))" }

    /// Fills `count` elements of `T` at `p` with uniform(-1, 1) values from
    /// a single `larnv` call.
    mutating func fillUniform<T: Scalar>(_: T.Type, _ p: UnsafeMutableRawPointer, count: Int) {
        var idist: Int32 = 2
        var n = lapackInt(count)
        switch T.kind {
        case .real: rla_dlarnv(&idist, &state, &n, p)
        case .complex: rla_zlarnv(&idist, &state, &n, p)
        }
    }
}

extension Matrix {
    /// A `rows x columns` matrix of uniform(-1, 1) entries drawn from `rng`
    /// with one LAPACK `larnv` call (column-major fill).
    public static func random(rows: Int, columns: Int, using rng: inout Rng) -> Matrix {
        precondition(rows >= 0 && columns >= 0, "matrix dimensions must be nonnegative")
        return Matrix(rows: rows, columns: columns) { rng.fillUniform(Element.self, $0, count: rows * columns) }
    }
}
