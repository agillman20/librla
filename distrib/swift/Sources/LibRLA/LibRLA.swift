// LibRLA - randomized algorithms for low-rank matrix approximations
//
// Randomized:
//   orthSketch - approximate orthonormal basis for the column space
//   qrSketch   - truncated QR factorization with column pivoting
//   svdSketch  - truncated singular value decomposition
//   idSketch   - interpolative decomposition
// Deterministic:
//   idQRPiv    - interpolative decomposition via QR with column pivoting
//
// Usage:
//
//     let (Q, flag, diagR) = orthSketch(A, rtol: rtol)
//     let (Q, R, p) = qrSketch(A, rtol: rtol)
//     let (U, s, Vh) = svdSketch(A, rtol: rtol)
//     let (k, piv, T) = idSketch(A, rtol: rtol)
//     // tolerance mode: rtol < 1, rank mode: rtol >= 1
//
// Experimental Swift port of librla, tracking librla 1.2.1 semantics.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

/// The version of this Swift port (the only place it is defined).
public let libRLAVersion = "0.1.0"

/// How `idSketch` and `idQRPiv` compute the interpolation matrix `T`.
public enum IDMethod: String, CaseIterable, Sendable {
    /// Triangular solve `R11 \ R12` (fastest; falls back to `.svd` when
    /// `R11` is numerically singular).
    case fast
    /// SVD-based pseudoinverse of `R11`.
    case svd
    /// Least squares on the columns of `A` itself (most accurate, slowest;
    /// materializes a matrix-free operator).
    case lstsq

    /// Parses `"fast"`, `"svd"` or `"lstsq"`; throws
    /// `LibRLAError.invalidMethod` for anything else.
    public init(parsing name: String) throws {
        guard let method = IDMethod(rawValue: name) else {
            throw LibRLAError.invalidMethod(name)
        }
        self = method
    }
}

/// Errors thrown by LibRLA.
public enum LibRLAError: Error, Equatable, CustomStringConvertible {
    /// An ID method name other than fast, svd or lstsq.
    case invalidMethod(String)
    /// A LAPACK seed that is not four integers in 0...4095 with the last one odd.
    case invalidSeed([Int])

    public var description: String {
        switch self {
        case .invalidMethod(let name):
            return "method must be one of fast, svd, lstsq, got '\(name)'"
        case .invalidSeed(let iseed):
            return "iseed must be four integers in 0...4095 with the last one odd, got \(iseed)"
        }
    }
}
