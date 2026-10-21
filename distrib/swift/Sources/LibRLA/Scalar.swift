// Scalar.swift - element types supported by LibRLA: Double and Complex<Double>
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import ComplexModule
import RealModule

/// A matrix element type: `Double` or `Complex<Double>`.
///
/// The factorizations call LAPACK `d*` routines for `Double` and `z*`
/// routines for `Complex<Double>` (whose memory layout matches Fortran
/// `complex*16`). Other conformances are not supported.
public protocol Scalar: AlgebraicField, Hashable, Sendable {
    /// Embeds a real number.
    init(_ real: Double)
    /// The complex conjugate (the value itself for real numbers).
    var conjugate: Self { get }
    /// The absolute value `|x|`, computed with `hypot` for complex values.
    var modulus: Double { get }
    /// `x * factor` for a real factor.
    func scaled(by factor: Double) -> Self
}

extension Double: Scalar {
    @inlinable public var conjugate: Double { self }
    @inlinable public var modulus: Double { Swift.abs(self) }
    @inlinable public func scaled(by factor: Double) -> Double { self * factor }
}

extension Complex: Scalar where RealType == Double {
    // `conjugate` and `init(_ real:)` come from swift-numerics.
    // swift-numerics' `length` is sqrt(x*x + y*y), which can differ from
    // hypot in the last bit, and its `magnitude` is the infinity norm; use
    // hypot like NumPy and the other ports so diagR agrees bit for bit.
    @inlinable public var modulus: Double { Double.hypot(real, imaginary) }
    @inlinable public func scaled(by factor: Double) -> Complex<Double> {
        Complex(real * factor, imaginary * factor)
    }
}

/// Which LAPACK precision prefix a scalar type maps to.
enum ScalarKind {
    case real, complex
}

extension Scalar {
    static var kind: ScalarKind {
        if Self.self == Double.self { return .real }
        if Self.self == Complex<Double>.self { return .complex }
        fatalError("LibRLA supports Double and Complex<Double> only, not \(Self.self)")
    }
}
