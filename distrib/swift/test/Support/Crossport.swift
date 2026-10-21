// Crossport.swift - the cross-port check lines shared by every librla port
//
// Every port uses the same LAPACK larnv generator, so the same seed gives
// the same sketch; these lines are compared verbatim between ports.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import ComplexModule
import Foundation
import LibRLA

private func fmt(_ values: [Double]) -> String {
    values.map { String(format: "%.15e", $0) }.joined(separator: " ")
}

/// The five CROSSPORT lines.
public func crossportLines() -> [String] {
    let seed = [1, 2, 3, 5]
    var lines: [String] = []

    var rng = try! Rng(iseed: seed)
    let ar: Matrix<Double> = crossportMatrix(50, 30)
    let orthReal = orthSketch(ar, rtol: 5, powerIter: 0, extraSamples: 3, rng: &rng)
    lines.append("CROSSPORT orth_real \(fmt(orthReal.diagR))")
    let seedAfter = rng.iseed

    var rngc = try! Rng(iseed: seed)
    let ac: Matrix<Complex<Double>> = crossportMatrix(50, 30)
    let orthCplx = orthSketch(ac, rtol: 5, powerIter: 0, extraSamples: 3, rng: &rngc)
    lines.append("CROSSPORT orth_cplx \(fmt(orthCplx.diagR))")
    lines.append("CROSSPORT seed_after \(seedAfter.map(String.init).joined(separator: " "))")

    var rngId = try! Rng(iseed: seed)
    let (k, piv, _) = idSketch(crossportMatrix(300, 200) as Matrix<Double>, rtol: 1e-10, rng: &rngId)
    lines.append("CROSSPORT id_real \(k) \(piv.prefix(10).map(String.init).joined(separator: " "))")

    var rngSvd = try! Rng(iseed: seed)
    let (_, s, _) = svdSketch(crossportMatrix(120, 80) as Matrix<Complex<Double>>, rtol: 10, powerIter: 1, rng: &rngSvd)
    lines.append("CROSSPORT svd_cplx \(s.count) \(fmt(Array(s.prefix(5))))")
    return lines
}
