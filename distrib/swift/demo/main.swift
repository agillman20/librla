// librla-demo - dense and matrix-free low-rank factorizations
//
//     swift run librla-demo
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import Foundation
import LibRLA

func sci(_ x: Double) -> String { String(format: "%.2e", x) }

print("LibRLA \(libRLAVersion) (Swift)\n")

// Dense: a 1000 x 500 Hilbert-like matrix (rapidly decaying singular values).
let A = Matrix<Double>(rows: 1000, columns: 500) { i, j in 1 / Double(i + j + 1) }
let normA = A.frobeniusNorm
var rng = Rng(seed: 2026)

let (Q, flag, _) = orthSketch(A, rtol: 1e-8, rng: &rng)
let spanErr = (A - Q * Matrix.adjointProduct(Q, A)).frobeniusNorm / normA
print("orthSketch  rtol 1e-8 : \(Q.columnCount) columns, flag \(flag), ||A - QQ'A||/||A|| = \(sci(spanErr))")

let (U, s, Vh) = svdSketch(A, rtol: 1e-10, rng: &rng)
let svdErr = (A - U.scalingColumns(by: s) * Vh).frobeniusNorm / normA
print("svdSketch   rtol 1e-10: rank \(s.count), s[0] = \(String(format: "%.6f", s[0])), error \(sci(svdErr))")

let (Qr, R, p) = qrSketch(A, rtol: 12, rng: &rng)
let qrErr = (A.columns(p) - Qr * R).frobeniusNorm / normA
print("qrSketch    rank 12   : error \(sci(qrErr))")

let (k, piv, T) = idSketch(A, rtol: 1e-8, method: .lstsq, rng: &rng)
let idErr = (A.columns(piv[k...]) - A.columns(piv[..<k]) * T).frobeniusNorm / normA
print("idSketch    rtol 1e-8 : \(k) skeleton columns \(Array(piv.prefix(6)))..., error \(sci(idErr))")

// Matrix-free: A = B C applied as B (C x), never formed.
let B = Matrix<Double>.random(rows: 3000, columns: 8, using: &rng)
let C = Matrix<Double>.random(rows: 8, columns: 2000, using: &rng)
let product = FunctionOperator<Double>(
    rows: 3000, columns: 2000,
    apply: { X in B * (C * X) },
    applyAdjoint: { W in Matrix.adjointProduct(C, Matrix.adjointProduct(B, W)) })
let (Uf, sf, Vhf) = svdSketch(product, rtol: 8, rng: &rng)
let explicit = B * C
let freeErr = (explicit - Uf.scalingColumns(by: sf) * Vhf).frobeniusNorm / explicit.frobeniusNorm
print("\nmatrix-free 3000x2000 rank-8 product, svdSketch rank 8: error \(sci(freeErr))")
print("  s = \(sf.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: " ")) ...")
