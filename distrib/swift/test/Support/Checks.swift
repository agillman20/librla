// Checks.swift - correctness checks for all five functions, real and complex
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import ComplexModule
import Foundation
import LibRLA

// MARK: - RNG

public func checkRng(_ report: CheckReport) {
    let expected = [3.7327920546847082E-01, 8.2093410748050388E-01, 5.5866811353917711E-01,
                    6.4291221902741569E-01, 6.8760847451716955E-01, 1.6449965895444763E-01]
    let expectedSeed = [2384, 3667, 635, 1229]

    var rng = try! Rng(iseed: [1, 2, 3, 5])
    let x = Matrix<Double>.random(rows: 6, columns: 1, using: &rng)
    let dx = maxAbsDiff(x.elements, expected)
    report.check("rng: dlarnv(2, {1,2,3,5}, 6) reference values", dx <= 1e-15, "max diff \(sci(dx))")
    report.check("rng: iseed after dlarnv is {2384,3667,635,1229}", rng.iseed == expectedSeed, "\(rng.iseed)")

    var rngc = try! Rng(iseed: [1, 2, 3, 5])
    let z = Matrix<Complex<Double>>.random(rows: 3, columns: 1, using: &rngc)
    let dz = maxAbsDiff(z.elements.flatMap { [$0.real, $0.imaginary] }, expected)
    report.check("rng: zlarnv(2, {1,2,3,5}, 3) reference values", dz <= 1e-15, "max diff \(sci(dz))")
    report.check("rng: iseed after zlarnv is {2384,3667,635,1229}", rngc.iseed == expectedSeed, "\(rngc.iseed)")

    report.check("rng: default iseed is {0,0,0,1}", Rng().iseed == [0, 0, 0, 1])
    let fromSeed = Rng(seed: 0x0123_4567_89AB_CDEF)
    report.check("rng: 64-bit seed maps to {1110,1929,2748,3567}",
                 fromSeed.iseed == [1110, 1929, 2748, 3567], "\(fromSeed.iseed)")
    report.check("rng: 64-bit seed forces iseed[3] odd", Rng(seed: 4096).iseed == [0, 0, 1, 1])
    let invalid = [[0, 0, 0, 2], [4096, 0, 0, 1], [-1, 0, 0, 1], [1, 2, 3]]
    let rejected = invalid.allSatisfy { seed in
        do { _ = try Rng(iseed: seed); return false } catch { return error as? LibRLAError == .invalidSeed(seed) }
    }
    report.check("rng: invalid iseed (even, out of range, wrong length) throws", rejected)
}

// MARK: - orthSketch

public func checkOrth<T: TestScalar>(_ report: CheckReport, _: T.Type) {
    let tag = "orth[\(T.tag)]"
    let (a, s) = fastDecayMatrix(T.self)
    var rng = Rng(seed: 11)

    for rtol in [1e-6, 1e-10] {
        let (q, flag, diagR) = orthSketch(a, rtol: rtol, rng: &rng)
        let err = spanError(a, q), orth = orthonormalityError(q)
        report.check("\(tag): tolerance \(sci(rtol)) accepted (flag 0, \(q.columnCount) columns)",
                     flag == 0 && q.columnCount == diagR.count && q.columnCount > 0)
        // The truncation error at the rtol-rank is about rtol by construction.
        report.check("\(tag): tolerance \(sci(rtol)) span error <= 3 rtol", err <= 3 * rtol, "span err \(sci(err))")
        report.check("\(tag): tolerance \(sci(rtol)) orthonormality", orth <= 1e-13, sci(orth))
        report.check("\(tag): tolerance \(sci(rtol)) diagR nonincreasing",
                     zip(diagR, diagR.dropFirst()).allSatisfy { $0.0 >= $0.1 * (1 - 1e-12) })
    }
    // rtol 1e-10 needs rank ~40 plus 12 buffer columns: 42 is too small, 168 is accepted.
    let (qGrow, _, _) = orthSketch(a, rtol: 1e-10, rng: &rng)
    report.check("\(tag): tolerance growth 42 -> 168 columns", qGrow.columnCount == 168, "\(qGrow.columnCount)")

    let (q20, flag20, diag20) = orthSketch(a, rtol: 20, rng: &rng)
    let err20 = spanError(a, q20)
    report.check("\(tag): rank 20 returns 20 + 12 columns", q20.columnCount == 32 && flag20 == 0 && diag20.count == 32)
    report.check("\(tag): rank 20 span error <= optimal rank-20 error", err20 <= optimalTail(s, 20),
                 "\(sci(err20)) vs \(sci(optimalTail(s, 20)))")
    let small: Matrix<T> = randomMatrix(30, 50, seed: 3)
    let (qSmall, flagSmall, _) = orthSketch(small, rtol: 25, rng: &rng)
    report.check("\(tag): rank 25 on 30x50 returns min(m, k + extra) = 30 columns",
                 qSmall.columnCount == 30 && flagSmall == 0 && orthonormalityError(qSmall) <= 1e-13)

    let (qLegacy, flagLegacy, diagLegacy) = orthSketch(a, rtol: 1e-6, extraSamples: 0, rng: &rng)
    report.check("\(tag): extraSamples 0 accepts the first 42-column sketch (legacy check)",
                 flagLegacy == 0 && qLegacy.columnCount == 42 && diagLegacy[41] <= 1e-6 * diagLegacy[0])
    report.check("\(tag): extraSamples 0 span error <= rtol", spanError(a, qLegacy) <= 1e-6,
                 sci(spanError(a, qLegacy)))
    let (qLegacyRank, _, _) = orthSketch(a, rtol: 20, extraSamples: 0, rng: &rng)
    report.check("\(tag): extraSamples 0 rank 20 returns 20 columns", qLegacyRank.columnCount == 20)

    let (qEps, flagEps, diagEps) = orthSketch(a, rtol: 1e-17, rng: &rng)
    report.check("\(tag): rtol < eps gives flag 1 and empty Q (m x 0)",
                 flagEps == 1 && qEps.shape == (400, 0) && diagEps.isEmpty)
    let (qBlock, flagBlock, _) = orthSketch(a, rtol: 1e-6, blockSize: 300, rng: &rng)
    report.check("\(tag): blockSize >= min(m,n) gives flag 1", flagBlock == 1 && qBlock.shape == (400, 0))
    let full: Matrix<T> = randomMatrix(100, 80, seed: 5)
    let before = rng.iseed
    let (qFull, flagFull, diagFull) = orthSketch(full, rtol: 1e-8, rng: &rng)
    report.check("\(tag): sketch growing to min(m,n) gives flag 1",
                 flagFull == 1 && qFull.shape == (100, 0) && diagFull.isEmpty && rng.iseed != before)
}

// MARK: - qrSketch

public func checkQR<T: TestScalar>(_ report: CheckReport, _: T.Type) {
    let (tall, _) = fastDecayMatrix(T.self, seed: 2)
    var rng = Rng(seed: 12)

    for (shape, a) in [("tall", tall), ("wide", tall.adjoint)] {
        let tag = "qr[\(T.tag)] \(shape) \(a.rowCount)x\(a.columnCount)"
        let n = a.columnCount
        // Deterministic pivoted QR as the reference for rank mode.
        let (qd, rd, pd) = a.pivotedQR()

        let rtol = 1e-8
        let (q, r, p) = qrSketch(a, rtol: rtol, rng: &rng)
        let err = relError(a.columns(p), q * r)
        report.check("\(tag): tolerance \(sci(rtol)) A[:,p] ~ QR within 3 rtol", err <= 3 * rtol, "err \(sci(err)), k \(q.columnCount)")
        report.check("\(tag): tolerance shapes, permutation, R upper triangular",
                     q.columnCount == r.rowCount && r.columnCount == n && isPermutation(p, n) && isUpperTriangular(r))
        report.check("\(tag): tolerance Q orthonormal", orthonormalityError(q) <= 1e-13, sci(orthonormalityError(q)))

        let k = 20
        let (qk, rk, pk) = qrSketch(a, rtol: Double(k), rng: &rng)
        let errK = relError(a.columns(pk), qk * rk)
        let errRef = relError(a.columns(pd), qd.columns(0..<k) * rd.rows(0..<k))
        report.check("\(tag): rank 20 shapes", qk.shape == (a.rowCount, k) && rk.shape == (k, n)
                     && isPermutation(pk, n) && isUpperTriangular(rk))
        report.check("\(tag): rank 20 error <= 4x deterministic pivoted QR", errK <= 4 * errRef,
                     "\(sci(errK)) vs \(sci(errRef))")
    }

    // Effectively full rank at rtol: deterministic fallback, independent of the seed.
    for (shape, a) in [("tall", randomMatrix(60, 40, seed: 7) as Matrix<T>), ("wide", randomMatrix(40, 60, seed: 7))] {
        let tag = "qr[\(T.tag)] fallback \(shape)"
        var rng1 = Rng(seed: 1), rng2 = Rng(seed: 2)
        let first = qrSketch(a, rtol: 1e-8, rng: &rng1)
        let second = qrSketch(a, rtol: 1e-8, rng: &rng2)
        let reference = a.pivotedQR()
        let k = min(a.rowCount, a.columnCount)
        report.check("\(tag): equals the deterministic pivoted QR",
                     first.q == reference.q.columns(0..<k) && first.r == reference.r.rows(0..<k) && first.p == reference.p)
        report.check("\(tag): independent of the seed",
                     first.q == second.q && first.r == second.r && first.p == second.p)
    }
}

// MARK: - svdSketch

public func checkSVD<T: TestScalar>(_ report: CheckReport, _: T.Type) {
    let (tall, _) = fastDecayMatrix(T.self, seed: 3)
    var rng = Rng(seed: 13)

    for (shape, a) in [("tall", tall), ("wide", tall.adjoint)] {
        let tag = "svd[\(T.tag)] \(shape) \(a.rowCount)x\(a.columnCount)"
        let sRef = a.svd().s
        let s0 = sRef[0]

        let rtol = 1e-10
        let (u, s, vh) = svdSketch(a, rtol: rtol, rng: &rng)
        let kRef = sRef.count { $0 >= rtol * s0 }
        let err = relError(a, svdReconstruction(u, s, vh))
        let ds = maxAbsDiff(s, Array(sRef.prefix(s.count))) / s0
        report.check("\(tag): tolerance \(sci(rtol)) reconstruction within 3 rtol", err <= 3 * rtol, "err \(sci(err))")
        report.check("\(tag): tolerance rank matches full SVD", abs(s.count - kRef) <= 1, "k \(s.count) vs \(kRef)")
        report.check("\(tag): tolerance singular values vs full SVD", ds <= 1e-12, "max rel diff \(sci(ds))")
        report.check("\(tag): tolerance U, Vh orthonormal",
                     orthonormalityError(u) <= 1e-13 && rowOrthonormalityError(vh) <= 1e-13)

        let k = 20
        let (uk, sk, vhk) = svdSketch(a, rtol: Double(k), rng: &rng)
        let errK = relError(a, svdReconstruction(uk, sk, vhk))
        let opt = optimalTail(sRef, k)
        let dsk = maxAbsDiff(sk, Array(sRef.prefix(k))) / s0
        report.check("\(tag): rank 20 shapes", uk.shape == (a.rowCount, k) && sk.count == k && vhk.shape == (k, a.columnCount))
        report.check("\(tag): rank 20 error <= 4x optimal", errK <= 4 * opt, "\(sci(errK)) vs \(sci(opt))")
        // s_20 ~ 1e-5 s_0, so this still checks the trailing values to ~1e-4 relative.
        report.check("\(tag): rank 20 singular values vs full SVD", dsk <= 1e-9, "max rel diff \(sci(dsk))")
        report.check("\(tag): rank 20 U, Vh orthonormal",
                     orthonormalityError(uk) <= 1e-13 && rowOrthonormalityError(vhk) <= 1e-13)
    }

    for (shape, a) in [("tall", randomMatrix(60, 40, seed: 8) as Matrix<T>), ("wide", randomMatrix(40, 60, seed: 8))] {
        let tag = "svd[\(T.tag)] fallback \(shape)"
        var rng1 = Rng(seed: 1), rng2 = Rng(seed: 2)
        let first = svdSketch(a, rtol: 1e-8, rng: &rng1)
        let second = svdSketch(a, rtol: 1e-8, rng: &rng2)
        let sRef = a.svd().s
        let err = relError(a, svdReconstruction(first.u, first.s, first.vh))
        report.check("\(tag): full deterministic SVD", first.s.count == 40 && err <= 1e-13
                     && maxAbsDiff(first.s, sRef) <= 1e-13 * sRef[0], "err \(sci(err))")
        report.check("\(tag): independent of the seed", first.u == second.u && first.s == second.s && first.vh == second.vh)
    }
}

// MARK: - idSketch and idQRPiv

public func checkID<T: TestScalar>(_ report: CheckReport, _: T.Type) {
    let (a, s) = spectrumMatrix(300, 200, seed: 4) { pow(10.0, -Double($0) / 4) } as (Matrix<T>, [Double])
    let n = a.columnCount
    var rng = Rng(seed: 14)

    for method in IDMethod.allCases {
        for rtol in [1e-8, 20] {
            let mode = rtol < 1 ? "tolerance \(sci(rtol))" : "rank 20"
            // Tolerance: error ~ rtol; rank: within 10x of the optimal rank-k error.
            let bound = rtol < 1 ? 10 * rtol : 10 * optimalTail(s, 20)
            let sketch = idSketch(a, rtol: rtol, method: method, rng: &rng)
            let qrpiv = idQRPiv(a, rtol: rtol, method: method)
            for (name, (k, piv, t)) in [("idSketch", sketch), ("idQRPiv", qrpiv)] {
                let tag = "id[\(T.tag)] \(name) \(method) \(mode)"
                let err = idError(a, k: k, piv: piv, t: t)
                report.check("\(tag): A[:,piv[k:]] ~ A[:,piv[:k]] T", err <= bound && isFinite(t),
                             "err \(sci(err)), k \(k)")
                report.check("\(tag): shapes", isPermutation(piv, n) && t.shape == (k, n - k)
                             && (rtol < 1 || k == 20))
            }
        }
    }

    // Requested rank above the true rank: R11 is singular, T must stay finite.
    var lowRng = Rng(seed: 21)
    let b = Matrix<T>.random(rows: 60, columns: 5, using: &lowRng)
    let c = Matrix<T>.random(rows: 5, columns: 40, using: &lowRng)
    let lowRank = b * c
    for method in IDMethod.allCases {
        let sketch = idSketch(lowRank, rtol: 10, method: method, rng: &rng)
        let qrpiv = idQRPiv(lowRank, rtol: 10, method: method)
        for (name, (k, piv, t)) in [("idSketch", sketch), ("idQRPiv", qrpiv)] {
            let err = idError(lowRank, k: k, piv: piv, t: t)
            report.check("id[\(T.tag)] \(name) \(method) rank 10 > true rank 5: T finite, reconstruction holds",
                         k == 10 && isFinite(t) && err <= 1e-12, "err \(sci(err))")
        }
    }

    // Zero matrix: tolerance mode finds rank 0.
    let zero = Matrix<T>(rows: 30, columns: 20)
    for method in IDMethod.allCases {
        let sketch = idSketch(zero, rtol: 1e-6, method: method, rng: &rng)
        let qrpiv = idQRPiv(zero, rtol: 1e-6, method: method)
        report.check("id[\(T.tag)] \(method) zero matrix gives k = 0, T 0x20",
                     sketch.k == 0 && sketch.t.shape == (0, 20) && qrpiv.k == 0 && qrpiv.t.shape == (0, 20)
                     && isPermutation(sketch.piv, 20) && isPermutation(qrpiv.piv, 20))
        let forced = idQRPiv(zero, rtol: 3, method: method)
        report.check("id[\(T.tag)] \(method) zero matrix rank 3 gives T = 0",
                     forced.k == 3 && forced.t.shape == (3, 17) && forced.t.elements.allSatisfy { $0 == .zero })
    }

    // k == n: every column is a skeleton column, T is k x 0.
    let narrow: Matrix<T> = randomMatrix(30, 10, seed: 9)
    for method in IDMethod.allCases {
        let sketch = idSketch(narrow, rtol: 10, method: method, rng: &rng)
        let qrpiv = idQRPiv(narrow, rtol: 12, method: method)
        report.check("id[\(T.tag)] \(method) k == n gives T 10x0",
                     sketch.k == 10 && sketch.t.shape == (10, 0) && qrpiv.k == 10 && qrpiv.t.shape == (10, 0)
                     && isPermutation(sketch.piv, 10) && isPermutation(qrpiv.piv, 10))
    }
}

// MARK: - Matrix-free operators

public func checkOperators<T: TestScalar>(_ report: CheckReport, _: T.Type) {
    let tag = "operator[\(T.tag)]"
    let (m, _) = spectrumMatrix(200, 150, seed: 5) { pow(10.0, -Double($0) / 4) } as (Matrix<T>, [Double])
    let op = hiddenDense(m)

    typealias Orth = (q: Matrix<T>, flag: Int, diagR: [Double])
    typealias QR = (q: Matrix<T>, r: Matrix<T>, p: [Int])
    typealias SVD = (u: Matrix<T>, s: [Double], vh: Matrix<T>)
    typealias ID = (k: Int, piv: [Int], t: Matrix<T>)

    func same<R>(_ name: String, seed: UInt64, _ run: (any LinearOperator<T>, inout Rng) -> R,
                 _ compare: (R, R) -> (Bool, String)) {
        var rngDense = Rng(seed: seed), rngOp = Rng(seed: seed)
        let dense = run(m, &rngDense)
        let free = run(op, &rngOp)
        let (ok, detail) = compare(dense, free)
        report.check("\(tag): \(name) matches the dense path", ok && rngDense == rngOp, detail)
    }

    same("orthSketch rank 20", seed: 1, { (a, rng) -> Orth in orthSketch(a, rtol: 20, rng: &rng) }) { (d: Orth, f: Orth) in
        (d.q == f.q && d.diagR == f.diagR, "")
    }
    same("orthSketch tolerance 1e-8", seed: 2, { (a, rng) -> Orth in orthSketch(a, rtol: 1e-8, rng: &rng) }) { (d: Orth, f: Orth) in
        (d.q == f.q && d.flag == f.flag, "")
    }
    for rtol in [20, 1e-8] {
        same("qrSketch rtol \(sci(rtol))", seed: 3, { (a, rng) -> QR in qrSketch(a, rtol: rtol, rng: &rng) }) { (d: QR, f: QR) in
            let diff = relError(d.q * d.r, f.q * f.r)
            return (d.p == f.p && d.r.shape == f.r.shape && diff <= 1e-13, "diff \(sci(diff))")
        }
        same("svdSketch rtol \(sci(rtol))", seed: 4, { (a, rng) -> SVD in svdSketch(a, rtol: rtol, rng: &rng) }) { (d: SVD, f: SVD) in
            let ds = maxAbsDiff(d.s, f.s) / d.s[0]
            return (d.s.count == f.s.count && ds <= 1e-13, "s diff \(sci(ds))")
        }
        same("idSketch rtol \(sci(rtol))", seed: 5, { (a, rng) -> ID in idSketch(a, rtol: rtol, rng: &rng) }) { (d: ID, f: ID) in
            let diff = relError(d.t, f.t)
            return (d.k == f.k && d.piv == f.piv && diff <= 1e-10, "T diff \(sci(diff))")
        }
    }
    let wideOp = hiddenDense(m.adjoint)
    var rngW1 = Rng(seed: 6), rngW2 = Rng(seed: 6)
    let wideDense = svdSketch(m.adjoint, rtol: 20, rng: &rngW1)
    let wideFree = svdSketch(wideOp, rtol: 20, rng: &rngW2)
    let dWide = relError(svdReconstruction(wideDense.u, wideDense.s, wideDense.vh),
                         svdReconstruction(wideFree.u, wideFree.s, wideFree.vh))
    report.check("\(tag): svdSketch wide (adjoint operator) matches the dense path", dWide <= 1e-13, "diff \(sci(dWide))")

    // Tolerance mode on a full-rank operator: deterministic fallback on the
    // materialized operator, identical to the dense result.
    let full: Matrix<T> = randomMatrix(60, 40, seed: 10)
    let fullOp = hiddenDense(full)
    var r1 = Rng(seed: 7), r2 = Rng(seed: 7)
    let qrD = qrSketch(full, rtol: 1e-8, rng: &r1), qrF = qrSketch(fullOp, rtol: 1e-8, rng: &r2)
    report.check("\(tag): qrSketch fallback materializes the operator exactly",
                 qrD.q == qrF.q && qrD.r == qrF.r && qrD.p == qrF.p)
    let svdD = svdSketch(full, rtol: 1e-8, rng: &r1), svdF = svdSketch(fullOp, rtol: 1e-8, rng: &r2)
    report.check("\(tag): svdSketch fallback materializes the operator exactly",
                 svdD.u == svdF.u && svdD.s == svdF.s && svdD.vh == svdF.vh)
    for method in IDMethod.allCases {
        let idD = idQRPiv(full, rtol: 1e-8, method: method), idF = idQRPiv(fullOp, rtol: 1e-8, method: method)
        let lsD = idSketch(full, rtol: 25, method: method, rng: &r1), lsF = idSketch(fullOp, rtol: 25, method: method, rng: &r2)
        let diff = relError(lsD.t, lsF.t)
        report.check("\(tag): idQRPiv / idSketch \(method) match the dense path",
                     idD.k == idF.k && idD.piv == idF.piv && idD.t == idF.t && lsD.piv == lsF.piv && diff <= 1e-10,
                     "idSketch T diff \(sci(diff))")
    }

    // Genuinely matrix-free low-rank product A = B C, applied as B (C x).
    var rng = Rng(seed: 8)
    for (rows, cols) in [(300, 200), (200, 300)] {
        let r = 15
        let bf = Matrix<T>.random(rows: rows, columns: r, using: &rng)
        let cf = Matrix<T>.random(rows: r, columns: cols, using: &rng)
        let product = FunctionOperator<T>(rows: rows, columns: cols,
                                          apply: { bf * (cf * $0) },
                                          applyAdjoint: { Matrix.adjointProduct(cf, Matrix.adjointProduct(bf, $0)) })
        let explicit = bf * cf
        let shape = "\(rows)x\(cols)"
        let (u, s, vh) = svdSketch(product, rtol: Double(r), rng: &rng)
        let errSVD = relError(explicit, svdReconstruction(u, s, vh))
        let ds = maxAbsDiff(s, Array(explicit.svd().s.prefix(r))) / s[0]
        report.check("\(tag): product B C \(shape) svdSketch rank 15", errSVD <= 1e-12 && ds <= 1e-12,
                     "err \(sci(errSVD)), s diff \(sci(ds))")
        let (q, rr, p) = qrSketch(product, rtol: Double(r), rng: &rng)
        let errQR = relError(explicit.columns(p), q * rr)
        report.check("\(tag): product B C \(shape) qrSketch rank 15", errQR <= 1e-12, "err \(sci(errQR))")
        let (k, piv, t) = idSketch(product, rtol: Double(r), rng: &rng)
        let errID = idError(explicit, k: k, piv: piv, t: t)
        report.check("\(tag): product B C \(shape) idSketch rank 15", k == r && errID <= 1e-12, "err \(sci(errID))")
        let (qo, flag, _) = orthSketch(product, rtol: Double(r), rng: &rng)
        report.check("\(tag): product B C \(shape) orthSketch rank 15",
                     flag == 0 && spanError(explicit, qo) <= 1e-12, "span err \(sci(spanError(explicit, qo)))")
    }
}

// MARK: - Power iteration

public func checkPowerIteration<T: TestScalar>(_ report: CheckReport, _: T.Type) {
    let tag = "power[\(T.tag)]"
    let (a, s) = spectrumMatrix(200, 150, seed: 6) { 1 / Double($0 + 1) } as (Matrix<T>, [Double])
    let k = 10
    var errors: [Double] = []
    for p in 0...2 {
        var rng = Rng(seed: 31)
        let (u, sk, vh) = svdSketch(a, rtol: Double(k), powerIter: p, rng: &rng)
        errors.append(relError(a, svdReconstruction(u, sk, vh)))
    }
    let opt = optimalTail(s, k)
    report.check("\(tag): powerIter 1 improves on 0", errors[1] <= errors[0], "\(sci(errors[0])) -> \(sci(errors[1]))")
    report.check("\(tag): powerIter 2 keeps or improves on 1", errors[2] <= errors[1] * (1 + 1e-12),
                 "\(sci(errors[1])) -> \(sci(errors[2])), optimal \(sci(opt))")
    report.check("\(tag): powerIter 2 within 1% of optimal", errors[2] <= 1.01 * opt)

    let (fast, _) = fastDecayMatrix(T.self, seed: 7)
    var rng = Rng(seed: 32)
    let (q, flag, _) = orthSketch(fast, rtol: 1e-10, powerIter: 2, rng: &rng)
    report.check("\(tag): tolerance mode with powerIter 2", flag == 0 && spanError(fast, q) <= 1e-10
                 && orthonormalityError(q) <= 1e-13, "span err \(sci(spanError(fast, q)))")
}

// MARK: - Reproducibility and options

public func checkReproducibility<T: TestScalar>(_ report: CheckReport, _: T.Type) {
    let tag = "seed[\(T.tag)]"
    let (a, _) = fastDecayMatrix(T.self, seed: 8)
    var rng1 = Rng(seed: 42), rng2 = Rng(seed: 42)
    let first = svdSketch(a, rtol: 1e-8, rng: &rng1)
    let second = svdSketch(a, rtol: 1e-8, rng: &rng2)
    report.check("\(tag): same seed gives identical output",
                 first.u == second.u && first.s == second.s && first.vh == second.vh && rng1 == rng2)
    report.check("\(tag): the Rng advances", rng1 != Rng(seed: 42))
    let third = svdSketch(a, rtol: 1e-8, rng: &rng1)
    report.check("\(tag): a second call with the advanced Rng draws a new sketch",
                 third.u != first.u
                 && maxAbsDiff(Array(third.s.prefix(20)), Array(first.s.prefix(20))) <= 1e-12 * first.s[0])
    let d1 = idSketch(a, rtol: 1e-8), d2 = idSketch(a, rtol: 1e-8)
    var fresh = Rng()
    let d3 = idSketch(a, rtol: 1e-8, rng: &fresh)
    report.check("\(tag): calls without rng use Rng() and are deterministic",
                 d1.piv == d2.piv && d1.t == d2.t && d1.piv == d3.piv && d1.t == d3.t)
}

public func checkMethodParsing(_ report: CheckReport) {
    let parsed = ["fast", "svd", "lstsq"].compactMap { try? IDMethod(parsing: $0) }
    report.check("method: fast, svd, lstsq parse", parsed == [.fast, .svd, .lstsq])
    var thrown: LibRLAError?
    do { _ = try IDMethod(parsing: "qr") } catch { thrown = error as? LibRLAError }
    report.check("method: invalid ID method throws", thrown == .invalidMethod("qr"), "\(String(describing: thrown))")
}

// MARK: - Everything

/// Runs every check group for Double and Complex<Double>.
public func runAllChecks(_ report: CheckReport) {
    checkRng(report)
    checkMethodParsing(report)
    checkOrth(report, Double.self)
    checkOrth(report, Complex<Double>.self)
    checkQR(report, Double.self)
    checkQR(report, Complex<Double>.self)
    checkSVD(report, Double.self)
    checkSVD(report, Complex<Double>.self)
    checkID(report, Double.self)
    checkID(report, Complex<Double>.self)
    checkOperators(report, Double.self)
    checkOperators(report, Complex<Double>.self)
    checkPowerIteration(report, Double.self)
    checkPowerIteration(report, Complex<Double>.self)
    checkReproducibility(report, Double.self)
    checkReproducibility(report, Complex<Double>.self)
}
