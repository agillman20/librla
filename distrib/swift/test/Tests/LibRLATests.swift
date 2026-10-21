// LibRLATests.swift - swift-testing suite: every check group, real and complex
//
// Each test runs one group of checks from LibRLACheckSuite (printing its
// PASS/FAIL lines) and records every failed check as an issue. The
// `crossport` test prints the CROSSPORT lines. `swift run librla-test`
// runs the same checks as a standalone program.
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import ComplexModule
import LibRLA
import LibRLACheckSuite
import Testing

private func run(_ group: (CheckReport) -> Void) {
    let report = CheckReport(echo: true)
    group(report)
    #expect(!report.results.isEmpty)
    for failure in report.failures {
        Issue.record("FAIL \(failure.name) \(failure.detail)")
    }
}

@Suite(.serialized) struct LibRLATests {
    @Test func rng() { run(checkRng) }
    @Test func idMethodParsing() { run(checkMethodParsing) }

    @Test func orthSketchReal() { run { checkOrth($0, Double.self) } }
    @Test func orthSketchComplex() { run { checkOrth($0, Complex<Double>.self) } }
    @Test func qrSketchReal() { run { checkQR($0, Double.self) } }
    @Test func qrSketchComplex() { run { checkQR($0, Complex<Double>.self) } }
    @Test func svdSketchReal() { run { checkSVD($0, Double.self) } }
    @Test func svdSketchComplex() { run { checkSVD($0, Complex<Double>.self) } }
    @Test func interpolativeReal() { run { checkID($0, Double.self) } }
    @Test func interpolativeComplex() { run { checkID($0, Complex<Double>.self) } }
    @Test func operatorsReal() { run { checkOperators($0, Double.self) } }
    @Test func operatorsComplex() { run { checkOperators($0, Complex<Double>.self) } }
    @Test func powerIterationReal() { run { checkPowerIteration($0, Double.self) } }
    @Test func powerIterationComplex() { run { checkPowerIteration($0, Complex<Double>.self) } }
    @Test func reproducibilityReal() { run { checkReproducibility($0, Double.self) } }
    @Test func reproducibilityComplex() { run { checkReproducibility($0, Complex<Double>.self) } }

    @Test func crossport() {
        let lines = crossportLines()
        for line in lines { print(line) }
        #expect(lines.count == 5)
        #expect(lines[2] == "CROSSPORT seed_after 3844 2983 1707 1605")
        #expect(lines[3] == "CROSSPORT id_real 16 0 3 28 1 168 10 72 5 199 2")
    }
}
