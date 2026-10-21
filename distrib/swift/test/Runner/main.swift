// librla-test - runs every check, printing one PASS/FAIL line per check
// followed by the CROSSPORT lines; exits 1 if any check fails.
//
//     swift run librla-test
//
// Author: Adrianna Gillman, Zydrunas Gimbutas
// SPDX-License-Identifier: MIT
// Assisted by: Claude Code (Anthropic)

import Foundation
import LibRLACheckSuite

let report = CheckReport(echo: true)
runAllChecks(report)
for line in crossportLines() { print(line) }

let failed = report.failures.count
print("\(report.results.count - failed) of \(report.results.count) checks passed")
exit(failed == 0 ? 0 : 1)
