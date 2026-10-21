#!/usr/bin/env python3
"""Check that the experimental ports print the same CROSSPORT lines.

Usage:
    python .github/scripts/check_crossport.py cpp.txt rust.txt fortran.txt

Each file is the output of one port's test suite or crossport example; lines
not starting with "CROSSPORT" are ignored. Every file must contain the same
set of CROSSPORT lines (by name, the second field) with the same number of
values. Integer values (pivots, ranks, RNG seeds) must match exactly; floating
point values must agree to RTOL relative, since the ports call the same
LAPACK in different BLAS summation orders.

Exits with status 1 on any mismatch.

SPDX-License-Identifier: MIT
"""

import re
import sys
from pathlib import Path

RTOL = 1e-9
INT = re.compile(r"^[+-]?\d+$")


def read(path):
    lines = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "CROSSPORT":
            lines[fields[1]] = fields[2:]
    return lines


def compare(ref, other):
    """Return (problems, worst relative float difference) for one line."""
    if len(ref) != len(other):
        return [f"{len(other)} values, expected {len(ref)}"], 0.0
    problems, worst = [], 0.0
    for i, (a, b) in enumerate(zip(ref, other)):
        if INT.match(a) or INT.match(b):
            if a != b:
                problems.append(f"value {i + 1}: {b} != {a}")
            continue
        x, y = float(a), float(b)
        scale = max(abs(x), abs(y))
        rel = abs(x - y) / scale if scale > 0 else 0.0
        worst = max(worst, rel)
        if rel > RTOL:
            problems.append(f"value {i + 1}: {b} vs {a} (rel {rel:.1e})")
    return problems, worst


def main(paths):
    if len(paths) < 2:
        sys.exit(__doc__)
    outputs = {Path(p).stem: read(p) for p in paths}
    ref_name, ref = next(iter(outputs.items()))
    failed = False
    if not ref:
        print(f"{ref_name}: no CROSSPORT lines")
        sys.exit(1)
    for name, lines in outputs.items():
        if set(lines) != set(ref):
            print(f"{name}: lines {sorted(lines)} != {ref_name}: {sorted(ref)}")
            failed = True
            continue
        if name == ref_name:
            continue
        for key in ref:
            problems, worst = compare(ref[key], lines[key])
            status = "FAIL" if problems else "ok"
            print(f"{status:4} {name} vs {ref_name} {key}: max rel diff {worst:.1e}")
            for p in problems:
                print(f"       {p}")
            failed = failed or bool(problems)
    print("CROSSPORT lines differ" if failed else
          f"CROSSPORT lines agree across {len(outputs)} ports")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
