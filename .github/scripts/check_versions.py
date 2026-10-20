#!/usr/bin/env python3
"""Check that the Python, MATLAB and Julia ports declare the same version.

Usage:
    python .github/scripts/check_versions.py          # ports must agree
    python .github/scripts/check_versions.py v1.2.2   # ...and match this tag

Each port has exactly one version source:
    Python  distrib/python/librla.py     __version__ = "X.Y.Z"
    MATLAB  distrib/matlab/librla.m      v = 'X.Y.Z';  (in librla.version)
    Julia   distrib/julia/Project.toml   version = "X.Y.Z"

Exits with status 1 on any mismatch.

SPDX-License-Identifier: MIT
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "Python": ("distrib/python/librla.py", r'^__version__ = "([^"]+)"$'),
    "MATLAB": ("distrib/matlab/librla.m", r"^  v = '([^']+)';$"),
    "Julia": ("distrib/julia/Project.toml", r'^version = "([^"]+)"$'),
}


def read_versions():
    versions = {}
    for port, (path, pattern) in SOURCES.items():
        found = re.findall(pattern, (ROOT / path).read_text(), flags=re.M)
        if len(found) != 1:
            sys.exit(f"{path}: expected one version line, found {len(found)}")
        versions[port] = found[0]
    return versions


def main(argv):
    versions = read_versions()
    for port, v in versions.items():
        print(f"{port:7s} {v}  ({SOURCES[port][0]})")

    ok = len(set(versions.values())) == 1
    if not ok:
        print("ERROR: port versions differ")

    if len(argv) > 1 and argv[1]:
        tag = argv[1]
        expected = "v" + versions["Python"]
        if tag != expected:
            print(f"ERROR: tag {tag} does not match version {expected}")
            ok = False
        else:
            print(f"Tag     {tag} matches")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
