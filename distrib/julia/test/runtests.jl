# runtests.jl - Entry point for `Pkg.test("LibRLA")`
#
# Runs the same suite as `julia --project=.. test_all.jl` and fails the
# test run if any module fails.
#
# SPDX-License-Identifier: MIT

include("test_all.jl")

main() == 0 || error("LibRLA tests failed")
