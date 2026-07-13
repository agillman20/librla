#!/usr/bin/env julia
#
# test_all.jl - Master test script for librla
#
# Runs all test tests and produces a unified summary:
# - test_id.jl    (ID implementations)
# - test_svd.jl   (SVD implementations)
# - test_qr.jl    (QR implementations)
# - test_orth.jl  (Orth implementations)
#
# Usage:
#     julia test_all.jl
#
# Returns exit code 0 if all tests pass, 1 otherwise.
#
# Author: Adrianna Gillman, Zydrunas Gimbutas
# SPDX-License-Identifier: MIT
# Version: 1.1.0
# Date: July 13, 2026
# Assisted by: Claude Code (Anthropic)

using Printf

# Get the directory containing this script
const SCRIPT_DIR = @__DIR__

# Include all test modules (JIT compiles once for all)
include(joinpath(SCRIPT_DIR, "test_id.jl"))
include(joinpath(SCRIPT_DIR, "test_svd.jl"))
include(joinpath(SCRIPT_DIR, "test_qr.jl"))
include(joinpath(SCRIPT_DIR, "test_orth.jl"))
include(joinpath(SCRIPT_DIR, "test_linop.jl"))

function run_test_module(test_func::Function, display_name::String)
    """
    Run a test module's test() function and return (exit_code, elapsed, success).
    """
    println()
    println("="^70)
    println("Running $display_name...")
    println("="^70)

    try
        t0 = time()
        exit_code = test_func()
        elapsed = time() - t0

        success = (exit_code == 0)
        return exit_code, elapsed, success

    catch e
        println("\n[ERROR] Failed to run $display_name: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return 1, 0.0, false
    end
end

function main()
    println()
    println("="^70)
    println("LIBRLA TEST SUITE - Julia")
    println("="^70)
    println()
    println("This script runs all test tests for librla functions:")
    println("  - test_id.jl    (id_sketch, id_qrpiv)")
    println("  - test_svd.jl   (svd_sketch)")
    println("  - test_qr.jl    (qr_sketch)")
    println("  - test_orth.jl  (orth_sketch)")
    println("  - test_linop.jl (LinearOperator, wide, method regressions)")
    println()
    println("Environment: Julia ", VERSION)
    println("="^70)

    # List of test modules (test function, display name)
    modules = [
        (TestID.test, "test_id"),
        (TestSVD.test, "test_svd"),
        (TestQR.test, "test_qr"),
        (TestOrth.test, "test_orth"),
        (TestLinop.test, "test_linop"),
    ]

    # Results tracking: (display_name, status, elapsed, success)
    results = []
    total_elapsed = 0.0

    for (test_func, display_name) in modules
        exit_code, elapsed, success = run_test_module(test_func, display_name)
        total_elapsed += elapsed

        if success
            status = exit_code == 0 ? "PASSED" : "FAILED"
        else
            status = "ERROR"
        end

        push!(results, (display_name, status, elapsed, success))
    end

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    println("\n")
    println("="^70)
    println("FINAL SUMMARY")
    println("="^70)
    println()

    # Per-module summary
    @printf("%-20s %-12s %-12s\n", "Module", "Status", "Time (s)")
    println("-"^50)

    all_passed = true
    modules_passed = 0

    for (display_name, status, elapsed, success) in results
        @printf("%-20s [%-6s]    %8.2fs\n", display_name, status, elapsed)

        if status != "PASSED"
            all_passed = false
        else
            modules_passed += 1
        end
    end

    println("-"^50)
    @printf("%-20s %12s %8.2fs\n", "Total", "", total_elapsed)

    # Overall status
    println()
    println("="^70)

    num_modules = length(modules)

    if all_passed
        println("[PASS] All $num_modules test modules passed!")
        println("="^70)
        return 0
    else
        println("[FAIL] $modules_passed/$num_modules test modules passed")
        println()
        println("Failed modules:")
        for (display_name, status, elapsed, success) in results
            if status != "PASSED"
                println("  - $display_name: $status")
            end
        end
        println("="^70)
        return 1
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit_code = main()
    exit(exit_code)
end
