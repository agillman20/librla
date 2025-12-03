#!/usr/bin/env julia
#
# validate_all.jl - Master validation script for librla
#
# Runs all validation tests and produces a unified summary:
# - validate_id.jl    (ID implementations)
# - validate_svd.jl   (SVD implementations)
# - validate_qr.jl    (QR implementations)
# - validate_orth.jl  (Orth implementations)
#
# Usage:
#     julia validate_all.jl
#
# Returns exit code 0 if all tests pass, 1 otherwise.

using Printf

function run_validation_module(script_path::String, display_name::String)
    """
    Run a validation script as a subprocess and return (exit_code, elapsed, success).
    """
    println()
    println("="^70)
    println("Running $display_name...")
    println("="^70)

    try
        t0 = time()

        # Run the script as a subprocess
        proc = run(`julia $script_path`, wait=true)
        exit_code = proc.exitcode

        elapsed = time() - t0

        success = (exit_code == 0)
        return exit_code, elapsed, success

    catch e
        println("\n[ERROR] Failed to run $display_name: $e")
        return 1, 0.0, false
    end
end

function main()
    println()
    println("="^70)
    println("LIBRLA VALIDATION SUITE - Julia")
    println("="^70)
    println()
    println("This script runs all validation tests for librla functions:")
    println("  - validate_id.jl    (id_sketch, id_qrpiv)")
    println("  - validate_svd.jl   (svd_sketch)")
    println("  - validate_qr.jl    (qr_sketch)")
    println("  - validate_orth.jl  (orth_sketch)")
    println()
    println("Environment: Julia ", VERSION)
    println("="^70)

    # Get the directory containing this script
    script_dir = dirname(@__FILE__)

    # List of validation modules
    modules = [
        ("validate_id.jl", "validate_id"),
        ("validate_svd.jl", "validate_svd"),
        ("validate_qr.jl", "validate_qr"),
        ("validate_orth.jl", "validate_orth"),
    ]

    # Results tracking: (display_name, status, elapsed, success)
    results = []
    total_elapsed = 0.0

    for (filename, display_name) in modules
        script_path = joinpath(script_dir, filename)

        if !isfile(script_path)
            println("\n[WARNING] $filename not found at $script_path")
            push!(results, (display_name, "not found", 0.0, false))
            continue
        end

        exit_code, elapsed, success = run_validation_module(script_path, display_name)
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
        println("[PASS] All $num_modules validation modules passed!")
        println("="^70)
        return 0
    else
        println("[FAIL] $modules_passed/$num_modules validation modules passed")
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
