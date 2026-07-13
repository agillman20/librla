#!/usr/bin/env julia
"""
demo05_methods.jl - T Matrix Computation Methods

This demo compares the three methods for computing the interpolation
matrix T in the ID factorization: A[:, piv[k+1:end]] = A[:, piv[1:k]] * T

Methods:
  - "fast":   Triangular solve (fastest, may have large T entries)
  - "svd":    SVD-based pseudoinverse
  - "lstsq":  Least squares from original A (most accurate, slowest)

The choice of method affects:
  - Speed (fast < svd < lstsq)
  - Stability (fast may produce large T, svd/lstsq are stable)
  - Accuracy (lstsq gives best reconstruction)

Try changing the CONFIGURATION parameters below to experiment!

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.1.0
Date: July 13, 2026
Assisted by: Claude Code (Anthropic)
"""

#==============================================================================
# CONFIGURATION - Modify these to experiment
==============================================================================#

const MATRIX_SIZE = (2000, 1000)    # (rows, columns)
const TARGET_RANK = 50            # Number of skeleton columns
const RANDOM_SEED = 42            # For reproducibility

# Matrix type: "lowrank", "fullrank", or "hilbert"
const MATRIX_TYPE = "fullrank"

#==============================================================================
# Demo code below
==============================================================================#

using LinearAlgebra
using Printf
using Random

include("../librla.jl")
include("demo_utils.jl")

using .librla: id_sketch
using .demo_utils: hilbert, lowrank, random_matrix, id_error
using .demo_utils: print_header, print_subheader


function main()
    if RANDOM_SEED !== nothing
        Random.seed!(RANDOM_SEED)
    end

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 05: T Matrix Computation Methods")
    @printf("\nMatrix: %d x %d\n", m, n)
    @printf("Target rank: %d\n", k)

    #--------------------------------------------------------------------------
    # Create test matrix
    #--------------------------------------------------------------------------
    if MATRIX_TYPE == "lowrank"
        println("Matrix type: LOW-RANK (true rank = 30)")
        A, _ = lowrank(m, n, 30; decay="exponential", gap=100.0)
    elseif MATRIX_TYPE == "fullrank"
        println("Matrix type: FULL-RANK RANDOM")
        println("(All columns are linearly independent)")
        A = random_matrix(m, n)
    else  # hilbert
        println("Matrix type: HILBERT (ill-conditioned)")
        A = hilbert(m, n)
    end

    normA = norm(A)
    s = svdvals(A)
    cnd = s[1] / s[end]
    @printf("Condition number: %.2e\n", cnd)

    # JIT warm-up
    println("\nJIT warm-up...")
    A_warmup = randn(50, 30)
    for method in ["fast", "svd", "lstsq"]
        id_sketch(A_warmup, 10.0; method=method)
    end
    println("JIT warm-up complete.")

    #--------------------------------------------------------------------------
    # Test all three methods
    #--------------------------------------------------------------------------
    methods = ["fast", "svd", "lstsq"]
    results = []

    for method in methods
        print_subheader("Method: '$method'")

        if method == "fast"
            println("   Triangular solve on R factor. Fastest but may be unstable.")
        elseif method == "svd"
            println("   SVD-based pseudoinverse. Stable for ill-conditioned R.")
        else
            println("   Least squares from original A. Most accurate, slowest.")
        end

        elapsed = @elapsed begin
            k_out, piv, T = id_sketch(A, Float64(k); method=method)
        end

        err = id_error(A, k_out, piv, T)
        maxT = isempty(T) ? 0.0 : maximum(abs.(T))

        push!(results, (method=method, k=k_out, error=err, maxT=maxT, time=elapsed))

        @printf("   Rank:     %d\n", k_out)
        @printf("   Error:    %.6e\n", err)
        @printf("   Max |T|:  %.3e\n", maxT)
        @printf("   Time:     %.4f s\n", elapsed)

        # Warn about large T entries
        if maxT > 10.0
            println("   [NOTE] Max|T| > 10 indicates potential instability")
        end
        if err > 1.0
            println("   [NOTE] Error > 1.0: relative error exceeds 100%")
        end
    end

    #--------------------------------------------------------------------------
    # Summary
    #--------------------------------------------------------------------------
    print_subheader("Summary")
    @printf("   %-8s %6s %14s %12s %10s\n", "Method", "Rank", "Error", "Max|T|", "Time")
    @printf("   %s %s %s %s %s\n", "-"^8, "-"^6, "-"^14, "-"^12, "-"^10)
    for r in results
        @printf("   %-8s %6d %14.6e %12.3e %9.4fs\n", r.method, r.k, r.error, r.maxT, r.time)
    end

    # Analysis
    println("\nAnalysis:")

    fast_err = results[1].error
    lstsq_err = results[3].error

    if fast_err > 1.0 && lstsq_err < 1.0
        println("  - 'fast' failed (error > 1) but 'lstsq' succeeded")
        println("  - This happens with full-rank matrices: skeleton columns")
        println("    cannot exactly represent other columns")
        println("  - Use method=\"lstsq\" for best least-squares approximation")
    elseif results[1].maxT > 100 * results[3].maxT
        println("  - 'fast' produced much larger T entries than 'lstsq'")
        println("  - This indicates numerical instability in triangular solve")
        println("  - Consider using method=\"svd\" or \"lstsq\" for stability")
    else
        println("  - All methods performed similarly")
        println("  - 'fast' is recommended for speed")
    end

    println("\nRecommendations:")
    println("  - Use \"fast\" (default) for low-rank matrices")
    println("  - Use \"svd\" when R factor is ill-conditioned")
    println("  - Use \"lstsq\" when best accuracy is needed")
    println("  - Use \"lstsq\" for full-rank matrices (guarantees error < 1)")

    return true
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
