#!/usr/bin/env julia
"""
demo01_basic.jl - Introduction to Interpolative Decomposition

This demo introduces the two core ID algorithms:
  - id_sketch: Randomized ID using QR sketching (fast, approximate)
  - id_qrpiv:  Deterministic ID using column-pivoted QR (exact, slower)

The ID factorizes a matrix A as:
  A[:, piv[k+1:end]] = A[:, piv[1:k]] * T

where piv[1:k] selects the "skeleton" columns and T is the interpolation matrix.

Try changing the CONFIGURATION parameters below to experiment!

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Assisted by: Claude Code (Anthropic)
"""

#==============================================================================
# CONFIGURATION - Modify these to experiment
==============================================================================#

const MATRIX_SIZE = (300, 200)    # (rows, columns)
const TARGET_RANK = 15            # Number of skeleton columns to select
const RANDOM_SEED = 42            # For reproducibility (set to nothing for random)

# Matrix type: "hilbert" or "kahan"
const MATRIX_TYPE = "hilbert"
const KAHAN_THETA = 1.2           # Kahan matrix parameter (only used if MATRIX_TYPE="kahan")

#==============================================================================
# Demo code below
==============================================================================#

using LinearAlgebra
using Printf
using Random

include("librla.jl")
include("demo_utils.jl")

using .librla: id_sketch, id_qrpiv
using .demo_utils: hilbert, kahan, id_error, print_header, print_subheader


function main()
    if RANDOM_SEED !== nothing
        Random.seed!(RANDOM_SEED)
    end

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 01: Basic Interpolative Decomposition")
    @printf("\nMatrix: %d x %d\n", m, n)
    @printf("Target rank: %d\n", k)

    # Generate test matrix
    if MATRIX_TYPE == "hilbert"
        println("Matrix type: Hilbert (ill-conditioned)")
        A = hilbert(m, n)
    else  # kahan
        @printf("Matrix type: Kahan (theta=%.1f)\n", KAHAN_THETA)
        A = kahan(m, n; theta=KAHAN_THETA)
    end

    normA = norm(A)
    @printf("||A||_F = %.3e\n", normA)

    # JIT warm-up
    println("\nJIT warm-up...")
    A_warmup = randn(50, 30)
    id_sketch(A_warmup, 10.0)
    id_qrpiv(A_warmup, 10.0)
    println("JIT warm-up complete.")

    #--------------------------------------------------------------------------
    # Method 1: id_sketch (randomized)
    #--------------------------------------------------------------------------
    print_subheader("1. id_sketch (randomized)")
    println("   Uses random projections + QR. Fast but approximate.")

    elapsed1 = @elapsed begin
        k1, piv1, T1 = id_sketch(A, Float64(k))
    end

    err1 = id_error(A, k1, piv1, T1)
    maxT1 = isempty(T1) ? 0.0 : maximum(abs.(T1))

    @printf("   Rank:     %d\n", k1)
    @printf("   Error:    %.3e\n", err1)
    @printf("   Max |T|:  %.3e\n", maxT1)
    @printf("   Time:     %.4f s\n", elapsed1)

    #--------------------------------------------------------------------------
    # Method 2: id_qrpiv (deterministic)
    #--------------------------------------------------------------------------
    print_subheader("2. id_qrpiv (deterministic)")
    println("   Uses LAPACK column-pivoted QR. More accurate but slower.")

    elapsed2 = @elapsed begin
        k2, piv2, T2 = id_qrpiv(A, Float64(k))
    end

    err2 = id_error(A, k2, piv2, T2)
    maxT2 = isempty(T2) ? 0.0 : maximum(abs.(T2))

    @printf("   Rank:     %d\n", k2)
    @printf("   Error:    %.3e\n", err2)
    @printf("   Max |T|:  %.3e\n", maxT2)
    @printf("   Time:     %.4f s\n", elapsed2)

    #--------------------------------------------------------------------------
    # Summary
    #--------------------------------------------------------------------------
    print_subheader("Summary")
    @printf("   %-12s %6s %12s %12s %10s\n", "Method", "Rank", "Error", "Max|T|", "Time")
    @printf("   %s %s %s %s %s\n", "-"^12, "-"^6, "-"^12, "-"^12, "-"^10)
    @printf("   %-12s %6d %12.3e %12.3e %9.4fs\n", "id_sketch", k1, err1, maxT1, elapsed1)
    @printf("   %-12s %6d %12.3e %12.3e %9.4fs\n", "id_qrpiv", k2, err2, maxT2, elapsed2)

    # Validate
    if err1 < 1.0 && err2 < 1.0
        println("\n   [PASS] Both methods produced valid decompositions.")
        return true
    else
        println("\n   [FAIL] Error > 1.0 detected!")
        return false
    end
end


# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
