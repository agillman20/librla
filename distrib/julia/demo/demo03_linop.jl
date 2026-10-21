#!/usr/bin/env julia
"""
demo03_linop.jl - Matrix-Free Computation with LinearOperator

This demo shows how to use librla with operators from LinearOperators.jl for
matrix-free computation. This is essential for large-scale problems where the
matrix doesn't fit in memory. Any operator that defines size, eltype, adjoint
and mul! works the same way (e.g. LinearMaps.jl).

Requires LinearOperators.jl in the active environment:
    using Pkg; Pkg.add("LinearOperators")

Three modes are demonstrated:
  1. Dense matrix (baseline)
  2. Explicit LinearOperator (matrix wrapped, accessed via products)
  3. Matrix-free LinearOperator (only in-place A*x and A'*x products provided)

Note: Matrix-free mode only supports rank mode (rtol >= 1).

Try changing the CONFIGURATION parameters below to experiment!

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.2.1
Date: July 30, 2026
Assisted by: Claude Code (Anthropic)
"""

#==============================================================================
# CONFIGURATION - Modify these to experiment
==============================================================================#

const MATRIX_SIZE = (1000, 2000)    # (rows, columns)
const TARGET_RANK = 15            # Must be >= 1 for matrix-free mode
const RANDOM_SEED = 42            # For reproducibility

#==============================================================================
# Demo code below
==============================================================================#

using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "src", "LibRLA.jl"))
include("demo_utils.jl")

using .LibRLA: id_sketch, svd_sketch
using LinearOperators: LinearOperator
using .demo_utils: hilbert, id_error, print_header, print_subheader


function main()
    if RANDOM_SEED !== nothing
        Random.seed!(RANDOM_SEED)
    end

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 03: Matrix-Free Computation")
    @printf("\nMatrix: %d x %d Hilbert matrix\n", m, n)
    @printf("Target rank: %d\n", k)

    # Generate test matrix (we'll wrap it in LinearOperators)
    A = hilbert(m, n)
    normA = norm(A)

    # JIT warm-up
    println("\nJIT warm-up...")
    A_warmup = randn(50, 30)
    id_sketch(A_warmup, 10.0)
    svd_sketch(A_warmup, 10.0)
    A_lo_warmup = LinearOperator(A_warmup)
    id_sketch(A_lo_warmup, 10.0)
    println("JIT warm-up complete.")

    #==========================================================================
    # Part 1: ID with different input types
    ==========================================================================#
    print_subheader("Part 1: id_sketch with LinearOperator")

    #--------------------------------------------------------------------------
    # Test 1: Dense matrix (baseline)
    #--------------------------------------------------------------------------
    println("\n   1a. Dense matrix (baseline)")

    elapsed1 = @elapsed begin
        k1, piv1, T1 = id_sketch(A, k)
    end

    err1 = id_error(A, k1, piv1, T1)
    @printf("       Rank: %d, Error: %.3e, Time: %.4fs\n", k1, err1, elapsed1)

    #--------------------------------------------------------------------------
    # Test 2: Explicit LinearOperator (wraps the stored matrix)
    #--------------------------------------------------------------------------
    println("\n   1b. Explicit LinearOperator (matrix wrapper)")
    println("       Matrix is wrapped by LinearOperator(A).")

    A_explicit = LinearOperator(A)

    elapsed2 = @elapsed begin
        k2, piv2, T2 = id_sketch(A_explicit, k)
    end

    err2 = id_error(A, k2, piv2, T2)
    @printf("       Rank: %d, Error: %.3e, Time: %.4fs\n", k2, err2, elapsed2)

    #--------------------------------------------------------------------------
    # Test 3: Matrix-free LinearOperator
    #--------------------------------------------------------------------------
    println("\n   1c. Matrix-free LinearOperator")
    println("       Only in-place A*x and A'*x products provided.")
    println("       Requires rank mode (rtol >= 1).")

    # 5-argument products y = α*op(A)*x + β*y; x may be a block of columns
    my_matvec = (y, x, α, β) -> mul!(y, A, x, α, β)
    my_rmatvec = (y, x, α, β) -> mul!(y, A', x, α, β)
    A_matfree = LinearOperator(Float64, m, n, false, false,
                               my_matvec, nothing, my_rmatvec)

    elapsed3 = @elapsed begin
        k3, piv3, T3 = id_sketch(A_matfree, k)
    end

    err3 = id_error(A, k3, piv3, T3)
    @printf("       Rank: %d, Error: %.3e, Time: %.4fs\n", k3, err3, elapsed3)

    #==========================================================================
    # Part 2: SVD with LinearOperator
    ==========================================================================#
    print_subheader("Part 2: svd_sketch with LinearOperator")

    # Dense baseline
    println("\n   2a. Dense matrix")
    elapsed_svd1 = @elapsed begin
        U1, s1, Vt1 = svd_sketch(A, k)
    end

    A_approx1 = U1 * Diagonal(s1) * Vt1
    err_svd1 = norm(A - A_approx1) / normA
    @printf("       Rank: %d, Error: %.3e, Time: %.4fs\n", length(s1), err_svd1, elapsed_svd1)

    # Matrix-free
    println("\n   2b. Matrix-free LinearOperator")
    elapsed_svd2 = @elapsed begin
        U2, s2, Vt2 = svd_sketch(A_matfree, k)
    end

    A_approx2 = U2 * Diagonal(s2) * Vt2
    err_svd2 = norm(A - A_approx2) / normA
    @printf("       Rank: %d, Error: %.3e, Time: %.4fs\n", length(s2), err_svd2, elapsed_svd2)

    #==========================================================================
    # Summary
    ==========================================================================#
    print_subheader("Summary: id_sketch")
    @printf("   %-28s %6s %12s %10s\n", "Input Type", "Rank", "Error", "Time")
    @printf("   %s %s %s %s\n", "-"^28, "-"^6, "-"^12, "-"^10)
    @printf("   %-28s %6d %12.3e %9.4fs\n", "Dense matrix", k1, err1, elapsed1)
    @printf("   %-28s %6d %12.3e %9.4fs\n", "Explicit LinearOperator", k2, err2, elapsed2)
    @printf("   %-28s %6d %12.3e %9.4fs\n", "Matrix-free LinearOperator", k3, err3, elapsed3)

    println("\nNotes:")
    println("  - LinearOperator allows matrix-free computation")
    println("  - Essential for large-scale problems (matrix doesn't fit in memory)")
    println("  - Matrix-free mode requires rank mode (rtol >= 1)")
    println("  - Define only the products A * x and A' * x")

    return true
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
