#!/usr/bin/env julia
"""
demo02_svd.jl - Truncated SVD via Randomized Sketching

This demo shows how to compute truncated SVD using librla:
  - svd_sketch: Randomized truncated SVD
  - qr_sketch:  Truncated QR factorization

Both functions use randomized sketching for efficiency on large matrices.

The SVD factorizes A as: A ≈ U * diagm(s) * Vt
The QR factorizes A as: A[:, p] ≈ Q * R

Try changing the CONFIGURATION parameters below to experiment!

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: NIST-PD
Version: 1.0.0
Date: January 5, 2026
Assisted by: Claude Code (Anthropic)
"""

#==============================================================================
# CONFIGURATION - Modify these to experiment
==============================================================================#

const MATRIX_SIZE = (1000, 2000)    # (rows, columns)
const TARGET_RANK = 30            # Number of singular values to compute
const RANDOM_SEED = 42            # For reproducibility

#==============================================================================
# Demo code below
==============================================================================#

using LinearAlgebra
using Printf
using Random

include("librla.jl")
include("demo_utils.jl")

using .librla: svd_sketch, qr_sketch
using .demo_utils: hilbert, svd_error, print_header, print_subheader


function main()
    if RANDOM_SEED !== nothing
        Random.seed!(RANDOM_SEED)
    end

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 02: Truncated SVD and QR")
    @printf("\nMatrix: %d x %d Hilbert matrix\n", m, n)
    @printf("Target rank: %d\n", k)

    # Generate test matrix
    A = hilbert(m, n)
    normA = norm(A)

    # Compute reference SVD for comparison
    s_true = svdvals(A)
    println("\nTrue singular values:")
    @printf("   s[1] = %.6e\n", s_true[1])
    @printf("   s[%d] = %.6e\n", k, s_true[k])
    @printf("   s[%d] = %.6e\n", k+1, s_true[k+1])
    @printf("   s[%d] = %.6e\n", min(m,n), s_true[min(m,n)])

    # JIT warm-up
    println("\nJIT warm-up...")
    A_warmup = randn(50, 30)
    svd_sketch(A_warmup, 10.0)
    qr_sketch(A_warmup, 10.0)
    println("JIT warm-up complete.")

    #--------------------------------------------------------------------------
    # Method 1: svd_sketch
    #--------------------------------------------------------------------------
    print_subheader("1. svd_sketch (truncated SVD)")
    println("   Returns U, s, Vt where A ≈ U * diagm(s) * Vt")

    elapsed = @elapsed begin
        U, s, Vt = svd_sketch(A, Float64(k))
    end

    err = svd_error(A, U, s, Vt)
    k_out = length(s)

    @printf("   Rank:      %d\n", k_out)
    @printf("   Error:     %.6e\n", err)
    @printf("   Time:      %.4f s\n", elapsed)

    # Compare singular values
    s_err = norm(s - s_true[1:k_out]) / norm(s_true[1:k_out])
    @printf("   SVal err:  %.6e (relative)\n", s_err)

    # Check orthogonality
    orth_U = norm(U' * U - I(k_out))
    orth_V = norm(Vt * Vt' - I(k_out))
    @printf("   ||U'U-I||: %.2e\n", orth_U)
    @printf("   ||VtVt'-I||: %.2e\n", orth_V)

    #--------------------------------------------------------------------------
    # Method 2: qr_sketch
    #--------------------------------------------------------------------------
    print_subheader("2. qr_sketch (truncated QR)")
    println("   Returns Q, R, piv where A[:, piv] ≈ Q * R")

    elapsed2 = @elapsed begin
        Q, R, piv = qr_sketch(A, Float64(k))
    end

    # Reconstruct: A[:, piv] = Q * R
    A_qr = zeros(size(A))
    A_qr[:, piv] = Q * R
    err2 = norm(A - A_qr) / normA
    k2 = size(Q, 2)

    @printf("   Rank:      %d\n", k2)
    @printf("   Error:     %.6e\n", err2)
    @printf("   Time:      %.4f s\n", elapsed2)

    # Check orthogonality
    orth_Q = norm(Q' * Q - I(k2))
    @printf("   ||Q'Q-I||: %.2e\n", orth_Q)

    #--------------------------------------------------------------------------
    # Summary
    #--------------------------------------------------------------------------
    print_subheader("Summary")
    @printf("   %-14s %6s %12s %10s\n", "Method", "Rank", "Error", "Time")
    @printf("   %s %s %s %s\n", "-"^14, "-"^6, "-"^12, "-"^10)
    @printf("   %-14s %6d %12.3e %9.4fs\n", "svd_sketch", k_out, err, elapsed)
    @printf("   %-14s %6d %12.3e %9.4fs\n", "qr_sketch", k2, err2, elapsed2)

    println("\nNotes:")
    println("  - svd_sketch gives U, s, Vt for best rank-k approximation")
    println("  - qr_sketch gives Q, R factorization with column pivoting")

    return true
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
