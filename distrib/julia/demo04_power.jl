#!/usr/bin/env julia
"""
demo04_power.jl - Power Iteration for Improved Accuracy

This demo shows how power iteration improves sketching accuracy.

Power iteration applies (A'*A)^q to the random sketch, which amplifies
the dominant singular components. This is especially helpful when:
  - The spectral gap is small
  - High accuracy is needed
  - The matrix has slowly decaying singular values

The demo tests a 2D grid of parameters:
  - extra_samples: How much oversampling (more = better accuracy)
  - power_iter: Number of power iterations (more = better accuracy)

Trade-off: More iterations/samples = better accuracy but more computation.

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

const MATRIX_SIZE = (2000, 1000)    # (rows, columns)
const TARGET_RANK = 50            # Number of singular values to compute
const RANDOM_SEED = 42            # For reproducibility

# Power iteration settings - test grid of values
const EXTRA_SAMPLES_LIST = [24, 18, 12, 6, 3]  # Oversampling values to test
const POWER_ITER_LIST = [0, 1, 2, 3, 4]  # Power iteration counts to test

# Matrix type: "structured" (clear spectral gap) or "random" (no gap)
const MATRIX_TYPE = "structured"

#==============================================================================
# Demo code below
==============================================================================#

using LinearAlgebra
using Printf
using Random

include("librla.jl")
include("demo_utils.jl")

using .librla: svd_sketch
using .demo_utils: print_header, print_subheader


function main()
    if RANDOM_SEED !== nothing
        Random.seed!(RANDOM_SEED)
    end

    m, n = MATRIX_SIZE
    k = TARGET_RANK

    print_header("Demo 04: Power Iteration")
    @printf("\nMatrix: %d x %d\n", m, n)
    @printf("Target rank: %d\n", k)

    #--------------------------------------------------------------------------
    # Create test matrix
    #--------------------------------------------------------------------------
    if MATRIX_TYPE == "structured"
        println("Matrix type: STRUCTURED")
        println("Singular values: logspace(0,-2,k) + logspace(-2,-10,n-k)")
        # Create matrix with decaying spectrum and clear gap at rank k
        U_full = Matrix(qr(randn(m, m)).Q)
        V_full = Matrix(qr(randn(n, n)).Q)
        s_true = vcat(10.0 .^ range(0, -2, length=k),
                      10.0 .^ range(-2, -10, length=n-k))
        U = U_full[:, 1:n]
        A = U * Diagonal(s_true) * V_full'
    else
        println("Matrix type: RANDOM (no spectral gap)")
        A = randn(m, n)
        s_true = svdvals(A)
    end

    # Matrix properties
    cnd = s_true[1] / s_true[end]
    gap = s_true[k] / s_true[k+1]

    println("\nSpectral properties:")
    @printf("   s[1]     = %.6e (largest)\n", s_true[1])
    @printf("   s[%d]   = %.6e (at target rank)\n", k, s_true[k])
    @printf("   s[%d]   = %.6e (first neglected)\n", k+1, s_true[k+1])
    @printf("   s[%d] = %.6e (smallest)\n", n, s_true[n])
    @printf("   Condition number: %.2e\n", cnd)
    @printf("   Spectral gap at k=%d: %.1fx\n", k, gap)

    # JIT warm-up
    println("\nJIT warm-up...")
    A_warmup = randn(50, 30)
    for p in 0:2
        svd_sketch(A_warmup, 10.0; power_iter=p, extra_samples=5)
    end
    println("JIT warm-up complete.")

    #--------------------------------------------------------------------------
    # Test grid of extra_samples and power_iter values
    #--------------------------------------------------------------------------
    print_subheader("Testing Parameter Grid")
    println("   power_iter=0 means no power iteration (baseline)")
    println("   Each power iteration costs 2 extra matrix-vector products")
    println("   extra_samples controls oversampling (block_size = k + extra_samples)")

    # Store results in 2D arrays
    num_extra = length(EXTRA_SAMPLES_LIST)
    num_power = length(POWER_ITER_LIST)
    errors = zeros(num_extra, num_power)
    sval_errors = zeros(num_extra, num_power)
    s_ref = s_true[1:k]

    for (idx, extra_samples) in enumerate(EXTRA_SAMPLES_LIST)
        block_size = k + extra_samples
        @printf("\n--- extra_samples = %d (block_size = %d) ---\n", extra_samples, block_size)

        for (jdx, power_iter) in enumerate(POWER_ITER_LIST)
            elapsed = @elapsed begin
                U, s, Vt = svd_sketch(A, Float64(k);
                                      power_iter=power_iter,
                                      extra_samples=extra_samples)
            end

            # Reconstruction error
            A_approx = U * Diagonal(s) * Vt
            recon_err = norm(A - A_approx) / norm(A)
            errors[idx, jdx] = recon_err

            # Singular value accuracy
            sval_err = norm(s - s_ref) / norm(s_ref)
            sval_errors[idx, jdx] = sval_err

            @printf("   power_iter=%d: err=%.2e, sval_err=%.2e, time=%.4fs\n",
                    power_iter, recon_err, sval_err, elapsed)
        end
    end

    #--------------------------------------------------------------------------
    # Summary tables
    #--------------------------------------------------------------------------
    print_subheader("Summary: Reconstruction Error")

    # Header row
    header = "extra_samples |"
    for p in POWER_ITER_LIST
        header *= @sprintf("  iter=%d  |", p)
    end
    println(header)
    println("-"^14 * "+" * ("-"^10 * "+")^num_power)

    # Data rows
    for (idx, extra_samples) in enumerate(EXTRA_SAMPLES_LIST)
        row = @sprintf("%13d |", extra_samples)
        for jdx in 1:num_power
            row *= @sprintf(" %.2e |", errors[idx, jdx])
        end
        println(row)
    end

    print_subheader("Summary: Singular Value Error")

    # Header row
    header = "extra_samples |"
    for p in POWER_ITER_LIST
        header *= @sprintf("  iter=%d  |", p)
    end
    println(header)
    println("-"^14 * "+" * ("-"^10 * "+")^num_power)

    # Data rows
    for (idx, extra_samples) in enumerate(EXTRA_SAMPLES_LIST)
        row = @sprintf("%13d |", extra_samples)
        for jdx in 1:num_power
            row *= @sprintf(" %.2e |", sval_errors[idx, jdx])
        end
        println(row)
    end

    println("\nNotes:")
    println("  - Power iteration amplifies dominant singular components")
    println("  - Extra samples (oversampling) improves subspace capture")

    return true
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
