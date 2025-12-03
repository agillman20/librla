"""
test7_power.jl - Test power iteration in svd_sketch (librla version)

Tests power iteration in the svd_sketch function.
Power iteration applies (A'*A)^n to improve sketch quality by amplifying
the dominant subspace.

This version uses librla instead of libid.

Usage:
    julia test7_power.jl           # structured matrix only
    julia test7_power.jl random    # both structured and random

Tests:
    Test 1: Power iteration in svd_sketch
        - Tests extra_samples: 24, 18, 12, 6, 3
        - Tests power_iter: 0-6
        - Measures reconstruction error and singular value accuracy
        - Can use structured or random matrix

Author: Power iteration tests (librla version)
SPDX-License-Identifier: TBD
"""

using LinearAlgebra
using Printf
using Random

include("librla.jl")

using .librla: svd_sketch


"""
    test_svd_sketch_power_iter(use_random_matrix::Bool=false)

Test 1: Power iteration in svd_sketch.

Tests how power iteration improves singular value accuracy in svd_sketch.
"""
function test_svd_sketch_power_iter(use_random_matrix::Bool=false)
    println("\n" * "="^70)
    println("TEST 1: Power iteration in svd_sketch")
    println("="^70)

    # Test matrix configuration (same as test6_power)
    m, n = 500, 300
    k = 50

    if use_random_matrix
        println("\nMatrix type: RANDOM (no prescribed singular values)")
        # Simple random matrix
        A = randn(m, n)

        # Compute SVD to get true singular values
        F = svd(A)
        s = F.S
    else
        println("\nMatrix type: STRUCTURED (prescribed singular values)")
        # Create matrix with decaying spectrum
        U_full = qr(randn(m, m)).Q
        V_full = qr(randn(n, n)).Q
        s = vcat(exp10.(range(0, -2, length=k)),
                exp10.(range(-2, -10, length=n-k)))
        # Use first n columns of U to match dimensions
        U = Matrix(U_full)[:, 1:n]
        V = Matrix(V_full)
        A = U * Diagonal(s) * V'
    end

    # Compute detailed matrix properties
    cond_number = s[1] / s[end]
    spectral_gap_k = s[k] / s[k+1]
    decay_rate_k = s[1] / s[k]

    println("\nMatrix Properties:")
    @printf("  Dimensions:       %dx%d\n", m, n)
    @printf("  Target rank:      %d (first %d singular values)\n", k, k)
    @printf("  Condition number: %.2e\n", cond_number)
    @printf("  Spectral gap at k=%d: %.2fx (s[%d]/s[%d])\n", k, spectral_gap_k, k, k+1)
    @printf("  Decay rate (s[1]/s[%d]): %.2fx\n", k, decay_rate_k)
    println("\nSingular value distribution:")
    @printf("  s[1]    = %.6e (largest)\n", s[1])
    @printf("  s[%3d]  = %.6e (target cutoff)\n", k, s[k])
    @printf("  s[%3d]  = %.6e (first neglected)\n", k+1, s[k+1])
    @printf("  s[%3d]  = %.6e (smallest)\n", n, s[n])

    # Test different extra_samples and power_iter values
    extra_samples_list = [24, 18, 12, 6, 3]
    power_iter_list = 0:6

    # Store results for summary
    errors = zeros(length(extra_samples_list), length(power_iter_list))
    sval_errors = zeros(length(extra_samples_list), length(power_iter_list))

    for (idx, extra_samples) in enumerate(extra_samples_list)
        println("\n" * "="^70)
        @printf("extra_samples = %d\n", extra_samples)
        println("="^70)

        for (jdx, power_iter) in enumerate(power_iter_list)
            @printf("\n--- power_iter = %d ---\n", power_iter)

            # Run svd_sketch
            t_start = time()
            U_sketch, s_sketch, Vt_sketch = svd_sketch(A, Float64(k);
                power_iter=power_iter, extra_samples=extra_samples)
            t_total = time() - t_start

            # Compute reconstruction error
            A_approx = U_sketch * Diagonal(s_sketch) * Vt_sketch
            err = norm(A - A_approx, 2) / norm(A, 2)
            errors[idx, jdx] = err

            # Singular value accuracy
            s_ref = s[1:k]
            sval_err = norm(s_sketch - s_ref) / norm(s_ref)
            sval_errors[idx, jdx] = sval_err

            @printf("  Rank:         k = %d\n", length(s_sketch))
            @printf("  Error:        %.12e\n", err)
            @printf("  SVal error:   %.12e\n", sval_err)
            @printf("  Time:         %.6fs\n", t_total)

            # Sanity check
            @assert length(s_sketch) == k "Expected rank $k, got $(length(s_sketch))"
        end
    end

    # Print matrix info before summary
    println("\n" * "="^70)
    if use_random_matrix
        println("Matrix type: RANDOM")
        println("Singular values: from SVD of randn(m,n)")
    else
        println("Matrix type: STRUCTURED")
        println("Singular values: logspace(0,-2,k) + logspace(-2,-10,n-k)")
    end
    @printf("Matrix: %dx%d, target rank: %d\n", m, n, k)
    @printf("Condition number: %.2e, Spectral gap: %.2fx\n", cond_number, spectral_gap_k)

    # Print summary table for reconstruction error
    println("\n" * "="^70)
    println("SUMMARY: Reconstruction error (Frobenius norm)")
    println("="^70)
    print("extra_samples |")
    for power_iter in power_iter_list
        @printf("  iter=%d  |", power_iter)
    end
    println()
    print("--------------+")
    for _ in power_iter_list
        print("----------+")
    end
    println()
    for (idx, extra_samples) in enumerate(extra_samples_list)
        @printf("%13d |", extra_samples)
        for jdx in 1:length(power_iter_list)
            @printf(" %.2e |", errors[idx, jdx])
        end
        println()
    end

    # Print summary table for singular value accuracy
    println("\n" * "="^70)
    println("SUMMARY: Singular value error (relative 2-norm)")
    println("="^70)
    print("extra_samples |")
    for power_iter in power_iter_list
        @printf("  iter=%d  |", power_iter)
    end
    println()
    print("--------------+")
    for _ in power_iter_list
        print("----------+")
    end
    println()
    for (idx, extra_samples) in enumerate(extra_samples_list)
        @printf("%13d |", extra_samples)
        for jdx in 1:length(power_iter_list)
            @printf(" %.2e |", sval_errors[idx, jdx])
        end
        println()
    end

    println("\n[PASS] Test 1 complete")
end


"""
    main(test_random::Bool=false)

Run power iteration in svd_sketch tests.
"""
function main(test_random::Bool=false)
    println("="^70)
    println("POWER ITERATION IN SVD_SKETCH TESTS (librla)")
    println("="^70)

    Random.seed!(42)  # For reproducibility

    # Test 1: Power iteration in svd_sketch
    if test_random
        # Run with structured matrix first
        test_svd_sketch_power_iter(false)
        # Then run with random matrix
        test_svd_sketch_power_iter(true)
    else
        # Default: structured matrix only
        test_svd_sketch_power_iter(false)
    end

    println("\n" * "="^70)
    println("ALL TESTS PASSED [PASS]")
    println("="^70)
end


# Run test
test_random = length(ARGS) > 0 && lowercase(ARGS[1]) == "random"
main(test_random)
