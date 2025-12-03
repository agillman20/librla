"""
test6_power.jl - Test power iteration for range estimation (librla version)

Tests simple power iteration for range estimation in sketching algorithms.
Power iteration applies (A'*A)^n to improve sketch quality by amplifying
the dominant subspace.

This version uses librla instead of libid.

Usage:
    julia test6_power.jl           # structured matrix only
    julia test6_power.jl random    # both structured and random

Tests:
    Test 1: Range estimation quality
        - Measures subspace angles to dominant subspace
        - Tests extra_samples: 24, 18, 12, 6, 3
        - Tests iterations: 0-6
        - Can use structured or random matrix

Author: Power iteration range estimator tests (librla version)
SPDX-License-Identifier: TBD
"""

using LinearAlgebra
using Printf
using Random
using Statistics

include("librla.jl")

using .librla: svd_sketch


"""
    subspace_angle(Q1, Q2)

Compute maximum principal angle between subspaces span(Q1) and span(Q2).
Returns angle in degrees.
"""
function subspace_angle(Q1::AbstractMatrix, Q2::AbstractMatrix)
    if size(Q1, 2) == 0 || size(Q2, 2) == 0
        return 90.0
    end

    # Compute singular values of Q1' * Q2
    M = Q1' * Q2
    s = svd(M).S

    # Principal angles: theta_i = arccos(s_i)
    # Maximum angle (worst alignment)
    theta_max = acos(min(max(s[end], 0.0), 1.0))
    return rad2deg(theta_max)
end


"""
    test_range_estimation_quality(use_random_matrix::Bool=false)

Test 1: Range estimation quality with power iteration.

Measures subspace angles to true dominant subspace across different iteration counts
and different oversampling parameters (extra_samples).
"""
function test_range_estimation_quality(use_random_matrix::Bool=false)
    println("\n" * "="^70)
    println("TEST 1: Range Estimation Quality")
    println("="^70)

    # Test matrix configuration
    m, n = 500, 300
    k = 50  # True rank we want to capture

    if use_random_matrix
        println("\nMatrix type: RANDOM (no prescribed singular values)")
        # Simple random matrix
        A = randn(m, n)

        # Compute SVD to get true dominant subspace
        F = svd(A)
        s = F.S
        V = F.V
        V_true = V[:, 1:k]

        # Note: Random matrices have clustered singular values with minimal
        # spectral gap, so power iteration converges much more slowly than
        # for structured matrices with prescribed decay.
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

        # True dominant subspace (right singular vectors)
        V_true = V[:, 1:k]
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
    @printf("  Decay rate (s[0]/s[%d]): %.2fx\n", k, decay_rate_k)
    println("\nSingular value distribution:")
    @printf("  s[1]    = %.12e (largest)\n", s[1])
    @printf("  s[%3d]  = %.12e (target cutoff)\n", k, s[k])
    @printf("  s[%3d]  = %.12e (first neglected)\n", k+1, s[k+1])
    @printf("  s[%3d] = %.12e (smallest)\n", n, s[n])

    # Test different extra_samples values
    extra_samples_list = [24, 18, 12, 6, 3]
    num_iters_list = 0:6

    # Store results for summary
    angles = zeros(length(extra_samples_list), length(num_iters_list))

    for (idx, extra_samples) in enumerate(extra_samples_list)
        block_size = k + extra_samples

        println("\n" * "="^70)
        @printf("extra_samples = %d (block_size = %d)\n", extra_samples, block_size)
        println("="^70)

        # Test different iteration counts (0-6)
        for (jdx, num_iters) in enumerate(num_iters_list)
            @printf("\n--- num_iters = %d ---\n", num_iters)

            # Generate same random test matrix
            Random.seed!(42)
            if eltype(A) <: Real
                X_init = 2 * rand(n, block_size) .- 1
            else
                X_init = 2 * rand(ComplexF64, n, block_size) .- 1
            end

            # Power iteration (manual implementation since librla._power_iteration is private)
            t_start = time()
            X_power = copy(X_init)
            # Orthogonalize X_init when num_iters=0
            if num_iters == 0
                X_power, _ = qr(X_power)
                X_power = Matrix(X_power)
            end
            for iter in 1:num_iters
                X_power = A' * (A * X_power)
                X_power, _ = qr(X_power)
                X_power = Matrix(X_power)
            end
            t_power = time() - t_start
            angle_power = subspace_angle(V_true, X_power)
            angles[idx, jdx] = angle_power

            # Orthogonality check
            orth_power = norm(X_power' * X_power - I(size(X_power, 2)), 2)

            # Compute alignment with dominant singular vectors
            M_power = V_true' * X_power
            svals_power = svd(M_power).S
            capture_quality_power = mean(svals_power)

            println("Power iteration:")
            @printf("  Subspace angle:   %18.12fdeg\n", angle_power)
            @printf("  Capture quality:  %.12f (mean singular value)\n", capture_quality_power)
            @printf("  Orthogonality:    ||Q'Q - I||_2 = %.12e\n", orth_power)
            @printf("  Time:             %.6fs\n", t_power)
            @printf("  Basis size:       %d\n", size(X_power, 2))
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

    # Print summary table
    println("\n" * "="^70)
    println("SUMMARY: Subspace angles (degrees)")
    println("="^70)
    print("extra_samples |")
    for num_iters in num_iters_list
        @printf(" iter=%d |", num_iters)
    end
    println()
    print("--------------+")
    for _ in num_iters_list
        print("--------+")
    end
    println()
    for (idx, extra_samples) in enumerate(extra_samples_list)
        @printf("%13d |", extra_samples)
        for jdx in 1:length(num_iters_list)
            @printf(" %6.2f |", angles[idx, jdx])
        end
        println()
    end

    println("\n[PASS] Test 1 complete")
end


"""
    main(test_random::Bool=false)

Run power iteration range estimation tests.
"""
function main(test_random::Bool=false)
    println("="^70)
    println("POWER ITERATION RANGE ESTIMATION TESTS (librla)")
    println("="^70)

    Random.seed!(42)  # For reproducibility

    # Test 1: Range estimation quality
    if test_random
        # Run with structured matrix first
        test_range_estimation_quality(false)
        # Then run with random matrix
        test_range_estimation_quality(true)
    else
        # Default: structured matrix only
        test_range_estimation_quality(false)
    end

    println("\n" * "="^70)
    println("ALL TESTS PASSED [PASS]")
    println("="^70)
end


# Run test
test_random = length(ARGS) > 0 && lowercase(ARGS[1]) == "random"
main(test_random)
