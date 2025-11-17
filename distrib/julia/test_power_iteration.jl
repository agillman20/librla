#!/usr/bin/env julia
#
# test_power_iteration.jl - Test power iteration for range estimation
#
#   Tests simple power iteration for range estimation in sketching algorithms.
#   Power iteration applies (A^H A)^n to improve sketch quality by amplifying
#   the dominant subspace.
#
#   Usage:
#       julia test_power_iteration.jl           # structured matrix only
#       julia test_power_iteration.jl random    # both structured and random
#
#   Tests:
#       Test 1: Range estimation quality
#           - Measures subspace angles to dominant subspace
#           - Tests extra_samples: 24, 18, 12, 6, 3
#           - Tests iterations: 0-6
#           - Can use structured or random matrix
#
#       Test 3: SVD sketch integration
#           - Tests flag_power: 0-6
#           - Measures reconstruction error and singular value accuracy
#
#   Author : Power iteration range estimator tests
#   SPDX-License-Identifier : TBD
#

using LinearAlgebra
using Printf
using Random
using Statistics

include("LibIDSketch.jl")
using .LibIDSketch


"""
    test_power_iteration(test_random=false)

Run power iteration range estimation tests.

# Arguments
- `test_random::Bool`: If true, run Test 1 with both structured and random matrices
"""
function test_power_iteration(test_random::Bool=false)
    # JIT warm-up: run small tests to compile both real and complex paths
    println("JIT warm-up...")
    A_warmup = randn(50, 30)
    X_warmup = LibIDSketch._uniform_omega(A_warmup, 30, 10)
    LibIDSketch._power_iteration(A_warmup, X_warmup, 1)
    LibIDSketch.svd_sketch(A_warmup; rtol=10.0, flag_power=1)
    # Complex warm-up
    A_warmup_complex = randn(ComplexF64, 50, 30)
    X_warmup_complex = LibIDSketch._uniform_omega(A_warmup_complex, 30, 10)
    LibIDSketch._power_iteration(A_warmup_complex, X_warmup_complex, 1)
    LibIDSketch.svd_sketch(A_warmup_complex; rtol=10.0, flag_power=1)
    println("Warm-up complete.\n")

    println("="^70)
    println("POWER ITERATION RANGE ESTIMATION TESTS")
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

    # Test 3: SVD sketch integration
    test_svd_sketch_integration()

    println("\n" * "="^70)
    println("ALL TESTS PASSED [PASS]")
    println("="^70)
end


"""
    test_range_estimation_quality(use_random_matrix=false)

Test 1: Range estimation quality with power iteration.
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
        svd_result = svd(A)
        s = svd_result.S
        V = svd_result.V
        V_true = V[:, 1:k]

        # Note: Random matrices have clustered singular values with minimal
        # spectral gap, so power iteration converges much more slowly than
        # for structured matrices with prescribed decay.
    else
        println("\nMatrix type: STRUCTURED (prescribed singular values)")
        # Create matrix with decaying spectrum
        U_full = Matrix(qr(randn(m, m)).Q)
        V_full = Matrix(qr(randn(n, n)).Q)
        s = vcat(exp10.(range(0, -2, length=k)),
                 exp10.(range(-2, -10, length=n-k)))

        # Use first n columns of U to match dimensions
        U = U_full[:, 1:n]
        V = V_full
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
    @printf("  s[0]    = %.12e (largest)\n", s[1])
    @printf("  s[%3d]  = %.12e (target cutoff)\n", k, s[k])
    @printf("  s[%3d]  = %.12e (first neglected)\n", k+1, s[k+1])
    @printf("  s[%3d] = %.12e (smallest)\n", n, s[n])

    # Test different extra_samples values
    extra_samples_list = [24, 18, 12, 6, 3]
    for extra_samples in extra_samples_list
        block_size = k + extra_samples

        println("\n" * "="^70)
        @printf("extra_samples = %d (block_size = %d)\n", extra_samples, block_size)
        println("="^70)

        # Test different iteration counts (0-6)
        for num_iters = 0:6
            @printf("\n--- num_iters = %d ---\n", num_iters)

            # Generate same random test matrix
            Random.seed!(42)
            X_init = LibIDSketch._uniform_omega(A, n, block_size)

            # Power iteration
            t_power = @elapsed begin
                X_power = LibIDSketch._power_iteration(A, copy(X_init), num_iters)
            end
            angle_power = subspace_angle(V_true, X_power)

            # Orthogonality check
            orth_power = norm(X_power' * X_power - I(size(X_power, 2)), 2)

            # Compute alignment with dominant singular vectors
            M_power = V_true' * X_power
            svals_power = svdvals(M_power)
            capture_quality_power = mean(svals_power)

            println("Power iteration:")
            @printf("  Subspace angle:   %18.12fdeg\n", angle_power)
            @printf("  Capture quality:  %.12f (mean singular value)\n", capture_quality_power)
            @printf("  Orthogonality:    ||Q^H Q - I||_F = %.12e\n", orth_power)
            @printf("  Time:             %.6fs\n", t_power)
            @printf("  Basis size:       %d\n", size(X_power, 2))
        end
    end

    println("\n[PASS] Test 1 complete")
end


"""
    test_svd_sketch_integration()

Test 3: Power iteration integration with svd_sketch pipeline.
"""
function test_svd_sketch_integration()
    println("\n" * "="^70)
    println("TEST 3: Integration with svd_sketch")
    println("="^70)

    # Test matrix with prescribed singular values
    m, n = 350, 200
    k = 40

    U_full = Matrix(qr(randn(m, m)).Q)
    V_full = Matrix(qr(randn(n, n)).Q)
    s = exp10.(range(0, -6, length=n))
    # Use first n columns of U to match dimensions
    U = U_full[:, 1:n]
    V = V_full
    A = U * Diagonal(s) * V'

    @printf("\nMatrix: %dx%d, target rank: %d\n", m, n, k)
    @printf("True singular values: s[0]=%.12e, s[%d]=%.12e\n", s[1], k+1, s[k+1])

    # Test with different flag_power values (extended to 6)
    for flag_power = 0:6
        @printf("\n--- flag_power = %d ---\n", flag_power)

        # Run svd_sketch
        t_total = @elapsed begin
            U_sketch, s_sketch, Vh_sketch = LibIDSketch.svd_sketch(A; rtol=Float64(k), block_size=42, flag_power=flag_power)
        end

        # Compute reconstruction error
        A_approx = U_sketch * Diagonal(s_sketch) * Vh_sketch
        err = norm(A - A_approx, 2) / norm(A, 2)

        # Singular value accuracy
        s_ref = s[1:k]
        sval_err = norm(s_sketch - s_ref) / norm(s_ref)

        @printf("  Rank:         k = %d\n", length(s_sketch))
        @printf("  Error:        %.12e\n", err)
        @printf("  SVal error:   %.12e\n", sval_err)
        @printf("  Time:         %.6fs\n", t_total)

        # Sanity check
        @assert length(s_sketch) == k "Expected rank $k, got $(length(s_sketch))"
        @assert err < 0.1 "Error $err too large"
    end

    println("\n[PASS] Test 3 complete")
end


"""
    subspace_angle(Q1, Q2)

Compute maximum principal angle between subspaces span(Q1) and span(Q2).
Returns angle in degrees.
"""
function subspace_angle(Q1::AbstractMatrix, Q2::AbstractMatrix)
    if size(Q1, 2) == 0 || size(Q2, 2) == 0
        return 90.0
    end

    # Compute singular values of Q1^H @ Q2
    M = Q1' * Q2
    s = svdvals(M)

    # Principal angles: theta_i = arccos(s_i)
    # Maximum angle (worst alignment)
    theta_max = acos(clamp(s[end], 0, 1))
    return rad2deg(theta_max)
end


# Run tests if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    test_random = length(ARGS) > 0 && lowercase(ARGS[1]) == "random"
    test_power_iteration(test_random)
end
