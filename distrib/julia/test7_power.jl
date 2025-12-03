"""
test7_power.jl - Test power iteration in svd_sketch (librla version)

Tests power iteration in the svd_sketch function.
Power iteration applies (A'*A)^n to improve sketch quality by amplifying
the dominant subspace.

This version uses librla instead of libid.

Usage:
    julia test7_power.jl

Tests:
    Test 1: Power iteration in svd_sketch
        - Tests power_iter: 0-6
        - Measures reconstruction error and singular value accuracy

Author: Power iteration tests (librla version)
SPDX-License-Identifier: TBD
"""

using LinearAlgebra
using Printf
using Random

include("librla.jl")

using .librla: svd_sketch


"""
    test_svd_sketch_power_iter()

Test 1: Power iteration in svd_sketch.

Tests how power iteration improves singular value accuracy in svd_sketch.
"""
function test_svd_sketch_power_iter()
    println("\n" * "="^70)
    println("TEST 1: Power iteration in svd_sketch")
    println("="^70)

    # Test matrix with prescribed singular values
    m, n = 350, 200
    k = 40

    U_full = qr(randn(m, m)).Q
    V_full = qr(randn(n, n)).Q
    s = exp10.(range(0, -6, length=n))
    # Use first n columns of U to match dimensions
    U = Matrix(U_full)[:, 1:n]
    V = Matrix(V_full)
    A = U * Diagonal(s) * V'

    @printf("\nMatrix: %dx%d, target rank: %d\n", m, n, k)
    @printf("True singular values: s[1]=%.12e, s[%d]=%.12e\n", s[1], k+1, s[k+1])

    # Test with different power_iter values (extended to 6)
    for power_iter in 0:6
        @printf("\n--- power_iter = %d ---\n", power_iter)

        # Run svd_sketch
        t_start = time()
        U_sketch, s_sketch, Vt_sketch = svd_sketch(A, Float64(k); power_iter=power_iter)
        t_total = time() - t_start

        # Compute reconstruction error
        A_approx = U_sketch * Diagonal(s_sketch) * Vt_sketch
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

    println("\n[PASS] Test 1 complete")
end


"""
    main()

Run power iteration in svd_sketch tests.
"""
function main()
    println("="^70)
    println("POWER ITERATION IN SVD_SKETCH TESTS (librla)")
    println("="^70)

    Random.seed!(42)  # For reproducibility

    # Test 1: Power iteration in svd_sketch
    test_svd_sketch_power_iter()

    println("\n" * "="^70)
    println("ALL TESTS PASSED [PASS]")
    println("="^70)
end


# Run test
main()
