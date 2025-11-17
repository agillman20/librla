"""
test1_hilbert.jl - Simple test with medium-size Hilbert matrix

Tests basic ID algorithms (id_sketch, id_rrqr) on an ill-conditioned
Hilbert matrix.
"""

using LinearAlgebra
using Printf

include("LibIDSketch.jl")
include("LibIDRRQR.jl")

using .LibIDSketch: id_sketch
using .LibIDRRQR: id_rrqr


"""
    hilb(m::Int, n::Int) -> Matrix{Float64}

Generate an m x n Hilbert matrix.

The Hilbert matrix is extremely ill-conditioned; it is useful for
testing numerical algorithms.
"""
function hilb(m::Int, n::Int)
    i = reshape(1:m, m, 1)
    j = reshape(1:n, 1, n)
    return 1.0 ./ (i .+ j .- 1)
end


function test_hilbert()
    """Test basic ID algorithms on Hilbert matrix."""

    println("="^70)
    println("TEST 1: Medium Hilbert Matrix")
    println("="^70)

    # Create medium-size Hilbert matrix (severely ill-conditioned)
    m, n = 300, 200
    @printf("\nMatrix size: %d x %d\n", m, n)
    println("Matrix type: Hilbert (severely ill-conditioned)")

    A = hilb(m, n)
    normA = norm(A)

    # Target rank
    k_target = 15
    @printf("Target rank: %d\n", k_target)

    # -------------------------------------------------------------------------
    # JIT warm-up: compile all methods before timing
    # -------------------------------------------------------------------------
    println("\nJIT warm-up (compiling methods)...")
    A_warmup = randn(50, 30)
    try
        id_sketch(A_warmup, rtol=10.0)
        id_rrqr(A_warmup, rtol=10.0)
    catch
        # Ignore warm-up errors
    end
    println("JIT warm-up complete.")
    println("="^70)

    # -------------------------------------------------------------------------
    # Method 1: id_sketch (randomized)
    # -------------------------------------------------------------------------
    println("\n1. LibIDSketch.id_sketch (randomized QR sketching)")
    println("-"^70)

    t1 = @elapsed k1, piv1, T1 = id_sketch(A, rtol=Float64(k_target))

    # Compute error
    A_skel1 = A[:, piv1[k1+1:end]]
    A_basis1 = A[:, piv1[1:k1]]
    if !isempty(T1)
        err1 = norm(A_skel1 - A_basis1 * T1) / normA
        maxT1 = maximum(abs.(T1))
    else
        err1 = 0.0
        maxT1 = 0.0
    end

    @printf("  Rank:      k = %d\n", k1)
    @printf("  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n", err1)
    @printf("  Max |T|:   %.3e\n", maxT1)
    @printf("  Time:      %.4f s\n", t1)

    # -------------------------------------------------------------------------
    # Method 2: id_rrqr (deterministic RRQR)
    # -------------------------------------------------------------------------
    println("\n2. LibIDRRQR.id_rrqr (deterministic RRQR via LAPACK)")
    println("-"^70)

    t2 = @elapsed k2, piv2, T2 = id_rrqr(A, rtol=Float64(k_target))

    # Compute error
    A_skel2 = A[:, piv2[k2+1:end]]
    A_basis2 = A[:, piv2[1:k2]]
    if !isempty(T2)
        err2 = norm(A_skel2 - A_basis2 * T2) / normA
        maxT2 = maximum(abs.(T2))
    else
        err2 = 0.0
        maxT2 = 0.0
    end

    @printf("  Rank:      k = %d\n", k2)
    @printf("  Error:     ||A_skel - A_basis @ T|| / ||A|| = %.3e\n", err2)
    @printf("  Max |T|:   %.3e\n", maxT2)
    @printf("  Time:      %.4f s\n", t2)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("  Method         Rank    Error        Max|T|       Time")
    println("-"^70)
    @printf("  id_sketch      %4d    %.3e    %.3e    %.4fs\n", k1, err1, maxT1, t1)
    @printf("  id_rrqr        %4d    %.3e    %.3e    %.4fs\n", k2, err2, maxT2, t2)
    println("="^70)

    # Validate
    if err1 > 1.0 || err2 > 1.0
        println("\n[FAIL] Error > 1.0 detected!")
        return false
    end

    println("\n[PASS] Test completed successfully!")
    return true
end


# Run test
success = test_hilbert()
exit(success ? 0 : 1)
