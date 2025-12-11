"""
test2_svd_hilbert.jl - Simple test with medium-size Hilbert matrix for SVD

Tests SVD algorithms (svd_sketch vs standard SVD) on an ill-conditioned
Hilbert matrix.

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

using LinearAlgebra
using Printf

include("librla.jl")

using .librla: svd_sketch


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


function test_svd_hilbert()
    """Test SVD algorithms on Hilbert matrix."""

    println("="^70)
    println("TEST 2: Medium Hilbert Matrix - SVD")
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
        svd_sketch(A_warmup, 10.0)
        svd(A_warmup)
    catch
        # Ignore warm-up errors
    end
    println("JIT warm-up complete.")
    println("="^70)

    # Reference: Full SVD for singular value comparison
    F_ref = svd(A)
    s_ref = F_ref.S

    # -------------------------------------------------------------------------
    # Method 1: svd_sketch (randomized)
    # -------------------------------------------------------------------------
    println("\n1. librla.svd_sketch (randomized SVD via sketching)")
    println("-"^70)

    t1 = @elapsed U1, s1, Vh1 = svd_sketch(A, Float64(k_target))

    k1 = length(s1)

    # Reconstruction error (Vh is already conjugate transposed)
    A1_recon = U1 * Diagonal(s1) * Vh1
    err1 = norm(A - A1_recon) / normA

    # Singular value accuracy
    s1_ref = s_ref[1:k1]
    sval_err1 = norm(s1 - s1_ref) / norm(s1_ref)

    @printf("  Rank:      k = %d\n", k1)
    @printf("  Error:     ||A - U @ S @ Vh|| / ||A|| = %.3e\n", err1)
    @printf("  SVal Err:  ||s - s_ref|| / ||s_ref|| = %.3e\n", sval_err1)
    @printf("  Time:      %.4f s\n", t1)

    # -------------------------------------------------------------------------
    # Method 2: svd (LAPACK, truncated)
    # -------------------------------------------------------------------------
    println("\n2. svd (LAPACK, deterministic, truncated)")
    println("-"^70)

    t2 = @elapsed begin
        F2 = svd(A)
        s2_full = F2.S
    end

    # Truncate to target rank
    k2 = k_target
    U2_k = F2.U[:, 1:k2]
    s2 = s2_full[1:k2]
    Vt2_k = F2.Vt[1:k2, :]

    # Reconstruction error
    A2_recon = U2_k * Diagonal(s2) * Vt2_k
    err2 = norm(A - A2_recon) / normA

    # Singular value accuracy
    sval_err2 = norm(s2 - s_ref[1:k2]) / norm(s_ref[1:k2])

    @printf("  Rank:      k = %d\n", k2)
    @printf("  Error:     ||A - U @ S @ Vh|| / ||A|| = %.3e\n", err2)
    @printf("  SVal Err:  ||s - s_ref|| / ||s_ref|| = %.3e\n", sval_err2)
    @printf("  Time:      %.4f s\n", t2)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("  Method         Rank    Recon Error   SVal Error    Time")
    println("-"^70)
    @printf("  svd_sketch     %4d    %.3e    %.3e    %.4fs\n", k1, err1, sval_err1, t1)
    @printf("  svd (LAPACK)   %4d    %.3e    %.3e    %.4fs\n", k2, err2, sval_err2, t2)
    println("="^70)

    # Validate
    if err1 > 1.0 || err2 > 1.0
        println("\n[FAIL] Reconstruction error > 1.0 detected!")
        return false
    end

    if sval_err1 > 1e-6 || sval_err2 > 1e-10
        println("\n[FAIL] Singular value error too large!")
        return false
    end

    println("\n[PASS] Test completed successfully!")
    return true
end


# Run test
success = test_svd_hilbert()
exit(success ? 0 : 1)
