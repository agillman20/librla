#!/usr/bin/env julia
#
# test_libid.jl - Test and demonstrate libid interpolative decomposition
#
#   Quick test script to validate LibIDSketch.id_sketch and LibIDRRQR.id_rrqr
#   implementations. Runs several test cases and prints results.
#
#   Usage:
#       julia test_libid.jl
#
#   Author : Your Name
#   SPDX-License-Identifier : TBD
#

using LinearAlgebra
using Printf
using Random

include("LibIDSketch.jl")
include("LibIDRRQR.jl")

using .LibIDSketch
using .LibIDRRQR


"""
    test_libid()

Run all tests.
"""
function test_libid()
    # JIT warm-up: run small tests to compile both real and complex paths
    println("JIT warm-up...")
    A_warmup = randn(10, 8)
    LibIDSketch.id_sketch(A_warmup; rtol=5)
    LibIDRRQR.id_rrqr(A_warmup; rtol=5)
    # Complex warm-up
    A_warmup_complex = randn(ComplexF64, 10, 8)
    LibIDSketch.id_sketch(A_warmup_complex; rtol=5)
    LibIDRRQR.id_rrqr(A_warmup_complex; rtol=5)
    println("Warm-up complete.\n")

    println("="^65)
    println("Testing libid Interpolative Decomposition")
    println("="^65)
    println()

    # Test 1: Random matrix
    println("Test 1: Random Matrix (500x300, rank=20)")
    println("-"^65)
    A1 = randn(500, 300)
    run_test(A1, 20, "Random 500x300")

    # Test 2: Low-rank matrix
    println("\nTest 2: Low-Rank Matrix (400x250, true rank~15)")
    println("-"^65)
    U = randn(400, 15)
    V = randn(250, 15)
    A2 = U * V' + 1e-10 * randn(400, 250)
    run_test(A2, 1e-8, "Low-Rank 400x250")

    # Test 3: Hilbert matrix (ill-conditioned)
    println("\nTest 3: Hilbert Matrix (200x100, ill-conditioned)")
    println("-"^65)
    A3 = hilb(200, 100)
    run_test(A3, 15, "Hilbert 200x100")

    # Test 4: Complex matrix
    println("\nTest 4: Complex Matrix (300x200, rank=25)")
    println("-"^65)
    A4 = randn(ComplexF64, 300, 200)
    run_test(A4, 25, "Complex 300x200")

    println("\n" * "="^65)
    println("All tests completed!")
    println("="^65)
end


"""
    hilb(m, n)

Generate an mxn Hilbert matrix.
"""
function hilb(m::Int, n::Int)
    H = zeros(m, n)
    for j = 1:n
        for i = 1:m
            H[i, j] = 1.0 / (i + j - 1)
        end
    end
    return H
end


"""
    run_test(A, rtol, name)

Helper function to run both methods and compare results.

# Arguments
- `A::AbstractMatrix`: Input matrix to decompose
- `rtol::Union{Float64,Int}`: Tolerance or target rank
- `name::String`: Descriptive name for this test
"""
function run_test(A::AbstractMatrix, rtol::Union{Float64,Int}, name::String)
    m, n = size(A)
    normA = norm(A, 2)

    # Test libid randomized version
    t_libid = @elapsed begin
        k_libid, piv_libid, T_libid = LibIDSketch.id_sketch(A; rtol=rtol)
    end

    # Compute error
    A_skel_libid = A[:, piv_libid[k_libid+1:end]]
    A_basis_libid = A[:, piv_libid[1:k_libid]]
    if !isempty(T_libid)
        err_libid = norm(A_skel_libid - A_basis_libid * T_libid, 2) / normA
        max_T_libid = maximum(abs.(T_libid))
    else
        err_libid = 0.0
        max_T_libid = 0.0
    end

    # Test RRQR deterministic version
    t_rrqr = @elapsed begin
        k_rrqr, piv_rrqr, T_rrqr = LibIDRRQR.id_rrqr(A; rtol=rtol)
    end

    # Compute error
    A_skel_rrqr = A[:, piv_rrqr[k_rrqr+1:end]]
    A_basis_rrqr = A[:, piv_rrqr[1:k_rrqr]]
    if !isempty(T_rrqr)
        err_rrqr = norm(A_skel_rrqr - A_basis_rrqr * T_rrqr, 2) / normA
        max_T_rrqr = maximum(abs.(T_rrqr))
    else
        err_rrqr = 0.0
        max_T_rrqr = 0.0
    end

    # Print results
    complex_str = eltype(A) <: Complex ? ", complex" : ""
    @printf("Matrix: %s (%dx%d%s)\n", name, m, n, complex_str)
    @printf("Target: rtol = %g\n\n", rtol)

    @printf("%-20s %-10s %-15s %-15s %-10s\n", "Method", "Rank", "Error", "max|T|", "Time (s)")
    @printf("%-20s %-10s %-15s %-15s %-10s\n", "-"^20, "-"^10, "-"^15, "-"^15, "-"^10)
    @printf("%-20s %-10d %-15.3e %-15.3e %-10.4f\n", "libid (randomized)", k_libid, err_libid, max_T_libid, t_libid)
    @printf("%-20s %-10d %-15.3e %-15.3e %-10.4f\n", "RRQR (deterministic)", k_rrqr, err_rrqr, max_T_rrqr, t_rrqr)

    # Compare ranks
    if k_libid == k_rrqr
        @printf("\n[OK] Ranks match (k=%d)\n", k_libid)
    else
        @printf("\n[WARNING] Different ranks: libid=%d, RRQR=%d (difference=%d)\n", k_libid, k_rrqr, abs(k_libid - k_rrqr))
    end

    # Compare speeds
    if t_libid < t_rrqr
        @printf("[OK] libid %.2fx faster than RRQR\n", t_rrqr/t_libid)
    else
        @printf("[WARNING] RRQR %.2fx faster than libid\n", t_libid/t_rrqr)
    end
end


# Run tests if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)  # For reproducibility
    test_libid()
end
