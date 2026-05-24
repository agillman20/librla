# test_linop.jl - LinearOperator, wide-matrix, and method="svd" regression tests
#
# Covers code paths not exercised by test_id/test_svd/test_qr/test_orth:
#   - Explicit LinearOperator (from_matrix — .matrix attribute attached)
#   - Matrix-free LinearOperator (matvec/rmatvec closures only)
#   - Wide matrices (m < n), which trigger svd_sketch's transpose branch
#   - id_sketch / id_qrpiv with method="svd" and method="lstsq"
#
# Usage:
#     julia test_linop.jl
#
# Author: Adrianna Gillman, Zydrunas Gimbutas
# SPDX-License-Identifier: MIT
# Version: 1.0.1
# Date: April 22, 2026
# Assisted by: Claude Code (Anthropic)

module TestLinop

using LinearAlgebra
using Printf
using Random

include(joinpath(@__DIR__, "..", "librla.jl"))

using .librla: orth_sketch, qr_sketch, svd_sketch, id_sketch, id_qrpiv,
               LinearOperator, from_matrix

export test


function wrap_matfree(M)
    m, n = size(M)
    T = eltype(M)
    return LinearOperator(x -> M * x, x -> M' * x, m, n;
                          dtype=T, is_complex=(T <: Complex))
end


function check_orth(M, k, label, errors)
    for (variant, A) in [("dense", M),
                         ("explicit", from_matrix(M)),
                         ("matfree",  wrap_matfree(M))]
        Q, flag, _ = orth_sketch(A, Float64(k))
        ortho = size(Q, 2) > 0 ? norm(Q' * Q - I) : 0.0
        ok = (flag == 0) && (size(Q, 2) == k) && (ortho < 1e-10)
        status = ok ? "PASS" : "FAIL"
        @printf("  [%s] orth_sketch %-26s %-9s flag=%d ortho=%.1e k=%d\n",
                status, label, variant, flag, ortho, size(Q, 2))
        ok || push!(errors, "orth_sketch $label $variant")
    end
end


function check_qr(M, k, label, errors)
    normM = norm(M)
    for (variant, A) in [("dense", M),
                         ("explicit", from_matrix(M)),
                         ("matfree",  wrap_matfree(M))]
        Q, R, p = qr_sketch(A, Float64(k))
        err = norm(M[:, p] - Q * R) / normM
        ortho = norm(Q' * Q - I)
        ok = (size(Q, 2) == k) && (ortho < 1e-10)
        status = ok ? "PASS" : "FAIL"
        @printf("  [%s] qr_sketch   %-26s %-9s err=%.2e ortho=%.1e k=%d\n",
                status, label, variant, err, ortho, size(Q, 2))
        ok || push!(errors, "qr_sketch $label $variant")
    end
end


function check_svd(M, k, label, errors)
    normM = norm(M)
    s_true = svdvals(M)
    err_opt = k < length(s_true) ? norm(s_true[(k+1):end]) / normM : 0.0
    tol = max(4.0 * err_opt, 1e-10)
    for (variant, A) in [("dense", M),
                         ("explicit", from_matrix(M)),
                         ("matfree",  wrap_matfree(M))]
        U, s, Vt = svd_sketch(A, Float64(k))
        err = norm(M - U * Diagonal(s) * Vt) / normM
        ortho_U = norm(U' * U - I)
        ortho_V = norm(Vt * Vt' - I)
        ok = (err < tol) && (ortho_U < 1e-10) && (ortho_V < 1e-10) && (length(s) == k)
        status = ok ? "PASS" : "FAIL"
        @printf("  [%s] svd_sketch  %-26s %-9s err=%.2e opt=%.2e orthU=%.1e k=%d\n",
                status, label, variant, err, err_opt, ortho_U, length(s))
        ok || push!(errors, "svd_sketch $label $variant")
    end
end


function check_id(M, k, label, errors)
    normM = norm(M)
    m, n = size(M)
    for (fn_name, fn) in [("id_sketch", id_sketch), ("id_qrpiv", id_qrpiv)]
        for method in ["fast", "svd", "lstsq"]
            kk, piv, T = fn(M, Float64(k); method=method)
            err = if !isempty(T) && kk < n
                A_skel = M[:, piv[(kk+1):end]]
                A_basis = M[:, piv[1:kk]]
                norm(A_skel - A_basis * T) / normM
            else
                0.0
            end
            ok = (kk == k) && (err < 1.5) && (size(T) == (k, n - k))
            status = ok ? "PASS" : "FAIL"
            @printf("  [%s] %-10s method=%-6s %-20s err=%.2e k=%d\n",
                    status, fn_name, method, label, err, kk)
            ok || push!(errors, "$fn_name method=$method $label")
        end
    end
end


function check_rng_kwarg(errors)
    # Seeded rng=... must produce deterministic output across calls, independent
    # of the global RNG state.
    M = randn(200, 60)
    rng1 = MersenneTwister(2024)
    rng2 = MersenneTwister(2024)
    rng3 = MersenneTwister(99)

    # Mutate global RNG between calls to prove rng= is fully isolated.
    U1, s1, Vt1 = svd_sketch(M, 10.0; rng=rng1)
    Random.seed!(0); randn(10_000)
    U2, s2, Vt2 = svd_sketch(M, 10.0; rng=rng2)
    U3, s3, Vt3 = svd_sketch(M, 10.0; rng=rng3)

    same_seed_match = isapprox(s1, s2; atol=1e-12) &&
                      isapprox(abs.(U1), abs.(U2); atol=1e-12) &&
                      isapprox(abs.(Vt1), abs.(Vt2); atol=1e-12)
    diff_seed_differs = !isapprox(s1, s3; atol=1e-6)

    status = (same_seed_match && diff_seed_differs) ? "PASS" : "FAIL"
    println("\n--- rng= kwarg determinism ---")
    @printf("  [%s] same_seed_match=%s diff_seed_differs=%s\n",
            status, same_seed_match, diff_seed_differs)
    if !(same_seed_match && diff_seed_differs)
        push!(errors, "rng= kwarg: same_seed=$same_seed_match, diff_seed=$diff_seed_differs")
    end
end


function test()
    Random.seed!(17)
    errors = String[]

    println("="^72)
    println("LINEAROPERATOR / WIDE-MATRIX / METHOD REGRESSION TESTS")
    println("="^72)

    shapes = [
        (200, 100,  8, "tall real  200x100", Float64),
        (100, 200,  8, "wide real  100x200", Float64),
        (150,  80,  6, "tall cplx  150x80",  ComplexF64),
        ( 80, 150,  6, "wide cplx   80x150", ComplexF64),
    ]

    for (m, n, k, label, T) in shapes
        M = T <: Complex ? randn(T, m, n) : randn(T, m, n)
        println("\n--- $label ---")
        check_orth(M, k, label, errors)
        check_qr(M, k, label, errors)
        check_svd(M, k, label, errors)
        if m >= n
            check_id(M, k, label, errors)
        end
    end

    check_rng_kwarg(errors)

    println()
    println("="^72)
    if !isempty(errors)
        println("[FAIL] $(length(errors)) check(s) failed:")
        for e in errors
            println("   - $e")
        end
        println("="^72)
        return 1
    end
    println("[PASS] All LinearOperator / wide / method regression checks passed.")
    println("="^72)
    return 0
end

end  # module TestLinop


# Run when invoked as a script
if abspath(PROGRAM_FILE) == @__FILE__
    exit(TestLinop.test())
end
