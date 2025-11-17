"""
compare_svd - Compare SVD implementations

Comprehensive comparison of two SVD implementations:
- LibIDSketch.svd_sketch:       Randomized SVD via sketching
- svd (LAPACK):                  Deterministic full SVD (truncated)

Compares on metrics:
- Accuracy (reconstruction error)
- Singular value accuracy
- Runtime

Author: Port from compare_svd.py
"""

using LinearAlgebra
using Random
using Printf
using Statistics

include("LibIDSketch.jl")
include("make_mat.jl")
include("make_linop.jl")

using .LibIDSketch

# Test result structure
struct SVDComparisonResult
    name::String
    rtol_or_rank::Float64

    # Rank for each method
    k_sketch::Int
    k_svd::Int

    # Reconstruction error
    err_sketch::Float64
    err_svd::Float64

    # Runtime
    t_sketch::Float64
    t_svd::Float64

    # Singular value accuracy
    sval_err_sketch::Float64
    sval_err_svd::Float64

    passed::Bool
end

"""
    hilb(m::Int, n::Int) -> Matrix{Float64}

Generate an mxn Hilbert matrix.

The Hilbert matrix is extremely ill-conditioned; it is useful for
testing numerical algorithms.

# Arguments
- `m::Int`: Number of rows
- `n::Int`: Number of columns

# Returns
- `H::Matrix{Float64}`: Hilbert matrix with entries H[i,j] = 1/(i+j-1)
"""
function hilb(m::Int, n::Int)
    i = reshape(1:m, m, 1)
    j = reshape(1:n, 1, n)
    return 1.0 ./ (i .+ j .- 1)
end

"""
    compare_on_matrix(A, rtol_or_rank, name, flag_power=0)

Compare SVD implementations on a single matrix.

# Arguments
- `A`: Input matrix to decompose (real or complex)
- `rtol_or_rank`: Tolerance (<1) or rank (>=1)
- `name`: Descriptive name for the test
- `flag_power`: Number of power iterations (default: 0)
"""
function compare_on_matrix(A, rtol_or_rank, name, flag_power=0)
    println("\nTesting: ", name)
    m, n = size(A)
    if flag_power > 0
        @printf("  Matrix shape: (%d, %d), rtol_or_rank=%g, flag_power=%d\n", m, n, rtol_or_rank, flag_power)
    else
        @printf("  Matrix shape: (%d, %d), rtol_or_rank=%g\n", m, n, rtol_or_rank)
    end

    is_rank_mode = (rtol_or_rank >= 1)
    normA = norm(A)

    # Reference: Full SVD for singular value comparison
    F_ref = svd(A)
    s_ref = F_ref.S

    # -------------------------------------------------------------------------
    # Method 1: svd_sketch (randomized)
    # -------------------------------------------------------------------------
    t0 = time()
    U1, s1, Vh1 = svd_sketch(A, rtol=rtol_or_rank, flag_power=flag_power)
    t_sketch = time() - t0
    k_sketch = length(s1)

    # Reconstruction error (Vh is already conjugate transposed)
    A1_recon = U1 * Diagonal(s1) * Vh1
    err_sketch = norm(A - A1_recon) / normA

    # Singular value accuracy
    s1_ref = s_ref[1:k_sketch]
    sval_err_sketch = norm(s1 - s1_ref) / norm(s1_ref)

    # -------------------------------------------------------------------------
    # Method 2: svd (LAPACK, deterministic)
    # -------------------------------------------------------------------------
    t0 = time()
    F2 = svd(A)
    t_svd = time() - t0
    s2 = F2.S

    # Truncate based on tolerance or rank
    if is_rank_mode
        k2 = min(Int(floor(rtol_or_rank)), length(s2))
    else
        k2 = sum(s2 .>= rtol_or_rank * s2[1])
        if k2 == 0
            k2 = 1  # At least one singular value
        end
    end
    k_svd = k2

    # Reconstruct with truncation
    U2_k = F2.U[:, 1:k2]
    s2_k = s2[1:k2]
    V2_k = F2.Vt[1:k2, :]
    A2_recon = U2_k * Diagonal(s2_k) * V2_k
    err_svd = norm(A - A2_recon) / normA

    # Singular value accuracy
    sval_err_svd = norm(s2_k - s_ref[1:k2]) / norm(s_ref[1:k2])

    # -------------------------------------------------------------------------
    # Display results
    # -------------------------------------------------------------------------
    @printf("  Ranks:  sketch=%d, svd=%d\n", k_sketch, k_svd)
    @printf("  Errors: sketch=%.3e, svd=%.3e\n", err_sketch, err_svd)
    @printf("  Times:  sketch=%.3fs, svd=%.3fs\n", t_sketch, t_svd)
    @printf("  SVal:   sketch=%.3e, svd=%.3e\n", sval_err_sketch, sval_err_svd)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    passed = true

    # Check reconstruction errors are reasonable
    if err_sketch > 1.0 || err_svd > 1.0
        println("  [ERROR] Reconstruction error > 1.0 (larger than input norm)")
        passed = false
    end

    # Check sketch is close to svd
    rel_err = abs(err_sketch - err_svd) / max(err_svd, 1e-15)
    if rel_err > 0.1  # 10% tolerance
        @printf("  [WARNING] sketch error differs from svd by %.1f%%\n", rel_err * 100)
    end

    # Build result struct
    return SVDComparisonResult(
        name, rtol_or_rank,
        k_sketch, k_svd,
        err_sketch, err_svd,
        t_sketch, t_svd,
        sval_err_sketch, sval_err_svd,
        passed
    )
end

"""
    print_summary(results)

Print comprehensive summary of all test results.
"""
function print_summary(results::Vector{SVDComparisonResult})
    println()
    println("="^70)
    println("SUMMARY")
    println("="^70)

    total_count = length(results)
    passed_count = count(r -> r.passed, results)

    println("\nTests: $passed_count/$total_count passed")

    # Timing statistics
    t_sketch_all = [r.t_sketch for r in results]
    t_svd_all = [r.t_svd for r in results]

    println("\nTiming Statistics:")
    @printf("  svd_sketch:  mean=%.3fs, min=%.3fs, max=%.3fs\n",
            mean(t_sketch_all), minimum(t_sketch_all), maximum(t_sketch_all))
    @printf("  LAPACK.svd:  mean=%.3fs, min=%.3fs, max=%.3fs\n",
            mean(t_svd_all), minimum(t_svd_all), maximum(t_svd_all))

    # Error statistics
    err_sketch_all = [r.err_sketch for r in results]
    err_svd_all = [r.err_svd for r in results]

    println("\nReconstruction Error Statistics:")
    @printf("  svd_sketch:  mean=%.3e, max=%.3e\n", mean(err_sketch_all), maximum(err_sketch_all))
    @printf("  LAPACK.svd:  mean=%.3e, max=%.3e\n", mean(err_svd_all), maximum(err_svd_all))

    # Singular value error statistics
    sval_err_sketch_all = [r.sval_err_sketch for r in results]
    sval_err_svd_all = [r.sval_err_svd for r in results]

    println("\nSingular Value Error Statistics:")
    @printf("  svd_sketch:  mean=%.3e, max=%.3e\n", mean(sval_err_sketch_all), maximum(sval_err_sketch_all))
    @printf("  LAPACK.svd:  mean=%.3e, max=%.3e\n", mean(sval_err_svd_all), maximum(sval_err_svd_all))

    println("="^70)
end

"""
    main()

Run comprehensive SVD comparison tests.
"""
function main()
    println("=" * "="^68 * "=")
    println("|" * " "^20 * "LIBID SVD COMPARISON TEST SUITE" * " "^16 * "|")
    println("=" * "="^68 * "=")

    results = SVDComparisonResult[]

    # JIT warm-up: compile all methods before timing
    println("\nJIT warm-up (compiling methods)...")
    A_warmup = randn(50, 30)
    try
        svd_sketch(A_warmup, rtol=20.0)
        svd(A_warmup)
        svds(A_warmup, nsv=5)
    catch
        # Ignore warm-up errors
    end
    println("JIT warm-up complete.")

    # -------------------------------------------------------------------------
    # Test 1: Random matrix (well-conditioned)
    # -------------------------------------------------------------------------
    Random.seed!(42)
    A1 = randn(400, 250)
    push!(results, compare_on_matrix(A1, 1e-8, "Random Matrix (well-conditioned)"))

    # -------------------------------------------------------------------------
    # Test 2: Low-rank matrix
    # -------------------------------------------------------------------------
    U = randn(400, 15)
    V = randn(250, 15)
    A2 = U * V' + 1e-10 * randn(400, 250)
    push!(results, compare_on_matrix(A2, 1e-8, "Low-Rank Matrix (rank~15)"))

    # -------------------------------------------------------------------------
    # Test 3: Hilbert matrix (severely ill-conditioned)
    # -------------------------------------------------------------------------
    A3 = hilb(2000, 1000)
    push!(results, compare_on_matrix(A3, 15, "Hilbert Matrix (severely ill-conditioned)"))

    # -------------------------------------------------------------------------
    # Test 4: Complex matrix
    # -------------------------------------------------------------------------
    A4 = randn(ComplexF64, 300, 200)
    push!(results, compare_on_matrix(A4, 1e-8, "Complex Matrix"))

    # -------------------------------------------------------------------------
    # Test 5: Power-law decay (slow decay)
    # -------------------------------------------------------------------------
    rank = 50
    s_decay = 1.0 ./ sqrt.(1:rank)  # s_k ~ 1/sqrt(k)
    U5, _ = qr(randn(300, rank))
    V5, _ = qr(randn(200, rank))
    A5 = U5 * Diagonal(s_decay) * V5'
    push!(results, compare_on_matrix(A5, 1e-6, "Power-Law Decay (slow)"))

    # -------------------------------------------------------------------------
    # Test 6: Rank mode test (fixed rank=20)
    # -------------------------------------------------------------------------
    A6 = randn(300, 200)
    push!(results, compare_on_matrix(A6, 20, "Rank Mode Test (k=20)"))

    # -------------------------------------------------------------------------
    # Test 7: Large low-rank matrix
    # -------------------------------------------------------------------------
    U7 = randn(800, 15)
    V7 = randn(500, 15)
    A7 = U7 * V7' + 1e-10 * randn(800, 500)
    push!(results, compare_on_matrix(A7, 1e-8, "Large Low-Rank Matrix (800x500, rank~15)"))

    # -------------------------------------------------------------------------
    # Test 8: Large Hilbert matrix
    # -------------------------------------------------------------------------
    A8 = hilb(4000, 2000)
    push!(results, compare_on_matrix(A8, 15, "Large Hilbert Matrix (4000x2000)"))

    # -------------------------------------------------------------------------
    # Test 9: Large complex matrix
    # -------------------------------------------------------------------------
    A9 = randn(ComplexF64, 600, 400)
    push!(results, compare_on_matrix(A9, 1e-8, "Large Complex Matrix"))

    # -------------------------------------------------------------------------
    # Test 10-12: Structured matrices
    # -------------------------------------------------------------------------
    for mat_type in ["gmm", "gaussexp", "snn"]
        A_struct = make_mat(400, 250, mat_type)
        push!(results, compare_on_matrix(A_struct, 1e-8, "Structured: $mat_type"))
    end

    # -------------------------------------------------------------------------
    # Test 13-14: Wide matrices
    # -------------------------------------------------------------------------
    A13 = randn(200, 500)
    push!(results, compare_on_matrix(A13, 1e-8, "Wide Random Matrix (200x500)"))

    U14 = randn(200, 15)
    V14 = randn(500, 15)
    A14 = U14 * V14' + 1e-10 * randn(200, 500)
    push!(results, compare_on_matrix(A14, 1e-8, "Wide Low-Rank Matrix (200x500, rank~15)"))

    # -------------------------------------------------------------------------
    # Test 15: XL Random matrix
    # -------------------------------------------------------------------------
    A15 = randn(1200, 800)
    push!(results, compare_on_matrix(A15, 1e-8, "XL Random Matrix (1200x800)"))

    # -------------------------------------------------------------------------
    # Test 16: XL Low-rank matrix
    # -------------------------------------------------------------------------
    U16 = randn(1600, 15)
    V16 = randn(1000, 15)
    A16 = U16 * V16' + 1e-10 * randn(1600, 1000)
    push!(results, compare_on_matrix(A16, 1e-8, "XL Low-Rank Matrix (1600x1000, rank~15)"))

    # -------------------------------------------------------------------------
    # Test 17: XL Hilbert matrix
    # -------------------------------------------------------------------------
    A17 = hilb(8000, 4000)
    push!(results, compare_on_matrix(A17, 15, "XL Hilbert Matrix (8000x4000)"))

    # -------------------------------------------------------------------------
    # Test 18: XL Complex matrix
    # -------------------------------------------------------------------------
    A18 = randn(ComplexF64, 1200, 800)
    push!(results, compare_on_matrix(A18, 1e-8, "XL Complex Matrix (1200x800)"))

    # =========================================================================
    # Power Iteration Tests (Rank Mode)
    # =========================================================================
    println("\n" * "="^70)
    println("POWER ITERATION TESTS (Rank Mode)")
    println("="^70)

    # Test matrix for power iteration tests
    A_power = randn(400, 300)
    target_rank = 30

    # Test 19: Power iteration = 0 (no power iteration)
    push!(results, compare_on_matrix(A_power, target_rank,
                                     "Rank Mode k=$target_rank, power=0", 0))

    # Test 20: Power iteration = 1
    push!(results, compare_on_matrix(A_power, target_rank,
                                     "Rank Mode k=$target_rank, power=1", 1))

    # Test 21: Power iteration = 2
    push!(results, compare_on_matrix(A_power, target_rank,
                                     "Rank Mode k=$target_rank, power=2", 2))

    # =========================================================================
    # Summary
    # =========================================================================
    print_summary(results)

    # Exit with appropriate code
    passed_count = count(r -> r.passed, results)
    total_count = length(results)

    if passed_count < total_count
        println("\n[FAIL] $(total_count - passed_count) tests failed")
        println("\nFailed tests:")
        for r in results
            if !r.passed
                println("  - $(r.name)")
            end
        end
        return 1
    else
        println("\n[PASS] ALL TESTS PASSED!")
        return 0
    end
end

# Run tests
main()
