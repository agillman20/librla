"""
LibIDRRQR - Deterministic Interpolative Decomposition using RRQR

This module provides ID implementation using Julia's QR with column pivoting,
which internally calls LAPACK's dgeqp3/zgeqp3 (strong rank-revealing QR).

This serves as a comparison baseline to randomized methods, representing
what deterministic methods achieve.

## Key Features

- **Deterministic**: Same results every run
- **LAPACK-based**: Uses highly optimized BLAS3 routines
- **Simple**: Straightforward QR -> ID conversion
- **Accurate**: Strong rank-revealing guarantees

## Algorithm

1. Compute full pivoted QR: A[:, jpiv] = Q * R
2. Determine rank k from diagonal of R
3. Extract R11 (kxk) and R12 (kx(n-k))
4. Solve R11 * T = R12 for interpolation matrix T

## Comparison with Randomized Methods

**Advantages**:
- Deterministic (reproducible)
- Uses highly optimized LAPACK
- Well-tested algorithm

**Disadvantages**:
- Must compute FULL QR (O(mn^2))
- No early detection for full-rank
- No adaptive block sizing

## References

- LAPACK dgeqp3/zgeqp3 documentation
- Golub & Van Loan, "Matrix Computations", 4th ed.
- Gu & Eisenstat, "Efficient algorithms for computing a strong rank-revealing QR factorization"

## Author

Port of Python libid_rrqr.py with Julia optimizations
"""
module LibIDRRQR

using LinearAlgebra
using Printf

export id_rrqr, compare_with_exact_rank

# --------------------------------------------------------------
# Type-dependent default tolerances
# --------------------------------------------------------------

"""
    _default_rtol(T)

Compute type-dependent default relative tolerance.

Returns sqrt(eps(T)) which gives reasonable precision for each float type:
- Float64: ~ 1.49e-8
- Float32: ~ 1.09e-4
- BigFloat: scales with precision
- Float16: ~ 0.0078
"""
_default_rtol(::Type{T}) where T = sqrt(eps(real(T)))

"""
    id_rrqr(A::AbstractMatrix{T}; rtol=nothing, kmax::Union{Int,Nothing}=nothing) where T

Compute interpolative decomposition using deterministic RRQR (LAPACK geqp3).

## Algorithm

This function uses Julia's `qr(A, ColumnNorm())`, which calls LAPACK's
geqp3 routine. Unlike randomized methods, this is:
- Deterministic (same results every run)
- Examines all columns sequentially
- No early stopping capability (must compute full QR)

The algorithm:
1. Compute full pivoted QR: A[:, jpiv] = Q * R
2. Determine rank k from diagonal: |R[k,k]| / |R[1,1]| >= rtol
3. Extract R11 (kxk) and R12 (kx(n-k))
4. Solve R11 * T = R12 for T

## Arguments

- `A::AbstractMatrix{T}`: Input matrix (mxn), supports any float type
- `rtol::Union{Real,Nothing}`: Relative tolerance for rank determination (default: sqrt(eps(T)))
    - If rtol < 1: |R[k,k]| / |R[1,1]| >= rtol
    - If rtol >= 1: Interpreted as fixed rank kmax
    - Float64: ~ 1.49e-8, Float32: ~ 1.09e-4, BigFloat: scales with precision
- `kmax::Union{Int,Nothing}`: Maximum rank (defaults to min(m,n))

## Returns

- `k::Int`: Determined rank
- `jpiv::Vector{Int}`: Column permutation (1-based Julia indexing)
    - A[:, jpiv] = A[:, jpiv[1:k]] * [I; T]
- `T::Matrix{T}`: Interpolation matrix (kx(n-k))

## Mathematical Decomposition

    A[:, jpiv] = Q * R  where R = [R11  R12]
                                   [0    R22]

- R11 (kxk): Well-conditioned, |R[k,k]| / |R[1,1]| >= rtol
- R22 ((m-k)x(n-k)): Small diagonal elements

Skeleton decomposition:
    A ~ A[:, jpiv[1:k]] * [I; T]

where T = R11 \\ R12 (kx(n-k))

## Complexity

- Time: O(mn^2) for full QR factorization
- Space: O(mn) for Q and R matrices

## Performance

Expected (based on LAPACK performance):
- 300x300: ~10-15ms (Julia)
- 300x300: ~12-20ms (Python)
- Julia ~1.2-1.5x faster due to lower overhead

## Example

```julia
using LibIDRRQR

# Low-rank matrix
A = randn(100, 80)
A[:, 20:end] .= 0.01 .* randn(100, 61)  # Rank ~ 20

# Compute ID
k, jpiv, T = id_rrqr(A)  # Uses sqrt(eps(Float64)) ~ 1.49e-8
# k ~ 20

# Verify decomposition
A_skel = A[:, jpiv[k+1:end]]
A_basis = A[:, jpiv[1:k]]
rel_err = norm(A_skel - A_basis * T) / norm(A)
# rel_err < rtol
```

## See Also

- `LibIDSketch.id_sketch`: Sketch-based ID (best for large-scale)
"""
function id_rrqr(A::AbstractMatrix{T};
                 rtol::Union{Real,Nothing}=nothing,
                 kmax::Union{Int,Nothing}=nothing) where T

    rtol_val = something(rtol, _default_rtol(T))
    m, n = size(A)

    # Handle rank mode (rtol >= 1 -> interpret as kmax)
    if rtol_val >= 1
        # Rank mode: rtol is actually kmax
        kmax_input = Int(floor(rtol_val))
        rtol_qr = max(m, n) * eps(real(T))
    else
        # Tolerance mode
        kmax_input = isnothing(kmax) ? min(m, n) : kmax
        rtol_qr = rtol_val
    end

    # Compute full pivoted QR factorization
    # This calls LAPACK's dgeqp3/zgeqp3
    F = qr(A, ColumnNorm())
    R = F.R
    jpiv = F.p  # Julia's QR returns 1-based permutation

    # Determine rank from diagonal of R
    # Use relative tolerance: |diag(R[k])| >= rtol * |diag(R[1])|
    diag_abs = abs.(diag(R))

    if diag_abs[1] == 0
        # Zero matrix
        k = 0
    else
        k = sum(diag_abs .>= rtol_qr * diag_abs[1])
    end

    # Apply maximum rank constraint
    k = min(k, kmax_input)

    # Handle edge cases
    if k == 0
        # Rank 0: return empty T
        return 0, jpiv, zeros(T, 0, n)
    end

    if k >= n
        # Full rank: no skeleton columns to approximate
        return k, jpiv, zeros(T, k, 0)
    end

    # Extract submatrices
    R11 = R[1:k, 1:k]  # Upper triangular (k x k)
    R12 = R[1:k, k+1:end]  # Rectangular (k x (n-k))

    # Solve R11 * Tmat = R12 for interpolation matrix
    # R11 is upper triangular, so use triangular solver
    Tmat = UpperTriangular(R11) \ R12

    return k, jpiv, Tmat
end


"""
    compare_with_exact_rank(A::Matrix{T}, true_rank::Int) where T

Compare RRQR rank detection with known true rank.

Useful for debugging and understanding rank detection behavior.

## Arguments

- `A::Matrix{T}`: Input matrix
- `true_rank::Int`: Known true rank (e.g., from SVD)

## Example

```julia
# Create rank-20 matrix
U = randn(100, 20)
V = randn(80, 20)
A = U * V'

k, jpiv, T = id_rrqr(A, rtol=1e-10)
compare_with_exact_rank(A, 20)
```
"""
function compare_with_exact_rank(A::Matrix{T}, true_rank::Int) where T
    println("Comparing RRQR with exact rank")
    println("="^60)

    # SVD for exact rank
    F = svd(A)
    s = F.S

    println("\nSingular values:")
    for i in 1:min(20, length(s))
        marker = (i == true_rank + 1) ? " <- expected gap" : ""
        @printf("  s[%2d] = %.3e%s\n", i, s[i], marker)
    end

    # RRQR at different tolerances
    tolerances = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]

    println("\nRRQR rank detection:")
    @printf("  %-10s %10s %15s\n", "rtol", "k", "Gap s[k]/s[k+1]")
    println("  " * "-"^40)

    for tol in tolerances
        k, _, _ = id_rrqr(A, rtol=tol)
        gap = (k > 0 && k < length(s)) ? s[k] / s[k+1] : Inf
        @printf("  %-10.0e %10d %15.3e\n", tol, k, gap)
    end

    println("\n  True rank: $true_rank")
    println("="^60)
end


end  # module LibIDRRQR
