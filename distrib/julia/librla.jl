"""
    librla - Randomized Linear Algebra Routines for Julia

Randomized algorithms for low-rank matrix approximations:
- `orth_sketch(A, rtol; kwargs...)` - Approximate orthonormal basis for column space
- `qr_sketch(A, rtol; kwargs...)` - Truncated QR factorization with column pivoting
- `svd_sketch(A, rtol; kwargs...)` - Truncated singular value decomposition (SVD)
- `id_sketch(A, rtol; kwargs...)` - Interpolative decomposition (ID)

Deterministic:
- `id_qrpiv(A, rtol; kwargs...)` - Interpolative decomposition via QR with pivoting

# Usage
```julia
Q, flag, diagR = orth_sketch(A, rtol)
Q, R, p = qr_sketch(A, rtol)
U, s, Vt = svd_sketch(A, rtol)
k, piv, T = id_sketch(A, rtol)
# tolerance mode: rtol < 1, rank mode: rtol >= 1
```

# Matrix-Free Operators
Use the LinearOperator type for matrix-free operators:
```julia
include("LinearOperator.jl")
A = LinearOperator(matvec_fun, rmatvec_fun, m, n; dtype=Float64)
U, s, Vt = svd_sketch(A, rank)  # both modes; tolerance mode may
                                # materialize A at O(n) matvec cost
```

# Author
Adrianna Gillman, Zydrunas Gimbutas

# SPDX-License-Identifier: MIT

# Version
1.2.0

# Date
July 26, 2026

# Assisted by
Claude Code (Anthropic)
"""
module librla

using LinearAlgebra
using Random

include("LinearOperator.jl")
import .LinearOperator, .matvec, .rmatvec, .from_matrix

export orth_sketch, qr_sketch, svd_sketch, id_sketch, id_qrpiv
export LinearOperator, from_matrix, matvec, rmatvec

# LinearOperator support:
# - m::Int, n::Int (dimensions)
# - matvec(A, x), rmatvec(A, x) (apply forward/adjoint)
# - matrix (explicit matrix or nothing for matrix-free)

# --------------------------------------------------------------
# Public API Functions
# --------------------------------------------------------------

"""
    orth_sketch(A, rtol; block_size=42, power_iter=0, extra_samples=12, rng=nothing)

Approximate orthonormal basis for column space using randomized sketching.

This function uses random test matrix multiplication (A*Ω where Ω has i.i.d.
uniform[-1,1] entries) followed by QR factorization to approximate the range
of A. The approach is particularly efficient for matrices with rapidly
decaying singular values.

The algorithm has two modes:
- Tolerance mode (rtol < 1): Adaptively grows the sketch size until it
  covers the numerical rank at rtol with at least extra_samples columns
  to spare, i.e. diagR[end-extra_samples] <= rtol * diagR[1].
- Rank mode (rtol >= 1): Performs a single sketch of
  floor(rtol) + extra_samples columns (rtol interpreted as target rank)

In both modes the returned basis includes at least extra_samples buffer
columns beyond the target rank (rank mode) or the rtol-rank of the sketch
(tolerance mode): callers that want the target rank exactly truncate after
projecting, as qr_sketch/svd_sketch do. The buffer guarantees the sketch
oversamples the numerical rank, so the accepted tolerance test is a
max-norm statistic over extra_samples + 1 sketch columns rather than the
single smallest one.

# Arguments
- `A`: Input matrix (m×n) or LinearOperator
- `rtol`: Relative tolerance (< 1) or target rank (>= 1)
- `block_size`: Initial number of random test vectors in tolerance mode
  (default: 42). Ignored in rank mode.
- `power_iter`: Number of power iterations to improve accuracy (default: 0)
  Setting power_iter=1 or 2 can improve results for matrices
  with slowly decaying singular values
- `extra_samples`: Number of buffer columns beyond the target rank
  (default: 12). Rank mode samples floor(rtol) + extra_samples columns;
  tolerance mode accepts a sketch only when at least extra_samples + 1 of
  its pivoted column norms are at or below rtol times the largest.
  extra_samples=0 reproduces the legacy last-column tolerance check.
- `rng`: Random number generator (default: nothing uses Random.default_rng())

# Returns
- `Q`: Orthonormal matrix (m×k) spanning approximate range of A, including
  the extra_samples buffer columns
- `flag`: Exit status:
  - 0: Success, Q contains valid orthonormal basis
  - 1: Early termination (tolerance mode only). Occurs when:
    (a) rtol < machine epsilon (tolerance too tight), or
    (b) sketch size grew to min(m,n) without meeting tolerance,
        indicating matrix is effectively full-rank (within extra_samples
        columns) at this tolerance
    When flag=1, Q is empty (m×0).
- `diagR`: Diagonal elements from pivoted QR factorization, representing
  column norms of the sketched matrix (sorted in decreasing order)

# Note
Higher-level functions (qr_sketch, svd_sketch, id_sketch) automatically
fall back to deterministic (full) QR or SVD when orth_sketch terminates
early, so users of those functions do not need to handle flag=1 explicitly.

See also [`qr_sketch`](@ref), [`svd_sketch`](@ref), [`id_sketch`](@ref).

# Examples
```julia
A = randn(200, 8) * randn(8, 100)

# Adaptive basis to 1e-6 relative tolerance
Q, flag, diagR = orth_sketch(A, 1e-6)

# Rank-20 basis (returned with the extra_samples buffer columns)
Q, flag, diagR = orth_sketch(A, 20)

# Reproducible run with a seeded generator
using Random
Q, flag, diagR = orth_sketch(A, 1e-6; rng=MersenneTwister(7))
```
"""
function orth_sketch(A, rtol; block_size=42, power_iter=0, extra_samples=12, rng=nothing)
    m, n = size(A)
    is_complex_op = _is_complex_type(A)
    dtype = _get_dtype(A)

    # Rank mode (rtol >= 1): single oversampled sketch, buffer included
    if rtol >= 1
        block_size = Int(floor(rtol)) + extra_samples
        x = _uniform_omega(n, block_size, is_complex_op, dtype, rng)
        x = _power_iteration(A, x, power_iter)
        y = _matvec(A, x)
        F = qr(y, ColumnNorm())
        R = F.R

        diagR = abs.(diag(R))

        # Materialize thin Q including the buffer columns (not full m×m);
        # R has min(m, block_size) rows
        Q = F.Q[:, 1:size(R, 1)]
        flag = 0
        return Q, flag, diagR
    end

    # Tolerance mode (rtol < 1): geometric growth with tolerance checking
    if rtol < _get_eps(dtype)
        Q = zeros(dtype, m, 0)
        flag = 1
        diagR = zeros(dtype, 0)
        return Q, flag, diagR
    end

    if block_size >= min(m, n)
        Q = zeros(dtype, m, 0)
        flag = 1
        diagR = zeros(dtype, 0)
        return Q, flag, diagR
    end

    # Main loop with geometric growth
    while true
        x = _uniform_omega(n, block_size, is_complex_op, dtype, rng)
        x = _power_iteration(A, x, power_iter)
        y = _matvec(A, x)
        F = qr(y, ColumnNorm())
        R = F.R

        # Buffered tolerance check (cross-multiplied form of
        # diagR[end-extra_samples]/diagR[1] <= rtol, avoiding the division;
        # diagR is sorted decreasing so diagR[1] is the max)
        diagR = abs.(diag(R))
        if isempty(diagR) || (length(diagR) > extra_samples &&
                              diagR[end-extra_samples] <= rtol * diagR[1])
            flag = 0
            Q = F.Q[:, 1:size(y, 2)]  # thin Q, not full m×m
            return Q, flag, diagR
        end

        # Grow block size
        block_size = min(block_size * 4, min(m, n))
        if block_size >= min(m, n)
            Q = zeros(dtype, m, 0)
            flag = 1
            diagR = zeros(dtype, 0)
            return Q, flag, diagR
        end
    end
end

"""
    qr_sketch(A, rtol; block_size=42, power_iter=0, extra_samples=12, rng=nothing)

Compute truncated QR factorization with column pivoting via randomized sketching.

The algorithm sketches an orthonormal basis for the column space of A,
projects A onto this basis, computes the QR of the smaller projected matrix,
and then expands back to the original space. If the matrix is effectively
full rank a deterministic QR is performed.

This is much faster than full QR for matrices where the target rank k is
much smaller than min(m,n).

# Arguments
- `A`: Input matrix (m×n) or LinearOperator
- `rtol`: Relative tolerance (< 1) or target rank (>= 1)
  - Tolerance mode: keep columns with norm >= rtol * max_norm
  - Rank mode: return k leading columns
- `block_size`: Sketch size for tolerance mode (default: 42)
- `power_iter`: Power iterations for accuracy (default: 0)
- `extra_samples`: Oversampling / buffer beyond the target rank (default: 12)
  Rank mode sketches rank + extra_samples columns; tolerance mode accepts a
  sketch only when at least extra_samples + 1 of its pivoted column norms
  are at or below rtol times the largest
- `rng`: Random number generator (default: nothing uses Random.default_rng())

# Returns
- `Q`: Orthonormal matrix (m×k), k ≤ min(m,n)
- `R`: Upper triangular matrix (k×n)
- `p`: Column permutation vector (1-based indexing), length n
  The decomposition satisfies A[:, p] ≈ Q*R

See also [`orth_sketch`](@ref), [`svd_sketch`](@ref), [`id_sketch`](@ref),
[`id_qrpiv`](@ref).

# Examples
```julia
A = randn(200, 8) * randn(8, 100)

# Adaptive rank to 1e-6 relative tolerance
Q, R, p = qr_sketch(A, 1e-6)

# Fixed rank 20; two power iterations for a slowly decaying spectrum
Q, R, p = qr_sketch(A, 20; power_iter=2)

# Reproducible run with a seeded generator
using Random
Q, R, p = qr_sketch(A, 1e-6; rng=MersenneTwister(7))
```
"""
function qr_sketch(A, rtol; block_size=42, power_iter=0, extra_samples=12, rng=nothing)
    m, n = size(A)
    dtype = _get_dtype(A)

    # Rank mode vs tolerance mode
    rank_mode = false
    if rtol >= 1
        rank_mode = true
        kmax = floor(Int, rtol)
    end

    # Compute sketch; the basis includes the extra_samples buffer columns
    # for better accuracy (truncated to the target rank after QR)
    Qs, flag, _ = orth_sketch(A, rtol; block_size=block_size, power_iter=power_iter,
                              extra_samples=extra_samples, rng=rng)

    k = size(Qs, 2)
    if flag != 0
        k = min(m, n)
    end

    # Fallback to full QR if needed
    needs_fallback = (flag != 0 || k >= min(m, n))
    if needs_fallback && rank_mode
        needs_fallback = false
    end

    if needs_fallback
        A_mat = _get_matrix(A)
        F = qr(A_mat, ColumnNorm())
        R = F.R
        p = F.p

        # Determine rank (cap by the number of rows in R, which equals min(m,n))
        if rank_mode
            rank = min(kmax, size(R, 1))
        else
            rank = _rank_from_diag(diag(R), rtol)
        end

        Q = F.Q[:, 1:rank]  # thin Q, not full m×m
        R = R[1:rank, :]
        return Q, R, p
    end

    # Project and compute QR
    B = _matmat_left(Qs, A)
    F = qr(B, ColumnNorm())
    k_proj = size(B, 1)
    Qproj = F.Q[:, 1:k_proj]  # thin Q
    R = F.R
    p = F.p
    Q = Qs * Qproj

    # Determine rank (cap by both available Q columns and available R rows)
    if rank_mode
        rank = min(kmax, size(Q, 2), size(R, 1))
    else
        rank = _rank_from_diag(diag(R), rtol)
    end

    Q = Q[:, 1:rank]
    R = R[1:rank, :]

    return Q, R, p
end

"""
    svd_sketch(A, rtol; block_size=42, power_iter=0, extra_samples=12, rng=nothing)

Compute truncated singular value decomposition (SVD) via randomized sketching.

The algorithm sketches an orthonormal basis for the column space of A,
projects A onto this basis, computes the SVD of the smaller projected matrix,
and then expands back to the original space. If the matrix is effectively
full rank a deterministic SVD is performed.

This is much faster than full SVD for matrices where the target rank k is
much smaller than min(m,n).

# Arguments
- `A`: Input matrix (m×n) or LinearOperator
- `rtol`: Relative tolerance (< 1) or target rank (>= 1)
  - Tolerance mode: keep singular values >= rtol * s[1]
  - Rank mode: return k leading singular triplets
- `block_size`: Sketch size for tolerance mode (default: 42)
- `power_iter`: Power iterations for accuracy (default: 0)
- `extra_samples`: Oversampling / buffer beyond the target rank (default: 12)
  Rank mode sketches rank + extra_samples columns; tolerance mode accepts a
  sketch only when at least extra_samples + 1 of its pivoted column norms
  are at or below rtol times the largest
- `rng`: Random number generator (default: nothing uses Random.default_rng())

# Returns
- `U`: Left singular vectors (m×k), orthonormal columns
- `s`: Singular values (length k), sorted descending
- `Vt`: Right singular vectors conjugate-transposed (k×n), orthonormal rows
  The decomposition satisfies A ≈ U*diagm(s)*Vt

See also [`orth_sketch`](@ref), [`qr_sketch`](@ref), [`id_sketch`](@ref),
`LinearAlgebra.svd`.

# Examples
```julia
A = randn(200, 8) * randn(8, 100)

# Adaptive rank to 1e-6 relative tolerance
U, s, Vt = svd_sketch(A, 1e-6)

# Fixed rank 20; two power iterations for a slowly decaying spectrum
U, s, Vt = svd_sketch(A, 20; power_iter=2)

# Reproducible run with a seeded generator
using Random
U, s, Vt = svd_sketch(A, 1e-6; rng=MersenneTwister(7))
```
"""
function svd_sketch(A, rtol; block_size=42, power_iter=0, extra_samples=12, rng=nothing)
    m, n = size(A)
    dtype = _get_dtype(A)

    # Handle wide matrices via transpose. For a LinearOperator, adjoint returns
    # a concrete new LinearOperator (Base.copy is undefined for it); for a dense
    # array, copy materializes the lazy Adjoint wrapper so recursive sketching
    # hits BLAS directly. If A' = U_at * diag(s) * Vt_at, then A = Vt_at' *
    # diag(s) * U_at', so U = Vt_at' and Vt = U_at'.
    if m < n
        At = isa(A, LinearOperator) ? A' : copy(A')
        U_at, s, Vt_at = svd_sketch(At, rtol; block_size=block_size,
                                    power_iter=power_iter, extra_samples=extra_samples, rng=rng)
        return Vt_at', s, U_at'
    end

    # Rank mode vs tolerance mode
    rank_mode = false
    if rtol >= 1
        rank_mode = true
        kmax = floor(Int, rtol)
    end

    # Compute sketch; the basis includes the extra_samples buffer columns
    # to get more accurate singular values (truncated to the target rank after SVD)
    Qs, flag, _ = orth_sketch(A, rtol; block_size=block_size, power_iter=power_iter,
                              extra_samples=extra_samples, rng=rng)

    k = size(Qs, 2)
    if flag != 0
        k = min(m, n)
    end

    # Fallback to full SVD if needed
    needs_fallback = (flag != 0 || k >= min(m, n))
    if needs_fallback && rank_mode
        needs_fallback = false
    end

    if needs_fallback
        A_mat = _get_matrix(A)
        F = svd(A_mat)
        U = F.U
        s = F.S
        Vt = F.Vt

        # Determine rank
        if rank_mode
            rank = min(kmax, length(s))
        else
            rank = _rank_from_svals(s, rtol)
        end

        U = U[:, 1:rank]
        Vt = Vt[1:rank, :]
        s = s[1:rank]
        return U, s, Vt
    end

    # Project and compute SVD
    Aproj = _matmat_left(Qs, A)
    F = svd(Aproj)
    Uproj = F.U
    s = F.S
    Vt = F.Vt
    U = Qs * Uproj

    # Determine rank
    if rank_mode
        rank = min(kmax, length(s))
    else
        rank = _rank_from_svals(s, rtol)
    end

    U = U[:, 1:rank]
    Vt = Vt[1:rank, :]
    s = s[1:rank]

    return U, s, Vt
end

"""
    id_sketch(A, rtol; block_size=42, power_iter=0, extra_samples=12, method="fast", rng=nothing)

Compute interpolative decomposition (ID) via randomized sketching.

An ID represents a matrix A by selecting k of its columns and expressing
the remaining columns as linear combinations of the selected ones:

    A[:, piv[k+1:n]] ≈ A[:, piv[1:k]] * T

where piv is a column permutation and T is a k×(n-k) interpolation matrix.
The selected columns (skeleton) capture the essential features of A, while
T provides the coefficients to reconstruct the other columns.

This function uses qr_sketch() to identify the column permutation.

# Arguments
- `A`: Input matrix (m×n) or LinearOperator
- `rtol`: Relative tolerance (< 1) or target rank (>= 1)
- `block_size`: Sketch size for tolerance mode (default: 42)
- `power_iter`: Power iterations for accuracy (default: 0)
- `extra_samples`: Oversampling / buffer beyond the target rank (default: 12)
  Rank mode sketches rank + extra_samples columns; tolerance mode accepts a
  sketch only when at least extra_samples + 1 of its pivoted column norms
  are at or below rtol times the largest
- `method`: Method for computing T matrix (default: "fast")
  - "fast": Triangular solve R11 \\ R12 (fastest)
  - "svd": SVD-based pseudoinverse
  - "lstsq": Least-squares from original A (most accurate, slowest)
- `rng`: Random number generator (default: nothing uses Random.default_rng())

# Returns
- `k`: Rank of the approximation (number of skeleton columns)
- `piv`: Column permutation (1-based indexing), length n
  - piv[1:k] are indices of skeleton columns
  - piv[k+1:n] are indices of interpolated columns
- `T`: Interpolation matrix (k×(n-k))
  The approximation is A[:, piv[k+1:n]] ≈ A[:, piv[1:k]] * T

See also [`id_qrpiv`](@ref), [`qr_sketch`](@ref), [`svd_sketch`](@ref),
[`orth_sketch`](@ref).

# Examples
```julia
A = randn(200, 8) * randn(8, 100)

# Adaptive rank to 1e-6 relative tolerance
k, piv, T = id_sketch(A, 1e-6)

# Fixed rank 20 with the most accurate T computation
k, piv, T = id_sketch(A, 20; method="lstsq")

# Reproducible run with a seeded generator
using Random
k, piv, T = id_sketch(A, 1e-6; rng=MersenneTwister(7))
```
"""
function id_sketch(A, rtol; block_size=42, power_iter=0, extra_samples=12, method="fast", rng=nothing)
    if !(method in ["fast", "svd", "lstsq"])
        error("method must be one of: 'fast', 'svd', 'lstsq'")
    end

    # Get QR factorization
    _, R, piv = qr_sketch(A, rtol; block_size=block_size,
                          power_iter=power_iter, extra_samples=extra_samples, rng=rng)

    k = size(R, 1)

    # Compute rtol for SVD filtering
    m, n = size(A)
    if rtol >= 1
        # Rank mode: minimal filtering (only exact zeros)
        rtol_for_svd = zero(typeof(rtol))
    else
        # Tolerance mode: use the provided tolerance
        rtol_for_svd = rtol
    end

    # Dispatch to helper functions
    if method == "lstsq"
        T = _compute_T_lstsq(A, R, piv, k)
    elseif method == "svd"
        T = _compute_T_svd(R, k, rtol_for_svd)
    elseif method == "fast"
        T = _compute_T_fast(R, k, rtol_for_svd)
    end

    return k, piv, T
end

"""
    id_qrpiv(A, rtol; method="fast")

Interpolative decomposition via deterministic QR with column pivoting.

An ID represents a matrix A by selecting k of its columns and expressing
the remaining columns as linear combinations of the selected ones:

    A[:, piv[k+1:n]] ≈ A[:, piv[1:k]] * T

where piv is a column permutation and T is a k×(n-k) interpolation matrix.
The selected columns (skeleton) capture the essential features of A, while
T provides the coefficients to reconstruct the other columns.

This function provides a deterministic alternative to id_sketch by
computing the interpolative decomposition using only QR with column
pivoting (LAPACK geqp3), without any randomized sketching. It preserves
LinearOperator support and uses the same T matrix computation logic as
id_sketch.

# Arguments
- `A`: Input matrix or LinearOperator
- `rtol`: Tolerance (< 1) or rank (>= 1)
- `method`: Method for computing T matrix (default: "fast")
  - "fast": Triangular solve R11 \\ R12 (fastest)
  - "svd": SVD-based pseudoinverse
  - "lstsq": Least-squares from original A (most accurate, slowest)

# Returns
- `k`: Rank
- `piv`: Column permutation (1-based)
- `T`: Interpolation matrix, size (k, n-k)

See also [`id_sketch`](@ref), [`qr_sketch`](@ref), [`svd_sketch`](@ref),
[`orth_sketch`](@ref).

# Examples
```julia
A = randn(200, 8) * randn(8, 100)

# Deterministic ID to 1e-6 relative tolerance
k, piv, T = id_qrpiv(A, 1e-6)

# Fixed rank 20 with the most accurate T computation
k, piv, T = id_qrpiv(A, 20; method="lstsq")
```
"""
function id_qrpiv(A, rtol; method="fast")
    if !(method in ["fast", "svd", "lstsq"])
        error("method must be one of: 'fast', 'svd', 'lstsq'")
    end

    m, n = size(A)

    # Determine rank mode vs tolerance mode
    rank_mode = false
    if rtol >= 1
        rank_mode = true
        kmax = floor(Int, rtol)
    end

    # Compute full QR with pivoting (deterministic)
    A_mat = _get_matrix(A)
    F = qr(A_mat, ColumnNorm())
    R = F.R
    piv = F.p

    # Determine rank
    if rank_mode
        rank = min(kmax, min(m, n))
        rtol_for_svd = zero(typeof(rtol))  # Minimal filtering in rank mode
    else
        rank = _rank_from_diag(diag(R), rtol)
        rtol_for_svd = rtol
    end

    k = rank
    Q = F.Q[:, 1:k]  # thin Q, not full m×m

    # Handle edge cases
    if k == 0
        T = zeros(eltype(R), 0, n)
        return k, piv, T
    end

    if k == n
        T = zeros(eltype(R), k, 0)
        return k, piv, T
    end

    # Dispatch to helper functions
    if method == "lstsq"
        T = _compute_T_lstsq(A, R, piv, k)
    elseif method == "svd"
        T = _compute_T_svd(R, k, rtol_for_svd)
    elseif method == "fast"
        T = _compute_T_fast(R, k, rtol_for_svd)
    end

    return k, piv, T
end

# --------------------------------------------------------------
# Private Helper Functions
# --------------------------------------------------------------

"""
    _power_iteration(A, x, power_iter)

Apply (A'*A)^power_iter to x with orthogonalization after each iteration.
"""
function _power_iteration(A, x, power_iter::Int)
    for i = 1:power_iter
        x = _rmatvec(A, _matvec(A, x))
        F = qr(x, ColumnNorm())
        # thin Q has min(rows, cols) columns; cap the slice so block_size > rows
        # (e.g. rank + extra_samples > n) does not overrun the available columns
        x = F.Q[:, 1:min(size(x, 1), size(x, 2))]
    end
    return x
end

"""
    _uniform_omega(n, block_size, is_complex, dtype, rng)

Generate uniform[-1,1] random test matrix.
"""
function _uniform_omega(n::Int, block_size::Int, is_complex::Bool, dtype::Type, rng)
    # Use provided rng or default
    r = isnothing(rng) ? Random.default_rng() : rng
    if is_complex
        if dtype <: Complex
            T = real(dtype)
        else
            T = dtype
        end
        omega = 2 * rand(r, T, n, block_size) .- 1 .+
                1im * (2 * rand(r, T, n, block_size) .- 1)
    else
        omega = 2 * rand(r, dtype, n, block_size) .- 1
    end
    return omega
end

"""
    _gaussian_omega(n, block_size, is_complex, dtype, rng)

Generate Gaussian random test matrix.
"""
function _gaussian_omega(n::Int, block_size::Int, is_complex::Bool, dtype::Type, rng)
    # Use provided rng or default
    r = isnothing(rng) ? Random.default_rng() : rng
    if is_complex
        if dtype <: Complex
            T = real(dtype)
        else
            T = dtype
        end
        omega = randn(r, T, n, block_size) .+ 1im * randn(r, T, n, block_size)
    else
        omega = randn(r, dtype, n, block_size)
    end
    return omega
end

"""
    _matvec(A, x)

Apply operator A to vector/matrix x.
"""
function _matvec(A, x)
    if isa(A, LinearOperator)
        return matvec(A, x)
    else
        return A * x
    end
end

"""
    _rmatvec(A, x)

Apply adjoint A' to vector/matrix x.
"""
function _rmatvec(A, x)
    if isa(A, LinearOperator)
        return rmatvec(A, x)
    else
        return A' * x
    end
end

"""
    _matmat_left(Q, A)

Compute Q' * A efficiently.
"""
function _matmat_left(Q, A)
    if isa(A, LinearOperator)
        # Use adjoint: Q' * A = (A' * Q)'
        return _rmatvec(A, Q)'
    else
        return Q' * A
    end
end

"""
    _rank_from_svals(s, rtol)

Determine numerical rank from singular values.
"""
function _rank_from_svals(s, rtol)
    if isempty(s) || s[1] <= 0
        return 0
    else
        return sum(s .>= rtol * s[1])
    end
end

"""
    _rank_from_diag(diag_vals, rtol)

Determine numerical rank from diagonal elements (e.g., from QR).
"""
function _rank_from_diag(diag_vals, rtol)
    diag_abs = abs.(diag_vals)
    if isempty(diag_abs) || diag_abs[1] <= 0
        return 0
    else
        return sum(diag_abs .>= rtol * diag_abs[1])
    end
end

"""
    _is_complex_type(A)

Check if A represents complex data.
"""
function _is_complex_type(A)
    # Use eltype which works for both matrices and LinearOperators
    T = eltype(A)
    return T <: Complex
end

"""
    _get_dtype(A)

Get data type for arrays and operations.
"""
function _get_dtype(A)
    # eltype works for both matrices and LinearOperators
    return eltype(A)
end

"""
    _get_eps(dtype)

Get machine epsilon for a data type, handling complex types.
"""
function _get_eps(dtype)
    if dtype <: Complex
        return eps(real(dtype))
    else
        return eps(dtype)
    end
end

"""
    _is_matrix_free_linop(A)

Check if A is a matrix-free LinearOperator.
"""
function _is_matrix_free_linop(A)
    return isa(A, LinearOperator) && isnothing(A.matrix)
end

"""
    _get_matrix(A)

Return A as a dense matrix, materializing it if necessary.

Dense matrices and LinearOperators backed by a dense `Matrix` are returned
directly; other raw `AbstractMatrix` types (sparse, `Adjoint`) are converted
via `Matrix()`. LinearOperators without a dense backing matrix are
materialized one column at a time via matvec, at an O(n)-matvec cost —
callers are dense (LAPACK) algorithms, so a dense result is required.
"""
function _get_matrix(A)
    if isa(A, LinearOperator)
        if A.matrix isa Matrix
            return A.matrix
        end
        # Materialize one column at a time via matvec (matrix-free or
        # sparse/lazy-backed operators)
        m, n = size(A)
        T = eltype(A)
        A_mat = Matrix{T}(undef, m, n)
        e_j = zeros(T, n)
        for j = 1:n
            e_j[j] = one(T)
            A_mat[:, j] = _matvec(A, e_j)
            e_j[j] = zero(T)
        end
        return A_mat
    else
        return A isa Matrix ? A : Matrix(A)
    end
end

"""
    _compute_T_lstsq(A, R, piv, k)

Compute T using least-squares from original A columns.
"""
function _compute_T_lstsq(A, R, piv, k)
    m, n = size(A)

    if k == 0 || k >= n
        return zeros(eltype(R), k, n - k)
    end

    A_mat = _get_matrix(A)
    cols = piv[1:k]
    remaining = piv[(k+1):end]
    T = A_mat[:, cols] \ A_mat[:, remaining]

    return T
end

"""
    _compute_T_svd(R, k, rtol_for_svd)

Compute T using SVD-based pseudoinverse of R11.

# Arguments
- `R`: R factor from QR decomposition
- `k`: Rank
- `rtol_for_svd`: Tolerance for filtering small singular values
"""
function _compute_T_svd(R, k, rtol_for_svd)
    m_r, n = size(R)

    if k == 0
        return zeros(eltype(R), k, n - k)
    end

    R11 = R[1:k, 1:k]
    R12 = R[1:k, (k+1):end]

    F = svd(R11)
    U = F.U
    s = F.S
    Vh = F.Vt

    # Filter small singular values. Floor the relative threshold at machine
    # precision so a rank-deficient R11 (e.g. requested rank exceeds the true
    # rank) drops near-zero singular values instead of inverting them, which
    # would otherwise produce Inf/NaN. An all-zero R11 yields T = 0.
    smax = isempty(s) ? zero(eltype(s)) : maximum(s)
    if smax == 0
        return zeros(eltype(R), size(R12)...)
    end
    tol = max(rtol_for_svd, eps(real(eltype(R))))
    keep = s .>= tol * smax
    if !any(keep)
        T = zeros(eltype(R), size(R12)...)
    else
        inv_s = one(eltype(R)) ./ s[keep]
        T = Vh[keep, :]' * Diagonal(inv_s) * (U[:, keep]' * R12)
    end

    return T
end

"""
    _compute_T_fast(R, k, rtol_for_svd)

Compute T using fast triangular solve, falling back to the SVD-based
minimum-norm solve when R11 is (near-)singular.
"""
function _compute_T_fast(R, k, rtol_for_svd)
    m_r, n = size(R)

    if k == 0
        return zeros(eltype(R), k, n - k)
    end

    R11 = R[1:k, 1:k]
    R12 = R[1:k, (k+1):end]

    # R11 is upper-triangular from QR; if it is (near-)singular the fast
    # triangular solve would divide by ~0 (crash or Inf/NaN). Detect via the
    # diagonal and fall back to the SVD-based minimum-norm solve, which stays
    # finite and lets the decomposition still reconstruct A.
    d = abs.(diag(R11))
    if minimum(d) <= eps(real(eltype(R))) * maximum(d)
        return _compute_T_svd(R, k, rtol_for_svd)
    end

    T = R11 \ R12

    return T
end

end # module librla
