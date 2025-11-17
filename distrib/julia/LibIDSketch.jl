"""
LibIDSketch - Randomized Sketching for Interpolative Decomposition

Direct port of Python libid.py reference implementation to Julia.
This version matches the Python/MATLAB algorithm exactly.

## Algorithm Overview

**orth_sketch**: Single-iteration random sketching with geometric block growth
- Generate uniform[-1,1] random test matrix
- Optional power iteration: (A'A)^n
- QR factorization and residual check
- Replace Q each iteration (not accumulate)
- Stop when: d = |R[end,end]| / max(column_norms) <= rtol

**id_sketch**: Single-stage pipeline
- Call qr_sketch directly on A
- Extract R11, R12 from R
- Solve R11 * T = R12

## References

- Python: libid.py (lines 70-321)
- MATLAB: libid.m (lines 179-774)
- Paper: Halko, Martinsson, Tropp "Finding structure with randomness" (2011)

## Author

Port of Python libid.py reference implementation
"""
module LibIDSketch

using LinearAlgebra
using Random

export orth_sketch, qr_sketch, svd_sketch, id_sketch

# --------------------------------------------------------------
# LinearOperator support utilities
# --------------------------------------------------------------

# Accept any type with these properties as a LinearOperator
function _has_linop_interface(A)
    return hasproperty(A, :m) && hasproperty(A, :n) &&
           hasproperty(A, :apply) && hasproperty(A, :applyT) &&
           hasproperty(A, :is_explicit)
end

"""
Union type for matrix-like inputs (AbstractMatrix or anything with LinearOperator interface)
"""
const MatrixLike = Union{AbstractMatrix, Any}

"""
Check if A is a LinearOperator (duck typing - check for required interface)
"""
_is_linop(A::AbstractMatrix) = false
_is_linop(A) = _has_linop_interface(A)

"""
Check if A is a matrix-free LinearOperator (not backed by explicit matrix)
"""
_is_matrix_free_linop(A::AbstractMatrix) = false
_is_matrix_free_linop(A) = _is_linop(A) && !A.is_explicit

"""
Forward matvec or matmat: A * x (BLAS3-rich)

Detects if x is a vector or matrix and uses appropriate BLAS level.
For LinearOperators with explicit matrices, BLAS3 is used automatically.
"""
_matvec(A::AbstractMatrix, x) = A * x
_matvec(A, x) = _is_linop(A) ? A.apply(x) : A * x

"""
Adjoint matvec or rmatmat: A' * x (BLAS3-rich)

Detects if x is a vector or matrix and uses appropriate BLAS level.
For LinearOperators with explicit matrices, BLAS3 is used automatically.
"""
_rmatvec(A::AbstractMatrix, x) = A' * x
_rmatvec(A, x) = _is_linop(A) ? A.applyT(x) : A' * x

"""
Left matrix multiplication: Q' * A (BLAS3-rich for LinearOperator)

For LinearOperators: B = Q' * A  =>  B' = A' * Q  =>  B = (A' * Q)'
Uses adjoint matmat (BLAS3) when LinearOperator has explicit matrix.
"""
function _matmat_left(Q::AbstractMatrix, A::AbstractMatrix)
    return Q' * A
end

function _matmat_left(Q::AbstractMatrix, A)
    # B = Q' * A  =>  B' = A' * Q  =>  B = (A' * Q)'
    # Use _rmatvec which will call A.applyT(Q)
    # For explicit LinearOperators, this uses BLAS3 matmat
    C = _rmatvec(A, Q)  # C = A' * Q, shape (n, k)
    return C'           # B = C', shape (k, n)
end

"""
Get element type from matrix or LinearOperator
"""
function _get_eltype(A::AbstractMatrix{T}) where T
    return T
end

function _get_eltype(A)
    # For LinearOperator, use the type parameter T from LinearOperator{T}
    if _is_linop(A)
        # Julia's eltype() extracts T from LinearOperator{T}
        return eltype(A)
    else
        # This should not be reached for valid inputs
        error("Unable to determine element type. " *
              "For matrix-free LinearOperators, type must be specified during creation.")
    end
end

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

# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------

"""
    _uniform_omega(A, n, block_size; rng=Random.default_rng())

Generate uniform[-1,1] random test matrix (real or complex).

Matches Python `_uniform_omega` function exactly.

# Arguments
- `A`: Matrix or LinearOperator (only type inspected)
- `n`: Number of rows
- `block_size`: Number of columns
- `rng`: Random number generator

# Returns
- `Omega`: n x block_size random matrix with uniform[-1,1] entries
"""
function _uniform_omega(A::MatrixLike, n::Int, block_size::Int;
                        rng::AbstractRNG=Random.default_rng())

    T = _get_eltype(A)

    if T <: Complex
        # Complex case: real and imag parts independent
        real_part = (2 * one(real(T))) .* rand(rng, real(T), n, block_size) .- one(real(T))
        imag_part = (2 * one(real(T))) .* rand(rng, real(T), n, block_size) .- one(real(T))
        return complex.(real_part, imag_part)
    else
        # Real case: uniform[-1,1]
        return (2 * one(T)) .* rand(rng, T, n, block_size) .- one(T)
    end
end


"""
    _power_iteration(A, X, flag_power)

Apply (A'*A)^flag_power to X, re-orthogonalizing each step.

Matches Python `_power_iteration` function exactly.

# Algorithm
For each iteration:
1. X = A' * (A * X)
2. Orthogonalize via QR with column pivoting

# Arguments
- `A`: Input matrix or LinearOperator
- `X`: Test matrix
- `flag_power`: Number of power iterations (0 = none)

# Returns
- `X`: Updated test matrix after power iteration
"""
function _power_iteration(A::MatrixLike, X::AbstractMatrix, flag_power::Int)
    for _ in 1:flag_power
        X = _rmatvec(A, _matvec(A, X))
        # Re-orthogonalize with QR (economy, with pivoting)
        F = qr(X, ColumnNorm())
        X = Matrix(F.Q)
    end
    return X
end


"""
    _rank_from_svals(s, rtol)

Determine numerical rank from singular values.

Matches Python `_rank_from_svals` function exactly (libid.py lines 259-263).

# Algorithm
Count how many singular values satisfy: s[i] >= rtol * s[1]

# Arguments
- `s::Vector`: Singular values (descending order)
- `rtol::Real`: Relative tolerance (supports Float64, Float32, BigFloat, etc.)

# Returns
- `rank::Int`: Numerical rank

# Examples
```julia
s = [10.0, 5.0, 1.0, 0.01, 0.001]
rank = _rank_from_svals(s, 1e-2)  # rank = 3
```
"""
function _rank_from_svals(s::Vector{T}, rtol::Real) where T <: Real
    if isempty(s)
        return 0
    end
    return sum(s .>= rtol * s[1])
end


"""
    orth_sketch(A; rtol=1e-8, block_size=42, flag_power=0, skip_tol_check=false, rng=Random.default_rng())

Orthonormal basis Q for col(A) via random sketch.

Direct port of Python libid.py `orth_sketch` (lines 70-112).

# Algorithm (Single-Iteration Replacement)

1. Guard: if block_size >= min(m,n), early exit (flag=1)
2. While true:
   a) Generate uniform[-1,1] test matrix X (n x block_size)
   b) Optional power iteration: X = (A'A)^n X
   c) Sketch: Y = A * X
   d) QR: [Q, R] = qr(Y)
   e) Residual: d = |R[end,end]| / max(column_norms(Y))
   f) If d <= rtol: return (Q, 0)  [NORMAL EXIT]
   g) Else: block_size *= 4, continue
   h) If block_size >= min(m,n): return (empty, 1)  [EARLY EXIT]

# Arguments
- `A::MatrixLike`: Input matrix or LinearOperator (m x n), supports any float type
- `rtol::Union{Real,Nothing}`: Relative tolerance (default: sqrt(eps(T)))
  - Float64: ~ 1.49e-8, Float32: ~ 1.09e-4, BigFloat: scales with precision
- `block_size::Int`: Sketch block size (default: 42). Interpretation depends on mode:
  - **Tolerance mode** (rtol < 1): Starting value for geometric growth
  - **Rank mode** (rtol >= 1): Ignored. Total sketch size is always kmax + extra_samples. Control via extra_samples parameter
- `flag_power::Int`: Number of power iterations (default: 0)
- `skip_tol_check::Bool`: Skip tolerance check and geometric growth. Computes a single
  sketch of size block_size, then filters Q to remove vectors outside the operator's
  range (based on diagonal of R vs column norms of Y). Used for matrix-free rank mode.
- `rng`: Random number generator

# Returns
- `Q::Matrix`: Orthonormal basis (m x k), or empty (m x 0) if early exit
- `flag::Int`: 0=normal exit (tolerance met), 1=early exit (full space)

# Examples
```julia
A = randn(500, 300)
Q, flag = orth_sketch(A)  # Uses sqrt(eps(Float64)) ~ 1.49e-8
# Q is mxk, flag=0 if converged
```
"""
function orth_sketch(A::MatrixLike;
                     rtol::Union{Real,Nothing}=nothing,
                     block_size::Int=42,
                     flag_power::Int=0,
                     skip_tol_check::Bool=false,
                     rng::AbstractRNG=Random.default_rng())

    T = _get_eltype(A)
    rtol_val = something(rtol, _default_rtol(T))
    m, n = size(A)

    # For matrix-free operators in rank mode, skip tolerance check
    if skip_tol_check
        X = _uniform_omega(A, n, block_size, rng=rng)
        X = _power_iteration(A, X, flag_power)
        Y = _matvec(A, X)
        F = qr(Y, ColumnNorm())
        Q_full = Matrix(F.Q)
        R = F.R

        # Filter Q using same criterion as tolerance loop:
        # Keep columns where |R[i,i]| / max(||y_j||) > rtol
        col_norms = [norm(Y[:, j]) for j in 1:size(Y, 2)]
        max_col_norm = maximum(col_norms)
        if max_col_norm == 0
            max_col_norm = 1.0
        end
        diagR = abs.(diag(R))

        # Use machine epsilon as tolerance
        rtol_eps = max(m, n) * eps(real(T))

        # Filter: keep column i if |R[i,i]| / max_col_norm > rtol_eps
        if length(diagR) > 0 && max_col_norm > 0
            d_ratios = diagR / max_col_norm
            keep = d_ratios .> rtol_eps
            rank = sum(keep)
        else
            rank = 0
        end

        # Truncate to actual rank
        Q = Q_full[:, 1:rank]
        return Q, 0
    end

    # Guard against tolerance smaller than machine precision (normal mode only)
    if rtol_val < eps(real(T))
        return zeros(T, m, 0), 1
    end

    # Guard against initial block already covering whole space (normal mode only)
    if block_size >= min(m, n)
        # Early exit - whole space spanned
        return zeros(T, m, 0), 1
    end

    # Main loop - geometric block growth
    while true
        # a) Random test matrix: uniform[-1,1]
        X = _uniform_omega(A, n, block_size, rng=rng)

        # b) Optional power iteration
        X = _power_iteration(A, X, flag_power)

        # c) Sketch the column space
        Y = _matvec(A, X)  # m x block_size

        # d) QR factorization (economy, with pivoting)
        F = qr(Y, ColumnNorm())
        Q = Matrix(F.Q)
        R = F.R

        # e) Residual estimate: |R[end,end]| / max(column_norms)
        # Python: d = abs(R.diagonal()[-1:]) / max(norm(y, axis=0))
        col_norms = [norm(Y[:, j]) for j in 1:size(Y, 2)]
        d = abs(R[end, end]) / maximum(col_norms)

        # f) Check convergence
        if d <= rtol_val
            # Normal termination - tolerance satisfied
            return Q, 0
        end

        # g) Enlarge block size (geometric growth: x4)
        block_size = min(block_size * 4, min(m, n))

        # h) Check if we've exhausted the space
        if block_size >= min(m, n)
            # Early exit - can't grow further
            return zeros(T, m, 0), 1
        end
    end
end


"""
    qr_sketch(A; rtol=1e-8, block_size=42, flag_power=0, extra_samples=12)

Rank-revealing QR factorization via random sketch.

Direct port of Python libid.py `qr_sketch` (lines 117-162).

# Algorithm

1. Handle rank mode: if rtol >= 1, set kmax = floor(rtol), rtol = machine_eps
2. Call orth_sketch(A, rtol, block_size, flag_power) -> (Qs, flag)
3. Determine effective sketch size k
4. If flag != 0 or k >= min(m,n):
   - Fallback: deterministic QR on A
   - Rank from diagonal of R
5. Else:
   - Project: B = Qs' * A
   - Deterministic QR on B (thin matrix)
   - Lift: Q = Qs * Qproj
   - Rank from diagonal of R

# Arguments
- `A::MatrixLike`: Input matrix or LinearOperator (m x n), supports any float type
- `rtol::Union{Real,Nothing}`: Relative tolerance (default: sqrt(eps(T)))
  - If rtol < 1: tolerance mode
  - If rtol >= 1: rank mode (kmax = floor(rtol))
  - For matrix-free LinearOperator, only rank mode (rtol >= 1) is supported
  - Float64 default: ~ 1.49e-8
- `block_size::Int`: Sketch block size (default: 42). Interpretation depends on mode:
  - **Tolerance mode** (rtol < 1): Starting value for geometric growth
  - **Rank mode** (rtol >= 1): Ignored. Total sketch size is always kmax + extra_samples. Control via extra_samples parameter
- `flag_power::Int`: Power iterations (default: 0)
- `extra_samples::Int`: Extra samples for oversampling in rank mode (default: 20)

# Returns
- `Q::Matrix`: Orthogonal factor (m x k)
- `R::Matrix`: Upper triangular (k x n)
- `jpiv::Vector{Int}`: Column permutation (1-based)

# Examples
```julia
A = randn(500, 300)
Q, R, jpiv = qr_sketch(A)  # Uses sqrt(eps(Float64))
# A[:, jpiv] ~ Q * R
```
"""
function qr_sketch(A::MatrixLike;
                   rtol::Union{Real,Nothing}=nothing,
                   block_size::Int=42,
                   flag_power::Int=0,
                   extra_samples::Int=12)

    T = _get_eltype(A)
    rtol_val = something(rtol, _default_rtol(T))
    m, n = size(A)
    is_matrix_free = _is_matrix_free_linop(A)
    is_linop = _is_linop(A)

    # Handle rank mode (rtol >= 1)
    flag_kmax = false
    kmax = min(m, n)
    rtol_for_sketch = rtol_val
    if rtol_val >= 1
        flag_kmax = true
        kmax = Int(floor(rtol_val))

        # In rank mode, block_size is always kmax + extra_samples
        # User controls via extra_samples parameter
        block_size = kmax + extra_samples

        rtol_for_sketch = max(m, n) * eps(real(T))
    elseif is_matrix_free
        # Matrix-free LinearOperator in tolerance mode not supported
        error("Matrix-free LinearOperator only supported in rank mode (rtol >= 1). " *
              "Got rtol=$rtol_val. Please specify target rank as rtol.")
    end

    # In rank mode, skip tolerance check (user specifies exact rank)
    skip_tol = flag_kmax

    # 1) Cheap orthogonal sketch
    Qs, flag = orth_sketch(A, rtol=rtol_for_sketch, block_size=block_size,
                          flag_power=flag_power, skip_tol_check=skip_tol)
    k = size(Qs, 2)

    # Determine if we need fallback
    needs_fallback = (flag != 0 || k >= min(m, n))

    # In rank mode: skip fallback (user requested specific rank, use sketch as-is)
    if needs_fallback && flag_kmax
        needs_fallback = false
    end

    # 2) Full-rank fallback: deterministic QR
    if needs_fallback

        # For explicit LinearOperator, extract the underlying matrix
        A_mat = (is_linop && hasattr(A, :matrix) && A.matrix !== nothing) ? A.matrix : A
        F = qr(A_mat, ColumnNorm())
        Q_full = Matrix(F.Q)
        R_full = F.R
        jpiv = F.p

        # Determine rank from diagonal
        rtol_for_rank = flag_kmax ? max(m, n) * eps(real(T)) : rtol_val
        diag_abs = abs.(diag(R_full))
        if !isempty(diag_abs) && diag_abs[1] > 0
            rank = sum(diag_abs .>= rtol_for_rank * diag_abs[1])
        else
            rank = 0
        end

        # Apply rank limit
        if flag_kmax
            rank = min(kmax, rank)
        end

        return Q_full[:, 1:rank], R_full[1:rank, :], jpiv
    end

    # 3) Project onto sketch space and factor thin matrix
    B = _matmat_left(Qs, A)  # k x n

    # Deterministic QR of thin matrix
    F = qr(B, ColumnNorm())
    Qproj = Matrix(F.Q)
    R = F.R
    jpiv = F.p

    # Lift back to original space
    Q = Qs * Qproj

    # Determine rank from diagonal
    rtol_for_rank = flag_kmax ? max(m, n) * eps(real(T)) : rtol_val
    diag_abs = abs.(diag(R))
    if !isempty(diag_abs) && diag_abs[1] > 0
        rank = sum(diag_abs .>= rtol_for_rank * diag_abs[1])
    else
        rank = 0
    end

    # Apply rank limit
    if flag_kmax
        rank = min(kmax, rank)
    end

    return Q[:, 1:rank], R[1:rank, :], jpiv
end

# Helper function for checking if object has a field (like Python's hasattr)
function hasattr(obj, sym::Symbol)
    return hasproperty(obj, sym)
end


"""
    id_sketch(A; rtol=nothing, block_size=42, flag_power=0, extra_samples=12, use_svd=false, recompute_T=false)

Interpolative decomposition via random sketch using single-stage QR pipeline.

Direct port of Python libid.py `id_sketch` (lines 551-750).

# Algorithm

1. Call qr_sketch(A, rtol, block_size, flag_power) -> (_, R, jpiv)
2. Extract k = size(R, 1)
3. Partition R = [R11 R12] where R11 is kxk, R12 is kx(n-k)
4. Compute T (two modes):
   - recompute_T=true: Recompute from original A via lstsq
   - recompute_T=false: Use R matrix (triangular solve or SVD)
5. Return (k, jpiv, T)

# Mathematical Decomposition

    A[:, jpiv] ~ A[:, jpiv[1:k]] * [I; T]

where:
- k = numerical rank
- jpiv = column permutation
- T = k x (n-k) interpolation matrix

# Arguments
- `A::MatrixLike`: Input matrix or LinearOperator (m x n), supports any float type
- `rtol::Union{Real,Nothing}`: Relative tolerance (default: sqrt(eps(T)))
  - If rtol < 1: tolerance mode
  - If rtol >= 1: rank mode (k = floor(rtol))
  - For matrix-free LinearOperator, only rank mode (rtol >= 1) is supported
  - Float64: ~ 1.49e-8, Float32: ~ 1.09e-4, BigFloat: scales with precision
- `block_size::Int`: Sketch block size (default: 42). Interpretation depends on mode:
  - **Tolerance mode** (rtol < 1): Starting value for geometric growth
  - **Rank mode** (rtol >= 1): Ignored. Total sketch size is always kmax + extra_samples. Control via extra_samples parameter
- `flag_power::Int`: Power iterations (default: 0)
- `extra_samples::Int`: Extra samples for oversampling in rank mode (default: 12)
- `use_svd::Bool`: Use SVD for solving R11*T=R12 (default: false)
- `recompute_T::Bool`: Recompute interpolation matrix T from original A (default: false)
  - **false (default)**: Compute T from R matrix (Fortran's approach).
    Fast (6-9x speedup for matrix-free), but may give error > 1.0 on
    full-rank matrices. Use when speed is critical and higher error
    is acceptable.

  - **true**: Recompute T via least squares on original A.
    Ensures error < 1.0 (mathematically guaranteed).

    For explicit matrices: Direct column indexing.
    For matrix-free operators: Extracts all n columns via unit vectors
    (n matvecs). Slower but guarantees accuracy.

# Returns
- `k::Int`: Numerical rank
- `jpiv::Vector{Int}`: Column permutation (1-based)
- `T::Matrix`: Interpolation matrix (k x (n-k))

# Examples
```julia
A = randn(500, 300)
k, jpiv, T = id_sketch(A)  # Uses sqrt(eps(Float64)) ~ 1.49e-8

# Verify decomposition
A_skel = A[:, jpiv[k+1:end]]
A_basis = A[:, jpiv[1:k]]
err = norm(A_skel - A_basis * T) / norm(A)
# err should be small

# Fast mode for matrix-free operators (expert use)
k, jpiv, T = id_sketch(A_linop, rtol=50.0, recompute_T=false)  # 6-9x faster
```
"""
function id_sketch(A::MatrixLike;
                   rtol::Union{Real,Nothing}=nothing,
                   block_size::Int=42,
                   flag_power::Int=0,
                   extra_samples::Int=12,
                   use_svd::Bool=false,
                   recompute_T::Bool=false)

    T = _get_eltype(A)
    rtol_val = something(rtol, _default_rtol(T))
    is_linop = _is_linop(A)
    is_matrix_free = _is_matrix_free_linop(A)

    # Standard path: QR with column pivoting on sketched A
    _, R, jpiv = qr_sketch(A, rtol=rtol_val, block_size=block_size,
                          flag_power=flag_power, extra_samples=extra_samples)

    k = size(R, 1)
    elemtype = eltype(R)

    # Handle edge cases
    if k == 0
        # Zero rank
        n = size(A, 2)
        return 0, collect(1:n), zeros(elemtype, 0, n)
    end

    n = size(A, 2)
    if k >= n
        # Full rank: no interpolation needed
        return k, jpiv, zeros(elemtype, k, 0)
    end

    # Recompute T from original A for better accuracy
    if recompute_T && k > 0 && k < n
        m = size(A, 1)

        if is_matrix_free
            # Matrix-free path: Extract columns via unit vectors
            # This is slower (n matvecs) but guarantees error < 1.0
            skeleton_cols = zeros(elemtype, m, k)
            for j in 1:k
                e_j = zeros(elemtype, n)
                e_j[jpiv[j]] = 1.0
                skeleton_cols[:, j] = A * e_j
            end

            remaining_cols = zeros(elemtype, m, n - k)
            for j in 1:(n - k)
                e_j = zeros(elemtype, n)
                e_j[jpiv[k + j]] = 1.0
                remaining_cols[:, j] = A * e_j
            end

            # Solve: skeleton_cols * T ~ remaining_cols
            Tmat = skeleton_cols \ remaining_cols
            return k, jpiv, Tmat

        else
            # Explicit matrix path: Direct column indexing
            # Extract the underlying matrix if available
            A_mat = nothing
            if is_linop
                A_mat = A.matrix
            else
                A_mat = A
            end

            if A_mat !== nothing
                # Recompute T via least squares on original A
                # This ensures: A[:, jpiv[k+1:end]] ~ A[:, jpiv[1:k]] * Tmat
                cols = jpiv[1:k]
                remaining = jpiv[k+1:end]
                Tmat = A_mat[:, cols] \ A_mat[:, remaining]
                return k, jpiv, Tmat
            end
        end
    end

    # Fall back to using T from R matrix (may have error > 1.0 for sketches)
    R11 = R[1:k, 1:k]
    R12 = R[1:k, k+1:end]

    # Solve for interpolation matrix Tmat
    if use_svd
        # SVD-based solve (more stable for ill-conditioned R11)
        F = svd(R11)
        U, s, Vt = F.U, F.S, F.Vt
        keep = s .>= rtol_val * maximum(s)

        if !any(keep)
            Tmat = zeros(elemtype, size(R12))
        else
            inv_s = one(eltype(s)) ./ s[keep]
            Tmat = Vt[keep, :]' * (Diagonal(inv_s) * (U[:, keep]' * R12))
        end
    else
        # Triangular solve: R11 * Tmat = R12
        Tmat = UpperTriangular(R11) \ R12
    end

    return k, jpiv, Tmat
end


"""
    svd_sketch(A; rtol=nothing, block_size=42, flag_power=0)

Truncated SVD via random sketch.

Direct port of Python libid.py `svd_sketch` (lines 268-305).

# Algorithm

1. Transpose optimization: If m < n (wide), work on A' instead
2. Handle rank mode: if rtol >= 1, set kmax = floor(rtol), rtol = machine_eps
3. Call orth_sketch(A, rtol, block_size, flag_power) -> (Qs, flag)
4. If flag != 0 or k >= min(m,n):
   - Fallback: deterministic SVD on A
   - Rank from singular values
5. Else:
   - Project: Aproj = Qs' * A
   - SVD of thin matrix: svd(Aproj)
   - Lift: U = Qs * Uproj
   - Rank from singular values

# Transpose Optimization
For wide matrices (m < n), transpose to minimize SVD cost:
- Without: SVD of [kxn] costs O(kn^2)
- With: SVD of [kxm] costs O(km^2)
- Speedup: (n/m)^2

# Arguments
- `A::AbstractMatrix`: Input matrix (m x n), supports any float type
- `rtol::Union{Real,Nothing}`: Relative tolerance (default: sqrt(eps(T)))
  - If rtol < 1: tolerance mode
  - If rtol >= 1: rank mode (kmax = floor(rtol))
  - Float64: ~ 1.49e-8, Float32: ~ 1.09e-4, BigFloat: scales with precision
- `block_size::Int`: Sketch block size (default: 42). Interpretation depends on mode:
  - **Tolerance mode** (rtol < 1): Starting value for geometric growth
  - **Rank mode** (rtol >= 1): Ignored. Total sketch size is always kmax + extra_samples. Control via extra_samples parameter
- `flag_power::Int`: Power iterations (default: 0)
- `extra_samples::Int`: Extra samples for oversampling in rank mode (default: 12)

# Returns
- `U::Matrix`: Left singular vectors (m x k)
- `s::Vector`: Singular values (k,)
- `Vh::Matrix`: Right singular vectors (k x n), conjugate transposed

# Examples
```julia
A = randn(500, 300)
U, s, Vh = svd_sketch(A)  # Uses sqrt(eps(Float64)) ~ 1.49e-8
# A ~ U * Diagonal(s) * Vh
```
"""
function svd_sketch(A::AbstractMatrix{T};
                    rtol::Union{Real,Nothing}=nothing,
                    block_size::Int=42,
                    flag_power::Int=0,
                    extra_samples::Int=12) where T

    rtol_val = something(rtol, _default_rtol(T))
    m, n = size(A)

    # For wide matrices (m < n), transpose to minimize SVD cost
    # The projected matrix becomes kxm instead of kxn, reducing SVD
    # from O(kn^2) to O(km^2), giving speedup of (n/m)^2
    if m < n
        if _is_linop(A)
            # Create transposed LinearOperator with swapped matvec/rmatvec
            A_T = make_linop(T, n, m,
                            (x) -> _rmatvec(A, x),  # matvec uses A's rmatvec
                            (x) -> _matvec(A, x))   # rmatvec uses A's matvec
            # Preserve matrix-free status
            if _is_matrix_free_linop(A)
                A_T.is_explicit = false
            end
        else
            A_T = A'
        end
        V, s, U = svd_sketch(A_T, rtol=rtol_val, block_size=block_size,
                            flag_power=flag_power, extra_samples=extra_samples)
        return U', s, V'
    end

    # Handle rank mode (rtol >= 1)
    is_matrix_free = _is_matrix_free_linop(A)
    flag_kmax = false
    kmax = min(m, n)
    if rtol_val >= 1
        flag_kmax = true
        kmax = Int(floor(rtol_val))

        # In rank mode, block_size is always kmax + extra_samples
        # User controls via extra_samples parameter
        block_size = kmax + extra_samples

        rtol_val = max(m, n) * eps(real(T))
    end

    # In rank mode, skip tolerance check (user specifies exact rank)
    skip_tol = flag_kmax

    # 1) Cheap orthogonal sketch
    Qs, flag = orth_sketch(A, rtol=rtol_val, block_size=block_size,
                          flag_power=flag_power, skip_tol_check=skip_tol)
    k = size(Qs, 2)

    # 2) Full-rank fallback: deterministic SVD
    needs_fallback = (flag != 0 || k >= min(m, n))

    # In rank mode: skip fallback (user requested specific rank, use sketch as-is)
    if needs_fallback && flag_kmax
        needs_fallback = false
    end

    if needs_fallback
        F = svd(A)
        U_full = F.U
        s_full = F.S
        Vh_full = F.Vt

        # Determine rank from singular values
        rank = _rank_from_svals(s_full, rtol_val)

        # Apply rank limit
        if flag_kmax
            rank = min(kmax, rank)
        end

        return U_full[:, 1:rank], s_full[1:rank], Vh_full[1:rank, :]
    end

    # 3) Project onto sketch space and compute SVD
    Aproj = Qs' * A  # k x n

    # SVD of thin matrix
    F = svd(Aproj)
    Uproj = F.U
    s = F.S
    Vh = F.Vt

    # Lift left singular vectors back to original space
    U = Qs * Uproj

    # Determine rank from singular values
    rank = _rank_from_svals(s, rtol_val)

    # Apply rank limit
    if flag_kmax
        rank = min(kmax, rank)
    end

    return U[:, 1:rank], s[1:rank], Vh[1:rank, :]
end

end  # module LibIDSketch
