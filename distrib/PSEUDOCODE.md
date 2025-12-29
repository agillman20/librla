# librla Pseudocode

Pseudocode for the randomized linear algebra algorithms in librla.

## orth_sketch

Approximate orthonormal basis for column space via randomized sketching.

```
function orth_sketch(A, rtol, block_size, power_iter):
    Input: A ∈ ℝ^{m×n} or '$\mathbb{C}^{m \times n}$', rtol (tolerance or rank), block_size, power_iter
    Output: Q (orthonormal basis), flag, diagR

    if rtol ≥ 1:  # Rank mode
        k_max = floor(rtol)
        Ω = random_matrix(n, block_size)        # Uniform[-1,1]
        Ω = power_iteration(A, Ω, power_iter)   # Optional: (A^H A)^p Ω
        Y = A Ω
        Q, R, _ = qr_pivoted(Y)
        return Q[:, 1:k_max], 0, |diag(R)|

    # Tolerance mode (rtol < 1): adaptive rank
    while true:
        Ω = random_matrix(n, block_size)
        Ω = power_iteration(A, Ω, power_iter)
        Y = A Ω
        Q, R, _ = qr_pivoted(Y)

        diagR = |diag(R)|
        if diagR[end] / diagR[1] ≤ rtol:
            return Q, 0, diagR

        block_size = min(4 × block_size, min(m, n))
        if block_size ≥ min(m, n):
            return empty(m, 0), 1, []  # Early termination
```

## qr_sketch

Truncated QR factorization with column pivoting.

```
function qr_sketch(A, rtol, block_size, power_iter, extra_samples):
    Input: A ∈ ℝ^{m×n}, rtol, block_size, power_iter, extra_samples
    Output: Q, R, p  such that A[:, p] ≈ Q R

    if rtol ≥ 1:  # Rank mode
        k_max = floor(rtol)
        block_size = k_max + extra_samples

    # Step 1: Sketch orthonormal basis for range(A)
    Q_s, flag, _ = orth_sketch(A, rtol, block_size, power_iter)

    if flag ≠ 0:  # Fallback to full QR
        Q, R, p = qr_pivoted(A)
        k = rank_from_diagonal(diag(R), rtol)
        return Q[:, 1:k], R[1:k, :], p

    # Step 2: Project onto sketch basis
    B = Q_s^H A                    # k × n projected matrix

    # Step 3: QR of small projected matrix
    Q_proj, R, p = qr_pivoted(B)

    # Step 4: Expand back
    Q = Q_s Q_proj
    k = rank_from_diagonal(diag(R), rtol)

    return Q[:, 1:k], R[1:k, :], p
```

## svd_sketch

Truncated SVD via randomized sketching.

```
function svd_sketch(A, rtol, block_size, power_iter, extra_samples):
    Input: A ∈ ℝ^{m×n}, rtol, block_size, power_iter, extra_samples
    Output: U, s, V^H  such that A ≈ U diag(s) V^H

    if m < n:  # Handle wide matrices via transpose
        V, s, U = svd_sketch(A^H, rtol, ...)
        return U^H, s, V^H

    if rtol ≥ 1:  # Rank mode
        k_max = floor(rtol)
        block_size = k_max + extra_samples

    # Step 1: Sketch orthonormal basis for range(A)
    Q_s, flag, _ = orth_sketch(A, rtol, block_size, power_iter)

    if flag ≠ 0:  # Fallback to full SVD
        U, s, V = svd(A)
        k = rank_from_diagonal(s, rtol)
        return U[:, 1:k], s[1:k], V[1:k, :]

    # Step 2: Project onto sketch basis
    B = Q_s^H A                    # k × n projected matrix

    # Step 3: SVD of small projected matrix
    U_proj, s, V = svd(B)

    # Step 4: Expand left singular vectors
    U = Q_s U_proj
    k = rank_from_diagonal(s, rtol)

    return U[:, 1:k], s[1:k], V[1:k, :]
```

## id_sketch

Interpolative decomposition via randomized sketching.

```
function id_sketch(A, rtol, block_size, power_iter, extra_samples, method):
    Input: A ∈ ℝ^{m×n}, rtol, method ∈ {'fast', 'svd', 'lstsq'}
    Output: k, piv, T  such that A[:, piv[k+1:end]] ≈ A[:, piv[1:k]] T

    # Step 1: Get column permutation via QR sketch
    _, R, piv = qr_sketch(A, rtol, block_size, power_iter, extra_samples)
    k = rows(R)

    # Step 2: Compute interpolation matrix T
    R₁₁ = R[1:k, 1:k]
    R₁₂ = R[1:k, k+1:n]

    if method == 'fast':
        T = R₁₁ \ R₁₂              # Triangular solve

    else if method == 'svd':
        U, s, V^H = svd(R₁₁)
        T = V s⁻¹ U^H R₁₂          # Pseudoinverse

    else if method == 'lstsq':
        A_skel = A[:, piv[1:k]]
        A_rest = A[:, piv[k+1:end]]
        T = lstsq(A_skel, A_rest)  # Least squares

    return k, piv, T
```

## id_qrpiv

Interpolative decomposition via deterministic QR (no randomization).

```
function id_qrpiv(A, rtol, method):
    Input: A ∈ ℝ^{m×n}, rtol, method ∈ {'fast', 'svd', 'lstsq'}
    Output: k, piv, T  such that A[:, piv[k+1:end]] ≈ A[:, piv[1:k]] T

    # Step 1: Full QR with column pivoting (LAPACK geqp3)
    Q, R, piv = qr_pivoted(A)
    k = rank_from_diagonal(diag(R), rtol)

    # Step 2: Compute interpolation matrix T (same as id_sketch)
    R₁₁ = R[1:k, 1:k]
    R₁₂ = R[1:k, k+1:n]

    if method == 'fast':
        T = R₁₁ \ R₁₂
    else if method == 'svd':
        U, s, V^H = svd(R₁₁)
        T = V s⁻¹ U^H R₁₂
    else if method == 'lstsq':
        T = lstsq(A[:, piv[1:k]], A[:, piv[k+1:end]])

    return k, piv, T
```

## Helper: power_iteration

```
function power_iteration(A, Ω, p):
    Input: A ∈ ℝ^{m×n}, Ω ∈ ℝ^{n×k}, p (number of iterations)
    Output: Ω after p iterations of subspace iteration

    for i = 1 to p:
        Ω = A^H (A Ω)
        Ω, _, _ = qr_pivoted(Ω)    # Re-orthogonalize
    return Ω
```

## Helper: rank_from_diagonal

```
function rank_from_diagonal(d, rtol):
    Input: d (diagonal values, sorted descending), rtol
    Output: numerical rank

    if rtol ≥ 1:
        return min(floor(rtol), length(d))
    else:
        return count(|d| ≥ rtol × |d[1]|)
```

## Complexity

| Function | Complexity | Notes |
|----------|------------|-------|
| `orth_sketch` | O(mnk + nk²) | k = sketch size |
| `qr_sketch` | O(mnk + nk²) | k = target rank |
| `svd_sketch` | O(mnk + nk²) | k = target rank |
| `id_sketch` | O(mnk + nk²) | k = target rank |
| `id_qrpiv` | O(mn²) | Full QR, deterministic |

For k << min(m,n), randomized methods are O(mnk) vs O(mn·min(m,n)) for full decompositions.
