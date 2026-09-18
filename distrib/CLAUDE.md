# librla/distrib

Multi-language distribution of librla (randomized low-rank factorizations: `orth_sketch`, `qr_sketch`, `svd_sketch`, `id_sketch`, `id_qrpiv`) with matching Python (`python/librla.py`, packaged via `python/pyproject.toml`), MATLAB/Octave (`matlab/librla.m`, a classdef with static methods) and Julia (`julia/librla.jl`) implementations.

## Layout, tests, demos

Each language directory has a `test/` (run `test_all`) and a `demo/` folder plus its own README.md. `fortran/` holds a prebuilt Fortran `test_all` binary, `compare/` has scripts comparing librla against SciPy and PyTorch, and `notes/` is an AI-generated, unverified knowledge base (read its README.md warning before trusting anything there).

## Contracts

- Two modes: `rtol < 1` is tolerance mode (adaptive rank); `rtol >= 1` is rank mode with `k = floor(rtol)`.
- Matrix-free operators (scipy `LinearOperator`, custom `LinearOperator.m` / `LinearOperator.jl`) support both modes, but there is a performance trap: in tolerance mode the deterministic fallback (`_get_matrix` / `get_matrix`, used when the sketch terminates early) may materialize the operator as a dense matrix, one matvec per column, at O(n) matvecs. Prefer rank mode for genuinely matrix-free operators.
- `extra_samples` (default 12) is the buffer beyond the target rank in both modes: rank mode samples `floor(rtol) + extra_samples` columns; tolerance mode accepts a sketch only when at least `extra_samples + 1` pivoted column norms fall at or below `rtol` times the largest. `extra_samples=0` selects the legacy last-column tolerance check.
- Other optional parameters (`block_size`, `power_iter`, `method`, `rng`) are documented in each function's header.

## API differences between languages

svd_sketch return convention (preserve it):

| Language | Returns | Reconstruction |
|----------|---------|----------------|
| Python | `U, s, Vh` | `A = U @ np.diag(s) @ Vh` |
| MATLAB | `U, s, V` | `A = U * diag(s) * V'` |
| Julia | `U, s, Vt` | `A = U * diagm(s) * Vt` |

Python and Julia return V transposed; MATLAB returns V (not transposed).

ID reconstruction identity for `id_sketch` / `id_qrpiv` (`k` skeleton columns, `T` is `k x (n-k)`):

```python
A[:, piv[k:]] = A[:, piv[:k]] @ T          # Python (0-based)
```
```matlab
A(:, piv(k+1:end)) = A(:, piv(1:k)) * T    % MATLAB (1-based)
```
```julia
A[:, piv[k+1:end]] = A[:, piv[1:k]] * T    # Julia (1-based)
```

## Cross-language consistency

When making changes:

1. Maintain API consistency across all three languages; change all three together.
2. Keep function signatures equivalent (accounting for language idioms).
3. Update tests in all three languages when adding features.
4. Ensure documentation stays synchronized in the per-language README.md files.
5. Preserve the svd_sketch return convention (Python/Julia: transposed V; MATLAB: non-transposed V).
