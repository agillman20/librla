# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Repository Overview

This is the **distrib/** directory of the librla (randomized linear algebra) library. It contains a multi-language distribution with identical APIs across Python, MATLAB/Octave, and Julia for low-rank matrix approximations using randomized sketching algorithms.

## Core Architecture

### Unified API Design

All three language implementations expose the same five core functions:

| Function | Description |
|----------|-------------|
| `orth_sketch(A, rtol)` | Approximate orthonormal basis for column space |
| `qr_sketch(A, rtol)` | Truncated QR factorization |
| `svd_sketch(A, rtol)` | Truncated SVD |
| `id_sketch(A, rtol)` | Interpolative decomposition (randomized) |
| `id_qrpiv(A, rtol)` | Interpolative decomposition (deterministic) |

### Two Operating Modes

- **Tolerance mode** (rtol < 1): Adaptive rank selection to achieve specified accuracy
- **Rank mode** (rtol >= 1): Fixed-rank approximation

### LinearOperator Abstraction

All implementations support matrix-free computation through LinearOperator:

| Language | Implementation |
|----------|---------------|
| Python | `scipy.sparse.linalg.LinearOperator` (standard library) |
| MATLAB | Custom `LinearOperator.m` class |
| Julia | Custom `LinearOperator.jl` type |

Matrix-free operators only support rank mode (rtol >= 1).

## Language-Specific Details

### Python (python/)

- Main library: `librla.py`
- Uses NumPy and SciPy
- Functions are module-level exports
- LinearOperator via scipy (no custom class needed)
- 0-based indexing

### MATLAB/Octave (matlab/)

- Main library: `librla.m`
- Implemented as `classdef` with static methods
- All functions accessed as `librla.method_name(...)`
- Custom `LinearOperator.m` class
- 1-based indexing

### Julia (julia/)

- Main library: `librla.jl`
- Implemented as a module
- Functions exported at module level
- Custom `LinearOperator.jl` type
- 1-based indexing

## API Differences Between Languages

### svd_sketch Return Values

| Language | Returns | Reconstruction |
|----------|---------|----------------|
| Python | `U, s, Vh` | `A = U @ np.diag(s) @ Vh` |
| MATLAB | `U, s, V` | `A = U * diag(s) * V'` |
| Julia | `U, s, Vt` | `A = U * diagm(s) * Vt` |

Python and Julia return V transposed; MATLAB returns V (not transposed).

### Indexing in id_sketch/id_qrpiv

```python
# Python (0-based)
A[:, piv[k:]] = A[:, piv[:k]] @ T
```

```matlab
% MATLAB (1-based)
A(:, piv(k+1:end)) = A(:, piv(1:k)) * T
```

```julia
# Julia (1-based)
A[:, piv[k+1:end]] = A[:, piv[1:k]] * T
```

## Testing

Each language directory contains 8 test files:

| Test | Description |
|------|-------------|
| `test1_hilbert` | Hilbert matrix (ill-conditioned) |
| `test1_kahan` | Kahan matrix |
| `test2_svd_hilbert` | SVD on Hilbert matrices |
| `test3_linop_hilbert` | LinearOperator with Hilbert |
| `test4_linop_random` | LinearOperator with random matrices |
| `test5_linop_fullrank` | Full-rank matrix tests |
| `test5_method_comparison` | Compare T computation methods |
| `test6_power` | Power iteration tests |

### Running Tests

```bash
# Python
cd python && python test1_hilbert.py

# MATLAB
cd matlab
test1_hilbert

# Julia
cd julia && julia -e 'include("test1_hilbert.jl")'
```

## Utility Files

Each language directory includes:

| File | Description |
|------|-------------|
| `hilb.*` | Hilbert matrix generator |
| `kahan.*` | Kahan matrix generator |
| `make_mat.*` | Matrix generation utilities |
| `compare_id.*` | Comparison example for ID methods |

## Key Algorithmic Features

### id_sketch Method Options

The `method` parameter controls T matrix computation:

| Method | Description |
|--------|-------------|
| `'fast'` | Triangular solve (default, fastest) |
| `'svd'` | SVD-based pseudoinverse (stable for ill-conditioned) |
| `'lstsq'` | Least squares from original A (most accurate, slowest) |

### Optional Parameters

All functions accept:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 42 | Initial sketch size |
| `power_iter` | 0 | Power iterations for accuracy |
| `extra_samples` | 12 | Oversampling for rank mode |

## Cross-Language Consistency

When making changes:

1. Maintain API consistency across all three languages
2. Keep function signatures equivalent (accounting for language idioms)
3. Update tests in all three languages when adding features
4. Ensure documentation stays synchronized in README.md and INSTALL.md
5. Preserve the svd_sketch return convention (Python/Julia: transposed V; MATLAB: non-transposed V)

## File Organization

```
distrib/
├── README.md           # Main documentation
├── INSTALL.md          # Installation instructions
├── CLAUDE.md           # This file
├── TODO.md             # Development tasks
├── FILE_MANIFEST.txt   # File listing
├── python/             # Python implementation
├── matlab/             # MATLAB/Octave implementation
└── julia/              # Julia implementation
```
