# libid - Randomized Linear Algebra Routines

A compact, multi-language toolbox implementing randomized algorithms for low-rank matrix factorizations. Provides efficient sketching-based methods for large-scale matrix decompositions with high accuracy and performance.

## Features

### Core Algorithms

- **`orth_sketch`** - Orthonormal basis for the column space via random sketching
- **`qr_sketch`** - Truncated QR factorization with column pivoting
- **`utv_sketch`** - Truncated UTV factorization (A = U x T x V<sup>H</sup>)
- **`cwr_sketch`** - CWR decomposition (A ~ C x (W \ R) or (C / W) x R)
- **`svd_sketch`** - Truncated SVD via random sketching
- **`id_sketch`** - Interpolative decomposition (ID) using random sketching

### LinearOperator Support

All sketching algorithms support **LinearOperator** abstraction for matrix-free computation:

- **Dense matrices** - Standard NumPy/MATLAB/Julia arrays
- **Explicit LinearOperators** - Matrix wrappers with unified API
- **Matrix-free LinearOperators** - Function handles for `A*x` and `A'*x` only

Matrix-free operators enable sketching of **implicit matrices** (FFT, convolution, Toeplitz, etc.) without explicit storage.

### Interpolative Decomposition (ID) Implementations

Multiple ID implementations for different use cases:

1. **`libid_sketch`** - Randomized QR sketching (default, fastest)
2. **`libid_rrqr`** - Deterministic RRQR using LAPACK geqp3

### Utility Functions

- **`gaussian_omega` / `_gaussian_omega`** - Generate Gaussian test matrices (real or complex)
- **`uniform_omega` / `_uniform_omega`** - Generate uniform[-1,1] test matrices
- **`hilb` / `_hilb`** - Generate rectangular Hilbert matrices
- **`safe_max_abs` / `_safe_max_abs`** - Safe maximum absolute value computation
- **`power_iteration` / `_power_iteration`** - Apply (A<sup>H</sup>A)<sup>n</sup> for sketch improvement

## Language Implementations

### Python
- **Main library:** `libid.py`
- **ID variants:** `libid_rrqr.py`
- **LinearOperator:** `make_linop.py`
- **Comparison:** `compare_id.py`, `compare_svd.py`
- **Matrix generation:** `make_mat.py`

### MATLAB/Octave
- **Main library:** `libid.m` (classdef with static methods)
- **ID variants:** `libid_rrqr.m`
- **LinearOperator:** `make_linop.m`
- **Comparison:** `compare_id.m`, `compare_svd.m`
- **Matrix generation:** `make_mat.m`

### Julia
- **Modules:** `LibIDRRQR.jl`, `LibIDSketch.jl`
- **LinearOperator:** `make_linop.jl`
- **Comparison:** `compare_id.jl`, `compare_svd.jl`

## Quick Start

### Python

```python
import numpy as np
from libid import id_sketch, qr_sketch, svd_sketch, _hilb

# Create a Hilbert matrix (ill-conditioned, rapidly-decaying singular values)
A = _hilb(1000, 500)

# Interpolative decomposition with relative tolerance
k, piv, T = id_sketch(A, rtol=1e-6)
# Reconstruction: A[:, piv[k:]] ~ A[:, piv[:k]] @ T

# Truncated QR factorization
Q, R, p = qr_sketch(A, rtol=1e-6, block_size=42)
# A[:, p] ~ Q @ R

# Truncated SVD
U, s, Vh = svd_sketch(A, rtol=1e-6)
# A ~ U @ np.diag(s) @ Vh
```

### MATLAB/Octave

```matlab
% Create a Hilbert matrix (ill-conditioned, rapidly-decaying singular values)
A = libid.hilb(1000, 500);

% Interpolative decomposition
[k, piv, T] = libid.id_sketch(A, 1e-6);
% Reconstruction: A(:, piv(k+1:end)) ~ A(:, piv(1:k)) * T

% Truncated QR factorization
[Q, R, p] = libid.qr_sketch(A, 1e-6, 42);
% A(:, p) ~ Q * R

% Truncated SVD
[U, s, Vh] = libid.svd_sketch(A, 1e-6);
% A ~ U * s * Vh
```

### Julia

```julia
using LibIDSketch: id_sketch, qr_sketch, svd_sketch

# Create a Hilbert matrix
A = [1.0/(i+j-1) for i=1:1000, j=1:500]

# Interpolative decomposition
k, piv, T = id_sketch(A, rtol=1e-6)
# Reconstruction: A[:, piv[k+1:end]] ~ A[:, piv[1:k]] * T

# Truncated SVD
U, s, Vh = svd_sketch(A, rtol=1e-6)
# A ~ U * Diagonal(s) * Vh
```

## LinearOperator Examples

### Python - Matrix-Free Operators

```python
from libid import id_sketch
from make_linop import make_linop

# Matrix-free operator with function handles
def matvec(x):
    return A @ x  # Your custom forward operation

def rmatvec(x):
    return A.conj().T @ x  # Your custom adjoint operation

A_linop = make_linop(m, n, matvec, rmatvec)

# Sketch without storing the matrix (rank mode only)
k, piv, T = id_sketch(A_linop, rtol=50.0)
```

### MATLAB - Matrix-Free Operators

```matlab
% Matrix-free operator with function handles
Afun = @(x) A * x;        % Forward operation
ATfun = @(x) A' * x;      % Adjoint operation

A_linop = make_linop(m, n, Afun, ATfun, 'double');

% Sketch without storing the matrix (rank mode only)
[k, piv, T] = libid.id_sketch(A_linop, 50);
```

## Algorithm Details

### Random Sketching

The library uses **random sketching** to efficiently compute low-rank approximations:

1. **Test matrix generation** - Create random test matrix Omega (Gaussian or uniform)
2. **Power iteration** (optional) - Apply (A<sup>H</sup>A)<sup>n</sup> to improve sketch quality
3. **Sketch construction** - Compute Y = A x Omega
4. **Orthogonalization** - QR factorization to obtain orthonormal basis
5. **Adaptive expansion** - Geometrically increase sketch size until tolerance is met

### Tolerance and Rank Selection

- **`rtol < 1`**: Relative tolerance for rank determination
  - Automatically selects rank k such that truncation error <= rtol
  - **Note**: Matrix-free operators only support rank mode (rtol >= 1)
- **`rtol >= 1`**: Maximum rank constraint
  - Returns decomposition with rank k = floor(rtol)

### recompute_T Parameter (ID Sketch)

The `recompute_T` parameter controls interpolation matrix computation:

- **`recompute_T=False`** (default) - Fast, uses R matrix from sketch (Fortran approach)
  - 6-9x faster for matrix-free operators
  - May give error > 1.0 on full-rank matrices
  - Matches original Fortran libid behavior

- **`recompute_T=True`** - Accurate, recomputes T via least squares
  - Guarantees error < 1.0 (mathematically proven)
  - For matrix-free: Extracts all n columns via unit vectors (slower)
  - Use when accuracy is critical

### Performance Optimizations

- **Adaptive sketching** - Geometric growth of block size (x4 each iteration)
- **Early termination** - Stops when sketch spans full space
- **Tall matrix optimization** - For m > n, CWR works on A<sup>H</sup> to reduce cost from O(mk<sup>2</sup>) to O(nk<sup>2</sup>)
- **Power iteration** - Optional A<sup>H</sup>A iteration to improve sketch accuracy
- **Matrix-free mode** - For LinearOperators, uses optimized adjoint matvec for projection

## Test Suite

Comprehensive demonstration tests available for all languages:

### Basic Tests
- **`test1_hilbert`** - Basic ID algorithms (id_sketch vs id_rrqr) on Hilbert matrix
- **`test2_svd_hilbert`** - Basic SVD algorithms (svd_sketch vs LAPACK SVD)

### LinearOperator Tests
- **`test3_linop_hilbert`** - LinearOperators on ill-conditioned Hilbert matrix
- **`test4_linop_random`** - LinearOperators on low-rank random matrix
- **`test5_linop_fullrank`** - LinearOperators on full-rank matrix (demonstrates recompute_T)

### Advanced Tests
- **`test_power_iteration`** - Power iteration for range estimation

### Running Tests

```bash
# Python
cd python
python3 test1_hilbert.py
python3 test2_svd_hilbert.py
python3 test3_linop_hilbert.py
python3 test4_linop_random.py
python3 test5_linop_fullrank.py
python3 test_power_iteration.py

# MATLAB/Octave
cd matlab
octave --no-gui --eval "test1_hilbert"
octave --no-gui --eval "test2_svd_hilbert"
octave --no-gui --eval "test3_linop_hilbert"
octave --no-gui --eval "test4_linop_random"
octave --no-gui --eval "test5_linop_fullrank"
octave --no-gui --eval "test_power_iteration"

# Julia
cd julia
julia test1_hilbert.jl
julia test2_svd_hilbert.jl
julia test3_linop_hilbert.jl
julia test4_linop_random.jl
julia test5_linop_fullrank.jl
julia test_power_iteration.jl
```

## Benchmarking

Compare different ID implementations using the comparison scripts:

```bash
# Python
python compare_id.py
python compare_svd.py

# MATLAB/Octave
matlab -batch "compare_id"
matlab -batch "compare_svd"

# Julia
julia compare_id.jl
julia compare_svd.jl
```

Comparison metrics:
- **Accuracy** - Reconstruction error ||A - A_approx||
- **Conditioning** - max|T| for interpolation matrix
- **Runtime** - Execution time
- **Rank selection** - Effective rank k chosen by each method

## Test Matrices

The `make_mat` module generates various test matrices:

- **Exponential decay** - `exp_decay(m, n, sigma_values)`
- **Power-law decay** - `power_decay(m, n, alpha)`
- **Hilbert matrices** - Ill-conditioned matrices with sigma_i ~ 1/i
- **Random low-rank** - Controlled rank-k matrices
- **Polynomial decay** - Various decay rates

## Documentation

Detailed documentation available for specific topics:

- **`TEST1_HILBERT.md`** - Basic ID test documentation
- **`TEST2_SVD_HILBERT.md`** - Basic SVD test documentation
- **`TEST3_LINOP_HILBERT.md`** - LinearOperator test with Hilbert matrix
- **`TEST4_LINOP_RANDOM.md`** - LinearOperator test with low-rank random matrix
- **`TEST5_LINOP_FULLRANK.md`** - LinearOperator test with full-rank matrix
- **`TEST_POWER_ITERATION.md`** - Power iteration for range estimation

## Dependencies

### Python
- NumPy
- SciPy

### MATLAB/Octave
- Base MATLAB/Octave (no additional toolboxes required)
- Compatible with Octave 6.0+

### Julia
- LinearAlgebra (standard library)
- LAPACK bindings

## Performance Notes

### When to Use Each Method

- **`id_sketch`** (default) - Best for large matrices, fastest, good accuracy
- **`id_rrqr`** - Deterministic, reproducible results, moderate speed

### Block Size Selection

Default `block_size=42` works well for most cases. Consider adjusting:
- **Larger** (64-128) - For very large matrices or slowly-decaying spectra
- **Smaller** (16-32) - For small matrices or when memory is constrained

### Power Iteration

Use `flag_power > 0` when:
- Matrix has slowly-decaying singular values
- Higher accuracy is needed
- Sketch quality needs improvement

**Typical values:**
- `flag_power=0` - Default, fast
- `flag_power=1-2` - Good balance (2-3x cost, ~100-1000x accuracy improvement)
- `flag_power=3-4` - High accuracy (diminishing returns beyond 4)

Trade-off: Each power iteration doubles the cost but squares the spectral gap.

### LinearOperator Performance

**Matrix-free operators:**
- Use rank mode only (`rtol >= 1`)
- `recompute_T=False`: 6-9x faster, may have error > 1.0
- `recompute_T=True`: Guarantees error < 1.0, requires n matvecs
- Best for large structured matrices (FFT, Toeplitz, sparse, etc.)

## License

SPDX-License-Identifier: TBD

## Author

Your Name

## References

The algorithms implemented here are based on:

1. Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217-288.

2. Liberty, E., Woolfe, F., Martinsson, P. G., Rokhlin, V., & Tygert, M. (2007). Randomized algorithms for the low-rank approximation of matrices. *PNAS*, 104(51), 20167-20172.

3. Martinsson, P. G., & Voronin, S. (2016). A randomized blocked algorithm for efficiently computing rank-revealing factorizations of matrices. *SIAM Journal on Scientific Computing*, 38(5), S485-S507.

## Repository Structure

```
distrib/
|---- README.md                  # This file
|---- python/
|     |---- libid.py             # Python main library
|     |---- make_linop.py        # LinearOperator factory
|     |---- compare_id.py        # ID benchmark
|     |---- compare_svd.py       # SVD benchmark
|     |---- test1_hilbert.py     # Basic ID test
|     |---- test2_svd_hilbert.py # Basic SVD test
|     |---- test3_linop_hilbert.py    # LinearOperator test (Hilbert)
|     |---- test4_linop_random.py     # LinearOperator test (low-rank)
|     |---- test5_linop_fullrank.py   # LinearOperator test (full-rank)
|     `---- test_power_iteration.py   # Power iteration demo
|---- matlab/
|     |---- libid.m              # MATLAB/Octave main library
|     |---- make_linop.m         # LinearOperator factory
|     |---- compare_id.m         # ID benchmark
|     |---- compare_svd.m        # SVD benchmark
|     |---- test1_hilbert.m      # Basic ID test
|     |---- test2_svd_hilbert.m  # Basic SVD test
|     |---- test3_linop_hilbert.m     # LinearOperator test (Hilbert)
|     |---- test4_linop_random.m      # LinearOperator test (low-rank)
|     |---- test5_linop_fullrank.m    # LinearOperator test (full-rank)
|     `---- test_power_iteration.m    # Power iteration demo
|---- julia/
|     |---- LibIDSketch.jl       # Julia sketch module
|     |---- make_linop.jl        # LinearOperator factory
|     |---- compare_id.jl        # ID benchmark
|     |---- compare_svd.jl       # SVD benchmark
|     |---- test1_hilbert.jl     # Basic ID test
|     |---- test2_svd_hilbert.jl # Basic SVD test
|     |---- test3_linop_hilbert.jl    # LinearOperator test (Hilbert)
|     |---- test4_linop_random.jl     # LinearOperator test (low-rank)
|     |---- test5_linop_fullrank.jl   # LinearOperator test (full-rank)
|     `---- test_power_iteration.jl   # Power iteration demo
`---- Documentation
      |---- TEST1_HILBERT.md          # Basic ID test guide
      |---- TEST2_SVD_HILBERT.md      # Basic SVD test guide
      |---- TEST3_LINOP_HILBERT.md    # LinearOperator (Hilbert) guide
      |---- TEST4_LINOP_RANDOM.md     # LinearOperator (low-rank) guide
      |---- TEST5_LINOP_FULLRANK.md   # LinearOperator (full-rank) guide
      `---- TEST_POWER_ITERATION.md   # Power iteration guide
```
