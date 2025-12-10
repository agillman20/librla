# librla - Randomized Linear Algebra Library

A unified, multi-language library implementing randomized algorithms for low-rank matrix approximations. Provides efficient sketching-based methods for large-scale matrix decompositions with consistent APIs across Python, MATLAB/Octave, and Julia.

## Features

### Core Algorithms

All algorithms support both **tolerance mode** (rtol < 1) for adaptive rank selection and **rank mode** (rtol >= 1) for fixed-rank approximations:

| Function | Description |
|----------|-------------|
| `orth_sketch` | Orthonormal basis for column space via randomized range finding |
| `qr_sketch` | Truncated QR factorization with column pivoting |
| `svd_sketch` | Truncated singular value decomposition |
| `id_sketch` | Interpolative decomposition via randomized sketching |
| `id_qrpiv` | Interpolative decomposition via deterministic QR pivoting |

### LinearOperator Support

All algorithms support **LinearOperator** abstraction for matrix-free computation:

- **Dense matrices** - Standard NumPy/MATLAB/Julia arrays
- **Explicit LinearOperators** - Matrix wrappers with unified interface
- **Matrix-free LinearOperators** - Function handles for `A*x` and `A'*x` operations only

Matrix-free operators enable sketching of **implicit matrices** (FFT, convolution, Toeplitz, circulant, etc.) without explicit storage.

### Method Options for ID

The `id_sketch` and `id_qrpiv` functions support three methods for computing the interpolation matrix T:

| Method | Description |
|--------|-------------|
| `'fast'` | Triangular solve (fastest, default) |
| `'svd'` | SVD-based pseudoinverse (stable for ill-conditioned matrices) |
| `'lstsq'` | Least-squares from original A (most accurate, slowest) |

## Quick Start

### Python

```python
import numpy as np
from librla import orth_sketch, qr_sketch, svd_sketch, id_sketch
from hilb import hilb

# Create a test matrix (Hilbert matrix is ill-conditioned)
A = hilb(1000, 500)

# Orthonormal basis with relative tolerance
Q, flag, err = orth_sketch(A, rtol=1e-6)

# Truncated QR factorization
Q, R, p = qr_sketch(A, rtol=1e-6)
# A[:, p] approx Q @ R

# Truncated SVD
U, s, Vh = svd_sketch(A, rtol=1e-6)
# A approx U @ np.diag(s) @ Vh

# Interpolative decomposition
k, piv, T = id_sketch(A, rtol=1e-6, method='lstsq')
# A[:, piv[k:]] approx A[:, piv[:k]] @ T
```

### MATLAB/Octave

```matlab
% Create a test matrix
A = hilb(1000, 500);

% Orthonormal basis
[Q, flag, err] = librla.orth_sketch(A, 1e-6);

% Truncated QR factorization
[Q, R, p] = librla.qr_sketch(A, 1e-6);
% A(:, p) approx Q * R

% Truncated SVD
[U, s, V] = librla.svd_sketch(A, 1e-6);
% A approx U * diag(s) * V'

% Interpolative decomposition
[k, piv, T] = librla.id_sketch(A, 1e-6, 'method', 'lstsq');
% A(:, piv(k+1:end)) approx A(:, piv(1:k)) * T
```

### Julia

```julia
include("librla.jl")
using .librla

# Create a test matrix
A = hilb(1000, 500)

# Orthonormal basis
Q, flag, err = orth_sketch(A, 1e-6)

# Truncated QR factorization
Q, R, p = qr_sketch(A, 1e-6)
# A[:, p] approx Q * R

# Truncated SVD
U, s, Vt = svd_sketch(A, 1e-6)
# A approx U * diagm(s) * Vt

# Interpolative decomposition
k, piv, T = id_sketch(A, 1e-6, method="lstsq")
# A[:, piv[k+1:end]] approx A[:, piv[1:k]] * T
```

## Usage Modes

### Tolerance Mode (rtol < 1)

Adaptively determines rank to achieve specified relative accuracy:

```python
# Python: Adaptive rank selection to achieve 10^-6 relative accuracy
Q, flag, err = orth_sketch(A, 1e-6)
```

```matlab
% MATLAB: Same behavior
[Q, flag, err] = librla.orth_sketch(A, 1e-6);
```

### Rank Mode (rtol >= 1)

Returns fixed-rank approximation:

```python
# Python: Rank-20 approximation
U, s, Vh = svd_sketch(A, 20.0)
```

```matlab
% MATLAB: Rank-20 approximation
[U, s, V] = librla.svd_sketch(A, 20);
```

## API Notes

### svd_sketch Return Values

The `svd_sketch` function returns slightly different formats across languages:

| Language | Returns | Reconstruction |
|----------|---------|----------------|
| Python | `U, s, Vh` | `A approx U @ np.diag(s) @ Vh` |
| MATLAB | `U, s, V` | `A approx U * diag(s) * V'` |
| Julia | `U, s, Vt` | `A approx U * diagm(s) * Vt` |

Python and Julia return V transposed/adjoint; MATLAB returns V (requiring explicit transpose in reconstruction).

### Indexing

- **Python**: 0-based indexing (piv[:k] for skeleton columns)
- **MATLAB/Julia**: 1-based indexing (piv(1:k) or piv[1:k] for skeleton columns)

## Algorithm Details

### Randomized Sketching (id_sketch)

- Uses random test matrix multiplication for fast column space approximation
- Geometric block growth for adaptive rank determination in tolerance mode
- Optional power iterations for improved accuracy (`power_iter` parameter)
- Typically 2-5x faster than deterministic methods
- Stochastic (results vary slightly between runs)

### Deterministic QR Pivoting (id_qrpiv)

- Uses LAPACK geqp3 column-pivoted QR factorization
- Deterministic and reproducible results
- Slower than randomized sketching but guaranteed behavior
- Same interface as `id_sketch`
- Useful for verification and when reproducibility is critical

### Accuracy of Randomized Methods

Randomized sketching algorithms have inherent variance in reconstruction error. In rank mode (rtol >= 1), the reconstruction error of randomized methods is typically within a small factor of the optimal (deterministic) error. The validation tests use these thresholds:

| Function | Threshold | Description |
|----------|-----------|-------------|
| `svd_sketch` | 4x | Error within 4x of truncated SVD |
| `qr_sketch` | 4x | Error within 4x of pivoted QR |
| `orth_sketch` | 8x | Span error within 8x of optimal |
| `id_sketch` | 10.0 | Absolute error < 10.0 (lenient for full-rank matrices) |

This variance is expected for randomized algorithms and does not indicate a bug.

For ill-conditioned matrices (e.g., Hilbert matrices), use **power iterations** to improve accuracy:

```python
# Use power_iter=2 for ill-conditioned matrices
U, s, Vh = svd_sketch(A, 20, power_iter=2)
```

With `power_iter=2`, randomized methods achieve near-optimal accuracy even for severely ill-conditioned matrices.

## LinearOperator Usage

Create matrix-free operators for implicit matrices:

### Python

Python uses `scipy.sparse.linalg.LinearOperator`:

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator
from librla import orth_sketch

# Define matrix-vector products
def matvec(x):
    return np.fft.fft(x)

def rmatvec(x):
    return np.fft.ifft(x).real

# Create LinearOperator
n = 1000
A_op = LinearOperator(shape=(n, n), matvec=matvec, rmatvec=rmatvec, dtype=complex)

# Use with librla (rank mode only for matrix-free operators)
Q, flag, err = orth_sketch(A_op, 20.0)  # Rank-20 approximation
```

### MATLAB

MATLAB uses the custom `LinearOperator` class:

```matlab
% Create LinearOperator
n = 1000;
matvec_fun = @(x) fft(x);
rmatvec_fun = @(x) ifft(x);
A_op = LinearOperator(matvec_fun, rmatvec_fun, n, n);

% Use with librla
[Q, flag, err] = librla.orth_sketch(A_op, 20);
```

### Julia

Julia uses the custom `LinearOperator` type:

```julia
include("LinearOperator.jl")

# Create LinearOperator
n = 1000
matvec_fun(x) = fft(x)
rmatvec_fun(x) = ifft(x)
A_op = LinearOperator(matvec_fun, rmatvec_fun, n, n; dtype=ComplexF64)

# Use with librla
Q, flag, err = orth_sketch(A_op, 20.0)
```

**Note:** Matrix-free LinearOperators only support **rank mode** (rtol >= 1).

## Optional Parameters

All sketching functions support these optional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 42 | Initial sketch size for tolerance mode |
| `power_iter` | 0 | Number of power iterations for accuracy |
| `extra_samples` | 12 | Oversampling for rank mode |

For `id_sketch` and `id_qrpiv` only:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | 'fast' | T matrix computation: 'fast', 'svd', or 'lstsq' |

Example with optional parameters:

```python
# Python: Use power iterations for better accuracy
k, piv, T = id_sketch(A, 1e-6, power_iter=2, method='svd')
```

```matlab
% MATLAB: Same options
[k, piv, T] = librla.id_sketch(A, 1e-6, 'power_iter', 2, 'method', 'svd');
```

```julia
# Julia: Named arguments
k, piv, T = id_sketch(A, 1e-6, power_iter=2, method="svd")
```

## File Structure

```
distrib/
├── README.md              # This file
├── INSTALL.md             # Installation instructions
├── FILE_MANIFEST.txt      # Complete file listing
├── python/
│   ├── librla.py          # Main library
│   ├── hilb.py            # Hilbert matrix generator
│   ├── kahan.py           # Kahan matrix generator
│   ├── make_mat.py        # Matrix generation utilities
│   ├── compare_id.py      # Comparison example
│   └── test*.py           # Test suite (9 files)
├── matlab/
│   ├── librla.m           # Main library
│   ├── LinearOperator.m   # Matrix-free operator class
│   ├── hilb.m             # Hilbert matrix generator
│   ├── kahan.m            # Kahan matrix generator
│   ├── make_mat.m         # Matrix generation utilities
│   ├── compare_id.m       # Comparison example
│   └── test*.m            # Test suite (9 files)
└── julia/
    ├── librla.jl          # Main library
    ├── LinearOperator.jl  # Matrix-free operator type
    ├── hilb.jl            # Hilbert matrix generator
    ├── kahan.jl           # Kahan matrix generator
    ├── make_mat.jl        # Matrix generation utilities
    ├── compare_id.jl      # Comparison example
    └── test*.jl           # Test suite (9 files)
```

## Requirements

| Language | Requirements |
|----------|-------------|
| Python | Python 3.8+, NumPy >= 1.20, SciPy >= 1.7 |
| MATLAB | R2018a or later, OR Octave 6.0+ |
| Julia | 1.6 or later (LinearAlgebra, Random standard libraries) |

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

Quick summary:

```python
# Python
import sys
sys.path.append('/path/to/distrib/python')
from librla import id_sketch, svd_sketch
```

```matlab
% MATLAB
addpath('/path/to/distrib/matlab');
```

```julia
# Julia
include("/path/to/distrib/julia/librla.jl")
using .librla
```

## Testing

Run the test suite to verify installation:

```bash
# Python
cd distrib/python
python test1_hilbert.py
python test2_svd_hilbert.py

# MATLAB (in MATLAB command window)
cd distrib/matlab
test1_hilbert
test2_svd_hilbert

# Julia
cd distrib/julia
julia -e 'include("test1_hilbert.jl")'
julia -e 'include("test2_svd_hilbert.jl")'
```

## Examples

See the `compare_id.*` files for comprehensive examples comparing the randomized sketching (`id_sketch`) and deterministic QR pivoting (`id_qrpiv`) methods.

The test suite provides examples for:

| Test | Description |
|------|-------------|
| test1_hilbert | Basic Hilbert matrix tests |
| test1_kahan | Kahan matrix tests |
| test2_svd_hilbert | SVD on Hilbert matrices |
| test3_linop_hilbert | LinearOperator with Hilbert matrix |
| test4_linop_random | LinearOperator with random matrices |
| test5_linop_fullrank | Full-rank matrix tests |
| test5_method_comparison | Compare T computation methods |
| test6_power | Power iteration effects |
| test7_power | Power iteration in svd_sketch |

## References

For the mathematical foundations and algorithms, see:

- Halko, Martinsson, Tropp. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." SIAM Review, 2011.
- Martinsson, Rokhlin, Tygert. "A randomized algorithm for the decomposition of matrices." Applied and Computational Harmonic Analysis, 2011.

## License

See LICENSE file in the parent repository.

## Contact

For questions, issues, or contributions, please contact the repository maintainers.
