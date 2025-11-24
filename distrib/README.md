# librla - Randomized Linear Algebra Library

A unified, multi-language library implementing randomized algorithms for low-rank matrix approximations. Provides efficient sketching-based methods for large-scale matrix decompositions with consistent APIs across Python, MATLAB/Octave, and Julia.

## Features

### Core Algorithms

All algorithms support both **tolerance mode** (rtol < 1) for adaptive rank selection and **rank mode** (rtol ≥ 1) for fixed-rank approximations:

- **`orth_sketch`** - Orthonormal basis for column space via randomized range finding
- **`qr_sketch`** - Truncated QR factorization with column pivoting
- **`svd_sketch`** - Truncated singular value decomposition
- **`id_sketch`** - Interpolative decomposition via randomized sketching
- **`id_qrpiv`** - Interpolative decomposition via deterministic QR pivoting

### LinearOperator Support

All algorithms support **LinearOperator** abstraction for matrix-free computation:

- **Dense matrices** - Standard NumPy/MATLAB/Julia arrays
- **Explicit LinearOperators** - Matrix wrappers with unified interface
- **Matrix-free LinearOperators** - Function handles for `A*x` and `A'*x` operations only

Matrix-free operators enable sketching of **implicit matrices** (FFT, convolution, Toeplitz, circulant, etc.) without explicit storage.

### Method Options for ID

The `id_sketch` function supports three methods for computing the interpolation matrix T:

- **`'fast'`** - Triangular solve (fastest, default)
- **`'svd'`** - SVD-based pseudoinverse (stable for ill-conditioned matrices)
- **`'lstsq'`** - Least-squares from original A (most accurate, slowest)

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
# Q spans approximate column space of A

# Truncated QR factorization
Q, R, p = qr_sketch(A, rtol=1e-6)
# A[:, p] ≈ Q @ R

# Truncated SVD
U, s, Vt = svd_sketch(A, rtol=1e-6)
# A ≈ U @ np.diag(s) @ Vt

# Interpolative decomposition
k, piv, T = id_sketch(A, rtol=1e-6, method='lstsq')
# A[:, piv[k:]] ≈ A[:, piv[:k]] @ T
```

### MATLAB/Octave

```matlab
% Create a test matrix
A = hilb(1000, 500);

% Orthonormal basis
[Q, flag, err] = librla.orth_sketch(A, 1e-6);

% Truncated QR factorization
[Q, R, p] = librla.qr_sketch(A, 1e-6);
% A(:, p) ≈ Q * R

% Truncated SVD
[U, s, V] = librla.svd_sketch(A, 1e-6);
% A ≈ U * diag(s) * V'

% Interpolative decomposition
[k, piv, T] = librla.id_sketch(A, 1e-6, 'method', 'lstsq');
% A(:, piv(k+1:end)) ≈ A(:, piv(1:k)) * T
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
# A[:, p] ≈ Q * R

# Truncated SVD
U, s, Vt = svd_sketch(A, 1e-6)
# A ≈ U * diagm(s) * Vt

# Interpolative decomposition
k, piv, T = id_sketch(A, 1e-6, method="lstsq")
# A[:, piv[k+1:end]] ≈ A[:, piv[1:k]] * T
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

### Rank Mode (rtol ≥ 1)

Returns fixed-rank approximation:

```python
# Python: Rank-20 approximation
U, s, Vt = svd_sketch(A, 20.0)
```

```matlab
% MATLAB: Rank-20 approximation
[U, s, V] = librla.svd_sketch(A, 20);
```

## Algorithm Details

### Randomized Sketching (id_sketch)

- Uses random test matrix multiplication for fast column space approximation
- Geometric block growth for adaptive rank determination in tolerance mode
- Optional power iterations for improved accuracy (`power_iter` parameter)
- Typically 2-5× faster than deterministic methods
- Stochastic (results vary slightly between runs)

### Deterministic QR Pivoting (id_qrpiv)

- Uses LAPACK geqp3 column-pivoted QR factorization
- Deterministic and reproducible results
- Slower than randomized sketching but guaranteed behavior
- Same interface as `id_sketch`
- Useful for verification and when reproducibility is critical

## LinearOperator Usage

Create matrix-free operators for implicit matrices:

### Python

```python
import numpy as np
from scipy.sparse.linalg import LinearOperator as ScipyLinOp

# Define matrix-vector products
def matvec(x):
    return np.fft.fft(x)

def rmatvec(x):
    return np.fft.ifft(x)

# Create LinearOperator
n = 1000
A_op = ScipyLinOp(shape=(n, n), matvec=matvec, rmatvec=rmatvec)

# Use with librla (rank mode only for matrix-free operators)
Q, flag, err = orth_sketch(A_op, 20.0)  # Rank-20 approximation
```

### MATLAB

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

```julia
include("LinearOperator.jl")

# Create LinearOperator
n = 1000
matvec_fun(x) = fft(x)
rmatvec_fun(x) = ifft(x)
A_op = LinearOperator(matvec_fun, rmatvec_fun, n, n, dtype=ComplexF64)

# Use with librla
Q, flag, err = orth_sketch(A_op, 20.0)
```

## Optional Parameters

All sketching functions support these optional parameters:

- **`block_size`** (default: 42) - Initial sketch size for tolerance mode
- **`power_iter`** (default: 0) - Number of power iterations for accuracy
- **`extra_samples`** (default: 12) - Oversampling for rank mode

For `id_sketch` only:

- **`method`** (default: 'fast') - T matrix computation method: 'fast', 'svd', or 'lstsq'

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
├── README.md              - This file
├── INSTALL.md             - Installation instructions
├── FILE_MANIFEST.txt      - Complete file listing
├── python/
│   ├── librla.py          - Main library (26K)
│   ├── LinearOperator.py  - Matrix-free operators (planned)
│   ├── compare_id.py      - Comparison example
│   ├── test*.py           - Test suite (8 files)
│   └── utilities...
├── matlab/
│   ├── librla.m           - Main library (21K)
│   ├── LinearOperator.m   - Matrix-free operators
│   ├── compare_id.m       - Comparison example
│   ├── test*.m            - Test suite (8 files)
│   └── utilities...
└── julia/
    ├── librla.jl          - Main library (21K)
    ├── LinearOperator.jl  - Matrix-free operators
    ├── compare_id.jl      - Comparison example
    ├── test*.jl           - Test suite (8 files)
    └── utilities...
```

## Requirements

- **Python**: NumPy ≥ 1.20, SciPy ≥ 1.7, Python 3.7+
- **MATLAB**: R2018a or later, OR Octave 6.0+
- **Julia**: 1.6 or later (LinearAlgebra, Random standard libraries)

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions for each language.

Quick summary:

```python
# Python
import sys
sys.path.append('/path/to/distrib/python')
```

```matlab
% MATLAB
addpath('/path/to/distrib/matlab');
```

```julia
# Julia
include("/path/to/distrib/julia/librla.jl")
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
include("test1_hilbert.jl")
include("test2_svd_hilbert.jl")
```

## Comparison with Previous Versions

This version (librla) replaces the previous libid/LibIDSketch/LibIDRRQR implementation:

### Changes

**Removed:**
- `libid.py`, `libid.m`, `LibIDSketch.jl`, `LibIDRRQR.jl` - replaced by unified `librla`
- `lsqr_simple.*` - obsolete LSQR utilities
- `make_linop.*` - replaced by LinearOperator classes
- `compare_svd.*` - consolidated into `compare_id`

**Added:**
- `librla.{py,m,jl}` - Unified library with consistent API across languages
- LinearOperator classes implemented in all languages
- Enhanced test suite with method comparison tests
- Power iteration tests (test6_power)

**API Changes:**
- All functions now use unified naming: `orth_sketch`, `qr_sketch`, `svd_sketch`, `id_sketch`, `id_qrpiv`
- Consistent parameter naming across languages
- LinearOperator interface standardized

## Examples

See the `compare_id.*` files for comprehensive examples comparing the randomized sketching (`id_sketch`) and deterministic QR pivoting (`id_qrpiv`) methods.

The test suite provides examples for:
- Hilbert matrices (test1_hilbert, test2_svd_hilbert)
- Kahan matrices (test1_kahan)
- LinearOperator usage (test3_linop_hilbert, test4_linop_random)
- Full-rank matrices (test5_linop_fullrank)
- Method comparisons (test5_method_comparison)
- Power iteration effects (test6_power)

## References

For the mathematical foundations and algorithms, see the parent repository documentation.

## License

See LICENSE file in the parent repository.

## Contact

For questions, issues, or contributions, please contact the repository maintainers.
