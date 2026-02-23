# librla - Randomized Linear Algebra Library

A unified, multi-language library implementing randomized algorithms for low-rank matrix factorizations of both real and complex matrices. The library provides efficient sketching-based methods for large-scale matrix decompositions with consistent APIs across Python, MATLAB/Octave, and Julia.

## Features

### Core Algorithms

All algorithms support both **tolerance mode** (rtol < 1) for adaptive rank selection and **rank mode** (rtol >= 1) for fixed-rank approximations:

| Function | Description |
|----------|-------------|
| `orth_sketch` | Approximate orthonormal basis for column space via randomized sketching |
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
| `'svd'` | SVD-based pseudoinverse |
| `'lstsq'` | Least-squares from original A (most accurate, slowest) |

## Quick Start
### MATLAB/Octave

```matlab
% Create a test matrix
A = demo_utils.hilbert(1000, 500);

% Orthonormal basis
[Q, flag, diagR] = librla.orth_sketch(A, 1e-6);

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

## Usage Modes

### Tolerance Mode (rtol < 1)

Adaptively determines rank to achieve specified relative tolerance:

```matlab
% MATLAB: Adaptive rank selection to achieve 10^-6 tolerance
[Q, flag, diagR] = librla.orth_sketch(A, 1e-6);
```


### Rank Mode (rtol >= 1)

Returns fixed-rank approximation:

```matlab
% MATLAB: Rank-20 approximation
[U, s, V] = librla.svd_sketch(A, 20);
```
## API Notes

### svd_sketch Return Values

The `svd_sketch` function returns slightly different formats across languages:

| Language | Returns | Reconstruction |
|----------|---------|----------------|
| MATLAB | `U, s, V` | `A approx U * diag(s) * V'` |

MATLAB returns V (requiring explicit transpose/adjoint in reconstruction).

### Indexing

- **MATLAB**: 1-based indexing (piv(1:k) or piv[1:k] for skeleton columns)



## Algorithm Details

### Randomized Sketching (id_sketch)

- Uses random test matrix multiplication for fast column space approximation
- Geometric block growth for adaptive rank determination in tolerance mode
- Optional power iterations for improved accuracy (`power_iter` parameter)
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

For matrices with slowly decaying singular values (small spectral gap), use **power iterations** to improve accuracy:

```python
# Use power_iter=2 for matrices with slowly decaying singular values
U, s, Vh = svd_sketch(A, 20, power_iter=2)
```


## LinearOperator Usage

Create matrix-free operators for implicit matrices:

### MATLAB

MATLAB uses the custom `LinearOperator` class:

```matlab
% Create LinearOperator
n = 1000;
matvec_fun = @(x) fft(x);
rmatvec_fun = @(x) ifft(x);
A_op = LinearOperator(matvec_fun, rmatvec_fun, n, n);

% Use with librla
[Q, flag, diagR] = librla.orth_sketch(A_op, 20);
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


```matlab
% MATLAB: Same options
[k, piv, T] = librla.id_sketch(A, 1e-6, 'power_iter', 2, 'method', 'svd');
```

## Requirements

| Language | Requirements |
|----------|-------------|
| MATLAB | R2018a or later, OR Octave 6.0+ |

## Installation

### MATLAB/Octave

**Requirements:** MATLAB R2018a+ or Octave 6.0+

**Option 1: Temporary path (session only)**
```matlab
addpath('/path/to/distrib/matlab');
```

**Option 2: Permanent path** - Add to `startup.m` (MATLAB) or `~/.octaverc` (Octave):
```matlab
addpath('/path/to/distrib/matlab');
```

**Option 3: Copy files to your project**
```bash
cp distrib/matlab/librla.m your_project/
cp distrib/matlab/LinearOperator.m your_project/  # Optional
```

**Troubleshooting:**
- `Undefined function 'librla'` - Run `which librla` to check path; verify `addpath()` was called
- Octave 5.x may not fully support all features; upgrade to 6.0+


## Demos

Run the demos to see the library in action:

# MATLAB (in MATLAB command window)
cd distrib/matlab/demo
demo01_basic
demo02_svd



## Examples

The demo suite provides examples for:

| Demo | Description |
|------|-------------|
| demo01_basic | Basic ID algorithms (id_sketch, id_qrpiv) |
| demo02_svd | SVD and QR sketching |
| demo03_linop | LinearOperator abstraction |
| demo04_power | Power iteration effects |
| demo05_methods | T computation methods comparison |

## Tests

Tests for validating that the algorithms are working are:

| Test | Description |
|------|-------------|
| test_id | Validates Interpolatory decomposition | 
| test_svd | Validates the SVD| 
| test_qr | Validates the randomized QR |
| test_orth | Validates the orthogonal sketch |
| test_all | runs all the test listed above |




