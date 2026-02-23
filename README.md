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



## API Notes

### svd_sketch Return Values

The `svd_sketch` function returns slightly different formats across languages:

| Language | Returns | Reconstruction |
|----------|---------|----------------|
| Python | `U, s, Vh` | `A approx U @ np.diag(s) @ Vh` |
| MATLAB | `U, s, V` | `A approx U * diag(s) * V'` |
| Julia | `U, s, Vt` | `A approx U * diagm(s) * Vt` |

Python and Julia return V transposed/adjoint; MATLAB returns V (requiring explicit transpose/adjoint in reconstruction).

### Indexing

- **Python**: 0-based indexing (piv[:k] for skeleton columns)
- **MATLAB/Julia**: 1-based indexing (piv(1:k) or piv[1:k] for skeleton columns)

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

## Linear Operators

 Matrix-free LinearOperators only support **rank mode** (rtol >= 1).

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


## File Structure

```
README.md                  # This file
distrib/
├── CLAUDE.md              # AI assistant instructions
├── python/
│   ├── librla.py          # Main library
│   ├── demo_utils.py      # Demo utilities (includes hilbert matrix)
│   ├── test_utils.py      # Test utilities
│   ├── demo*.py           # Demo suite (demo01-demo05)
│   └── test*.py           # Test suite (test_id, test_svd, test_qr, test_orth, test_all)
├── matlab/
│   ├── librla.m           # Main library
│   ├── LinearOperator.m   # Matrix-free operator class
│   ├── demo_utils.m       # Demo utilities (includes hilbert matrix)
│   ├── test_utils.m       # Test utilities
│   ├── demo*.m            # Demo suite (demo01-demo05)
│   └── test*.m            # Test suite (test_id, test_svd, test_qr, test_orth, test_all)
├── julia/
│   ├── librla.jl          # Main library
│   ├── LinearOperator.jl  # Matrix-free operator type
│   ├── demo_utils.jl      # Demo utilities (includes hilbert matrix)
│   ├── test_utils.jl      # Test utilities
│   ├── demo*.jl           # Demo suite (demo01-demo05)
│   └── test*.jl           # Test suite (test_id, test_svd, test_qr, test_orth, test_all)
├── compare/               # External library comparisons
├── climate_analysis/      # Climate data analysis examples
└── image_analysis/        # Image compression examples
```

## Requirements

| Language | Requirements |
|----------|-------------|
| Python | Python 3.8+, NumPy >= 1.20, SciPy >= 1.7 |
| MATLAB | R2018a or later, OR Octave 6.0+ |
| Julia | 1.6 or later (LinearAlgebra, Random standard libraries) |


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

## Additional resources

Additional resources provided for users are:

- The folder '/distrib/compare' provides codes that compare 'librla' with other randomized linear algebra packages:
  - `compare_id_scipy.py` - Python: librla vs SciPy (ID)
  - `compare_svd_torch.py` - Python: librla vs PyTorch (SVD)
- The folder '/distrib/image_analysis' provides codes that illustrate the use of the different randomized methods for image compression.
- The folder '/distrib/climate_analysis' provides codes that illustrate the use of randomized methods for data compression.

The folders contain documentation.  The last two folders illustrate the use of 'librla' for common applications of interest.  The examples considered are taken from the randomized linear algebra literature.

## References

For the mathematical foundations and algorithms, see:

- Halko, Martinsson, Tropp. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." SIAM Review, 2011.
- Martinsson, Rokhlin, Tygert. "A randomized algorithm for the decomposition of matrices." Applied and Computational Harmonic Analysis, 2011.

## License

See COPYING file in the parent repository.

## Contact

For questions, issues, or contributions, please contact the repository maintainers.
