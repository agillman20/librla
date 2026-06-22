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

### Python

For Python, it is often a good idea to run in a virtual environment.  A script named 'setup-venv.sh' is available for users.  To utilize it, run the shell script in the terminal.  To end the virtual environment, type 'deactivate' in the terminal.

The package liblra can be used by either calling the .py file or using the following pip install command:

```pip install librla==1.0.1```

Here is an example using the library.

```python
import numpy as np
from librla import orth_sketch, qr_sketch, svd_sketch, id_sketch

# Create a test matrix (Hilbert matrix is ill-conditioned)
A = 1.0 / (np.arange(1, 1001)[:, None] + np.arange(1, 501) - 1)

# Orthonormal basis with relative tolerance
Q, flag, diagR = orth_sketch(A, rtol=1e-6)

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
## Usage Modes

### Tolerance Mode (rtol < 1)

Adaptively determines rank to achieve specified relative tolerance:

```python
# Python: Adaptive rank selection to achieve 10^-6 tolerance
Q, flag, diagR = orth_sketch(A, 1e-6)
```

### Rank Mode (rtol >= 1)

Returns fixed-rank approximation:

```python
# Python: Rank-20 approximation
U, s, Vh = svd_sketch(A, 20)
```
## API Notes

### svd_sketch Return Values

The `svd_sketch` function returns slightly different formats across languages:

| Language | Returns | Reconstruction |
|----------|---------|----------------|
| Python | `U, s, Vh` | `A approx U @ np.diag(s) @ Vh` |

Python returns V transposed/adjoint

### Indexing

- **Python**: 0-based indexing (piv[:k] for skeleton columns)

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
Q, flag, diagR = orth_sketch(A_op, 20)  # Rank-20 approximation
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

## Requirements

| Language | Requirements |
|----------|-------------|
| Python | Python 3.9+, NumPy >= 1.20, SciPy >= 1.7 |

## Installation

### Python

**Requirements:** Python 3.9+, NumPy >= 1.20, SciPy >= 1.7

**Option 1: Add to Python path (recommended for testing)**
```python
import sys
sys.path.append('/path/to/distrib/python')
from librla import id_sketch, svd_sketch
```

**Option 2: Add to PYTHONPATH environment variable**
```bash
export PYTHONPATH="/path/to/distrib/python:$PYTHONPATH"
```

**Option 3: Copy files to your project**
```bash
cp distrib/python/librla.py your_project/
```

**Troubleshooting:**
- `ImportError: No module named 'librla'` - Verify path is correct and use absolute paths
- `ImportError: No module named 'numpy'` - Run `pip install numpy scipy`


## Demos

Run the demos to see the library in action:

```bash
# Python
cd distrib/python/demo
python demo01_basic.py
python demo02_svd.py
```

The `pip install librla` wheel contains only the library module. To obtain
the demos and tests after installing from PyPI, either clone the GitHub
repository or download and unpack the source distribution:

```bash
pip download --no-binary :all: --no-deps librla
tar xf librla-*.tar.gz
cd librla-*/demo && python demo01_basic.py
cd ../test && python test_all.py
```


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
 - The repo [librla_appliations](https://github.com/agillman20/librla_applications) contains examples of using 'librla' for data compression. The examples are taken from common applications of interest and are taken from the randomized linear algebra literature.

