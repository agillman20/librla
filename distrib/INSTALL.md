# Installation Instructions - librla

This distribution contains a unified randomized linear algebra library (librla) implemented in Python, MATLAB/Octave, and Julia. All three implementations provide the same core functions with consistent APIs.

## Quick Start

### Python
```python
import sys
sys.path.append('/path/to/distrib/python')
from librla import orth_sketch, qr_sketch, svd_sketch, id_sketch
```

### MATLAB/Octave
```matlab
addpath('/path/to/distrib/matlab');
[Q, R, p] = librla.qr_sketch(A, 1e-6);
```

### Julia
```julia
include("/path/to/distrib/julia/librla.jl")
using .librla
Q, R, p = qr_sketch(A, 1e-6)
```

## Python

### Requirements
- Python 3.7 or later
- NumPy >= 1.20
- SciPy >= 1.7

### Installation

**Option 1: Add to Python path (recommended for testing)**
```python
import sys
sys.path.append('/path/to/distrib/python')
```

**Option 2: Add to PYTHONPATH environment variable**
```bash
export PYTHONPATH="/path/to/distrib/python:$PYTHONPATH"
```

**Option 3: Copy files to your project**
```bash
cp distrib/python/librla.py your_project/
cp distrib/python/LinearOperator.py your_project/  # Optional
cp distrib/python/hilbert.py your_project/  # Optional utilities
```

### Usage
```python
from librla import (
    orth_sketch,    # Orthonormal basis via random sketching
    qr_sketch,      # Truncated QR factorization
    svd_sketch,     # Truncated SVD
    id_sketch,      # Interpolative decomposition (randomized)
    id_qrpiv        # Interpolative decomposition (deterministic)
)

import numpy as np

# Create a test matrix
A = np.random.randn(1000, 500)

# Tolerance mode: adaptive rank selection
k, piv, T = id_sketch(A, rtol=1e-6)

# Rank mode: fixed-rank approximation
U, s, Vt = svd_sketch(A, rtol=20.0)  # Rank-20 approximation
```

### Testing Python Installation
```bash
cd distrib/python
python demo01_basic.py
python demo02_svd.py
```

### Troubleshooting Python

**ImportError: No module named 'librla'**
- Verify the path in `sys.path.append()` is correct
- Use absolute paths, not relative paths
- Check that `librla.py` exists in the specified directory

**ImportError: No module named 'numpy' or 'scipy'**
```bash
pip install numpy scipy
# or with conda:
conda install numpy scipy
```

## MATLAB/Octave

### Requirements
- **MATLAB**: R2018a or later (earlier versions may work but are untested)
- **Octave**: 6.0 or later

### Installation

**Option 1: Temporary path (session only)**
```matlab
addpath('/path/to/distrib/matlab');
```

**Option 2: Permanent path**

Add to your `startup.m` file (create if it doesn't exist):
```matlab
% File: ~/Documents/MATLAB/startup.m (MATLAB)
% File: ~/.octaverc (Octave)
addpath('/path/to/distrib/matlab');
```

To find startup file location in MATLAB:
```matlab
userpath  % Shows MATLAB user path
```

**Option 3: Copy files to your project**
```bash
cp distrib/matlab/librla.m your_project/
cp distrib/matlab/LinearOperator.m your_project/  # Optional
cp distrib/matlab/hilbert.m your_project/  # Optional utilities
```

### Usage

All librla functions are static methods of the `librla` class:

```matlab
% Create a test matrix
A = randn(1000, 500);

% Tolerance mode: adaptive rank selection
[k, piv, T] = librla.id_sketch(A, 1e-6);

% Rank mode: fixed-rank approximation
[U, s, V] = librla.svd_sketch(A, 20);  % Rank-20 approximation

% With optional parameters
[Q, R, p] = librla.qr_sketch(A, 1e-6, 'power_iter', 2, 'block_size', 50);

% Interpolative decomposition with method selection
[k, piv, T] = librla.id_sketch(A, 1e-6, 'method', 'lstsq');
```

### Testing MATLAB Installation
```matlab
cd distrib/matlab
demo01_basic
demo02_svd
```

### Troubleshooting MATLAB/Octave

**Undefined function or variable 'librla'**
- Run `which librla` to check if it's in the path
- Verify `addpath()` was called with correct directory
- Check that `librla.m` exists in the specified directory

**Error using librla (line...)**
- Ensure you're using MATLAB R2018a or Octave 6.0+
- Older versions may not support classdef syntax used by librla

**Octave-specific issues**
- Octave 5.x may not fully support all features
- Upgrade to Octave 6.0 or later for best compatibility

## Julia

### Requirements
- Julia 1.6 or later
- LinearAlgebra package (standard library)
- Random package (standard library)

### Installation

**Option 1: Include in your project (recommended)**
```julia
include("/path/to/distrib/julia/librla.jl")
using .librla
```

**Option 2: Add to LOAD_PATH**
```julia
push!(LOAD_PATH, "/path/to/distrib/julia")
using librla
```

**Option 3: Copy files to your project**
```bash
cp distrib/julia/librla.jl your_project/
cp distrib/julia/LinearOperator.jl your_project/  # Optional
cp distrib/julia/hilbert.jl your_project/  # Optional utilities
```

### Usage
```julia
using LinearAlgebra
include("/path/to/distrib/julia/librla.jl")
using .librla

# Create a test matrix
A = randn(1000, 500)

# Tolerance mode: adaptive rank selection
k, piv, T = id_sketch(A, 1e-6)

# Rank mode: fixed-rank approximation
U, s, Vt = svd_sketch(A, 20.0)  # Rank-20 approximation

# With optional parameters
Q, R, p = qr_sketch(A, 1e-6, power_iter=2, block_size=50)

# Interpolative decomposition with method selection
k, piv, T = id_sketch(A, 1e-6, method="lstsq")
```

### Testing Julia Installation
```julia
cd("/path/to/distrib/julia")
include("demo01_basic.jl")
include("demo02_svd.jl")
```

### Troubleshooting Julia

**UndefVarError: librla not defined**
- Ensure you called `using .librla` (note the dot `.` prefix)
- Verify the path to `librla.jl` in the `include()` statement
- Check that `librla.jl` exists at the specified location

**MethodError or type errors**
- Ensure you're using Julia 1.6 or later
- Run `versioninfo()` to check Julia version

**LoadError while including librla.jl**
- Check that `LinearAlgebra` and `Random` are available:
  ```julia
  using LinearAlgebra
  using Random
  ```

## LinearOperator Support

All three implementations support **LinearOperator** for matrix-free computations.

### Python - Using scipy.sparse.linalg.LinearOperator

```python
from scipy.sparse.linalg import LinearOperator as ScipyLinOp
from librla import orth_sketch
import numpy as np

# Define matrix-vector products
def matvec(x):
    return np.fft.fft(x)

def rmatvec(x):
    return np.fft.ifft(x)

# Create LinearOperator
n = 1000
A_op = ScipyLinOp(shape=(n, n), matvec=matvec, rmatvec=rmatvec)

# Use with librla (rank mode only for matrix-free)
Q, flag, err = orth_sketch(A_op, 20.0)
```

### MATLAB - Using LinearOperator class

```matlab
% Create LinearOperator
n = 1000;
matvec_fun = @(x) fft(x);
rmatvec_fun = @(x) ifft(x);
A_op = LinearOperator(matvec_fun, rmatvec_fun, n, n);

% Use with librla (rank mode only for matrix-free)
[Q, flag, err] = librla.orth_sketch(A_op, 20);
```

### Julia - Using LinearOperator type

```julia
include("LinearOperator.jl")
using .librla

# Create LinearOperator
n = 1000
matvec_fun(x) = fft(x)
rmatvec_fun(x) = ifft(x)
A_op = LinearOperator(matvec_fun, rmatvec_fun, n, n, dtype=ComplexF64)

# Use with librla (rank mode only for matrix-free)
Q, flag, err = orth_sketch(A_op, 20.0)
```

**Note:** Matrix-free LinearOperators only support **rank mode** (rtol ≥ 1). Tolerance mode requires access to the full matrix.

## Running All Demos

### Python
```bash
cd distrib/python
python demo01_basic.py
python demo02_svd.py
python demo03_linop.py
python demo04_power.py
python demo05_methods.py
```

### MATLAB/Octave
```matlab
cd distrib/matlab
demo01_basic
demo02_svd
demo03_linop
demo04_power
demo05_methods
```

### Julia
```julia
cd("/path/to/distrib/julia")
include("demo01_basic.jl")
include("demo02_svd.jl")
include("demo03_linop.jl")
include("demo04_power.jl")
include("demo05_methods.jl")
```

## Usage Modes

### Tolerance Mode (rtol < 1)
Automatically determines rank to achieve specified relative accuracy:

```python
# Python
Q, flag, err = orth_sketch(A, 1e-6)  # Adaptive rank to achieve 10^-6 accuracy
```

```matlab
% MATLAB
[Q, flag, err] = librla.orth_sketch(A, 1e-6);
```

```julia
# Julia
Q, flag, err = orth_sketch(A, 1e-6)
```

### Rank Mode (rtol ≥ 1)
Returns fixed-rank approximation:

```python
# Python
U, s, Vt = svd_sketch(A, 20.0)  # Rank-20 SVD
```

```matlab
% MATLAB
[U, s, V] = librla.svd_sketch(A, 20);  # Rank-20 SVD
```

```julia
# Julia
U, s, Vt = svd_sketch(A, 20.0)  # Rank-20 SVD
```

## Optional Parameters

All functions support these optional parameters:

- **`block_size`** (default: 42) - Sketch size for tolerance mode
- **`power_iter`** (default: 0) - Number of power iterations for accuracy
- **`extra_samples`** (default: 12) - Oversampling for rank mode

For `id_sketch` only:
- **`method`** (default: 'fast') - T computation method: 'fast', 'svd', or 'lstsq'

### Python
```python
k, piv, T = id_sketch(A, 1e-6, power_iter=2, block_size=50, method='svd')
```

### MATLAB
```matlab
[k, piv, T] = librla.id_sketch(A, 1e-6, 'power_iter', 2, ...
                                'block_size', 50, 'method', 'svd');
```

### Julia
```julia
k, piv, T = id_sketch(A, 1e-6, power_iter=2, block_size=50, method="svd")
```

## Documentation

- **README.md** - Comprehensive overview, examples, and API reference
- **FILE_MANIFEST.txt** - Complete file listing with descriptions
- **This file (INSTALL.md)** - Installation and testing instructions

## License

See LICENSE file in the parent repository.

## Support

For questions, issues, or bug reports, please contact the repository maintainers or check the parent repository documentation.
