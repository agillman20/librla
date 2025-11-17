# Installation Instructions

This distribution contains two algorithms for interpolative decomposition (ID) in Python, MATLAB/Octave, and Julia:

- **libid** / **LibIDSketch**: Randomized QR sketching (recommended - fast, adaptive)
- **libid_rrqr** / **LibIDRRQR**: Deterministic RRQR via LAPACK geqp3 (reproducible)

## Python

### Requirements
- Python 3.7 or later
- NumPy >= 1.20
- SciPy >= 1.7

### Installation
No installation needed. Simply copy the files from the `python/` directory to your project or add the directory to your Python path:

```python
import sys
sys.path.append('/path/to/distrib/python')
from libid import id_sketch          # Randomized sketching (recommended)
from libid_rrqr import id_rrqr       # Deterministic RRQR
from make_linop import make_linop    # LinearOperator support (optional)
```

## MATLAB/Octave

### Requirements
- MATLAB R2018a or later, OR
- GNU Octave 6.0 or later

### Installation
Add the directory to your MATLAB/Octave path:

```matlab
addpath('/path/to/distrib/matlab');
```

To make this permanent, add these lines to your `startup.m` file.

### Usage
All functions are static methods of the `libid` class:

```matlab
% Randomized sketching (recommended)
[k, piv, T] = libid.id_sketch(A, rtol);

% Deterministic RRQR
[k, piv, T] = libid_rrqr(A, rtol);
```

## Julia

### Requirements
- Julia 1.6 or later
- LinearAlgebra (standard library)
- Random (standard library)

### Installation
Include the module files in your project:

```julia
# Randomized sketching (recommended)
include("/path/to/distrib/julia/LibIDSketch.jl")
using .LibIDSketch

# Deterministic RRQR
include("/path/to/distrib/julia/LibIDRRQR.jl")
using .LibIDRRQR

# LinearOperator support (optional)
include("/path/to/distrib/julia/make_linop.jl")
```

## Running Tests

### Python
```bash
cd python
python test_libid.py
python test_power_iteration.py
```

### MATLAB/Octave
```matlab
cd matlab
test_libid
test_power_iteration
```

### Julia
```julia
cd("julia")
include("test_libid.jl")
include("test_power_iteration.jl")
```

## Running Examples

Comparison benchmarks are provided in each language directory:

### Python
```bash
cd python
python compare_id.py    # Compare libid vs libid_rrqr
python compare_svd.py   # Compare SVD methods
```

### MATLAB/Octave
```matlab
cd matlab
compare_id   # Compare libid vs libid_rrqr
compare_svd  # Compare SVD methods
```

### Julia
```julia
cd("julia")
include("compare_id.jl")
include("compare_svd.jl")
```

## Algorithm Selection Guide

### When to use `libid` (Randomized Sketching)
**Recommended for most users**
- Default choice for production use
- Fastest implementation (typically 2-5x faster than RRQR)
- Adaptive rank selection via geometric growth
- Excellent accuracy in practice
- Supports both tolerance mode and rank mode
- Matrix-free operation support via LinearOperators

**Trade-off:**  
- Results are stochastic (vary slightly between runs)

### When to use `libid_rrqr` (Deterministic RRQR)
**Use when reproducibility is critical**
- Deterministic results (always identical for same input)
- Verification and validation
- When you need bit-exact reproducibility
- Regression testing
- Debugging

**Trade-off:**  
- Slower than randomized sketching
- No adaptive rank selection

## Usage Modes

Both algorithms support two modes:

### Tolerance Mode (rtol < 1)
Automatically selects rank based on relative error tolerance:
```python
k, piv, T = id_sketch(A, rtol=1e-8)   # Keep rank where error <= 1e-8
```

### Rank Mode (rtol >= 1)
Specify exact target rank:
```python
k, piv, T = id_sketch(A, rtol=50.0)   # Force rank = 50
```

## LinearOperator Framework

LinearOperator support for matrix-free computations is included in each language directory:

- **`make_linop`** - Create LinearOperator from functions or matrices
- **`parse_linop`** - Extract properties from LinearOperator
- **`lsqr_simple`** - LSQR solver supporting LinearOperators

See README.md for detailed LinearOperator documentation.

## Documentation

- **README.md** - Comprehensive overview of algorithms and usage
- **FILE_MANIFEST.txt** - Complete file listing and descriptions

## Quick Start Example

### Python
```python
from libid import id_sketch
import numpy as np

# Create a matrix
A = np.random.randn(1000, 500)

# Compute ID with tolerance mode
k, piv, T = id_sketch(A, rtol=1e-6)

# Verify: A[:, piv[k:]] ~ A[:, piv[:k]] @ T
err = np.linalg.norm(A[:, piv[k:]] - A[:, piv[:k]] @ T) / np.linalg.norm(A)
print(f"Rank: {k}, Error: {err:.3e}")
```

### MATLAB/Octave
```matlab
% Create a matrix
A = randn(1000, 500);

% Compute ID with tolerance mode
[k, piv, T] = libid.id_sketch(A, 1e-6);

% Verify: A(:, piv(k+1:end)) ~ A(:, piv(1:k)) * T
err = norm(A(:, piv(k+1:end)) - A(:, piv(1:k)) * T, 'fro') / norm(A, 'fro');
fprintf('Rank: %d, Error: %.3e\n', k, err);
```

### Julia
```julia
using LinearAlgebra
include("julia/LibIDSketch.jl")
using .LibIDSketch

# Create a matrix
A = randn(1000, 500)

# Compute ID with tolerance mode
k, piv, T = id_sketch(A, rtol=1e-6)

# Verify: A[:, piv[k+1:end]] ~ A[:, piv[1:k]] * T
err = norm(A[:, piv[k+1:end]] - A[:, piv[1:k]] * T) / norm(A)
println("Rank: $k, Error: $(err)")
```

## License

This software is distributed under the BSD 3-Clause License (if LICENSE file is found in distribution).
