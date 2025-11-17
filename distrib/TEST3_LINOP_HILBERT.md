# Test 3: LinearOperator Test with Hilbert Matrix

## Overview

`test3_linop_hilbert` demonstrates `id_sketch` with LinearOperators on a medium-size Hilbert matrix. This test validates three usage patterns:

1. **Dense matrix** (baseline) - Standard NumPy/MATLAB/Julia array
2. **Explicit LinearOperator** (matrix wrapper) - LinearOperator wrapping a matrix
3. **Matrix-free LinearOperator** (function handles) - True matrix-free operators

## What are LinearOperators?

LinearOperators provide a unified interface for matrix-like objects, supporting both:

- **Explicit operators**: Backed by an actual matrix (useful for unified APIs)
- **Matrix-free operators**: Defined only by `matvec` (A*x) and `rmatvec` (A'*x) functions

This abstraction enables memory-efficient algorithms for large-scale problems where storing the full matrix is infeasible.

## Test Matrix

- **Type**: Hilbert matrix (severely ill-conditioned)
- **Size**: 300 x 200
- **Target rank**: k = 15

## Running the Tests

### Python
```bash
cd distrib/python
python3 test3_linop_hilbert.py
```

### MATLAB/Octave
```bash
cd distrib/matlab
octave --no-gui --eval "test3_linop_hilbert"
# or in MATLAB:
matlab -batch "test3_linop_hilbert"
```

### Julia
```bash
cd distrib/julia
julia test3_linop_hilbert.jl
```

## Expected Results

All three implementations should produce identical results:

```
======================================================================
TEST 3: LinearOperators - Hilbert Matrix
======================================================================

Matrix size: 300 x 200
Matrix type: Hilbert (severely ill-conditioned)
Target rank: 15

JIT warm-up (compiling methods)...  [Julia only]
JIT warm-up complete.
======================================================================

1. Dense Matrix (baseline)
----------------------------------------------------------------------
  Rank:      k = 15
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 2.956e-10
  Max |T|:   1.379e+00
  Time:      0.0010 s

2. Explicit LinearOperator (matrix wrapper)
----------------------------------------------------------------------
  Operator: 300 x 200
  Is explicit: True
  Rank:      k = 15
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 2.956e-10
  Max |T|:   1.379e+00
  Time:      0.0009 s
  [OK] Explicit LinearOperator produces same rank and error as dense!

3. Matrix-free LinearOperator (function handles)
----------------------------------------------------------------------
  Operator: 300 x 200
  Is explicit: False
  Mode: Rank mode only (rtol >= 1)
  Rank:      k = 15
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 2.956e-10
  Max |T|:   1.379e+00
  Time:      0.0009 s
  [OK] Matrix-free returns target rank k=15

======================================================================
SUMMARY
======================================================================
  Method              Rank    Error        Max|T|       Time
----------------------------------------------------------------------
  Dense (baseline)      15    2.956e-10    1.379e+00    0.0010s
  Explicit LinOp        15    2.956e-10    1.379e+00    0.0009s
  Matrix-free LinOp     15    2.956e-10    1.379e+00    0.0009s
======================================================================

[PASS] All LinearOperator tests passed!
```

## Key Concepts

### 1. Explicit LinearOperator (Matrix Wrapper)

**Purpose**: Provides a unified API for algorithms that accept both matrices and operators.

**Python:**
```python
from make_linop import make_linop
A_linop = make_linop(A)  # Wraps matrix A
k, piv, T = id_sketch(A_linop, rtol=15.0)
```

**MATLAB:**
```matlab
A_linop = make_linop(A);
[k, piv, T] = libid.id_sketch(A_linop, 15);
```

**Julia:**
```julia
A_linop = make_linop(A)
k, piv, T = id_sketch(A_linop, rtol=15.0)
```

**Expected behavior**:
- Should produce **identical rank and error** as dense matrix
- Pivots may differ due to randomness in `id_sketch`
- Slightly faster or same speed as dense (minimal overhead)

### 2. Matrix-Free LinearOperator (Function Handles)

**Purpose**: Enables sketching without storing the full matrix - only need matrix-vector products.

**Python:**
```python
def matvec(x):   return A @ x      # Forward: A*x
def rmatvec(x):  return A.conj().T @ x  # Adjoint: A'*x

A_linop = make_linop(m, n, matvec, rmatvec, dtype=A.dtype)
k, piv, T = id_sketch(A_linop, rtol=15.0)  # Rank mode only!
```

**MATLAB:**
```matlab
Afun = @(x) A * x;        % Forward: A*x
ATfun = @(x) A' * x;      % Adjoint: A'*x

A_linop = make_linop(m, n, Afun, ATfun, 'double');
[k, piv, T] = libid.id_sketch(A_linop, 15);  % Rank mode only!
```

**Julia:**
```julia
Afun(x) = A * x          # Forward: A*x
ATfun(y) = A' * y        # Adjoint: A'*x

A_linop = make_linop(Float64, m, n, Afun, ATfun)
k, piv, T = id_sketch(A_linop, rtol=15.0)  # Rank mode only!
```

**Important constraints**:
- **Rank mode only** (`rtol >= 1`) - cannot auto-select rank without the full matrix
- Requires `matvec` (A*x) and `rmatvec` (A'*x) function handles
- Must specify element type for matrix-free operators
- Efficient: only k matrix-vector products needed

## Performance Comparison

**Typical results:**

| Method              | Python   | MATLAB/Octave | Julia    |
|---------------------|----------|---------------|----------|
| Dense (baseline)    | 1.0 ms   | 1.2 ms        | 0.3 ms   |
| Explicit LinOp      | 0.7 ms   | 1.0 ms        | 0.3 ms   |
| Matrix-free LinOp   | 0.8 ms   | 0.8 ms        | 2.2 ms   |

**Observations:**
- Explicit LinearOperator has minimal overhead
- Matrix-free is similar speed for small problems
- For large problems, matrix-free saves memory (doesn't store n×m matrix)

## Use Cases for LinearOperators

### When to use Explicit LinearOperators:
- Unified API: algorithm accepts both matrices and operators
- Easy migration: wrap existing matrices without changing algorithm code
- Testing: validate that operator and matrix paths produce identical results

### When to use Matrix-Free LinearOperators:
- **Large structured matrices**: FFT, convolution, Toeplitz, circulant matrices
- **Memory constraints**: Matrix doesn't fit in RAM, but matvec is cheap
- **Implicit operators**: Matrix never formed explicitly (e.g., Jacobian-free methods)
- **Sparse operators**: Only O(n) storage instead of O(mn)

**Example applications:**
- PDE discretizations (sparse, structured)
- Image processing (convolution, FFT-based)
- Signal processing (Toeplitz, circulant)
- Machine learning (implicit Hessian-vector products)

## Notes

1. **Identical errors**: All three methods achieve ~3e-10 reconstruction error on the Hilbert matrix, demonstrating that LinearOperator abstraction doesn't compromise accuracy.

2. **Randomness**: `id_sketch` is randomized, so column selections (pivots) may differ between runs, but rank and error should be consistent.

3. **Rank mode restriction**: Matrix-free operators only support rank mode (`rtol >= 1`) because tolerance mode requires adaptive rank selection, which needs explicit matrix access.

4. **JIT warm-up (Julia)**: Julia version pre-compiles all three operator types to avoid compilation overhead in timing measurements.

5. **Validation**: The test validates:
   - Reconstruction error < 1.0 (all methods)
   - Explicit operator matches dense exactly (rank and error)
   - Matrix-free returns target rank k=15

## Files

- `python/test3_linop_hilbert.py` - Python implementation
- `matlab/test3_linop_hilbert.m` - MATLAB/Octave implementation
- `julia/test3_linop_hilbert.jl` - Julia implementation (with JIT warm-up)

## See Also

- **LinearOperator infrastructure**:
  - `python/make_linop.py` - Python LinearOperator factory
  - `matlab/make_linop.m` - MATLAB/Octave LinearOperator factory
  - `julia/make_linop.jl` - Julia LinearOperator factory

- **Comprehensive tests** (in `compare/` directory):
  - `compare/test_linop_simple.py` - Detailed LinearOperator validation
  - `compare/test_matfree_python.py` - Matrix-free operator tests
  - `compare/test_matfree_matlab.m` - MATLAB matrix-free tests
  - `compare/test_matfree_julia.jl` - Julia matrix-free tests
