# Test 4: LinearOperator Test with Random Matrix

## Overview

`test4_linop_random` demonstrates `id_sketch` with LinearOperators on a medium-size low-rank random matrix. This complements test3 (Hilbert matrix, ill-conditioned) by testing on a well-conditioned low-rank matrix.

This test validates three usage patterns:

1. **Dense matrix** (baseline) - Standard NumPy/MATLAB/Julia array
2. **Explicit LinearOperator** (matrix wrapper) - LinearOperator wrapping a matrix
3. **Matrix-free LinearOperator** (function handles) - True matrix-free operators

## Test Matrix

- **Type**: Low-rank random matrix (well-conditioned)
- **Size**: 500 x 300
- **True rank**: ~30 (generated as U*V' + small noise)
- **Target rank**: k = 20

## Why Low-Rank Random Matrix?

Unlike test3 (Hilbert), this test uses a clean low-rank structure:
- **Well-conditioned**: No numerical stability issues
- **Known rank structure**: True rank ~30, target k=20
- **Moderate error**: Expected error ~0.5 (reasonable for rank-20 approximation of rank-30 matrix)
- **Larger size**: 500x300 vs 300x200 (demonstrates scalability)

## Running the Tests

### Python
```bash
cd distrib/python
python3 test4_linop_random.py
```

### MATLAB/Octave
```bash
cd distrib/matlab
octave --no-gui --eval "test4_linop_random"
# or in MATLAB:
matlab -batch "test4_linop_random"
```

### Julia
```bash
cd distrib/julia
julia test4_linop_random.jl
```

## Expected Results

All three implementations should produce similar results (exact results may vary due to randomness):

```
======================================================================
TEST 4: LinearOperators - Random Matrix
======================================================================

Matrix size: 500 x 300
Matrix type: Low-rank random (rank ~30)
Target rank: 20

JIT warm-up (compiling methods)...  [Julia only]
JIT warm-up complete.
======================================================================

1. Dense Matrix (baseline)
----------------------------------------------------------------------
  Rank:      k = 20
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 5.284e-01
  Max |T|:   6.698e-01
  Time:      0.0014 s

2. Explicit LinearOperator (matrix wrapper)
----------------------------------------------------------------------
  Operator: 500 x 300
  Is explicit: True
  Rank:      k = 20
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 5.284e-01
  Max |T|:   6.698e-01
  Time:      0.0010 s
  [OK] Explicit LinearOperator produces same rank and error as dense!

3. Matrix-free LinearOperator (function handles)
----------------------------------------------------------------------
  Operator: 500 x 300
  Is explicit: False
  Mode: Rank mode only (rtol >= 1)
  Rank:      k = 20
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 5.284e-01
  Max |T|:   6.698e-01
  Time:      0.0019 s
  [OK] Matrix-free returns target rank k=20

======================================================================
SUMMARY
======================================================================
  Method              Rank    Error        Max|T|       Time
----------------------------------------------------------------------
  Dense (baseline)      20    5.284e-01    6.698e-01    0.0014s
  Explicit LinOp        20    5.284e-01    6.698e-01    0.0010s
  Matrix-free LinOp     20    5.284e-01    6.698e-01    0.0019s
======================================================================

[PASS] All LinearOperator tests passed!
```

## Key Observations

### 1. Error Level
The reconstruction error ~0.53 is expected because:
- True matrix rank: ~30
- Target rank: 20
- Error represents information lost by discarding 10 singular values
- Error < 1.0 confirms the approximation is valid

### 2. Identical Results
All three methods (dense, explicit LinOp, matrix-free LinOp) produce:
- **Same rank**: k = 20 (exact)
- **Same error**: ~0.53 (identical to machine precision)
- **Same max|T|**: ~0.67 (interpolation coefficients well-conditioned)

This validates that LinearOperator abstraction preserves accuracy.

### 3. Performance
**Typical results:**

| Method              | Python   | MATLAB   | Julia    |
|---------------------|----------|----------|----------|
| Dense (baseline)    | 1.4 ms   | 5.0 ms   | 0.6 ms   |
| Explicit LinOp      | 1.0 ms   | 3.7 ms   | 0.6 ms   |
| Matrix-free LinOp   | 1.9 ms   | 4.2 ms   | 2.5 ms   |

- All methods are very fast (<5ms) for this medium-sized problem
- Matrix-free has slight overhead for small problems
- For large problems, matrix-free saves memory

## Comparison: test3 vs test4

| Property             | test3 (Hilbert)      | test4 (Random)       |
|----------------------|----------------------|----------------------|
| Matrix type          | Hilbert              | Low-rank random      |
| Conditioning         | Severely ill-cond.   | Well-conditioned     |
| Size                 | 300 x 200            | 500 x 300            |
| True rank            | ~15-20               | ~30                  |
| Target rank          | 15                   | 20                   |
| Expected error       | ~3e-10 (excellent)   | ~0.53 (moderate)     |
| Purpose              | Numerical stability  | Scalability          |

Both tests are complementary:
- **test3**: Validates numerical stability on ill-conditioned problems
- **test4**: Validates scalability and typical use case

## Matrix Construction

The low-rank matrix is constructed as:

**Python:**
```python
U = np.random.randn(m, true_rank)
V = np.random.randn(n, true_rank)
A = U @ V.T + 1e-10 * np.random.randn(m, n)
```

**MATLAB:**
```matlab
U = randn(m, true_rank);
V = randn(n, true_rank);
A = U * V' + 1e-10 * randn(m, n);
```

**Julia:**
```julia
U = randn(m, true_rank)
V = randn(n, true_rank)
A = U * V' + 1e-10 * randn(m, n)
```

This creates a matrix with:
- **Rank ~30**: Dominant structure from U*V'
- **Small noise**: 1e-10 factor ensures numerical rank ~30
- **Well-conditioned**: Random matrices have good condition number

## Use Cases

This test demonstrates LinearOperator usage for:

1. **Low-rank approximation**: Target rank k < true rank
2. **Memory efficiency**: Matrix-free for large structured matrices
3. **Unified API**: Same code works for dense, explicit, and matrix-free operators
4. **Scalability**: Larger size (500x300) shows performance characteristics

## Notes

1. **Randomness**: Results vary between runs due to random matrix generation. The seed (42) ensures reproducibility within each language, but cross-language results may differ slightly.

2. **Error interpretation**: Error ~0.53 means the rank-20 approximation captures ~47% of the Frobenius norm. This is expected when true rank (~30) > target rank (20).

3. **JIT warm-up (Julia)**: Pre-compiles all three operator types to avoid compilation overhead in timing.

4. **Validation**: The test validates:
   - Reconstruction error < 1.0 (all methods)
   - Explicit operator matches dense exactly (rank and error)
   - Matrix-free returns target rank k=20

## Files

- `python/test4_linop_random.py` - Python implementation
- `matlab/test4_linop_random.m` - MATLAB/Octave implementation
- `julia/test4_linop_random.jl` - Julia implementation (with JIT warm-up)

## See Also

- **test3_linop_hilbert** - LinearOperator test with ill-conditioned Hilbert matrix
- **test1_hilbert** - Basic ID test (dense only)
- **test2_svd_hilbert** - Basic SVD test (dense only)
