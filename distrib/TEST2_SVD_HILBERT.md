# Test 2: Simple Hilbert Matrix Test for SVD

## Overview

`test2_svd_hilbert` is a simple test demonstrating SVD algorithms on a medium-size Hilbert matrix. This test validates:

- `svd_sketch` - Randomized SVD via sketching (fast, randomized)
- `svd` (LAPACK) - Deterministic full SVD (reproducible, deterministic, truncated)

## Test Matrix

- **Type**: Hilbert matrix (severely ill-conditioned)
- **Size**: 300 x 200
- **Target rank**: k = 15

The Hilbert matrix is chosen because:
1. It's severely ill-conditioned (good stress test for SVD)
2. Results are reproducible across implementations
3. Medium size makes it fast to run

## Running the Tests

### Python
```bash
cd distrib/python
python3 test2_svd_hilbert.py
```

### MATLAB/Octave
```bash
cd distrib/matlab
octave --no-gui --eval "test2_svd_hilbert"
# or in MATLAB:
matlab -batch "test2_svd_hilbert"
```

### Julia
```bash
cd distrib/julia
julia test2_svd_hilbert.jl
```

## Expected Results

All three implementations should produce identical results:

```
======================================================================
TEST 2: Medium Hilbert Matrix - SVD
======================================================================

Matrix size: 300 x 200
Matrix type: Hilbert (severely ill-conditioned)
Target rank: 15
======================================================================

1. svd_sketch (randomized SVD via sketching)
----------------------------------------------------------------------
  Rank:      k = 15
  Error:     ||A - U @ S @ Vh|| / ||A|| = 1.806e-10
  SVal Err:  ||s - s_ref|| / ||s_ref|| = 7.439e-16
  Time:      0.0010 s

2. svd (LAPACK, deterministic, truncated)
----------------------------------------------------------------------
  Rank:      k = 15
  Error:     ||A - U @ S @ Vh|| / ||A|| = 1.806e-10
  SVal Err:  ||s - s_ref|| / ||s_ref|| = 0.000e+00
  Time:      0.0149 s

======================================================================
SUMMARY
======================================================================
  Method         Rank    Recon Error   SVal Error    Time
----------------------------------------------------------------------
  svd_sketch       15    1.806e-10    7.439e-16    0.0010s
  svd (LAPACK)     15    1.806e-10    0.000e+00    0.0149s
======================================================================

[PASS] Test completed successfully!
```

## Key Metrics

- **Rank**: Both methods return exactly k=15 (rank mode)
- **Reconstruction Error**: Both achieve ~1.8e-10 (excellent)
- **Singular Value Error**:
  - `svd_sketch`: ~7e-16 (near machine precision)
  - `svd (LAPACK)`: 0 (exact, used as reference)
- **Time**:
  - `svd_sketch`: ~0.3-1.0ms (very fast)
  - `svd (LAPACK)`: ~15-22ms (slower for full SVD)

## Performance Comparison

**Speed comparison (typical results):**

| Implementation | svd_sketch | svd (LAPACK) | Speedup |
|----------------|------------|--------------|---------|
| Python         | 1.0ms      | 14.9ms       | 15x     |
| MATLAB/Octave  | 2.1ms      | 22.4ms       | 11x     |
| Julia          | 0.3ms      | 2.1ms        | 7x      |

`svd_sketch` is significantly faster because it only computes k=15 singular vectors, while LAPACK SVD computes all min(m,n)=200 singular values/vectors before truncation.

## Notes

1. **Identical reconstruction errors**: Both methods produce identical reconstruction errors (~1.8e-10) because they're both computing the rank-15 approximation accurately.

2. **Singular value accuracy**: `svd_sketch` achieves near-machine-precision accuracy (~7e-16) for the singular values, which is excellent for a randomized method.

3. **Speed advantage**: `svd_sketch` is 7-15x faster depending on implementation, making it ideal for large-scale problems where only leading singular vectors are needed.

4. **JIT warm-up (Julia)**: Julia version includes JIT warm-up to avoid compilation overhead in timing.

5. **Validation**: The test validates:
   - Reconstruction error < 1.0 (mathematically required)
   - Singular value error < 1e-6 for svd_sketch (high accuracy)
   - Singular value error < 1e-10 for LAPACK svd (exact reference)

## Files

- `python/test2_svd_hilbert.py` - Python implementation
- `matlab/test2_svd_hilbert.m` - MATLAB/Octave implementation
- `julia/test2_svd_hilbert.jl` - Julia implementation (with JIT warm-up)
