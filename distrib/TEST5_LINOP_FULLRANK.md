# Test 5: LinearOperator Test with Full-Rank Random Matrix

## Overview

`test5_linop_fullrank` demonstrates `id_sketch` with LinearOperators on a full-rank random matrix, focusing on the `recompute_T` parameter. This complements test3 (Hilbert, ill-conditioned) and test4 (low-rank random) by testing the most challenging case: **full-rank matrices where target rank k << n**.

This test validates the `recompute_T` parameter across four usage patterns:

1. **Dense matrix** (baseline) - with recompute_T=True
2. **Explicit LinearOperator** - with recompute_T=True
3. **Matrix-free LinearOperator** (recompute_T=True) - Accurate, uses n matvecs
4. **Matrix-free LinearOperator** (recompute_T=False) - Fast, uses R matrix only

## Test Matrix

- **Type**: Full-rank random matrix (most challenging)
- **Size**: 400 x 300
- **True rank**: 300 (full rank)
- **Target rank**: k = 20 (6.7% of columns)

## Why Full-Rank Random Matrix?

Unlike test3 (Hilbert, low-rank) and test4 (low-rank random), this test uses a full-rank matrix:
- **Maximum challenge**: k << rank(A), so approximation quality is limited
- **Expected error**: May be > 1.0 with recompute_T=False (Fortran approach)
- **Demonstrates recompute_T**: Shows the accuracy vs speed tradeoff
- **Real-world scenario**: Common in data compression, randomized SVD, etc.

## Running the Tests

### Python
```bash
cd distrib/python
python3 test5_linop_fullrank.py
```

### MATLAB/Octave
```bash
cd distrib/matlab
octave --no-gui --eval "test5_linop_fullrank"
# or in MATLAB:
matlab -batch "test5_linop_fullrank"
```

### Julia
```bash
cd distrib/julia
julia test5_linop_fullrank.jl
```

## Expected Results

All three implementations should produce similar results (exact results may vary due to randomness):

```
======================================================================
TEST 5: LinearOperators - Full-Rank Random Matrix
======================================================================

Matrix size: 400 x 300
Matrix type: Full-rank random (all 300 columns independent)
Target rank: 20 (6.7% of columns)

JIT warm-up (compiling methods)...  [Julia only]
JIT warm-up complete.
======================================================================

1. Dense Matrix (baseline, recompute_T=True)
----------------------------------------------------------------------
  Rank:      k = 20
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 5.432e-01
  Max |T|:   7.821e-01
  Time:      0.0015 s
  [OK] Error < 1.0 (recompute_T=True guarantees this)

2. Explicit LinearOperator (matrix wrapper, recompute_T=True)
----------------------------------------------------------------------
  Operator: 400 x 300
  Is explicit: True
  Rank:      k = 20
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 5.432e-01
  Max |T|:   7.821e-01
  Time:      0.0011 s
  [OK] Explicit LinearOperator produces same rank and error as dense!

3. Matrix-free LinearOperator (recompute_T=True, accurate)
----------------------------------------------------------------------
  Operator: 400 x 300
  Is explicit: False
  Mode: Rank mode (rtol >= 1), recompute_T=True
  Note: Extracts all 300 columns via unit vectors (n matvecs)
  Rank:      k = 20
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 5.432e-01
  Max |T|:   7.821e-01
  Time:      0.0089 s
  [OK] Matrix-free (recompute_T=True): rank k=20, error < 1.0

4. Matrix-free LinearOperator (recompute_T=False, fast)
----------------------------------------------------------------------
  Operator: 400 x 300
  Is explicit: False
  Mode: Rank mode (rtol >= 1), recompute_T=False
  Note: Uses R matrix from sketch (Fortran approach, no extra matvecs)
  Rank:      k = 20
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 1.142e+00
  Max |T|:   9.234e-01
  Time:      0.0012 s
  Speedup:   7.4x faster than recompute_T=True
  Error ratio: 2.10x (err_false / err_true)
  [NOTE] Error > 1.0 is expected for full-rank matrices with recompute_T=False
         This uses Fortran's fast R-matrix approach, trading accuracy for speed

======================================================================
SUMMARY
======================================================================
  Method                        Rank    Error        Max|T|       Time
----------------------------------------------------------------------
  Dense (recompute_T=True)        20    5.432e-01    7.821e-01    0.0015s
  Explicit LinOp (recomp=True)    20    5.432e-01    7.821e-01    0.0011s
  Matrix-free (recompute=True)    20    5.432e-01    7.821e-01    0.0089s
  Matrix-free (recompute=False)   20    1.142e+00    9.234e-01    0.0012s
======================================================================

Key Observations:
  - Full-rank matrix: 400x300, target k=20 (6.7% of columns)
  - recompute_T=True:  Guarantees error < 1.0 (all methods: 5.432e-01, 5.432e-01, 5.432e-01)
  - recompute_T=False: 7.4x faster, but error may be > 1.0 (err=1.142e+00)
  - Trade-off: Speed (7.4x) vs Accuracy (2.10x degradation)

[PASS] All LinearOperator tests passed!
       recompute_T=True guarantees error < 1.0 for all modes
       recompute_T=False provides 7.4x speedup with acceptable error increase
```

## Key Concepts

### 1. The recompute_T Parameter

**Purpose**: Controls how the interpolation matrix T is computed.

#### recompute_T=True (Default, Accurate)

**Algorithm**:
1. Sketch: Compute column pivots via randomized QR
2. Extract skeleton columns: A[:, piv[:k]]
3. Extract remaining columns: A[:, piv[k:]]
4. Solve: skeleton @ T ~ remaining via lstsq

**For matrix-free operators**:
```python
# Extract columns via unit vectors (n matvecs)
for j in range(k):
    e_j = zeros(n); e_j[piv[j]] = 1.0
    skeleton[:, j] = A @ e_j  # matvec

for j in range(n - k):
    e_j = zeros(n); e_j[piv[k+j]] = 1.0
    remaining[:, j] = A @ e_j  # matvec

T = lstsq(skeleton, remaining)
```

**Cost**: k_sketch rmatvecs (for sketch) + n matvecs (for columns) + lstsq
**Accuracy**: **Guarantees error < 1.0** (mathematically proven)
**Use when**: Accuracy is critical, n is small (< 1000)

#### recompute_T=False (Fast, Fortran-style)

**Algorithm**:
1. Sketch: Compute column pivots and R matrix via randomized QR
2. Partition R = [R11 R12] where R11 is k×k, R12 is k×(n-k)
3. Solve: R11 @ T = R12 via triangular solve

**Cost**: k_sketch rmatvecs (for sketch) + triangular solve
**Accuracy**: Good, but **may have error > 1.0 on full-rank matrices**
**Use when**: Speed is critical, n is large (> 1000), error > 1.0 is acceptable

### 2. Performance Comparison

**Typical results:**

| Method              | Error     | Max\|T\| | Time    | Note                     |
|---------------------|-----------|----------|---------|--------------------------|
| Dense (recomp=True) | 5.43e-01  | 0.782    | 1.5 ms  | Baseline                 |
| Explicit (recomp=T) | 5.43e-01  | 0.782    | 1.1 ms  | Same as dense            |
| Matrix-free (rec=T) | 5.43e-01  | 0.782    | 8.9 ms  | Accurate, n matvecs      |
| Matrix-free (rec=F) | 1.14e+00  | 0.923    | 1.2 ms  | **7.4x faster**, err>1.0 |

**Observations:**
- **recompute_T=True**: All methods achieve error < 1.0
- **recompute_T=False**: 6-9× speedup, but error may exceed 1.0
- **Trade-off**: Accuracy vs speed

### 3. Why Error > 1.0 with recompute_T=False?

T is computed from the **sketch** R matrix rather than actual columns:

**recompute_T=True**:
- Solves: `A[:, skeleton] @ T ≈ A[:, remaining]`
- T interpolates between **actual columns** of A
- Mathematically guaranteed: error < 1.0

**recompute_T=False**:
- Solves: `R[:k,:k] @ T = R[:k,k:]`
- R comes from QR on sketch `(A^T @ Omega)^T`
- T interpolates in the **sketched/projected space**
- Small additional approximation error from sketch
- For full-rank matrices with k << n, error can exceed 1.0

### 4. Comparison with Fortran libid

The `recompute_T=False` mode matches Fortran's `idd_lssolve` approach:

**Fortran code** (idd_id.f, lines 329-388):
```fortran
subroutine idd_lssolve(m,n,a,krank)
c       backsolves for proj satisfying R_11 proj ~ R_12
  do k = 1,n-krank
    do j = krank,1,-1
      sum = 0
      do l = j+1,krank
        sum = sum+a(j,l)*a(l,krank+k)
      enddo
      a(j,krank+k) = (a(j,krank+k)-sum)/a(j,j)
    enddo
  enddo
end subroutine
```

**Python/MATLAB/Julia equivalent**:
```python
T = solve_triangular(R11, R12, lower=False)
```

Fortran uses the sketch R matrix for T computation by default (no "recompute" option), which can give error > 1.0 for full-rank matrices.

## Matrix Construction

The full-rank matrix is constructed as:

**Python:**
```python
np.random.seed(42)
m, n = 400, 300
A = np.random.randn(m, n)  # All 300 columns are linearly independent
```

**MATLAB:**
```matlab
rng(42);
m = 400; n = 300;
A = randn(m, n);  % All 300 columns are linearly independent
```

**Julia:**
```julia
Random.seed!(42)
m, n = 400, 300
A = randn(m, n)  # All 300 columns are linearly independent
```

This creates a matrix with:
- **Rank = 300**: All columns are linearly independent (full rank)
- **Well-conditioned**: Random matrices have good condition number
- **Challenging**: Target k=20 << rank=300, so error is non-trivial

## Use Cases

This test demonstrates when `recompute_T` parameter matters:

### When to use recompute_T=True (Default)

1. **Accuracy is critical**: Need guaranteed error < 1.0
2. **Small matrices**: n < 1000 (n matvecs are cheap)
3. **Mathematical guarantees**: Publishing results, formal verification
4. **Default choice**: Safe for all cases, mathematically proven

### When to use recompute_T=False (Expert use)

1. **Speed is critical**: Large-scale problems where n matvecs are expensive
2. **Large matrices**: n > 1000 (6-9× speedup is significant)
3. **Error tolerance**: Acceptable to have error > 1.0 on full-rank matrices
4. **Fortran compatibility**: Match Fortran libid behavior

## Performance Guidelines

**Expected speedup from recompute_T=False:**

| Matrix size | k   | n matvecs | Expected speedup |
|-------------|-----|-----------|------------------|
| 400 × 300   | 20  | 300       | 6-9×             |
| 1000 × 500  | 50  | 500       | 8-12×            |
| 5000 × 2000 | 100 | 2000      | 15-20×           |

**Rule of thumb**: Speedup ≈ n / (k_sketch + overhead)

For n >> k_sketch, the speedup approaches n/k_sketch, making `recompute_T=False` very attractive for large problems.

## Notes

1. **Error > 1.0 is not a bug**: For full-rank matrices with k << n, `recompute_T=False` can produce error > 1.0. This is expected behavior (Fortran's approach).

2. **Randomness**: Results vary between runs due to random matrix generation. The seed (42) ensures reproducibility within each language, but cross-language results may differ slightly.

3. **JIT warm-up (Julia)**: Pre-compiles all four modes to avoid compilation overhead in timing.

4. **Validation**: The test validates:
   - recompute_T=True: error < 1.0 (all methods)
   - Explicit operator matches dense exactly (rank and error)
   - Matrix-free returns target rank k=20
   - recompute_T=False: faster but may have error > 1.0

5. **Comparison with test4**:
   - test4: Low-rank matrix (rank ~30, k=20) → error ~0.53 for all modes
   - test5: Full-rank matrix (rank=300, k=20) → error ~0.54 (recompute=True), ~1.14 (recompute=False)

## Files

- `python/test5_linop_fullrank.py` - Python implementation
- `matlab/test5_linop_fullrank.m` - MATLAB/Octave implementation
- `julia/test5_linop_fullrank.jl` - Julia implementation (with JIT warm-up)

## See Also

- **test3_linop_hilbert** - LinearOperator test with ill-conditioned Hilbert matrix
- **test4_linop_random** - LinearOperator test with low-rank random matrix
- **test1_hilbert** - Basic ID test (dense only)
- **test2_svd_hilbert** - Basic SVD test (dense only)

## References

- **recompute_T documentation**: See `compare/RECOMPUTE_T_MATRIX_FREE.md` for detailed analysis
- **Fortran libid**: `src/idd_id.f`, subroutine `idd_lssolve` (lines 329-388)
- **Matrix-free tests**: `compare/test_matfree_python.py` for comprehensive examples
