# Error Threshold Analysis: `max(m,n) * eps`

This document analyzes the error threshold `max(m,n) * eps(dtype)` used in librla for numerical rank determination.

## Current Implementation

In `librla.py`, the threshold is computed as:

```python
rtol_eps = max(m, n) * np.finfo(dtype).eps
```

This is used to filter small diagonal elements when determining numerical rank after QR factorization.

## Machine Epsilon Values

| Precision | `eps` |
|-----------|-------|
| float64 (double) | 2.22e-16 |
| float32 (single) | 1.19e-07 |

## Threshold Values for Typical Matrix Sizes

### Double Precision (float64)

| Matrix Size | `max(m,n)*eps` | `sqrt(max(m,n))*eps` | `10*eps` |
|-------------|----------------|----------------------|----------|
| 100x100 | 2.22e-14 | 2.22e-15 | 2.22e-15 |
| 500x300 | 1.11e-13 | 4.97e-15 | 2.22e-15 |
| 1000x600 | 2.22e-13 | 7.02e-15 | 2.22e-15 |
| 4000x2000 | 8.88e-13 | 1.40e-14 | 2.22e-15 |

### Single Precision (float32)

| Matrix Size | `max(m,n)*eps` | `sqrt(max(m,n))*eps` | `10*eps` |
|-------------|----------------|----------------------|----------|
| 100x100 | 1.19e-05 | 1.19e-06 | 1.19e-06 |
| 500x300 | 5.96e-05 | 2.67e-06 | 1.19e-06 |
| 1000x600 | 1.19e-04 | 3.77e-06 | 1.19e-06 |
| 4000x2000 | 4.77e-04 | 7.54e-06 | 1.19e-06 |

## LAPACK Theory

According to the [LAPACK Users' Guide](https://www.netlib.org/lapack/lug/):

### Backward Stability

LAPACK algorithms are backward stable, meaning the computed result is the exact result for a slightly perturbed input:

```
||E|| / ||A|| <= p(m,n) * eps
```

where `p(m,n)` is a "modestly growing function" of the matrix dimensions.

### The Function p(m,n)

From [LAPACK Standard Error Analysis](https://www.netlib.org/lapack/lug/node78.html):

> "Many of LAPACK's error bounds contain a factor p(n) (or p(m,n)), which grows as a function of matrix dimension n. It represents a potentially different function for each problem. In practice, the true errors usually grow just linearly; using p(n) = 10n in the error bound formulas will often give a reasonable bound."

LAPACK code typically uses `p(m,n) = 1` for simplicity, which may slightly underestimate the true error.

### SVD Error Bounds

From [LAPACK SVD Error Bounds](https://www.netlib.org/lapack/lug/node97.html):

> "Each computed singular value differs from the true singular value by at most `p(m,n) * eps * sigma_1`"

where `sigma_1` is the largest singular value.

## Language Conventions

### MATLAB

From [MATLAB rank() documentation](https://www.mathworks.com/help/matlab/ref/rank.html):

```matlab
s = svd(A);
tol = max(size(A)) * s(1) * eps;
r = sum(s > tol);
```

**Default tolerance:** `max(size(A)) * eps(norm(A))` = `max(m,n) * eps * sigma_1`

Key points:
- Scales by the largest singular value `sigma_1 = norm(A)`
- Each row/column processed is allowed to introduce roundoff error of `eps(norm(A))`
- This is the standard approach described in LINPACK Users' Guide

### NumPy (Python)

From [numpy.linalg.matrix_rank() documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html):

```python
tol = S.max() * max(M, N) * eps
```

**Default tolerance:** `max(M, N) * eps * S.max()` (same as MATLAB)

Key points:
- Identical to MATLAB's approach
- The `rtol` parameter defaults to `max(M, N) * eps`
- Designed to detect rank deficiency accounting for SVD numerical errors

### Julia

From [Julia LinearAlgebra.rank() documentation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/):

```julia
tol = max(size(M)...) * eps(eltype(M))
```

**Default tolerance:** `max(m, n) * eps` (without scaling by largest singular value)

Key points:
- Counts singular values with magnitude greater than `tol`
- Parameters `atol` and `rtol` available for control
- Simpler than MATLAB/NumPy (doesn't scale by `sigma_1`)

### Comparison Summary

| Language | Relative Precision | Implementation |
|----------|-------------------|----------------|
| MATLAB | `max(m,n) * eps` | `σᵢ > max(m,n) * eps * σ₁` |
| NumPy | `max(m,n) * eps` | `σᵢ > max(m,n) * eps * σ₁` |
| Julia | `max(m,n) * eps` | `σᵢ > max(m,n) * eps` |
| librla | `max(m,n) * eps` | `σᵢ > max(m,n) * eps` |

All four languages use the same **relative precision** `max(m,n) * eps`.

**MATLAB/NumPy:** Rank is determined by `σᵢ / σ₁ > rtol`. The multiplication by `σ₁` converts the relative tolerance to absolute units for the internal comparison against singular values.

**Julia/librla:** Compare singular values directly against the tolerance. This is equivalent to MATLAB/NumPy for normalized matrices (σ₁ ≈ 1).

## Analysis

### Is `max(m,n) * eps` Too Conservative?

**Conservative means:** The threshold is large, so more values are filtered out as "numerically zero." This may underestimate the numerical rank.

| Threshold | Conservatism | Risk |
|-----------|--------------|------|
| `max(m,n) * eps` | High | May underestimate rank |
| `sqrt(max(m,n)) * eps` | Medium | Balanced |
| `10 * eps` | Low | May include numerical noise |

### Justification for `max(m,n) * eps`

1. **Worst-case accumulation**: In matrix operations, rounding errors can accumulate across all `O(mn)` operations. Using `max(m,n)` accounts for linear error growth.

2. **LAPACK guidance**: While LAPACK uses `p=1` in code, it notes that `p(n) = 10n` is reasonable in practice. Using `max(m,n)` is within this range for most practical sizes.

3. **Safety margin**: A conservative threshold ensures robustness at the cost of potentially selecting a lower rank.

### When It May Be Too Conservative

- For well-conditioned matrices where errors don't accumulate to `O(n)` levels
- When maximum accuracy in rank detection is needed
- For small matrices where `max(m,n) * eps` is still very small

### Alternative Approaches

1. **MATLAB's rank()**: Uses `max(m,n) * eps * max(s)` where `max(s)` is the largest singular value
2. **NumPy's matrix_rank()**: Uses `max(m,n) * eps * max(s)` (same as MATLAB)
3. **Relative threshold**: Use `rtol * ||A||` where `rtol` is user-specified

## Conclusion

The threshold `max(m,n) * eps` is a reasonable conservative choice that:

- Aligns with LAPACK's guidance that errors grow modestly with dimension
- Matches the approach used by MATLAB and NumPy for rank determination
- Provides a safety margin against including numerical noise

For applications requiring tighter thresholds, consider using `sqrt(max(m,n)) * eps` or making the threshold a configurable parameter.

## References

1. [LAPACK Users' Guide: Accuracy and Stability](https://www.netlib.org/lapack/lug/node72.html)
2. [LAPACK: Standard Error Analysis](https://www.netlib.org/lapack/lug/node78.html)
3. [LAPACK: SVD Error Bounds](https://www.netlib.org/lapack/lug/node97.html)
4. [Wikipedia: Machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon)
5. [MATLAB rank() documentation](https://www.mathworks.com/help/matlab/ref/rank.html)
6. [NumPy matrix_rank() documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)
7. [Julia LinearAlgebra documentation](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
