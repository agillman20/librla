# MATLAB zeros() Type Preservation - Implementation Note

## Problem

MATLAB's current approach in librla.m uses `zeros(m, n, dtype_str)` or `zeros(m, n, class(R))` for typed zeros allocation. This approach has a critical limitation: **it loses complex type information**.

### Why class() Loses Complex Information

```matlab
R = randn(5, 5) + 1i*randn(5, 5);  % Complex matrix
class(R)                            % Returns: 'double' (NOT 'complex double')

% Current approach:
T = zeros(3, 5, class(R));         % Creates REAL zeros (WRONG!)

% Correct approach:
T = zeros(3, 5, 'like', R);        % Creates COMPLEX zeros (CORRECT!)
```

The problem: `class()` only returns the base class ('double' or 'single'), not the full type including complex/real nature. When initializing matrices with `zeros()`, this can cause complex operations to fail or produce incorrect results.

## MATLAB's 'like' Syntax (The Better Way)

MATLAB provides the `'like'` syntax which preserves:
1. ✅ Base data type (double, single, int32, etc.)
2. ✅ Complex/real nature
3. ✅ GPU/sparse properties (if applicable)
4. ✅ Future-proof for new types

### Syntax
```matlab
zeros(m, n, 'like', prototype)
```

Where `prototype` is any existing matrix whose type properties should be matched.

## Cross-Language Comparison

| Language | Current Approach | Complex Safe? | Better Alternative |
|----------|-----------------|---------------|-------------------|
| **MATLAB** | `zeros(m, n, class(R))` | ❌ NO | `zeros(m, n, 'like', R)` ✅ |
| **Python** | `np.zeros((m, n), dtype=R.dtype)` | ✅ YES | `np.zeros_like(R, shape=(m,n))` ✅ |
| **Julia** | `zeros(eltype(R), m, n)` | ✅ YES | Already correct ✅ |

### Key Insight
- **Python's dtype**: Includes full type info (float64, complex128, etc.) → Safe
- **Julia's eltype**: Includes full type info (Float64, ComplexF64, etc.) → Safe
- **MATLAB's class()**: Only base class (double, single) → **NOT SAFE**
- **MATLAB's 'like'**: Full type preservation → Safe

## Current Implementation in librla.m

After the recent dtype fix (commit 77ec0c3), all 16 `zeros()` calls in librla.m use typed allocation:
- 6 calls use `zeros(m, n, dtype_str)` where `dtype_str = get_dtype_string(A)`
- 10 calls use `zeros(m, n, class(R))`

However, **all of these lose complex type information**.

### All zeros() Locations

#### 1. orth_sketch function (6 calls, lines 136-173)
**Current:**
```matlab
Q = zeros(m, 0, dtype_str);
diagR = zeros(0, 1, dtype_str);
```

**Improved:**
```matlab
Q = zeros(m, 0, 'like', A);
diagR = zeros(0, 1, 'like', A);
```

**Benefit:** Preserves complex type when A is complex.

#### 2. id_qrpiv function (2 calls, lines 532, 537)
**Current:**
```matlab
T = zeros(0, n, class(R));
T = zeros(k, 0, class(R));
```

**Improved:**
```matlab
T = zeros(0, n, 'like', R);
T = zeros(k, 0, 'like', R);
```

**Benefit:** Preserves complex type from QR factorization.

#### 3. compute_T_lstsq function (5 calls, lines 718, 729, 731, 736, 738)
**Current:**
```matlab
T = zeros(k, n - k, class(R));
skeleton_cols = zeros(m, k, class(R));
e_j = zeros(n, 1, class(R));
remaining_cols = zeros(m, n - k, class(R));
e_j = zeros(n, 1, class(R));
```

**Improved:**
```matlab
T = zeros(k, n - k, 'like', R);
skeleton_cols = zeros(m, k, 'like', R);
e_j = zeros(n, 1, 'like', R);
remaining_cols = zeros(m, n - k, 'like', R);
e_j = zeros(n, 1, 'like', R);
```

**Benefit:** Ensures unit vectors e_j and intermediate matrices match A's type.

#### 4. compute_T_svd function (2 calls, lines 768, 779)
**Current:**
```matlab
T = zeros(k, n - k, class(R));
T = zeros(size(R12), class(R));
```

**Improved:**
```matlab
T = zeros(k, n - k, 'like', R);
T = zeros(size(R12), 'like', R12);  % Even better: use R12 as prototype
```

**Benefit:** Line 779 can use R12 directly as prototype for perfect type matching.

#### 5. compute_T_fast function (1 call, line 793)
**Current:**
```matlab
T = zeros(k, n - k, class(R));
```

**Improved:**
```matlab
T = zeros(k, n - k, 'like', R);
```

**Benefit:** Consistent with other T computation methods.

## Summary of Locations

| Location | Current Syntax | Lines | Count |
|----------|---------------|-------|-------|
| orth_sketch | `zeros(m/0, 0/1, dtype_str)` | 136, 138, 143, 145, 170, 172 | 6 |
| id_qrpiv | `zeros(k/0, n/0, class(R))` | 532, 537 | 2 |
| compute_T_lstsq | `zeros(..., class(R))` | 718, 729, 731, 736, 738 | 5 |
| compute_T_svd | `zeros(..., class(R))` | 768, 779 | 2 |
| compute_T_fast | `zeros(k, n-k, class(R))` | 793 | 1 |
| **TOTAL** | | | **16** |

**Unified approach:** Replace all with `zeros(..., 'like', prototype)` where prototype is the appropriate source matrix (A or R).

## Benefits of 'like' Syntax

1. **Correctness**: Properly handles complex matrices throughout
2. **Simplicity**: Cleaner syntax, no need for get_dtype_string() in most places
3. **Consistency**: Matches Python's dtype approach and Julia's eltype approach
4. **Future-proof**: Works with GPU arrays, sparse matrices, and future types
5. **Documentation**: Best practice for MATLAB typed zeros allocation

## Testing Considerations

If migrating to 'like' syntax, test with:
1. **Complex matrix test**: Create a complex Hilbert matrix and verify ID decomposition
2. **Single precision test**: Verify single('double') matrices work correctly
3. **Mixed test**: Complex single-precision matrices
4. **Existing tests**: All current tests should continue to pass

Example test case:
```matlab
% Complex matrix test
A = hilb(100, 50) + 1i * hilb(100, 50);
[k, piv, T] = librla.id_sketch(A, 10);
A_skel = A(:, piv(k+1:end));
A_basis = A(:, piv(1:k));
err = norm(A_skel - A_basis * T) / norm(A);
assert(err < 1.0, 'Complex matrix ID failed');
assert(~isreal(T), 'T should be complex but is real');
```

## Current Status

This library currently uses the `class()` approach for consistency with the existing codebase. It works correctly for real matrices but may have issues with complex inputs. Future versions should consider migrating to 'like' syntax for full type safety.

## References

- MATLAB Documentation: [Create arrays of specified class](https://www.mathworks.com/help/matlab/ref/zeros.html#btv5x5l-1-like)
- See also: TODO.md for implementation roadmap
