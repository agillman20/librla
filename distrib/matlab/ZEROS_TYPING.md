# MATLAB zeros() Type Preservation - Implementation Note

## Problem

MATLAB's current approach in librla.m uses `zeros(m, n, dtype_str)` or `zeros(m, n, class(R))` for typed zeros allocation. This approach has a critical limitation: **it loses complex type information**.

### Why class() Loses Complex Information

```matlab
R = randn(5, 5) + 1i*randn(5, 5);  % Complex matrix
class(R)                            % Returns: 'double' (NOT 'complex double')

% Current approach:
T = zeros(3, 5, class(R));         % Creates REAL zeros

% Correct approach:
T = zeros(3, 5, 'like', R);        % Creates COMPLEX zeros (CORRECT!)
```

## Current Status

This library currently uses the `class()` approach for consistency with the existing codebase. It works correctly for real matrices but may have issues with complex inputs. Future versions should consider migrating to 'like' syntax for full type safety.

## References

- MATLAB Documentation: [Create arrays of specified class](https://www.mathworks.com/help/matlab/ref/zeros.html#btv5x5l-1-like)
- See also: TODO.md for implementation roadmap
