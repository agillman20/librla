# Test 1: Simple Hilbert Matrix Test

## Overview

`test1_hilbert` is a simple test demonstrating basic ID algorithms on a medium-size Hilbert matrix. This test validates the core functionality of:

- `id_sketch` - Randomized QR sketching (fast, randomized)
- `id_rrqr` - Deterministic RRQR via LAPACK (reproducible, deterministic)

## Test Matrix

- **Type**: Hilbert matrix (severely ill-conditioned)
- **Size**: 300 x 200
- **Target rank**: k = 15

The Hilbert matrix is chosen because:
1. It's severely ill-conditioned (good stress test)
2. Results are reproducible across implementations
3. Medium size makes it fast to run

## Running the Tests

### Python
```bash
cd distrib/python
python3 test1_hilbert.py
```

### MATLAB/Octave
```bash
cd distrib/matlab
octave --no-gui --eval "test1_hilbert"
# or in MATLAB:
matlab -batch "test1_hilbert"
```

### Julia
```bash
cd distrib/julia
julia test1_hilbert.jl
```

## Expected Results

All three implementations should produce identical results:

```
======================================================================
TEST 1: Medium Hilbert Matrix
======================================================================

Matrix size: 300 x 200
Matrix type: Hilbert (severely ill-conditioned)
Target rank: 15
======================================================================

1. id_sketch (randomized QR sketching)
----------------------------------------------------------------------
  Rank:      k = 15
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 2.956e-10
  Max |T|:   1.379e+00
  Time:      ~0.001-0.002 s

2. id_rrqr (deterministic RRQR via LAPACK)
----------------------------------------------------------------------
  Rank:      k = 15
  Error:     ||A_skel - A_basis @ T|| / ||A|| = 2.956e-10
  Max |T|:   1.379e+00
  Time:      ~0.001-0.013 s

======================================================================
SUMMARY
======================================================================
  Method         Rank    Error        Max|T|       Time
----------------------------------------------------------------------
  id_sketch        15    2.956e-10    1.379e+00    0.0009s
  id_rrqr          15    2.956e-10    1.379e+00    0.0017s
======================================================================

[PASS] Test completed successfully!
```

## Key Metrics

- **Rank**: Both methods select exactly k=15 (rank mode)
- **Error**: Both achieve ~3e-10 reconstruction error (excellent)
- **Max|T|**: Interpolation coefficients ~1.4 (well-conditioned)
- **Time**: Sub-second execution (~1-15ms depending on implementation)

## Notes

1. **Identical results**: Both methods produce identical errors because the Hilbert matrix has a clear rank structure, and rank mode (rtol >= 1) forces both to select exactly k=15 columns.

2. **Speed**: `id_sketch` is typically faster for large matrices, but for this small example both are very fast.

3. **First run (Julia)**: Julia's first run includes JIT compilation time (~0.15s). Subsequent runs are much faster.

4. **Validation**: The test validates that reconstruction error < 1.0 (mathematically required for relative error).

## Files

- `python/test1_hilbert.py` - Python implementation
- `matlab/test1_hilbert.m` - MATLAB/Octave implementation
- `julia/test1_hilbert.jl` - Julia implementation
