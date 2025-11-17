# Power Iteration for Range Estimation

## Overview

`test_power_iteration` demonstrates how **power iteration** improves the quality of randomized range estimation for low-rank matrix approximation. Power iteration applies `(A^H A)^n` to amplify the dominant subspace, making sketching algorithms more accurate.

## What is Power Iteration?

### Basic Concept

Given a matrix `A` and a random test matrix `Ω`, instead of directly forming:
```
Y = A * Ω
```

Power iteration applies the operator `(A^H A)^n`:
```
Y = (A * A^H)^n * A * Ω
```

This amplifies singular values by their power: `σ_i → σ_i^(2n+1)`

### Why It Works

**Spectral amplification**: If matrix `A` has singular values `σ_1 > σ_2 > ... > σ_n`, power iteration increases the spectral gap:

| Original Gap | After 1 iteration | After 2 iterations | After 3 iterations |
|--------------|-------------------|--------------------|--------------------|
| σ₁/σ₂        | (σ₁/σ₂)³          | (σ₁/σ₂)⁵           | (σ₁/σ₂)⁷           |

**Example**: If `σ₁/σ₂ = 2`:
- Original gap: 2×
- After 1 iteration: 2³ = 8×
- After 2 iterations: 2⁵ = 32×
- After 3 iterations: 2⁷ = 128×

This makes the dominant subspace much more distinguishable from noise.

## Test Results Explanation

### Test 1: Range Estimation Quality

This test measures how well power iteration captures the dominant subspace.

**Matrix properties:**
- Size: 500×300
- Target rank: k=50 (want to capture first 50 singular directions)
- Condition number: 1.00e+10
- Spectral gap at k=50: 1.00× (no gap - challenging!)
- Singular values: Exponential decay from 1.0 to 1e-10

**Key metrics:**

1. **Subspace angle** (degrees)
   - Measures alignment between estimated subspace and true dominant subspace
   - Lower is better (0° = perfect alignment)
   - 90° = orthogonal (worst case)

2. **Capture quality** (0 to 1)
   - Mean singular value of projection onto true subspace
   - Higher is better (1.0 = perfect capture)
   - Measures how well the basis captures the dominant directions

3. **Orthogonality** (near machine precision ~1e-15)
   - Verifies `Q^H Q ≈ I` (basis remains orthonormal)

### Results Analysis

#### extra_samples = 24 (block_size = 74)

```
num_iters = 0:  angle = 83.56°,  quality = 0.459  (POOR - random basis)
num_iters = 1:  angle =  2.24°,  quality = 1.000  (GOOD - rapid improvement!)
num_iters = 2:  angle =  0.12°,  quality = 1.000  (EXCELLENT)
num_iters = 3:  angle =  0.001°, quality = 1.000  (NEAR-PERFECT)
num_iters = 4-6: angle < 0.001°, quality = 1.000  (CONVERGED)
```

**Observations:**
- **0 iterations**: Random basis is almost orthogonal to target (83°)
- **1 iteration**: Dramatic improvement to 2.24° (38× better!)
- **2 iterations**: Near-perfect alignment at 0.12°
- **3+ iterations**: Converged (further iterations don't help much)

#### Smaller oversampling (extra_samples = 12, 6, 3)

With less oversampling, convergence is slower:

```
extra_samples = 12:
  num_iters = 0:  angle = 86.78°  (worse starting point)
  num_iters = 1:  angle =  8.43°  (slower convergence)
  num_iters = 2:  angle =  1.87°  (still improving)
  num_iters = 6:  angle =  0.002° (eventually converges)

extra_samples = 3:
  num_iters = 0:  angle = 88.79°  (nearly orthogonal)
  num_iters = 1:  angle = 67.60°  (slow improvement)
  num_iters = 6:  angle =  3.76°  (still not converged!)
```

**Lesson**: Less oversampling requires more power iterations to achieve same accuracy.

### Test 3: SVD Sketch Integration

Tests how `flag_power` parameter in `svd_sketch` improves singular value accuracy.

**Matrix**: 350×200, target rank k=40, exponential decay (10⁶ condition number)

**Results:**

```
flag_power = 0:  error = 8.98e-02, s[40] error = 5.98e-03  (baseline)
flag_power = 1:  error = 6.23e-02, s[40] error = 5.01e-05  (120× better!)
flag_power = 2:  error = 6.22e-02, s[40] error = 2.64e-06  (2265× better)
flag_power = 3:  error = 6.22e-02, s[40] error = 2.29e-07  (26k× better)
flag_power = 6:  error = 6.22e-02, s[40] error = 1.04e-13  (MACHINE PRECISION)
```

**Observations:**
- **Reconstruction error**: Improves from 8.98e-02 → 6.22e-02 (30% reduction)
- **Singular value accuracy**: Improves by **57 million×** (from 6e-3 to 1e-13)
- **Convergence**: 3-4 iterations usually sufficient

## When to Use Power Iteration?

### Use Cases

✅ **Use power iteration when:**
1. Matrix has **poorly separated spectrum** (small spectral gaps)
2. Need **high accuracy** singular values (scientific computing)
3. Matrix has **exponential or polynomial decay** (natural signals)
4. Willing to pay **2× cost per iteration** for better accuracy

❌ **Skip power iteration when:**
1. Matrix has **fast decay** (low-rank structure already clear)
2. Speed is critical (power iteration doubles cost per iteration)
3. Only need **rough approximation** (tolerance mode with loose rtol)

### Cost vs Benefit

**Computational cost:**
- Each power iteration requires: **2 matrix-vector products**
  - Forward: `Y = A * X`
  - Adjoint: `Z = A^H * Y`
  - Total: `X_new = (A^H A) * X_old`

**Typical strategy:**
- `flag_power = 0`: Fast, ~10% error on singular values
- `flag_power = 1`: 2× cost, ~100× better accuracy
- `flag_power = 2`: 3× cost, ~1000× better accuracy
- `flag_power = 3-4`: Diminishing returns

## Relationship to Sketching Parameters

### Oversampling vs Power Iteration

These are **complementary** strategies:

| Strategy         | Effect                    | Cost        | Use When              |
|------------------|---------------------------|-------------|-----------------------|
| **Oversampling** | Increases block_size      | Memory      | Have RAM, want speed  |
| **Power iteration** | Amplifies spectral gap | 2× time/iter | Low memory, need accuracy |

**Rule of thumb:**
- **Large extra_samples** (20-30): Use 0-1 power iterations
- **Small extra_samples** (5-10): Use 2-4 power iterations
- **Minimal extra_samples** (3): Use 4-6 power iterations

### Example Trade-offs

To achieve similar accuracy, choose one:
1. `extra_samples=24, flag_power=0` → Fast, uses more memory
2. `extra_samples=12, flag_power=2` → Balanced
3. `extra_samples=6,  flag_power=4` → Slow, low memory

## Implementation Details

### Python
```python
from libid import id_sketch, svd_sketch

# Use power iteration in id_sketch
k, piv, T = id_sketch(A, rtol=1e-6, flag_power=2)

# Use power iteration in svd_sketch
U, s, Vh = svd_sketch(A, rtol=1e-6, flag_power=2)
```

### MATLAB
```matlab
% Use power iteration
[k, piv, T] = libid.id_sketch(A, 1e-6, 42, 2);  % flag_power=2

% SVD sketch
[U, s, Vh] = libid.svd_sketch(A, 1e-6, 42, 2);  % flag_power=2
```

### Julia
```julia
using LibIDSketch: id_sketch, svd_sketch

# Use power iteration
k, piv, T = id_sketch(A, rtol=1e-6, flag_power=2)

# SVD sketch
U, s, Vh = svd_sketch(A, rtol=1e-6, flag_power=2)
```

## Mathematical Background

### Algorithm

Given matrix `A` (m×n) and test matrix `Ω` (n×l):

**Without power iteration (flag_power=0):**
```
1. Y = A * Ω
2. [Q, ~] = qr(Y, 'econ')
```

**With power iteration (flag_power=p):**
```
1. Y = A * Ω
2. for i = 1 to p:
     Y = A^H * Y
     Y = A * Y
3. [Q, ~] = qr(Y, 'econ')
```

### Convergence Rate

For matrix with singular values `σ₁ ≥ σ₂ ≥ ... ≥ σₙ`, after `p` power iterations:

**Effective singular value ratios:**
```
(σᵢ / σⱼ)^(2p+1)
```

**Convergence criterion:**
- Stop when subspace angle < target tolerance
- Typically 2-4 iterations sufficient
- Diminishing returns after 4-6 iterations

### Numerical Stability

Power iteration maintains **excellent numerical stability**:
- Orthogonality preserved: `||Q^H Q - I||_F ≈ 1e-15` (machine precision)
- No catastrophic cancellation
- QR factorization after each iteration maintains orthonormality

## Running the Tests

### Python
```bash
cd distrib/python
python3 test_power_iteration.py              # Structured matrix only
python3 test_power_iteration.py --random     # Include random matrix
```

### MATLAB/Octave
```bash
cd distrib/matlab
octave --no-gui --eval "test_power_iteration"
# or in MATLAB:
matlab -batch "test_power_iteration"
```

### Julia
```bash
cd distrib/julia
julia test_power_iteration.jl
```

## Key Takeaways

1. **Power iteration dramatically improves sketch quality** with modest cost (2× per iteration)
2. **1-2 iterations usually sufficient** for well-conditioned problems
3. **3-4 iterations recommended** for ill-conditioned or poorly separated spectra
4. **Trade-off**: Oversampling (memory) vs power iteration (time)
5. **Diminishing returns** after 4-6 iterations
6. **Numerically stable**: Orthogonality maintained to machine precision

## See Also

- **test1_hilbert.py** - Basic ID test (uses flag_power parameter)
- **test2_svd_hilbert.py** - Basic SVD test (uses flag_power parameter)
- **compare/test_adaptive_qb.py** - Adaptive QB factorization (power iteration variant)

## References

- **Halko, Martinsson, Tropp (2011)**: "Finding structure with randomness"
  - Algorithm 4.4 (Randomized QB with power iteration)
  - Section 4.5 (Analysis of power iteration scheme)

- **Liberty, Woolfe, et al. (2007)**: "Randomized algorithms for low-rank approximation"
  - Power iteration for range estimation
