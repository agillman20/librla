# compare/ - External Library Comparisons

This directory contains scripts to compare librla's Python implementations against external libraries.

## Usage

### Setup

```bash
# Create virtual environment with dependencies
./setup_venv.sh

# Install PARLA (optional, for PARLA comparison)
./setup_parla.sh

# Activate the environment
source venv/bin/activate
```

### Running Comparisons

```bash
# Run all comparisons
python compare_all.py

# Run individual comparisons
python compare_id_scipy.py    # ID: librla vs scipy
python compare_svd_torch.py   # SVD: librla vs torch (CPU + GPU)
python compare_svd_parla.py   # SVD: librla vs PARLA
python compare_id_parla.py    # ID: librla vs PARLA

# When done, deactivate the virtual environment
deactivate
```

## Features

### compare_id_scipy.py

Compares `librla.id_sketch` vs `scipy.linalg.interpolative.interp_decomp`:

- **Metrics**: reconstruction error, conditioning (max|T|), timing
- **Modes**: tolerance mode (rtol < 1) and rank mode (rtol >= 1)
- **Test matrices**: random, low-rank, Hilbert, complex, decaying spectrum, structured (gaussexp, gmm, snn)

### compare_svd_torch.py

Compares `librla.svd_sketch` vs `torch.svd_lowrank`:

- **Metrics**: reconstruction error, singular value accuracy, orthonormality, timing
- **Devices**: CPU always, GPU (CUDA) if available
- **Fair comparison settings**:
  - `power_iter=0` for both (torch defaults to `niter=2`, librla defaults to `power_iter=0`)
  - `extra_samples=12` (librla default oversampling)
  - torch `q = rank + extra_samples` to match librla's oversampled sketch size
- **Test matrices**: same suite as ID comparison (real matrices only, torch doesn't support complex)

### compare_svd_parla.py

Compares `librla.svd_sketch` vs [PARLA's](https://github.com/BallisticLA/parla) `svd1`:

- **Metrics**: reconstruction error, singular value accuracy, orthonormality, timing
- **Test matrices**: random, low-rank, Hilbert, complex, decaying spectrum, structured

### compare_id_parla.py

Compares `librla.id_sketch` vs [PARLA's](https://github.com/BallisticLA/parla) `osid1/osid2`:

- **Metrics**: reconstruction error, timing
- **Test matrices**: comprehensive suite (19 matrices) matching compare_id_scipy.py

#### PARLA ID Strategies (osid1 vs osid2)

PARLA provides two approaches for computing the interpolative coefficient matrix:

| Method | Strategy | Speed | Accuracy on Full-Rank |
|--------|----------|-------|----------------------|
| `osid1` | Triangular solve on **sketch** | Fast | Lower (error can exceed 1.0) |
| `osid2` | Least squares on **original A** | Slower | Higher (always accurate) |

**osid1** (Voronin & Martinsson, 2016):
```
1. Sketch: Y = Sk @ A
2. QRCP on sketch: Q, R, J = qr(Y, pivoting=True)
3. Triangular solve: T = R11^{-1} @ R12
```

**osid2** (Dong & Martinsson, 2021):
```
1. Sketch: Y = Sk @ A
2. QRCP for skeleton indices: J = qrcp(Y)[:k]
3. Least squares on original: T = A[:, J]^+ @ A  (pseudoinverse)
```

**librla equivalents**:
- `id_sketch(..., method='fast')` ≈ osid1 (triangular solve)
- `id_sketch(..., method='lstsq')` ≈ osid2 (least squares)

librla automatically falls back to `method='lstsq'` when the fast method produces error > 1.0.

#### Benchmark Results

**SVD Comparison** (6 test matrices):
| Metric | librla | PARLA |
|--------|--------|-------|
| Average Time | 0.0010s | 0.0017s |
| **Speedup** | **1.7x faster** | baseline |
| Average Error | 4.63e-01 | 4.59e-01 |

**ID Comparison** (5 test matrices, lstsq methods):
| Metric | librla | PARLA osid2 |
|--------|--------|-------------|
| Average Time | 0.0013s | 0.0021s |
| **Speedup** | **1.7x faster** | baseline |
| Average Error | 3.81e-01 | 3.61e-01 |

### compare_all.py

Unified runner that executes all comparisons and produces a combined summary.

## Dependencies

- NumPy
- SciPy (for `scipy.linalg.interpolative`)
- PyTorch (for `torch.svd_lowrank`, optional but recommended)
- PARLA (for `compare_svd_parla.py` and `compare_id_parla.py`, install via `./setup_parla.sh`)

### PARLA Installation Note

PARLA requires patching for NumPy 2.0 compatibility (`np.NaN` → `np.nan`).
The `setup_parla.sh` script handles this automatically.

## Output

Each script produces:
- Per-matrix detailed comparison
- Summary tables with pass/fail status
- Performance comparisons (speedup ratios)
- Accuracy statistics (mean/max errors)
