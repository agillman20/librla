# compare/ - External Library Comparisons

This directory contains scripts to compare librla's Python implementations against two existing Python libraries: SciPy and PyTorch. These libraries were chosen because they are maintained and widely available.

## Usage

### Setup

```bash
# Create virtual environment with dependencies
./setup_venv.sh

# Activate the environment
source venv/bin/activate
```

### Running Comparisons

```bash
# Run comparisons
python compare_id_scipy.py    # ID: librla vs scipy
python compare_svd_torch.py   # SVD: librla vs torch (CPU + GPU)

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

## Dependencies

- NumPy
- SciPy (for `scipy.linalg.interpolative`)
- PyTorch (for `torch.svd_lowrank`, optional but recommended)

## Output

Each script produces:
- Per-matrix detailed comparison
- Summary tables with pass/fail status
- Performance comparisons (speedup ratios)
- Accuracy statistics (mean/max errors)
