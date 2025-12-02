# TODO - librla Distribution

## Python pip Distribution

Currently there's no pip packaging. Here's what's needed to create a pip-distributable package:

### Recommended Package Structure

```
distrib/python/
├── pyproject.toml          # Package metadata (modern approach)
├── src/
│   └── librla/
│       ├── __init__.py     # Exports public API
│       └── _core.py        # Current librla.py content
├── tests/                  # Test files
│   ├── test_hilbert.py
│   ├── test_kahan.py
│   └── ...
└── README.md               # Package-specific readme
```

### Minimal pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "librla"
version = "0.1.0"
description = "Randomized linear algebra for low-rank matrix approximations"
readme = "README.md"
license = {text = "TBD"}
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["src"]
```

### Quick Option: Single-file package

For a simpler approach with the current flat structure:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "librla"
version = "0.1.0"
description = "Randomized linear algebra for low-rank matrix approximations"
requires-python = ">=3.8"
dependencies = ["numpy>=1.20", "scipy>=1.7"]

[tool.setuptools]
py-modules = ["librla", "hilb", "kahan", "make_mat"]
```

### Commands to Build and Distribute

```bash
# Install build tools
pip install build twine

# Build the package
cd distrib/python
python -m build

# Creates dist/librla-0.1.0.tar.gz and dist/librla-0.1.0-py3-none-any.whl

# Upload to PyPI (requires account)
twine upload dist/*

# Or upload to TestPyPI first
twine upload --repository testpypi dist/*
```

---

## Documentation Issues to Fix

### FILE_MANIFEST.txt
- File count is wrong (says 47, actual count differs)
- CLAUDE.md not listed
- Python `local/` directory should not be distributed

### README.md (line 250-251)
```
│   ├── LinearOperator.py  - Matrix-free operators (planned)
```
Should say **(via scipy)** or be removed since Python uses `scipy.sparse.linalg.LinearOperator` directly.

### INSTALL.md (line 50-51)
```bash
cp distrib/python/LinearOperator.py your_project/  # Optional
```
This file doesn't exist - remove this line since scipy provides LinearOperator.

### test1_hilbert.py (line 96)
Summary output refers to `id_rrqr` but the function is actually `id_qrpiv`.

---

## API Note: svd_sketch Return Values

Cross-language API difference to document clearly:
- **Python**: Returns `U, s, Vh` (Vh is V conjugate-transpose)
- **MATLAB**: Returns `U, s, V` (V is NOT transposed)
- **Julia**: Returns `U, s, Vt` (Vt is V transpose)

---

## MATLAB: Consider Migrating to 'like' Syntax for zeros()

### Current Issue
MATLAB code uses `zeros(m, n, class(R))` or `zeros(m, n, dtype_str)` for typed zeros allocation. This approach loses complex type information since `class()` only returns the base class ('double', 'single') without the complex/real distinction.

### Better Approach
Use MATLAB's `'like'` syntax: `zeros(m, n, 'like', R)` which preserves:
- Complex/real nature
- Precision (single/double)
- GPU/sparse properties
- Future type safety

### Details
See extensive documentation comment in `matlab/librla.m` (lines ~28-65) explaining:
- Why 'like' is better than class()
- Cross-language comparison (Python dtype, Julia eltype)
- Examples and benefits

### Locations to Update
All 16 zeros() calls in `matlab/librla.m`:
- orth_sketch: lines 137, 139, 144, 146, 171, 173 (6 calls)
- id_qrpiv: lines 532, 537 (2 calls)
- compute_T_lstsq: lines 722, 729, 731, 736, 738 (5 calls)
- compute_T_svd: lines 768, 780 (2 calls)
- compute_T_fast: line 793 (1 call)

### Priority
Low - current code works correctly for real matrices. Mainly improves correctness for complex matrix inputs and future-proofs the implementation.

---

## Cleanup

- Remove `distrib/python/local/` directory (virtual environment)
- Add `local/` and `__pycache__/` to `.gitignore`
