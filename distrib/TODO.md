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
py-modules = ["librla", "hilbert", "kahan"]
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

## MATLAB: Consider Migrating to 'like' Syntax for zeros()

### Current Issue
MATLAB code uses `zeros(m, n, class(R))` or `zeros(m, n, dtype_str)` for typed zeros allocation. This approach loses complex type information since `class()` only returns the base class ('double', 'single') without the complex/real distinction.

### Better Approach
Use MATLAB's `'like'` syntax: `zeros(m, n, 'like', R)` which preserves:
- Complex/real nature
- Precision (single/double)
- GPU/sparse properties
- Future type safety

### Priority
Low - current code works correctly for real matrices. Mainly improves correctness for complex matrix inputs and future-proofs the implementation.

---

## Cleanup

- Remove `distrib/python/venv/` directory (virtual environment)
- Add `venv/` and `__pycache__/` to `.gitignore`
