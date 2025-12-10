# GPU Support for Linear Algebra Operations

This document analyzes GPU support for key linear algebra operations used in librla.

## Summary

| Operation | CUDA (NVIDIA) | Metal (Apple) | MPS (PyTorch) |
|-----------|---------------|---------------|---------------|
| QR with pivoting | No* | No | No |
| SVD | Yes | Partial | Yes |
| Matrix multiply | Yes | Yes | Yes |
| float64 support | Yes | No | Yes |

*MAGMA has pivoted QR, but PyTorch doesn't expose it.

## Pivoted QR Decomposition

librla relies heavily on `linalg.qr(..., pivoting=True)` for rank-revealing decompositions.

### PyTorch / CUDA

From [PyTorch GitHub issue #82092](https://github.com/pytorch/pytorch/issues/82092):

> "Pivoted QR is implemented in many libraries on the CPU (LAPACK, etc.). On the GPU there seems to be only MAGMA. In particular, there is no pivoted QR in cuSOLVER (only pivoted LU)."

**Status:** Not supported. PyTorch's `torch.linalg.qr()` has no pivoting option.

### JAX / CUDA

From [JAX documentation](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.qr.html):

> "Pivoting is only implemented on the CPU and GPU backends."

**Status:** Supported on CUDA via MAGMA (with `use_magma=True`).

### JAX / Metal (Apple Silicon)

From [Apple JAX Metal documentation](https://developer.apple.com/metal/jax/):

- Experimental status
- **Does not support float64, complex64, complex128**
- Pivoted QR support unclear

**Status:** Not recommended due to float64 limitation.

## SVD Operations

### PyTorch `torch.svd_lowrank`

Used in `compare_svd_torch.py` for randomized low-rank SVD comparison.

**Supported on:**
- CPU (LAPACK)
- CUDA (cuSOLVER)
- MPS (Metal Performance Shaders on Apple Silicon)

### Single Precision Anomalies

When running with `--precision single`, tests may show:

1. **Orthogonality loss**: ~1 digit lost in `||U'U - I||` and `||V'V - I||`
2. **Test failures**: Orthogonality checks may fail with single precision

This is expected behavior because:
- Single precision (float32) has ~7 significant digits vs ~16 for double (float64)
- Accumulated rounding errors in Gram-Schmidt orthogonalization
- GPU implementations may use different algorithms than CPU

**Example:** Orthogonality error of `1e-5` in single precision vs `1e-14` in double precision.

## Recommendations

### For librla (pivoted QR-based)

1. **CPU with Accelerate** (macOS) or **MKL** (Linux/Windows) - best option
2. **JAX with CUDA** - if GPU acceleration needed and MAGMA available
3. **Not recommended:** PyTorch CUDA, JAX Metal

### For SVD comparison (`compare_svd_torch.py`)

1. **Double precision** (`--precision double`): Use for accuracy comparisons
2. **Single precision** (`--precision single`): Expect ~1 digit orthogonality loss
3. **CUDA** (`--cuda`): Supported via `torch.svd_lowrank`

## References

- [torch.linalg.qr documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html)
- [PyTorch issue #82092: Add pivoted QR from MAGMA](https://github.com/pytorch/pytorch/issues/82092)
- [JAX scipy.linalg.qr documentation](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.qr.html)
- [Apple JAX Metal documentation](https://developer.apple.com/metal/jax/)
