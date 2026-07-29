# GPU Support for Linear Algebra Operations

This document analyzes GPU support for key linear algebra operations used in librla.

## Summary

| Operation | CUDA (NVIDIA) | Metal (Apple) | MPS (PyTorch) |
|-----------|---------------|---------------|---------------|
| QR with pivoting* | JAX only | No | No |
| SVD | Yes | Partial | Yes |
| Matrix multiply | Yes | Yes | Yes |
| float64 support | Yes | No | No |

*PyTorch pivoted QR is in development (CPU-only PR as of March 2026). JAX supports pivoted QR on GPU.

## Pivoted QR Decomposition

librla relies heavily on `linalg.qr(..., pivoting=True)` for rank-revealing decompositions.
Column pivoting is not merely an optimization — it is structurally required by the
algorithms.  Without it, three failure modes arise:

1. **Rank detection breaks.**  Pivoted QR sorts columns by decreasing norm, so the
   diagonal of R decays monotonically.  The buffered tolerance test
   `diagR[end − extra_samples] / diagR[1] ≤ rtol` (used in `orth_sketch` to
   adaptively grow the sketch until the desired accuracy is reached) relies
   on this ordering.  With
   unpivoted QR the diagonal has no guaranteed ordering; small and large values
   can appear in any position, making the ratio unreliable.  In practice this
   causes the adaptive loop to either terminate too early (returning an
   under-rank basis) or never terminate (growing the sketch to full size and
   falling back unnecessarily).

2. **Column selection fails.**  The interpolative decomposition (`id_sketch`,
   `id_qrpiv`) uses the permutation vector returned by pivoted QR to identify
   *skeleton columns* — the most linearly independent subset of columns of A.
   Unpivoted QR returns no permutation; the leading columns of R correspond to
   the original column order, which is generally unrelated to linear
   independence.  Selecting the first k columns as the skeleton can yield a
   badly conditioned interpolation matrix T, or an outright rank-deficient one,
   causing the reconstruction `A[:, piv[k:]] ≈ A[:, piv[:k]] @ T` to blow up.

3. **Power iteration re-orthogonalization degrades.**  The `_power_iteration`
   helper orthogonalizes via pivoted QR at each step.  When the sketch matrix
   is nearly rank-deficient (common after several subspace iterations on a
   matrix with rapid singular-value decay), pivoting relegates the
   near-dependent directions to the trailing columns of Q, keeping the leading
   block well-conditioned.  Without pivoting the near-zero directions are mixed
   into the leading columns, and rounding errors amplify across iterations,
   producing an orthonormal basis that poorly approximates the dominant
   subspace.

### PyTorch / CUDA

From [PyTorch GitHub issue #82092](https://github.com/pytorch/pytorch/issues/82092):

> "Pivoted QR is implemented in many libraries on the CPU (LAPACK, etc.). On the GPU there seems to be only MAGMA. In particular, there is no pivoted QR in cuSOLVER (only pivoted LU)."

**Status:** Not yet supported. [PR #170051](https://github.com/pytorch/pytorch/pull/170051) adds `torch.linalg.qr_piv` but is CPU-only (CUDA support was dropped due to the ongoing MAGMA deprecation effort). As of March 2026, the PR is under review.

### JAX / CUDA

From [JAX documentation](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.qr.html):

> "Pivoting is only implemented on the CPU and GPU backends."

**Status:** Supported on CPU and GPU backends via `jax.scipy.linalg.qr(pivoting=True)`.

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

### GPU Orthogonality Anomalies

When running on GPU with `--precision single`, tests show:

1. **Orthogonality loss**: ~2 digits lost in `||U'U - I||` and `||V'V - I||`
2. **Test failures**: Orthogonality checks may fail

These results were observed in single precision only — we did not have access to
hardware supporting float64 on GPU to test double precision.

Possible causes:
- GPU implementations may use different algorithms than CPU (e.g. different
  Householder panel strategies, batched vs column-by-column Gram-Schmidt)
- Non-deterministic reduction order on massively parallel hardware
- Fused-multiply-add (FMA) usage differences between CPU and GPU paths

**Observed:** Orthogonality error of `1e-5` on GPU (single precision) vs `1e-7` on
CPU (single precision) — a loss of roughly 2 digits at the same precision, i.e.
a GPU-vs-CPU implementation effect rather than a float32 vs float64 effect.

## Recommendations

### For librla (pivoted QR-based)

1. **CPU with Accelerate** (macOS) or **MKL** (Linux/Windows) - best option
2. **JAX with CUDA** - if GPU acceleration needed (pivoted QR supported on GPU)
3. **Not recommended:** PyTorch CUDA, JAX Metal

### For SVD comparison (`compare_svd_torch.py`)

1. **Double precision** (`--precision double`): Use for accuracy comparisons
2. **Single precision** (`--precision single`): Expect ~2 digits orthogonality loss on GPU relative to CPU at the same precision
3. **CUDA** (`--cuda`): Supported via `torch.svd_lowrank`; note the GPU orthogonality anomalies above

## References

- [torch.linalg.qr documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html)
- [PyTorch issue #82092: Add pivoted QR from MAGMA](https://github.com/pytorch/pytorch/issues/82092)
- [PyTorch PR #170051: Add pivoted QR to ATen and torch.linalg](https://github.com/pytorch/pytorch/pull/170051)
- [JAX scipy.linalg.qr documentation](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.qr.html)
- [Apple JAX Metal documentation](https://developer.apple.com/metal/jax/)

*Last updated: 2026-07-29*
