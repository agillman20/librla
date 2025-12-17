#!/usr/bin/env python3
"""
compare_svd_torch.py - Compare librla svd_sketch vs torch.svd_lowrank

Compares two low-rank SVD implementations:
- librla.svd_sketch: Randomized SVD (our implementation)
- torch.svd_lowrank: PyTorch's randomized low-rank SVD

Tests on both CPU and GPU (if CUDA available).

Compares on metrics:
- Accuracy (reconstruction error)
- Singular value accuracy
- Orthonormality of U and V
- Runtime

Usage:
    python compare_svd_torch.py [--threads N] [--precision {double,single}] [--cuda] [--extra-samples N] [--power-iter N]

Options:
    --threads N        Number of threads (default: number of CPU cores)
                       Sets threading for both numpy/scipy (MKL/OpenBLAS) and torch
    --precision        Floating-point precision: double (default) or single
    --cuda             Run CUDA tests (default: CPU tests only)
    --extra-samples N  Number of extra samples for oversampling (default: 12)
    --power-iter N     Number of power iterations (default: 0)

Requires:
    - NumPy, SciPy, PyTorch
    - librla.py, test_utils.py from ../python/

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 0.1.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import argparse
import os
import sys

# Get number of CPU cores for default thread count (physical cores only)
NUM_CPUS = (os.cpu_count() or 2) // 2

# Parse arguments BEFORE any numeric library imports
# Threading env vars must be set before numpy/scipy/torch are imported
parser = argparse.ArgumentParser(description='Compare librla svd_sketch vs torch.svd_lowrank')
parser.add_argument('--threads', type=int, default=NUM_CPUS,
                    help=f'Number of threads (default: {NUM_CPUS})')
parser.add_argument('--precision', choices=['double', 'single'], default='double',
                    help='Floating-point precision (default: double)')
parser.add_argument('--cuda', action='store_true',
                    help='Run CUDA tests (default: CPU tests only)')
parser.add_argument('--extra-samples', type=int, default=12,
                    help='Number of extra samples for oversampling (default: 12)')
parser.add_argument('--power-iter', type=int, default=0,
                    help='Number of power iterations (default: 0)')
parser.add_argument('--verbose', action='store_true',
                    help='Show detailed results table (default: summary only)')
args = parser.parse_args()

# Set all threading environment variables BEFORE importing numpy/scipy/torch
# This ensures consistent threading across all libraries
thread_str = str(args.threads)
os.environ['OMP_NUM_THREADS'] = thread_str
os.environ['MKL_NUM_THREADS'] = thread_str
os.environ['OPENBLAS_NUM_THREADS'] = thread_str
os.environ['VECLIB_MAXIMUM_THREADS'] = thread_str  # macOS Accelerate
os.environ['NUMEXPR_NUM_THREADS'] = thread_str

# Now import numeric libraries
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional

# Set dtype and tolerances based on precision
DTYPE = np.float64 if args.precision == 'double' else np.float32
CDTYPE = np.complex128 if args.precision == 'double' else np.complex64
PRECISION = args.precision

# Precision-dependent constants
# Double: ~16 decimal digits, Single: ~7 decimal digits
if PRECISION == 'double':
    EPS = 1e-10      # Noise level for low-rank matrices
    ORTH_TOL = 1e-10    # Orthonormality tolerance
else:
    EPS = 1e-5       # Noise level for low-rank matrices (single precision)
    ORTH_TOL = 1e-5     # Orthonormality tolerance (single precision)

# Add parent python directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Import implementations
from librla import svd_sketch
from test_utils import make_mat

try:
    import torch
    TORCH_AVAILABLE = True
    # Set torch threads explicitly for Intel MKL compatibility
    # This ensures torch respects our thread setting even if MKL ignores env vars
    torch.set_num_threads(args.threads)
    # Also set inter-op threads to avoid nested parallelism (thread explosion)
    torch.set_num_interop_threads(args.threads)
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Run: pip install torch")


@dataclass
class ComparisonResult:
    """Results from comparing SVD methods on a single matrix."""
    name: str
    rank: int
    device: str  # 'cpu' or 'cuda'

    # Reconstruction errors
    err_librla: float
    err_torch: float

    # Singular value accuracy (vs numpy reference)
    sval_err_librla: float
    sval_err_torch: float

    # Orthonormality errors
    orth_U_librla: float
    orth_V_librla: float
    orth_U_torch: float
    orth_V_torch: float

    # Timing
    t_librla: float
    t_torch: float

    passed: bool


def hilbert(m, n, dtype=np.float64):
    """Generate an mxn Hilbert matrix."""
    i = np.arange(1, m + 1, dtype=dtype).reshape(-1, 1)
    j = np.arange(1, n + 1, dtype=dtype).reshape(1, -1)
    return 1.0 / (i + j - 1)


def compare_on_matrix(A, rank, name, device='cpu', power_iter=0, extra_samples=12):
    """
    Compare SVD implementations on a single matrix.

    Parameters
    ----------
    A : ndarray
        Input matrix to decompose (real or complex)
    rank : int
        Target rank for approximation
    name : str
        Test case name for display
    device : str
        'cpu' or 'cuda'
    power_iter : int
        Number of power iterations for both methods (default 0 for fair comparison)
    extra_samples : int
        Oversampling parameter (default 12, matching librla default)
        librla uses rank + extra_samples internally
        torch q parameter should be rank + extra_samples for fair comparison

    Returns
    -------
    result : ComparisonResult
        Comparison metrics
    """
    print("\n" + "="*70)
    print(f"Test: {name}")
    print(f"Matrix: {A.shape[0]}x{A.shape[1]}")
    print(f"Target rank: {rank}, oversampled rank (q): {rank + extra_samples}")
    print(f"Device: {device}")
    print(f"Power iterations: {power_iter}, extra_samples: {extra_samples}")
    print("="*70)

    normA = np.linalg.norm(A, 'fro')

    # Get reference singular values from numpy
    s_ref = np.linalg.svd(A, compute_uv=False)

    # -------------------------------------------------------------------------
    # 1. librla svd_sketch (always on CPU/numpy)
    # librla internally uses block_size = rank + extra_samples in rank mode
    # -------------------------------------------------------------------------
    print("\n--- librla svd_sketch ---")

    t0 = time.perf_counter()
    U_librla, s_librla, Vh_librla = svd_sketch(A, rtol=float(rank), power_iter=power_iter, extra_samples=extra_samples)
    t_librla = time.perf_counter() - t0

    k_librla = len(s_librla)

    # Reconstruction error
    A_recon_librla = U_librla @ np.diag(s_librla) @ Vh_librla
    err_librla = np.linalg.norm(A - A_recon_librla, 'fro') / normA

    # Singular value accuracy
    k_cmp = min(k_librla, len(s_ref))
    if k_cmp > 0:
        sval_err_librla = np.linalg.norm(s_librla[:k_cmp] - s_ref[:k_cmp]) / np.linalg.norm(s_ref[:k_cmp])
    else:
        sval_err_librla = 0.0

    # Orthonormality checks (use conjugate transpose for complex)
    orth_U_librla = np.linalg.norm(U_librla.conj().T @ U_librla - np.eye(k_librla), 'fro')
    orth_V_librla = np.linalg.norm(Vh_librla @ Vh_librla.conj().T - np.eye(k_librla), 'fro')

    print(f"Rank:       k = {k_librla}")
    print(f"Error:      ||A - U @ S @ Vh|| / ||A|| = {err_librla:.3e}")
    print(f"SVal Err:   ||s - s_ref|| / ||s_ref|| = {sval_err_librla:.3e}")
    print(f"Orth U:     ||U'U - I|| = {orth_U_librla:.3e}")
    print(f"Orth Vh:    ||Vh Vh' - I|| = {orth_V_librla:.3e}")
    print(f"Time:       {t_librla:.4f} s")

    # -------------------------------------------------------------------------
    # 2. torch.svd_lowrank
    # Note: torch defaults to niter=2, we use power_iter for fair comparison
    # torch q = oversampled rank, so we use q = rank + extra_samples to match librla
    # -------------------------------------------------------------------------
    q_torch = rank + extra_samples  # Match librla's oversampling
    print(f"\n--- torch svd_lowrank ({device}, q={q_torch}, niter={power_iter}) ---")

    # Convert to torch tensor (use same precision as input matrix)
    A_tensor = torch.from_numpy(A)
    if device == 'cuda':
        A_tensor = A_tensor.cuda()

    # Warm-up run for GPU
    if device == 'cuda':
        _ = torch.svd_lowrank(A_tensor, q=q_torch, niter=power_iter)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    U_torch, S_torch, V_torch = torch.svd_lowrank(A_tensor, q=q_torch, niter=power_iter)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_torch = time.perf_counter() - t0

    # Convert back to numpy and truncate to target rank (like librla does)
    # torch returns q columns, we truncate to rank for fair comparison
    # Use resolve_conj() for complex tensors (V may have conjugate bit set)
    U_torch_full = U_torch.resolve_conj().cpu().numpy()
    s_torch_full = S_torch.cpu().numpy()
    V_torch_full = V_torch.resolve_conj().cpu().numpy()

    # Truncate to target rank (matching librla behavior)
    k_torch = min(rank, len(s_torch_full))
    U_torch_np = U_torch_full[:, :k_torch]
    s_torch_np = s_torch_full[:k_torch]
    V_torch_np = V_torch_full[:, :k_torch]

    # Reconstruction error
    # torch returns V (not V'), so reconstruction is U @ diag(S) @ V.H
    A_recon_torch = U_torch_np @ np.diag(s_torch_np) @ V_torch_np.conj().T
    err_torch = np.linalg.norm(A - A_recon_torch, 'fro') / normA

    # Singular value accuracy
    k_cmp = min(k_torch, len(s_ref))
    if k_cmp > 0:
        sval_err_torch = np.linalg.norm(s_torch_np[:k_cmp] - s_ref[:k_cmp]) / np.linalg.norm(s_ref[:k_cmp])
    else:
        sval_err_torch = 0.0

    # Orthonormality checks (use conjugate transpose for complex)
    orth_U_torch = np.linalg.norm(U_torch_np.conj().T @ U_torch_np - np.eye(k_torch), 'fro')
    orth_V_torch = np.linalg.norm(V_torch_np.conj().T @ V_torch_np - np.eye(k_torch), 'fro')

    print(f"Rank:       k = {k_torch} (truncated from q={q_torch})")
    print(f"Error:      ||A - U @ S @ V^H|| / ||A|| = {err_torch:.3e}")
    print(f"SVal Err:   ||s - s_ref|| / ||s_ref|| = {sval_err_torch:.3e}")
    print(f"Orth U:     ||U'U - I|| = {orth_U_torch:.3e}")
    print(f"Orth V:     ||V'V - I|| = {orth_V_torch:.3e}")
    print(f"Time:       {t_torch:.4f} s")

    # -------------------------------------------------------------------------
    # Summary comparison
    # -------------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Method':<25s} {'Rank':<8s} {'Recon Err':<12s} {'SVal Err':<12s} {'Time (s)':<10s}")
    print("-"*75)
    print(f"{'librla svd_sketch':<25s} {k_librla:<8d} {err_librla:<12.3e} {sval_err_librla:<12.3e} {t_librla:<10.4f}")
    print(f"{'torch svd_lowrank':<25s} {k_torch:<8d} {err_torch:<12.3e} {sval_err_torch:<12.3e} {t_torch:<10.4f}")

    # Highlight fastest method
    methods = ['librla', 'torch']
    times = [t_librla, t_torch]
    fastest_idx = np.argmin(times)
    print(f"\nFastest method: {methods[fastest_idx]} ({times[fastest_idx]:.4f}s)")

    # Speedup
    if t_librla > 0 and t_torch > 0:
        if t_librla < t_torch:
            print(f"Speedup: librla is {t_torch/t_librla:.1f}x faster")
        else:
            print(f"Speedup: torch is {t_librla/t_torch:.1f}x faster")

    # Determine if test passed
    # Check orthonormality and that errors are reasonable
    orth_ok = (orth_U_librla < ORTH_TOL and orth_V_librla < ORTH_TOL and
               orth_U_torch < ORTH_TOL and orth_V_torch < ORTH_TOL)
    # Error ratio should be within 2x of each other (both are randomized)
    error_ok = max(err_librla, err_torch) < 1.0

    passed = orth_ok and error_ok

    return ComparisonResult(
        name=name,
        rank=rank,
        device=device,
        err_librla=err_librla, err_torch=err_torch,
        sval_err_librla=sval_err_librla, sval_err_torch=sval_err_torch,
        orth_U_librla=orth_U_librla, orth_V_librla=orth_V_librla,
        orth_U_torch=orth_U_torch, orth_V_torch=orth_V_torch,
        t_librla=t_librla, t_torch=t_torch,
        passed=passed
    )


def run_test_suite(device='cpu', power_iter=0, extra_samples=12):
    """Run the full test suite on a given device.

    Parameters
    ----------
    device : str
        'cpu' or 'cuda'
    power_iter : int
        Number of power iterations for both methods (0 for fair base comparison)
    extra_samples : int
        Number of extra samples for oversampling (default: 12)
    """

    print("\n" + "#"*80)
    print(f"# RUNNING TESTS ON: {device.upper()} (power_iter={power_iter})")
    print("#"*80)

    results = []

    # -------------------------------------------------------------------------
    # BASIC TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("BASIC TESTS")
    print("Testing fundamental rank-mode approximations")
    print("="*70)

    # Test 1: Random matrix (well-conditioned)
    np.random.seed(42)
    A1 = np.random.randn(500, 300).astype(DTYPE)
    results.append(compare_on_matrix(A1, 20, "Random Matrix (well-conditioned)", device, power_iter, extra_samples))

    # Test 2: Low-rank matrix
    U = np.random.randn(400, 15).astype(DTYPE)
    V = np.random.randn(250, 15).astype(DTYPE)
    A2 = (U @ V.T + EPS * np.random.randn(400, 250)).astype(DTYPE)
    results.append(compare_on_matrix(A2, 15, "Low-Rank Matrix (rank~15)", device, power_iter, extra_samples))

    # Test 3: Hilbert matrix (extremely ill-conditioned)
    A3 = hilbert(2000, 1000, dtype=DTYPE)
    results.append(compare_on_matrix(A3, 15, "Hilbert Matrix (ill-conditioned)", device, power_iter, extra_samples))

    # Test 4: Decaying spectrum
    A4 = np.random.randn(400, 300).astype(DTYPE)
    U4, S4, Vh4 = np.linalg.svd(A4, full_matrices=False)
    s4 = (1.0 / np.arange(1, 301)).astype(DTYPE)
    A4 = (U4 @ np.diag(s4) @ Vh4).astype(DTYPE)
    results.append(compare_on_matrix(A4, 50, "Decaying Spectrum (1/k)", device, power_iter, extra_samples))

    # -------------------------------------------------------------------------
    # LARGE MATRIX TESTS (2x larger)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("LARGE MATRIX TESTS (2x SCALE)")
    print("Testing scaling behavior with matrices twice as large")
    print("="*70)

    # Test 5: Large random matrix
    A5 = np.random.randn(1000, 600).astype(DTYPE)
    results.append(compare_on_matrix(A5, 20, "Large Random Matrix (1000x600)", device, power_iter, extra_samples))

    # Test 6: Large low-rank matrix
    U6 = np.random.randn(800, 15).astype(DTYPE)
    V6 = np.random.randn(500, 15).astype(DTYPE)
    A6 = (U6 @ V6.T + EPS * np.random.randn(800, 500)).astype(DTYPE)
    results.append(compare_on_matrix(A6, 15, "Large Low-Rank (800x500, rank~15)", device, power_iter, extra_samples))

    # Test 7: Large Hilbert matrix
    A7 = hilbert(4000, 2000, dtype=DTYPE)
    results.append(compare_on_matrix(A7, 15, "Large Hilbert Matrix (4000x2000)", device, power_iter, extra_samples))

    # Test 8: Large decaying spectrum
    A8 = np.random.randn(800, 600).astype(DTYPE)
    U8, S8, Vh8 = np.linalg.svd(A8, full_matrices=False)
    s8 = (1.0 / np.arange(1, 601)).astype(DTYPE)
    A8 = (U8 @ np.diag(s8) @ Vh8).astype(DTYPE)
    results.append(compare_on_matrix(A8, 50, "Large Decaying Spectrum (1/k, 800x600)", device, power_iter, extra_samples))

    # -------------------------------------------------------------------------
    # SLOW DECAYING SPECTRUM TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("SLOW DECAYING SPECTRUM TESTS")
    print("Testing harder rank-deficient problems with slow decay")
    print("="*70)

    # Test 9: Slow decay - sqrt
    A9 = np.random.randn(400, 300).astype(DTYPE)
    U9, S9, Vh9 = np.linalg.svd(A9, full_matrices=False)
    s9 = (1.0 / np.sqrt(np.arange(1, 301))).astype(DTYPE)
    A9 = (U9 @ np.diag(s9) @ Vh9).astype(DTYPE)
    results.append(compare_on_matrix(A9, 50, "Slow Decay - Sqrt (1/sqrtk)", device, power_iter, extra_samples))

    # Test 10: Slow decay - polynomial
    A10 = np.random.randn(400, 300).astype(DTYPE)
    U10, S10, Vh10 = np.linalg.svd(A10, full_matrices=False)
    s10 = (1.0 / (np.arange(1, 301) ** 0.7)).astype(DTYPE)
    A10 = (U10 @ np.diag(s10) @ Vh10).astype(DTYPE)
    results.append(compare_on_matrix(A10, 50, "Slow Decay - Polynomial (1/k^0.7)", device, power_iter, extra_samples))

    # Test 11: Slow decay - exponential
    A11 = np.random.randn(400, 300).astype(DTYPE)
    U11, S11, Vh11 = np.linalg.svd(A11, full_matrices=False)
    s11 = np.exp(-np.arange(1, 301) / 100.0).astype(DTYPE)
    A11 = (U11 @ np.diag(s11) @ Vh11).astype(DTYPE)
    results.append(compare_on_matrix(A11, 50, "Slow Decay - Exponential (exp(-k/100))", device, power_iter, extra_samples))

    # -------------------------------------------------------------------------
    # MAKE_MAT MATRIX TESTS (structured matrices from paper)
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("MAKE_MAT STRUCTURED MATRIX TESTS")
    print("Testing matrices from 'Robust blockwise random pivoting' paper")
    print("="*70)

    # Test 12: Gaussian Exponential Decay Matrix
    A12 = make_mat(500, 500, 'gaussexp').astype(DTYPE)
    results.append(compare_on_matrix(A12, 50, "Gaussexp (Gaussian Exponential Decay)", device, power_iter, extra_samples))

    # Test 13: Gaussian Mixture Model Matrix
    A13 = make_mat(400, 400, 'gmm').astype(DTYPE)
    results.append(compare_on_matrix(A13, 50, "GMM (Gaussian Mixture Model)", device, power_iter, extra_samples))

    # Test 14: Sparse Neural Network Matrix
    A14 = make_mat(300, 300, 'snn').astype(DTYPE)
    results.append(compare_on_matrix(A14, 50, "SNN (Sparse Neural Network)", device, power_iter, extra_samples))

    # -------------------------------------------------------------------------
    # COMPLEX MATRIX TESTS
    # -------------------------------------------------------------------------
    print("\n\n" + "="*70)
    print("COMPLEX MATRIX TESTS")
    print("Testing complex-valued matrices (both real and imaginary parts)")
    print("="*70)

    # Test 15: Complex random matrix
    A15 = (np.random.randn(500, 300) + 1j * np.random.randn(500, 300)).astype(CDTYPE)
    results.append(compare_on_matrix(A15, 20, "Complex Random Matrix", device, power_iter, extra_samples))

    # Test 16: Complex low-rank matrix
    U16 = (np.random.randn(400, 15) + 1j * np.random.randn(400, 15)).astype(CDTYPE)
    V16 = (np.random.randn(250, 15) + 1j * np.random.randn(250, 15)).astype(CDTYPE)
    A16 = U16 @ V16.conj().T + EPS * (np.random.randn(400, 250) + 1j * np.random.randn(400, 250)).astype(CDTYPE)
    results.append(compare_on_matrix(A16, 15, "Complex Low-Rank (rank~15)", device, power_iter, extra_samples))

    # Test 17: Complex decaying spectrum
    A17 = (np.random.randn(400, 300) + 1j * np.random.randn(400, 300)).astype(CDTYPE)
    U17, S17, Vh17 = np.linalg.svd(A17, full_matrices=False)
    s17 = (1.0 / np.arange(1, 301)).astype(DTYPE)
    A17 = (U17 @ np.diag(s17) @ Vh17).astype(CDTYPE)
    results.append(compare_on_matrix(A17, 50, "Complex Decaying Spectrum (1/k)", device, power_iter, extra_samples))

    # Test 18: Complex Hermitian-like matrix (A @ A.H is Hermitian)
    B18 = (np.random.randn(300, 150) + 1j * np.random.randn(300, 150)).astype(CDTYPE)
    A18 = B18 @ B18.conj().T  # Hermitian positive semi-definite
    results.append(compare_on_matrix(A18, 30, "Complex Hermitian (B @ B^H)", device, power_iter, extra_samples))

    # Test 19: Large complex matrix
    A19 = (np.random.randn(800, 500) + 1j * np.random.randn(800, 500)).astype(CDTYPE)
    results.append(compare_on_matrix(A19, 20, "Large Complex Random (800x500)", device, power_iter, extra_samples))

    return results


def print_summary(results: List[ComparisonResult], device: str, power_iter: int = 0):
    """Print summary for a set of results."""

    print("\n\n" + "="*80)
    print(f"TEST SUMMARY ({device.upper()}, power_iter={power_iter}) - {len(results)} tests completed")
    print("="*80)

    # Overall pass/fail
    passed_tests = sum(1 for r in results if r.passed)
    total_tests = len(results)
    pass_rate = 100.0 * passed_tests / total_tests

    print()
    print(f"Pass Rate: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")

    if passed_tests == total_tests:
        print("[PASS] All tests PASSED")
    else:
        print("[WARNING] Some tests FAILED")
        for r in results:
            if not r.passed:
                print(f"  [FAIL] {r.name}")

    # Performance summary
    print()
    print("="*80)
    print("Performance Summary")
    print("="*80)

    avg_time_librla = np.mean([r.t_librla for r in results])
    avg_time_torch = np.mean([r.t_torch for r in results])

    print()
    print(f"{'Method':<28s} {'Avg Time':<12s} {'Speedup':<15s}")
    print("-"*80)

    # Show speedup relative to the slower method
    if avg_time_librla < avg_time_torch:
        speedup_librla = avg_time_torch / avg_time_librla
        print(f"{'librla svd_sketch':<28s} {avg_time_librla:>8.4f}s    {speedup_librla:>6.1f}x faster")
        print(f"{'torch svd_lowrank':<28s} {avg_time_torch:>8.4f}s    {1.0:>6.1f}x (base)")
    else:
        speedup_torch = avg_time_librla / avg_time_torch
        print(f"{'librla svd_sketch':<28s} {avg_time_librla:>8.4f}s    {1.0:>6.1f}x (base)")
        print(f"{'torch svd_lowrank':<28s} {avg_time_torch:>8.4f}s    {speedup_torch:>6.1f}x faster")

    # Accuracy summary
    print()
    print("Reconstruction Error Summary:")
    print("-"*80)

    avg_err_librla = np.mean([r.err_librla for r in results])
    max_err_librla = np.max([r.err_librla for r in results])
    avg_err_torch = np.mean([r.err_torch for r in results])
    max_err_torch = np.max([r.err_torch for r in results])

    print(f"  librla svd_sketch: mean={avg_err_librla:.3e}, max={max_err_librla:.3e}")
    print(f"  torch svd_lowrank: mean={avg_err_torch:.3e}, max={max_err_torch:.3e}")

    # Singular value accuracy summary
    print()
    print("Singular Value Accuracy (vs numpy reference):")
    print("-"*80)

    avg_sval_err_librla = np.mean([r.sval_err_librla for r in results])
    avg_sval_err_torch = np.mean([r.sval_err_torch for r in results])

    print(f"  librla svd_sketch: mean={avg_sval_err_librla:.3e}")
    print(f"  torch svd_lowrank: mean={avg_sval_err_torch:.3e}")

    # Orthonormality summary
    print()
    print("Orthonormality Summary:")
    print("-"*80)

    max_orth_U_librla = np.max([r.orth_U_librla for r in results])
    max_orth_V_librla = np.max([r.orth_V_librla for r in results])
    max_orth_U_torch = np.max([r.orth_U_torch for r in results])
    max_orth_V_torch = np.max([r.orth_V_torch for r in results])

    print(f"  librla: max ||U'U - I||={max_orth_U_librla:.3e}, max ||Vh Vh' - I||={max_orth_V_librla:.3e}")
    print(f"  torch:  max ||U'U - I||={max_orth_U_torch:.3e}, max ||V'V - I||={max_orth_V_torch:.3e}")

    print()
    print("="*80)


def main():
    """Run comprehensive SVD comparison tests."""

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for this comparison.")
        print("Install with: pip install torch")
        return 1

    print("="*70)
    print("LOW-RANK SVD COMPARISON")
    print("librla svd_sketch vs torch.svd_lowrank")
    print("="*70)

    print("\nEnvironment:")
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  NumPy:      {np.__version__}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  Threads:    {torch.get_num_threads()} (set via torch.set_num_threads for MKL compatibility)")
    print(f"  Precision:  {PRECISION} ({DTYPE.__name__})")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA:       {'Available (' + torch.cuda.get_device_name(0) + ')' if cuda_available else 'Not available'}")
    print(f"  Mode:       {'CUDA' if args.cuda else 'CPU'}")

    print("\nThread configuration details:")
    print(torch.__config__.parallel_info())

    # Check CUDA availability if --cuda flag is set
    if args.cuda and not cuda_available:
        print("\nERROR: --cuda flag specified but CUDA is not available.")
        return 1

    print("\nComparison settings:")
    print(f"  - power_iter={args.power_iter} for both (torch defaults to niter=2, librla to 0)")
    print(f"  - extra_samples={args.extra_samples} (librla default: 12)")
    print(f"  - verbose={args.verbose}")
    print("  - torch q = rank + extra_samples to match librla's oversampling")
    print("="*70)

    all_results = []
    power_iter = args.power_iter
    extra_samples = args.extra_samples

    if args.cuda:
        # Run CUDA tests only
        gpu_results = run_test_suite(device='cuda', power_iter=power_iter, extra_samples=extra_samples)
        all_results.extend(gpu_results)
        print_summary(gpu_results, 'cuda', power_iter)
    else:
        # Run CPU tests only
        cpu_results = run_test_suite(device='cpu', power_iter=power_iter, extra_samples=extra_samples)
        all_results.extend(cpu_results)
        print_summary(cpu_results, 'cpu', power_iter)

    # Overall summary
    print("\n\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    passed_tests = sum(1 for r in all_results if r.passed)
    total_tests = len(all_results)
    pass_rate = 100.0 * passed_tests / total_tests

    print(f"\nTotal Pass Rate: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")

    if passed_tests == total_tests:
        print("[PASS] All tests PASSED")
        return 0
    else:
        print("[WARNING] Some tests FAILED")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
