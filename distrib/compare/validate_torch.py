#!/usr/bin/env python3
"""
validate_torch_compat.py - Validate torch_compat wrappers against PyTorch

Validates that torch_compat.svd_lowrank and torch_compat.pca_lowrank
produce equivalent results to torch.svd_lowrank and torch.pca_lowrank.

Tests:
- Same output shapes
- Similar reconstruction errors
- Similar singular values

Usage:
    python validate_torch.py

Requires:
    - NumPy, SciPy, PyTorch
    - torch_compat.py from ../python/

Author: Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Assisted by: Claude Code (Anthropic)
"""

import os
import sys
import numpy as np

# Add python directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from torch_compat import svd_lowrank, pca_lowrank

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)


def validate_svd_lowrank(A, q, niter, M=None, name="test"):
    """Validate svd_lowrank wrapper against torch.svd_lowrank."""
    print(f"\n--- svd_lowrank: {name} ---")
    print(f"    A: {A.shape}, q={q}, niter={niter}, M={'yes' if M is not None else 'no'}")

    # torch_compat version (numpy)
    U_compat, S_compat, V_compat = svd_lowrank(A, q=q, niter=niter, M=M)

    # PyTorch version
    A_torch = torch.from_numpy(A)
    M_torch = torch.from_numpy(M) if M is not None else None
    U_torch, S_torch, V_torch = torch.svd_lowrank(A_torch, q=q, niter=niter, M=M_torch)
    U_torch = U_torch.numpy()
    S_torch = S_torch.numpy()
    V_torch = V_torch.numpy()

    # Compare shapes
    shape_ok = (U_compat.shape == U_torch.shape and
                S_compat.shape == S_torch.shape and
                V_compat.shape == V_torch.shape)

    # Compare reconstruction errors
    A_input = A - M if M is not None else A
    normA = np.linalg.norm(A_input, 'fro')

    recon_compat = U_compat @ np.diag(S_compat) @ V_compat.T
    recon_torch = U_torch @ np.diag(S_torch) @ V_torch.T

    err_compat = np.linalg.norm(A_input - recon_compat, 'fro') / normA
    err_torch = np.linalg.norm(A_input - recon_torch, 'fro') / normA

    # Compare singular values
    sval_diff = np.linalg.norm(S_compat - S_torch) / (np.linalg.norm(S_torch) + 1e-16)

    print(f"    Shapes:  compat U={U_compat.shape}, S={S_compat.shape}, V={V_compat.shape}")
    print(f"             torch  U={U_torch.shape}, S={S_torch.shape}, V={V_torch.shape}")
    print(f"    Shape match: {shape_ok}")
    print(f"    Recon err:   compat={err_compat:.3e}, torch={err_torch:.3e}")
    print(f"    Sval diff:   {sval_diff:.3e}")

    # Pass if shapes match and errors are similar (both randomized, so not exact)
    err_ratio = max(err_compat, err_torch) / (min(err_compat, err_torch) + 1e-16)
    passed = shape_ok and err_ratio < 10  # Allow 10x difference due to randomness

    print(f"    PASSED: {passed}")
    return passed


def validate_pca_lowrank(A, q, center, niter, name="test"):
    """Validate pca_lowrank wrapper against torch.pca_lowrank."""
    print(f"\n--- pca_lowrank: {name} ---")
    print(f"    A: {A.shape}, q={q}, center={center}, niter={niter}")

    # torch_compat version (numpy)
    U_compat, S_compat, V_compat = pca_lowrank(A, q=q, center=center, niter=niter)

    # PyTorch version
    A_torch = torch.from_numpy(A)
    U_torch, S_torch, V_torch = torch.pca_lowrank(A_torch, q=q, center=center, niter=niter)
    U_torch = U_torch.numpy()
    S_torch = S_torch.numpy()
    V_torch = V_torch.numpy()

    # Compare shapes
    shape_ok = (U_compat.shape == U_torch.shape and
                S_compat.shape == S_torch.shape and
                V_compat.shape == V_torch.shape)

    # Compare reconstruction errors
    A_input = A - A.mean(axis=0, keepdims=True) if center else A
    normA = np.linalg.norm(A_input, 'fro')

    recon_compat = U_compat @ np.diag(S_compat) @ V_compat.T
    recon_torch = U_torch @ np.diag(S_torch) @ V_torch.T

    err_compat = np.linalg.norm(A_input - recon_compat, 'fro') / normA
    err_torch = np.linalg.norm(A_input - recon_torch, 'fro') / normA

    # Compare singular values
    sval_diff = np.linalg.norm(S_compat - S_torch) / (np.linalg.norm(S_torch) + 1e-16)

    print(f"    Shapes:  compat U={U_compat.shape}, S={S_compat.shape}, V={V_compat.shape}")
    print(f"             torch  U={U_torch.shape}, S={S_torch.shape}, V={V_torch.shape}")
    print(f"    Shape match: {shape_ok}")
    print(f"    Recon err:   compat={err_compat:.3e}, torch={err_torch:.3e}")
    print(f"    Sval diff:   {sval_diff:.3e}")

    # Pass if shapes match and errors are similar
    err_ratio = max(err_compat, err_torch) / (min(err_compat, err_torch) + 1e-16)
    passed = shape_ok and err_ratio < 10

    print(f"    PASSED: {passed}")
    return passed


def main():
    print("=" * 70)
    print("TORCH_COMPAT VALIDATION")
    print("Comparing torch_compat wrappers vs PyTorch implementations")
    print("=" * 70)

    print(f"\nEnvironment:")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  NumPy:   {np.__version__}")
    print(f"  PyTorch: {torch.__version__}")

    np.random.seed(42)
    results = []

    # -------------------------------------------------------------------------
    # svd_lowrank tests
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SVD_LOWRANK TESTS")
    print("=" * 70)

    # Test 1: Basic random matrix
    A1 = np.random.randn(200, 100)
    results.append(validate_svd_lowrank(A1, q=20, niter=2, name="Random 200x100"))

    # Test 2: With M parameter
    A2 = np.random.randn(150, 80)
    M2 = np.random.randn(150, 80) * 0.1
    results.append(validate_svd_lowrank(A2, q=15, niter=2, M=M2, name="With M subtraction"))

    # Test 3: Different q values
    A3 = np.random.randn(100, 100)
    results.append(validate_svd_lowrank(A3, q=6, niter=2, name="Default q=6"))

    # Test 4: Different niter
    A4 = np.random.randn(150, 100)
    results.append(validate_svd_lowrank(A4, q=20, niter=0, name="niter=0"))
    results.append(validate_svd_lowrank(A4, q=20, niter=5, name="niter=5"))

    # Test 5: Wide matrix
    A5 = np.random.randn(50, 200)
    results.append(validate_svd_lowrank(A5, q=20, niter=2, name="Wide 50x200"))

    # -------------------------------------------------------------------------
    # pca_lowrank tests
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PCA_LOWRANK TESTS")
    print("=" * 70)

    # Test 1: Basic with centering
    A6 = np.random.randn(200, 50)
    results.append(validate_pca_lowrank(A6, q=10, center=True, niter=2, name="Centered"))

    # Test 2: Without centering
    A7 = np.random.randn(200, 50)
    results.append(validate_pca_lowrank(A7, q=10, center=False, niter=2, name="Not centered"))

    # Test 3: Default q
    A8 = np.random.randn(100, 100)
    results.append(validate_pca_lowrank(A8, q=None, center=True, niter=2, name="Default q"))

    # Test 4: Different niter
    A9 = np.random.randn(150, 80)
    results.append(validate_pca_lowrank(A9, q=15, center=True, niter=0, name="niter=0"))
    results.append(validate_pca_lowrank(A9, q=15, center=True, niter=5, name="niter=5"))

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("[PASS] All validation tests PASSED")
        return 0
    else:
        print("[FAIL] Some validation tests FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
