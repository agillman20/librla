#!/usr/bin/env python3
"""
compare_all.py - Run all comparison tests

Unified runner for comparing librla against external libraries:
- compare_id_scipy.py: librla.id_sketch vs scipy.linalg.interpolative.interp_decomp
- compare_svd_torch.py: librla.svd_sketch vs torch.svd_lowrank

Usage:
    python compare_all.py [--precision {double,single}]

Options:
    --precision    Floating-point precision: double (default) or single

Requires:
    - NumPy, SciPy, PyTorch
    - librla.py, make_mat.py from ../python/

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import argparse
import sys
import os
import time
import subprocess

# Parse arguments
parser = argparse.ArgumentParser(description='Run all librla comparison tests')
parser.add_argument('--precision', choices=['double', 'single'], default='double',
                    help='Floating-point precision (default: double)')
args = parser.parse_args()

PRECISION = args.precision

# Add parent python directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


def run_module(module_path, module_name, precision='double'):
    """
    Run a comparison module and capture its exit code.

    Parameters
    ----------
    module_path : str
        Path to the Python file
    module_name : str
        Display name for the module
    precision : str
        Floating-point precision ('double' or 'single')

    Returns
    -------
    exit_code : int
        0 if passed, non-zero if failed
    elapsed : float
        Time taken in seconds
    """
    print("\n" + "#"*80)
    print(f"# RUNNING: {module_name}")
    print("#"*80 + "\n")

    t0 = time.perf_counter()

    # Run the module as a subprocess with precision argument
    try:
        result = subprocess.run(
            [sys.executable, module_path, '--precision', precision],
            check=False
        )
        exit_code = result.returncode
    except Exception as e:
        print(f"\n[ERROR] {module_name} failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        exit_code = 1

    elapsed = time.perf_counter() - t0

    return exit_code, elapsed


def main():
    """Run all comparison tests."""

    print("="*80)
    print("LIBRLA COMPARISON SUITE")
    print("Comparing librla against external libraries")
    print("="*80)

    print("\nEnvironment:")
    print(f"  Python: {sys.version.split()[0]}")

    import numpy as np
    print(f"  NumPy:  {np.__version__}")

    import scipy
    print(f"  SciPy:  {scipy.__version__}")
    print(f"  Precision: {PRECISION}")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  CUDA:    Available ({torch.cuda.get_device_name(0)})")
        else:
            print(f"  CUDA:    Not available")
    except ImportError:
        print(f"  PyTorch: Not installed (SVD comparison will be skipped)")
        torch = None

    print("="*80)

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define modules to run
    modules = [
        ('compare_id_scipy.py', 'ID Comparison (librla vs scipy)'),
    ]

    # Only add torch comparison if torch is available
    if torch is not None:
        modules.append(('compare_svd_torch.py', 'SVD Comparison (librla vs torch)'))
    else:
        print("\n[WARNING] Skipping SVD comparison (PyTorch not installed)")

    # Run each module
    results = []
    total_start = time.perf_counter()

    for filename, display_name in modules:
        module_path = os.path.join(script_dir, filename)

        if not os.path.exists(module_path):
            print(f"\n[ERROR] Module not found: {module_path}")
            results.append((display_name, 1, 0.0, 'NOT FOUND'))
            continue

        exit_code, elapsed = run_module(module_path, display_name, PRECISION)

        if exit_code == 0:
            status = 'PASS'
        else:
            status = 'FAIL'

        results.append((display_name, exit_code, elapsed, status))

    total_elapsed = time.perf_counter() - total_start

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print()
    print(f"{'Module':<45s} {'Status':<12s} {'Time (s)':<12s}")
    print("-"*80)

    all_passed = True
    for display_name, exit_code, elapsed, status in results:
        status_str = f"[{status}]"
        print(f"{display_name:<45s} {status_str:<12s} {elapsed:>8.2f}s")
        if exit_code != 0:
            all_passed = False

    print("-"*80)
    print(f"{'Total':<45s} {'':<12s} {total_elapsed:>8.2f}s")

    print()
    if all_passed:
        print("[PASS] All comparison tests PASSED")
        return 0
    else:
        print("[FAIL] Some comparison tests FAILED")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
