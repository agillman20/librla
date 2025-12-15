#!/usr/bin/env python3
"""
test_all.py - Master test script for librla

Runs all tests and produces a unified summary:
- test_id.py    (ID implementations)
- test_svd.py   (SVD implementations)
- test_qr.py    (QR implementations)
- test_orth.py  (Orth implementations)

Usage:
    python test_all.py

Requires:
    - NumPy, SciPy
    - librla.py, make_mat.py in Python path
    - test_id.py, test_svd.py, test_qr.py, test_orth.py

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
"""

import sys
import time
import importlib.util
import os


def run_comparison_module(module_name, module_path):
    """
    Run a comparison module and return (passed, total, elapsed_time).

    Parameters
    ----------
    module_name : str
        Name of the module (for display)
    module_path : str
        Path to the module file

    Returns
    -------
    passed : int
        Number of tests passed
    total : int
        Total number of tests
    elapsed : float
        Time elapsed in seconds
    success : bool
        Whether the module ran without errors
    """
    print(f"\n{'='*70}")
    print(f"Running {module_name}...")
    print("="*70)

    try:
        # Load and run the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)

        t0 = time.perf_counter()
        spec.loader.exec_module(module)

        # Run main() and capture exit code
        if hasattr(module, 'main'):
            exit_code = module.main()
        else:
            print(f"[ERROR] {module_name} has no main() function")
            return 0, 0, 0.0, False

        elapsed = time.perf_counter() - t0

        # Count results from the module's global state (if accessible)
        # Since we can't easily get this, we'll parse from exit code
        # exit_code 0 = all passed, 1 = some failed

        # For now, we'll re-run and count
        # This is a simplified approach - in production, modules should return counts
        return exit_code, elapsed, True

    except Exception as e:
        print(f"\n[ERROR] Failed to run {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return 1, 0.0, False


def run_module_with_count(module_path, module_display_name):
    """
    Import and run a comparison module, counting passed/total tests.

    Returns (passed, total, elapsed, success)
    """
    print(f"\n{'='*70}")
    print(f"Running {module_display_name}...")
    print("="*70)

    try:
        # Change to the directory containing the module
        module_dir = os.path.dirname(os.path.abspath(module_path))
        original_dir = os.getcwd()

        # Add module directory to path
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        # Load the module
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)

        t0 = time.perf_counter()
        spec.loader.exec_module(module)

        # The module defines main() which we need to run
        # We'll capture the results by running main() which prints and returns exit code
        if hasattr(module, 'main'):
            exit_code = module.main()
            elapsed = time.perf_counter() - t0

            # We can't easily get passed/total from the module
            # So we'll use a convention: modules return 0 if all passed, 1 otherwise
            # The actual counts are printed by the module itself
            success = (exit_code == 0)
            return exit_code, elapsed, success
        else:
            print(f"[ERROR] {module_display_name} has no main() function")
            return 1, 0.0, False

    except Exception as e:
        print(f"\n[ERROR] Failed to run {module_display_name}: {e}")
        import traceback
        traceback.print_exc()
        return 1, 0.0, False


def main():
    """Run all tests and produce summary."""

    print()
    print("="*70)
    print("LIBRLA TEST SUITE - Python")
    print("="*70)
    print()
    print("This script runs all tests for librla functions:")
    print("  - test_id.py    (id_sketch, id_qrpiv)")
    print("  - test_svd.py   (svd_sketch)")
    print("  - test_qr.py    (qr_sketch)")
    print("  - test_orth.py  (orth_sketch)")
    print()
    print(f"Python:     {sys.version.split()[0]}")
    print("="*70)

    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # List of modules to run
    modules = [
        ('test_id.py', 'test_id'),
        ('test_svd.py', 'test_svd'),
        ('test_qr.py', 'test_qr'),
        ('test_orth.py', 'test_orth'),
    ]

    # Results tracking
    results = []
    total_elapsed = 0.0

    for filename, display_name in modules:
        module_path = os.path.join(script_dir, filename)

        if not os.path.exists(module_path):
            print(f"\n[WARNING] {filename} not found at {module_path}")
            results.append((display_name, None, None, False, "not found"))
            continue

        exit_code, elapsed, success = run_module_with_count(module_path, display_name)
        total_elapsed += elapsed

        if success:
            status = "PASSED" if exit_code == 0 else "FAILED"
        else:
            status = "ERROR"

        results.append((display_name, exit_code, elapsed, success, status))

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n")
    print("="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print()

    # Per-module summary
    print(f"{'Module':<20s} {'Status':<12s} {'Time (s)':<12s}")
    print("-"*50)

    all_passed = True
    for display_name, exit_code, elapsed, success, status in results:
        if elapsed is not None:
            print(f"{display_name:<20s} [{status:<6s}]    {elapsed:>8.2f}s")
        else:
            print(f"{display_name:<20s} [{status:<6s}]    {'N/A':>8s}")

        if status != "PASSED":
            all_passed = False

    print("-"*50)
    print(f"{'Total':<20s} {'':<12s} {total_elapsed:>8.2f}s")

    # Overall status
    print()
    print("="*70)

    modules_passed = sum(1 for _, _, _, success, status in results if status == "PASSED")
    modules_total = len(results)

    if all_passed:
        print(f"[PASS] All {modules_total} test modules passed!")
        print("="*70)
        return 0
    else:
        print(f"[FAIL] {modules_passed}/{modules_total} test modules passed")
        print()
        print("Failed modules:")
        for display_name, exit_code, elapsed, success, status in results:
            if status != "PASSED":
                print(f"  - {display_name}: {status}")
        print("="*70)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
