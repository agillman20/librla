#!/bin/bash
# setup_parla.sh - Install PARLA for comparison tests
#
# PARLA (Python Algorithms for Randomized Linear Algebra) from BallisticLA
# https://github.com/BallisticLA/parla
#
# Note: PARLA requires patching for NumPy 2.0 compatibility (np.NaN -> np.nan)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up PARLA ==="

# Clone if not exists
if [ ! -d "parla" ]; then
    echo "Cloning PARLA repository..."
    git clone https://github.com/BallisticLA/parla.git
else
    echo "PARLA directory already exists"
fi

# Patch for NumPy 2.0 compatibility
# PARLA uses np.NaN which was removed in NumPy 2.0
echo "Patching PARLA for NumPy 2.0 compatibility (np.NaN -> np.nan)..."
find parla -name "*.py" -exec sed -i '' 's/np\.NaN/np.nan/g' {} \; 2>/dev/null || \
find parla -name "*.py" -exec sed -i 's/np\.NaN/np.nan/g' {} \;

# Install in the compare venv
if [ -d "venv" ]; then
    echo "Installing PARLA in compare venv..."
    cd parla
    ../venv/bin/pip install -e .
    cd ..
else
    echo "WARNING: venv not found. Run setup_venv.sh first, then run this script again."
    exit 1
fi

echo ""
echo "=== PARLA installation complete ==="
echo ""
echo "PARLA is now available. See README.md for usage."
