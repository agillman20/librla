#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install numpy scipy netCDF4 matplotlib requests pillow
