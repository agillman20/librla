#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m venv local
source local/bin/activate
pip install numpy scipy matplotlib pillow requests
