"""
download_sst_data.py - Download NOAA ERSST v5 Sea Surface Temperature data.

Description
-----------
Downloads monthly SST data from NOAA's Extended Reconstructed Sea Surface
Temperature (ERSST) Version 5 dataset.

Data source: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/
Product info: https://www.ncei.noaa.gov/products/extended-reconstructed-sst

ERSST v5 specifications:
  - Grid: 2° × 2° (89 lat × 180 lon)
  - Coverage: 1854-present
  - Variables: SST (sea surface temperature)
  - Format: NetCDF (CF conventions)

Citation:
  Huang et al. (2017): NOAA Extended Reconstructed Sea Surface Temperature
  (ERSST), Version 5. doi:10.7289/V5T72FNM

Requirements
------------
Install required packages:
  pip install requests

For analysis scripts (test_sst_*.py), also install:
  pip install numpy scipy netCDF4 matplotlib

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
"""

import os
import requests
from pathlib import Path

BASE_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/"
DATA_DIR = Path(__file__).parent / "data"


def download_ersst_year(year):
    """Download all monthly ERSST files for a given year.

    Files are named: ersst.v5.YYYYMM.nc
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    files_downloaded = []

    for month in range(1, 13):
        filename = f"ersst.v5.{year:04d}{month:02d}.nc"
        url = BASE_URL + filename
        outpath = DATA_DIR / filename

        if outpath.exists():
            print(f"  {filename} already exists, skipping")
            files_downloaded.append(str(outpath))
            continue

        print(f"  Downloading {filename}...", end="", flush=True)
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(outpath, 'wb') as f:
                f.write(response.content)
            print(" done")
            files_downloaded.append(str(outpath))
        except Exception as e:
            print(f" failed ({type(e).__name__})")
            # File might not exist yet for recent months

    return files_downloaded


def download_ersst_range(start_year, end_year):
    """Download ERSST data for a range of years."""
    print("=" * 60)
    print("Downloading NOAA ERSST v5 data")
    print(f"  Years: {start_year} - {end_year}")
    print(f"  Source: {BASE_URL}")
    print("=" * 60)

    all_files = []

    for year in range(start_year, end_year + 1):
        print(f"\nYear {year}:")
        files = download_ersst_year(year)
        all_files.extend(files)

    print("\n" + "=" * 60)
    print(f"Download complete: {len(all_files)} files")
    print("=" * 60)

    return all_files


def list_local_files():
    """List all downloaded ERSST files."""
    if not DATA_DIR.is_dir():
        return []
    files = sorted([str(f) for f in DATA_DIR.glob("*.nc")])
    return files


if __name__ == "__main__":
    print("=" * 60)
    print("NOAA ERSST v5 Data Downloader")
    print("=" * 60)
    print("""
    This script downloads monthly Sea Surface Temperature data.

    Grid: 2° × 2° (89 lat × 180 lon = 16,020 ocean points)
    Time: Monthly, 1854-present (~2000 months)

    Each file is ~200KB, full dataset ~400MB.
    """)

    # Download recent decades for demo (1980-2023)
    # For full analysis, use 1854-present
    print("\nDownloading 1980-2023 (44 years, ~530 files)...")
    print("This will take a few minutes...\n")

    files = download_ersst_range(1980, 2023)

    print("\nTo download more years, run:")
    print("  download_ersst_range(1854, 1979)")

    # Show what we have
    local_files = list_local_files()
    if local_files:
        print(f"\nLocal files: {len(local_files)}")
        print(f"  First: {os.path.basename(local_files[0])}")
        print(f"  Last:  {os.path.basename(local_files[-1])}")
