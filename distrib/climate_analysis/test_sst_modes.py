"""
test_sst_modes.py - Modal decomposition of raw NOAA SST data using SVD.

Description
-----------
Applies SVD/EOF analysis to raw Sea Surface Temperature data (no anomaly
preprocessing). The dominant modes capture:

  1. EOF1: Mean spatial pattern (warm tropics, cold poles)
  2. EOF2-4: Seasonal variations
  3. EOF5: ENSO (El Niño-Southern Oscillation)

This demonstrates randomized SVD (svd_sketch) on real climate data.

Data: Raw SST values in °C (no anomaly computation, no detrending)

EOF (Empirical Orthogonal Functions) = PCA = POD = SVD

Prerequisites
-------------
Run download_sst_data.py first:
  python download_sst_data.py

Requirements
------------
* librla.py in the path
* pip install numpy scipy netCDF4 matplotlib

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
"""

import sys
import os
import re
import time
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.signal import detrend
import netCDF4 as nc
import matplotlib.pyplot as plt

# Add parent directory to path for librla
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
import librla
from test_sst_utils import north_test, print_north_test

# ========================================================================
# Load SST data
# ========================================================================

print("=" * 70)
print("NOAA SST Climate Mode Analysis (EOF/SVD)")
print("=" * 70)

DATA_DIR = Path(__file__).parent / "data"

# Find all downloaded files
nc_files = sorted(DATA_DIR.glob("ersst.v5.*.nc"))

if not nc_files:
    raise FileNotFoundError(f"""
    No data files found in {DATA_DIR}

    Please run download_sst_data.py first:
      python download_sst_data.py
    """)

print(f"\nFound {len(nc_files)} monthly SST files")
print(f"  First: {nc_files[0].name}")
print(f"  Last:  {nc_files[-1].name}")

# Load first file to get grid info
ds = nc.Dataset(nc_files[0])
lon = ds.variables['lon'][:]
lat = ds.variables['lat'][:]
ds.close()

n_lon = len(lon)
n_lat = len(lat)
n_time = len(nc_files)

print(f"\nGrid dimensions:")
print(f"  Longitude: {n_lon} points ({lon[0]}° to {lon[-1]}°)")
print(f"  Latitude:  {n_lat} points ({lat[0]}° to {lat[-1]}°)")
print(f"  Time:      {n_time} months")

# Load all SST data
print("\nLoading SST data...")
SST = np.zeros((n_lon, n_lat, n_time), dtype=np.float32)
dates = []

for i, f in enumerate(nc_files):
    ds = nc.Dataset(f)
    sst_raw = ds.variables['sst'][0, 0, :, :]  # time × lev × lat × lon -> lat × lon
    # NetCDF returns masked array - convert to NaN
    SST[:, :, i] = np.ma.filled(sst_raw.T, np.nan)  # Transpose to lon × lat

    # Extract date from filename (ersst.v5.YYYYMM.nc)
    m = re.match(r"ersst\.v5\.(\d{4})(\d{2})\.nc", f.name)
    if m:
        dates.append(datetime(int(m.group(1)), int(m.group(2)), 1))
    ds.close()

    if (i + 1) % 100 == 0:
        print(f"  Loaded {i + 1} / {n_time}\r", end="", flush=True)

print(f"  Loaded {n_time} months              ")

# Handle missing values (land = NaN in ERSST)
SST[SST < -900] = np.nan  # Missing value flag

# Center on Greenwich meridian: convert 0-360 to -180 to 180
print("\nCentering on Greenwich meridian...")
shift_idx = np.argmax(lon >= 180)
lon = np.concatenate([lon[shift_idx:] - 360, lon[:shift_idx]])
SST = np.concatenate([SST[shift_idx:, :, :], SST[:shift_idx, :, :]], axis=0)

print(f"  Longitude: {lon[0]}° to {lon[-1]}°")
valid_sst = SST[~np.isnan(SST)]
print(f"\nSST range: {np.min(valid_sst):.1f}°C to {np.max(valid_sst):.1f}°C")

# ========================================================================
# Create ocean mask and reshape for SVD
# ========================================================================

print("\n" + "=" * 70)
print("Preparing data matrix...")
print("=" * 70)

# Ocean mask (where we have valid data for all times)
ocean_mask = ~np.any(np.isnan(SST), axis=2)
n_ocean = np.sum(ocean_mask)

print(f"\n  Ocean points: {n_ocean} / {n_lon * n_lat} ({100*n_ocean/(n_lon*n_lat):.1f}%)")
print(f"  Area weighting: None (unweighted EOF analysis)")

# Reshape to 2D matrix: ocean_points × time
# Each column is one month's SST field (ocean points only)
SST_matrix = np.zeros((n_ocean, n_time), dtype=np.float32)

for t in range(n_time):
    SST_matrix[:, t] = SST[:, :, t][ocean_mask]

print(f"  Data matrix: {n_ocean} × {n_time}")

# ========================================================================
# EOF Analysis via SVD
# ========================================================================

print("\n" + "=" * 70)
print("EOF Analysis (SVD)")
print("=" * 70)

# Number of modes to compute
n_modes = 30

print(f"\nComputing {n_modes}-mode SVD using svd_sketch (extra_samples=10, power_iter=2)...")
t0 = time.time()
U, s, Vh = librla.svd_sketch(SST_matrix, n_modes, power_iter=2, extra_samples=10)
V = Vh.T
print(f"  Elapsed time: {time.time() - t0:.4f}s")

# Additional SVD configurations for paper figure comparison
print("\nComputing additional SVD configurations...")
t0 = time.time()
U_a, s_a, Vh_a = librla.svd_sketch(SST_matrix, n_modes, extra_samples=0)
print(f"  (a) Basic (extra_samples=0, power_iter=0): {time.time() - t0:.4f}s")

t0 = time.time()
U_b, s_b, Vh_b = librla.svd_sketch(SST_matrix, n_modes, extra_samples=10)
print(f"  (b) extra_samples=10, power_iter=0: {time.time() - t0:.4f}s")

t0 = time.time()
U_c, s_c, Vh_c = librla.svd_sketch(SST_matrix, n_modes, extra_samples=0, power_iter=2)
print(f"  (c) extra_samples=0, power_iter=2: {time.time() - t0:.4f}s")

# (d) is the existing computation: extra_samples=10, power_iter=2
s_d = s

# Also compute reference SVD
print("\nComputing reference SVD...")
t0 = time.time()
U_full, s_full, Vh_full = np.linalg.svd(SST_matrix, full_matrices=False)
V_full = Vh_full.T
print(f"  Elapsed time: {time.time() - t0:.2f}s")

# Variance explained
total_var = np.sum(s_full**2)
var_explained = s_full**2 / total_var
cumulative_var = np.cumsum(var_explained)

print("\nEOF variance explained:")
print("  Mode    Variance %   Cumulative %")
print("  " + "-" * 40)
for i in range(min(10, n_modes)):
    print(f"  EOF{i+1:2d}   {100*var_explained[i]:6.2f}%      {100*cumulative_var[i]:6.2f}%")

n90 = np.argmax(cumulative_var >= 0.90) + 1
n95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\n  Modes for 90%: {n90}, 95%: {n95}")

# North's rule of thumb for mode separability
print_north_test(s_full, n_samples=n_time, n_modes=10)

# ========================================================================
# Reshape EOFs back to spatial maps
# ========================================================================

def reshape_to_map(vec, mask):
    """Reshape ocean vector back to lon×lat map with NaN for land."""
    field = np.full((len(lon), len(lat)), np.nan, dtype=np.float32)
    field[mask] = vec
    return field


# Compute Niño 3.4 index for ENSO validation (not for sign fixing)
# Niño 3.4 region: 5°N-5°S, 170°W-120°W (-170° to -120° in -181.0.0 system)
nino34_lon_idx = np.where((lon >= -170) & (lon <= -120))[0]
nino34_lat_idx = np.where((lat >= -5) & (lat <= 5))[0]

nino34_index = np.zeros(n_time)
for t in range(n_time):
    region = SST[np.ix_(nino34_lon_idx, nino34_lat_idx, [t])][:, :, 0]
    nino34_index[t] = np.nanmean(region)


def fix_svd_signs(U_modes, V_modes):
    """Fix signs based on spatial pattern mean.

    Convention: spatial pattern should have positive mean over ocean.
    This ensures positive PC = positive anomaly contribution.
    """
    U_modes = U_modes.copy()
    V_modes = V_modes.copy()
    for i in range(U_modes.shape[1]):
        if np.mean(U_modes[:, i]) < 0:
            U_modes[:, i] *= -1
            V_modes[:, i] *= -1
    return U_modes, V_modes


# Deterministic SVD
U_det, V_det = fix_svd_signs(U_full[:, :n_modes], V_full[:, :n_modes])

# Randomized SVD
U_rand, V_rand = fix_svd_signs(U, V)

# Spatial: U (non-dimensional pattern)
# Temporal: projection of data onto U, normalized by √n_ocean for °C units
EOF_maps_det = [reshape_to_map(U_det[:, i], ocean_mask) for i in range(n_modes)]
PC_det = [(U_det[:, i] @ SST_matrix) / np.sqrt(n_ocean) for i in range(n_modes)]

EOF_maps_rand = [reshape_to_map(U_rand[:, i], ocean_mask) for i in range(n_modes)]
PC_rand = [(U_rand[:, i] @ SST_matrix) / np.sqrt(n_ocean) for i in range(n_modes)]

# ========================================================================
# Visualize Deterministic SVD results
# ========================================================================

print("\n" + "=" * 70)
print("Visualizing EOF patterns...")
print("=" * 70)

# Convert dates to decimal years for plotting
years = np.array([d.year + (d.month - 1) / 12 for d in dates])

# Mode labels
mode_names = ["EOF1", "EOF2", "EOF3", "EOF4", "EOF5"]

fig1, axes1 = plt.subplots(5, 2, figsize=(12, 14))
fig1.suptitle("Deterministic SVD", fontsize=14, fontweight='bold')

for i in range(5):
    # Left: Spatial EOF pattern
    ax_spatial = axes1[i, 0]
    clim = np.nanmax(np.abs(EOF_maps_det[i])) * 0.8
    im = ax_spatial.pcolormesh(lon, lat, EOF_maps_det[i].T, cmap='RdBu_r',
                                vmin=-clim, vmax=clim, shading='auto')
    ax_spatial.set_xlabel("Longitude")
    ax_spatial.set_ylabel("Latitude")
    ax_spatial.set_title(f"{mode_names[i]} ({100*var_explained[i]:.1f}%)")

    # Right: Temporal PC coefficient
    ax_temporal = axes1[i, 1]
    ax_temporal.plot(years, PC_det[i], linewidth=0.8, color='steelblue')
    ax_temporal.set_xlabel("Year")
    ax_temporal.set_ylabel(f"PC{i+1}")
    if i > 0:  # Only show zero line for oscillating modes
        ax_temporal.axhline(0, color='gray', linestyle='--', linewidth=0.5)

fig1.subplots_adjust(top=0.95, bottom=0.05, hspace=0.35, wspace=0.25)

# ========================================================================
# Visualize Randomized SVD results
# ========================================================================

var_explained_rand = s**2 / total_var

fig2, axes2 = plt.subplots(5, 2, figsize=(12, 14))
fig2.suptitle("Randomized SVD", fontsize=14, fontweight='bold')

for i in range(5):
    # Left: Spatial EOF pattern
    ax_spatial = axes2[i, 0]
    clim = np.nanmax(np.abs(EOF_maps_rand[i])) * 0.8
    im = ax_spatial.pcolormesh(lon, lat, EOF_maps_rand[i].T, cmap='RdBu_r',
                                vmin=-clim, vmax=clim, shading='auto')
    ax_spatial.set_xlabel("Longitude")
    ax_spatial.set_ylabel("Latitude")
    ax_spatial.set_title(f"{mode_names[i]} ({100*var_explained_rand[i]:.1f}%)")

    # Right: Temporal PC coefficient
    ax_temporal = axes2[i, 1]
    ax_temporal.plot(years, PC_rand[i], linewidth=0.8, color='steelblue')
    ax_temporal.set_xlabel("Year")
    ax_temporal.set_ylabel(f"PC{i+1}")
    if i > 0:
        ax_temporal.axhline(0, color='gray', linestyle='--', linewidth=0.5)

fig2.subplots_adjust(top=0.95, bottom=0.05, hspace=0.35, wspace=0.25)

# ========================================================================
# Singular value spectrum and variance comparison
# ========================================================================

fig3, ax3 = plt.subplots(figsize=(8, 4))

# Singular value spectrum
n_plot = min(50, len(s_full))
ax3.semilogy(range(1, n_plot + 1), s_full[:n_plot],
             'o', markersize=4, label="Deterministic")
ax3.semilogy(range(1, n_modes + 1), s, 'x', markersize=8, color='red',
             label="Randomized")
ax3.set_xlabel("Mode")
ax3.set_ylabel("σᵢ")
ax3.set_title("Singular Value Spectrum")
ax3.legend()

fig3.tight_layout()

# ========================================================================
# ENSO region analysis
# ========================================================================

print("\n" + "=" * 70)
print("ENSO Analysis")
print("=" * 70)

# Niño 3.4 index was computed earlier for sign determination
# Correlation without detrending
corr_pc5_nino_raw = pearsonr(PC_det[4], nino34_index)[0]
print(f"\nCorrelation between PC5 and Niño 3.4 index: {corr_pc5_nino_raw:.3f}")

# Detrend to remove warming trend and reveal ENSO signal
pc5_detrend = detrend(PC_det[4])
nino34_detrend = detrend(nino34_index)

# Correlation between detrended PC5 (ENSO mode) and Niño 3.4
corr_pc5_nino = pearsonr(pc5_detrend, nino34_detrend)[0]
print(f"Correlation (detrended): {corr_pc5_nino:.3f}")

# Plot comparison (detrended and normalized)
fig4, ax4 = plt.subplots(figsize=(10, 3))
ax4.plot(years, pc5_detrend / np.std(pc5_detrend), linewidth=1.5, label="PC5 (det)", color='blue')
ax4.plot(years, nino34_detrend / np.std(nino34_detrend), linewidth=1.5,
         label="Niño 3.4", color='red', linestyle='--')
ax4.axhline(0, color='gray', linestyle=':', linewidth=0.5)
ax4.set_xlabel("Year")
ax4.set_ylabel("Index (detrended, normalized)")
ax4.set_title(f"PC5 (ENSO) vs Niño 3.4 Index (r = {corr_pc5_nino:.2f}, detrended)")
ax4.legend()

fig4.tight_layout()

# ========================================================================
# Summary
# ========================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"  Data: NOAA ERSST v5")
print(f"  Grid: {n_lon} × {n_lat} (2° resolution)")
print(f"  Time: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')} ({n_time} months)")
print(f"  Ocean points: {n_ocean}")
print()
print("  EOF Analysis:")
print(f"    EOF1 variance: {100*var_explained[0]:.1f}% (mean pattern)")
print(f"    EOF5 variance: {100*var_explained[4]:.1f}% (ENSO)")
print(f"    Modes for 90% variance: {n90}")
separable = north_test(s_full, n_samples=n_time)
n_well_separated = np.sum(separable[:10])
print(f"    Well-separated modes (North's rule): {n_well_separated}/10")
print()
print("  Randomized SVD accuracy:")
rel_err = norm(s - s_full[:n_modes]) / norm(s_full[:n_modes])
print(f"    Singular value error: {100*rel_err:.3f}%")
print()
print("  ENSO validation (PC5 - Niño 3.4 correlation):")
pc5_rand_detrend = detrend(PC_rand[4])
corr_pc5_rand_raw = pearsonr(PC_rand[4], nino34_index)[0]
corr_pc5_rand = pearsonr(pc5_rand_detrend, nino34_detrend)[0]
print(f"    Deterministic SVD: {corr_pc5_nino_raw:.3f} (detrended: {corr_pc5_nino:.3f})")
print(f"    Randomized SVD:    {corr_pc5_rand_raw:.3f} (detrended: {corr_pc5_rand:.3f})")
print("=" * 70)

#fig1.savefig("sst_modes.png", dpi=150)
#print("\nFigure saved to sst_modes.png")
#fig2.savefig("sst_modes_sketch.png", dpi=150)
#print("\nFigure saved to sst_modes_sketch.png")

# ========================================================================
# Publication-quality figures
# ========================================================================

SCRIPT_DIR = Path(__file__).parent
PAPER_DIR = SCRIPT_DIR.parent.parent / "paper"

pub_rcParams = {
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'font.family': 'serif',
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
}

# --- svdAll.png: 2x2 singular value spectrum ---
with plt.rc_context(pub_rcParams):
    fig5, axes5 = plt.subplots(2, 2, figsize=(6.5, 5.0), sharey=True)

    configs = [
        (s_a, "(a)"),
        (s_b, "(b) extra_samples=10"),
        (s_c, "(c) power_iter=2"),
        (s_d, "(d) extra_samples=10, power_iter=2"),
    ]

    n_plot = min(50, len(s_full))
    for ax, (s_config, label) in zip(axes5.flat, configs):
        ax.semilogy(range(1, n_plot + 1), s_full[:n_plot],
                     'o', markersize=2.5, label="Deterministic")
        ax.semilogy(range(1, n_modes + 1), s_config, 'x', markersize=4,
                     color='red', label="Randomized")
        ax.set_xlabel("Mode")
        ax.set_title(label, fontsize=8)
        ax.legend(fontsize=7)

    axes5[0, 0].set_ylabel(r"$\sigma_i$")
    axes5[1, 0].set_ylabel(r"$\sigma_i$")

    fig5.tight_layout()
    fig5.savefig(SCRIPT_DIR / "svdAll.png", dpi=600, bbox_inches='tight')
    print("\nSaved svdAll.png")

    shutil.copy2(SCRIPT_DIR / "svdAll.png", PAPER_DIR / "svdAll.png")
    print(f"Copied to {PAPER_DIR / 'svdAll.png'}")

# --- sstModes.png: 5-mode randomized SVD EOF figure ---
with plt.rc_context(pub_rcParams):
    fig6, axes6 = plt.subplots(5, 2, figsize=(7.0, 8.0))

    for i in range(5):
        ax_spatial = axes6[i, 0]
        clim = np.nanmax(np.abs(EOF_maps_rand[i])) * 0.8
        ax_spatial.pcolormesh(lon, lat, EOF_maps_rand[i].T, cmap='RdBu_r',
                              vmin=-clim, vmax=clim, shading='auto')
        ax_spatial.set_xlabel("Longitude")
        ax_spatial.set_ylabel("Latitude")
        ax_spatial.set_title(f"{mode_names[i]} ({100*var_explained_rand[i]:.1f}%)")

        ax_temporal = axes6[i, 1]
        ax_temporal.plot(years, PC_rand[i], linewidth=0.6, color='steelblue')
        ax_temporal.set_xlabel("Year")
        ax_temporal.set_ylabel(r"$\degree$C")
        ax_temporal.set_title(f"PC{i+1}")
        if i > 0:
            ax_temporal.axhline(0, color='gray', linestyle='--', linewidth=0.3)

    fig6.subplots_adjust(top=0.97, bottom=0.04, hspace=0.55, wspace=0.30)
    fig6.savefig(SCRIPT_DIR / "sstModes.png", dpi=600, bbox_inches='tight')
    print("\nSaved sstModes.png")

    shutil.copy2(SCRIPT_DIR / "sstModes.png", PAPER_DIR / "sstModes.png")
    print(f"Copied to {PAPER_DIR / 'sstModes.png'}")

plt.show(block=False)
