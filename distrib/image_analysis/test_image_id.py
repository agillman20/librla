"""
test_image_id.py - Compare low-rank image compression methods from librla.

Description
-----------
This script compares low-rank factorization methods on RGB images:

  1. Randomized SVD (svd_sketch)
  2. Interpolative Decomposition (id_sketch)
  3. Randomized SVD with oversampling and power iteration
  4. ID with oversampling and power iteration
  5. QR with column pivoting (qr_sketch)

The image is reshaped to a 2D matrix (m x 3n) for processing.

ID selects k skeleton columns and expresses remaining columns as
linear combinations: A[:, piv[k:]] = A[:, piv[:k]] @ T

Set use_single=True to run in single precision (faster, less memory).

Requirements
------------
* librla.py in the path
* Image file (see below)
* matplotlib, numpy, PIL

See also: librla.id_sketch, librla.svd_sketch, librla.qr_sketch

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
"""

import sys
import os
import time
import shutil
from pathlib import Path
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for librla
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import librla

# Load image
image_file = 'pexels-flickr-149387.jpg'
# image_file = 'pexels-anniroenkae-4793404.jpg'
A = np.array(Image.open(image_file))

# A = np.transpose(A, (1, 0, 2))

m, n, nc = A.shape
print(f'Image: {image_file}, size: {m} x {n} x {nc}')

plt.figure(1)
plt.imshow(A)
plt.title('Original (RGB)')
plt.draw()
plt.pause(0.1)

k = 30
use_single = True  # set to True for single precision

if use_single:
    conv = np.float32
else:
    conv = np.float64

# Reshape RGB image to 2D matrix: m x (n*nc)
A2 = conv(A).reshape(m, n * nc)

# ========================================================================
# Method 1: Randomized SVD for comparison
# ========================================================================
t0 = time.time()
U, s, Vh = librla.svd_sketch(A2, k)
B2_svd = U @ np.diag(s) @ Vh
elapsed_svd = time.time() - t0

B_svd = B2_svd.reshape(m, n, nc)

plt.figure(2)
plt.imshow(np.clip(B_svd, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k} svd_sketch')
plt.draw()

rel_error_svd = norm(A2 - B2_svd, 'fro') / norm(A2, 'fro')
print(f'svd_sketch(k={k}): {elapsed_svd:.3f}s, error {rel_error_svd:.6e}')

# ========================================================================
# Method 2: Interpolative Decomposition (randomized)
# ========================================================================
t0 = time.time()
k_id, piv, T = librla.id_sketch(A2, k)
elapsed_id = time.time() - t0

# Reconstruct: skeleton columns + interpolated columns
# A[:, piv[:k]] are skeleton columns (kept exactly)
# A[:, piv[k:]] = A[:, piv[:k]] @ T

# Build reconstruction in permuted order, then unpermute
skeleton = A2[:, piv[:k_id]]
interpolated = skeleton @ T

# Unpermute to original column order
B2_id = np.zeros_like(A2)
B2_id[:, piv[:k_id]] = skeleton
B2_id[:, piv[k_id:]] = interpolated

B_id = B2_id.reshape(m, n, nc)

plt.figure(3)
plt.imshow(np.clip(B_id, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k_id} id_sketch')
plt.draw()

rel_error_id = norm(A2 - B2_id, 'fro') / norm(A2, 'fro')
print(f'id_sketch(k={k_id}): {elapsed_id:.3f}s, error {rel_error_id:.6e}')

# ========================================================================
# Method 3: Randomized SVD with oversampling and power iteration
# ========================================================================
extra = k // 2  # 50% oversampling
piter = 2       # power iterations
t0 = time.time()
U3, s3, Vh3 = librla.svd_sketch(A2, k, extra_samples=extra, power_iter=piter)
B2_svd2 = U3 @ np.diag(s3) @ Vh3
elapsed_svd2 = time.time() - t0

B_svd2 = B2_svd2.reshape(m, n, nc)

plt.figure(4)
plt.imshow(np.clip(B_svd2, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k} svd_sketch (extra_samples={extra}, power_iter={piter})')
plt.draw()

rel_error_svd2 = norm(A2 - B2_svd2, 'fro') / norm(A2, 'fro')
print(f'svd_sketch(k={k}, extra_samples={extra}, power_iter={piter}): {elapsed_svd2:.3f}s, error {rel_error_svd2:.6e}')

# ========================================================================
# Method 4: Interpolative Decomposition with oversampling and power iteration
# ========================================================================
t0 = time.time()
k_id2, piv2, T2 = librla.id_sketch(A2, k, extra_samples=extra, power_iter=piter)
elapsed_id2 = time.time() - t0

# Reconstruct
skeleton2 = A2[:, piv2[:k_id2]]
interpolated2 = skeleton2 @ T2

B2_id2 = np.zeros_like(A2)
B2_id2[:, piv2[:k_id2]] = skeleton2
B2_id2[:, piv2[k_id2:]] = interpolated2

B_id2 = B2_id2.reshape(m, n, nc)

plt.figure(5)
plt.imshow(np.clip(B_id2, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k_id2} id_sketch (extra_samples={extra}, power_iter={piter})')
plt.draw()

rel_error_id2 = norm(A2 - B2_id2, 'fro') / norm(A2, 'fro')
print(f'id_sketch(k={k_id2}, extra_samples={extra}, power_iter={piter}): {elapsed_id2:.3f}s, error {rel_error_id2:.6e}')

# ========================================================================
# Method 5: QR with column pivoting (via qr_sketch)
# ========================================================================
t0 = time.time()
Q_qr, R_qr, p_qr = librla.qr_sketch(A2, k, extra_samples=extra, power_iter=piter)
elapsed_qr = time.time() - t0

k_qr = Q_qr.shape[1]

# Reconstruct: A[:, p] = Q @ R, so unpermute columns
B2_qr_perm = Q_qr @ R_qr  # columns in permuted order
B2_qr = np.zeros_like(A2)
B2_qr[:, p_qr] = B2_qr_perm

B_qr = B2_qr.reshape(m, n, nc)

plt.figure(6)
plt.imshow(np.clip(B_qr, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k_qr} qr_sketch (extra_samples={extra}, power_iter={piter})')
plt.draw()

rel_error_qr = norm(A2 - B2_qr, 'fro') / norm(A2, 'fro')
print(f'qr_sketch(k={k_qr}, extra_samples={extra}, power_iter={piter}): {elapsed_qr:.3f}s, error {rel_error_qr:.6e}')

# ========================================================================
# Summary
# ========================================================================
print(f'\n{"Method":<40} {"Rank":>4}    {"Error"}')
print('-' * 55)
print(f'{"svd_sketch(k=" + str(k) + ")":<40} {k:4d}    {rel_error_svd:.6e}')
print(f'{"id_sketch(k=" + str(k_id) + ")":<40} {k_id:4d}    {rel_error_id:.6e}')
print(f'{"svd_sketch(k=" + str(k) + ", extra, power)":<40} {k:4d}    {rel_error_svd2:.6e}')
print(f'{"id_sketch(k=" + str(k_id2) + ", extra, power)":<40} {k_id2:4d}    {rel_error_id2:.6e}')
print(f'{"qr_sketch(k=" + str(k_qr) + ", extra, power)":<40} {k_qr:4d}    {rel_error_qr:.6e}')

# ========================================================================
# Publication-quality composite figure
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

images = [
    (A, "(a) Original"),
    (np.clip(B_svd, 0, 255).astype(np.uint8), "(b) SVD"),
    (np.clip(B_id, 0, 255).astype(np.uint8), "(c) ID"),
    (np.clip(B_svd2, 0, 255).astype(np.uint8), "(d) SVD + power iter"),
    (np.clip(B_id2, 0, 255).astype(np.uint8), "(e) ID + power iter"),
]

with plt.rc_context(pub_rcParams):
    fig_pub, axes_pub = plt.subplots(3, 2, figsize=(6.5, 9.0))

    # Top row: (a) original in left cell, right cell empty
    axes_pub[0, 0].imshow(images[0][0])
    axes_pub[0, 0].set_title(images[0][1])
    axes_pub[0, 0].axis('off')
    axes_pub[0, 1].axis('off')

    # Middle row: (b) SVD, (c) ID
    for j in range(2):
        axes_pub[1, j].imshow(images[1 + j][0])
        axes_pub[1, j].set_title(images[1 + j][1])
        axes_pub[1, j].axis('off')

    # Bottom row: (d) SVD + power iter, (e) ID + power iter
    for j in range(2):
        axes_pub[2, j].imshow(images[3 + j][0])
        axes_pub[2, j].set_title(images[3 + j][1])
        axes_pub[2, j].axis('off')

    fig_pub.subplots_adjust(hspace=0.02, wspace=0.05)

    fig_pub.savefig(SCRIPT_DIR / "imageEX.png", dpi=600, bbox_inches='tight')
    print("\nSaved imageEX.png")

    shutil.copy2(SCRIPT_DIR / "imageEX.png", PAPER_DIR / "imageEX.png")
    print(f"Copied to {PAPER_DIR / 'imageEX.png'}")

plt.show(block=False)
