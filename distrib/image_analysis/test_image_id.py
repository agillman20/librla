"""
test_image_id.py - Compare low-rank image compression methods from librla.

Description
-----------
This script compares low-rank approximation methods on RGB images:

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
"""

import sys
import os
import time
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for librla
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import librla

# Load image
# A = np.array(Image.open('pexels-flickr-149387.jpg'))
A = np.array(Image.open('pexels-anniroenkae-4793404.jpg'))

# A = np.transpose(A, (1, 0, 2))

m, n, nc = A.shape
print(f'Image size: {m} x {n} x {nc}')

plt.figure(1)
plt.imshow(A)
plt.title('Original (RGB)')
plt.draw()
plt.pause(0.1)

k = 60*2
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
print('\n--- Randomized SVD ---')
t0 = time.time()
U, s, Vh = librla.svd_sketch(A2, k)
B2_svd = U @ np.diag(s) @ Vh
print(f'Elapsed time: {time.time() - t0:.3f}s')

B_svd = B2_svd.reshape(m, n, nc)

plt.figure(2)
plt.imshow(np.clip(B_svd, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k} randomized SVD')
plt.draw()

rel_error_svd = norm(A2 - B2_svd, 'fro') / norm(A2, 'fro')
print(f'SVD relative error: {rel_error_svd:.6e}')

# ========================================================================
# Method 2: Interpolative Decomposition (randomized)
# ========================================================================
print('\n--- Interpolative Decomposition (id_sketch) ---')
t0 = time.time()
k_id, piv, T = librla.id_sketch(A2, k)
print(f'Elapsed time: {time.time() - t0:.3f}s')

print(f'ID rank: {k_id}')

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
plt.title(f'Rank-{k_id} interpolative decomposition')
plt.draw()

rel_error_id = norm(A2 - B2_id, 'fro') / norm(A2, 'fro')
print(f'ID relative error: {rel_error_id:.6e}')

# ========================================================================
# Method 3: Randomized SVD with oversampling and power iteration
# ========================================================================
extra = k // 2  # 50% oversampling
piter = 2       # power iterations
print(f'\n--- Randomized SVD (extra_samples={extra}, power_iter={piter}) ---')
t0 = time.time()
U3, s3, Vh3 = librla.svd_sketch(A2, k, extra_samples=extra, power_iter=piter)
B2_svd2 = U3 @ np.diag(s3) @ Vh3
print(f'Elapsed time: {time.time() - t0:.3f}s')

B_svd2 = B2_svd2.reshape(m, n, nc)

plt.figure(4)
plt.imshow(np.clip(B_svd2, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k} SVD (extra={extra}, piter={piter})')
plt.draw()

rel_error_svd2 = norm(A2 - B2_svd2, 'fro') / norm(A2, 'fro')
print(f'SVD (oversampled+piter) relative error: {rel_error_svd2:.6e}')

# ========================================================================
# Method 4: Interpolative Decomposition with oversampling and power iteration
# ========================================================================
print(f'\n--- ID (extra_samples={extra}, power_iter={piter}) ---')
t0 = time.time()
k_id2, piv2, T2 = librla.id_sketch(A2, k, extra_samples=extra, power_iter=piter)
print(f'Elapsed time: {time.time() - t0:.3f}s')

print(f'ID rank: {k_id2}')

# Reconstruct
skeleton2 = A2[:, piv2[:k_id2]]
interpolated2 = skeleton2 @ T2

B2_id2 = np.zeros_like(A2)
B2_id2[:, piv2[:k_id2]] = skeleton2
B2_id2[:, piv2[k_id2:]] = interpolated2

B_id2 = B2_id2.reshape(m, n, nc)

plt.figure(5)
plt.imshow(np.clip(B_id2, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k_id2} ID (extra={extra}, piter={piter})')
plt.draw()

rel_error_id2 = norm(A2 - B2_id2, 'fro') / norm(A2, 'fro')
print(f'ID (oversampled+piter) relative error: {rel_error_id2:.6e}')

# ========================================================================
# Method 5: QR with column pivoting (via qr_sketch)
# ========================================================================
print(f'\n--- QR (extra_samples={extra}, power_iter={piter}) ---')
t0 = time.time()
Q_qr, R_qr, p_qr = librla.qr_sketch(A2, k, extra_samples=extra, power_iter=piter)
print(f'Elapsed time: {time.time() - t0:.3f}s')

k_qr = Q_qr.shape[1]
print(f'QR rank: {k_qr}')

# Reconstruct: A[:, p] = Q @ R, so unpermute columns
B2_qr_perm = Q_qr @ R_qr  # columns in permuted order
B2_qr = np.zeros_like(A2)
B2_qr[:, p_qr] = B2_qr_perm

B_qr = B2_qr.reshape(m, n, nc)

plt.figure(6)
plt.imshow(np.clip(B_qr, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k_qr} QR (extra={extra}, piter={piter})')
plt.draw()

rel_error_qr = norm(A2 - B2_qr, 'fro') / norm(A2, 'fro')
print(f'QR relative error: {rel_error_qr:.6e}')

# ========================================================================
# Summary
# ========================================================================
print('\n=== Summary ===')
print('Method                              Rank    Relative Error')
print('-----------------------------------------------------------')
print(f'Randomized SVD                      {k:4d}    {rel_error_svd:.6e}')
print(f'ID (randomized)                     {k_id:4d}    {rel_error_id:.6e}')
print(f'SVD (extra={extra}, piter={piter})            {k:4d}    {rel_error_svd2:.6e}')
print(f'ID  (extra={extra}, piter={piter})            {k_id2:4d}    {rel_error_id2:.6e}')
print(f'QR  (extra={extra}, piter={piter})            {k_qr:4d}    {rel_error_qr:.6e}')

plt.show(block=False)
