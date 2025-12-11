"""
test_image_rgb.py - Demonstrate RGB image compression using truncated SVD.

Description
-----------
This script demonstrates low-rank image compression using the
randomized truncated SVD from librla. It loads an RGB image,
reshapes it to a 2D matrix (m x 3n) for SVD, and displays results.

Two compression methods are compared:
  1. Basic truncated SVD with rank k
  2. Truncated SVD with power iterations and extra samples for
     improved accuracy

Set use_single=True to run in single precision (faster, less memory).

Requirements
------------
* librla.py in the path
* Image file: pexels-flickr-149387.jpg
* matplotlib, numpy, PIL

See also: librla.svd_sketch, test_image.py

Image source: https://www.pexels.com/photo/silver-metal-round-gears-connected-to-each-other-149387/

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: TBD
Version: 1.0.0
Date: TBD
Assisted by: Claude Code (Anthropic)
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
# image_file = 'pexels-flickr-149387.jpg'
# image_file = 'pexels-anniroenkae-4793404.jpg'
# image_file = 'b_29667.jpg'
image_file = 'pexels-andre-ulysses-de-salis-2100065-7824822.jpg'
A = np.array(Image.open(image_file))

# A = np.transpose(A, (1, 0, 2))

m, n, nc = A.shape
print(f'Image: {image_file}, size: {m} x {n} x {nc}')

plt.figure(1)
plt.imshow(A)
plt.title('Original (RGB)')
plt.draw()
plt.pause(0.1)

k = 120*2
use_single = True  # set to True for single precision

if use_single:
    conv = np.float32
else:
    conv = np.float64

# Reshape RGB image to 2D matrix: m x (n*nc)
A2 = conv(A).reshape(m, n * nc)

t0 = time.time()
U, s, Vh = librla.svd_sketch(A2, k)
B2 = U @ np.diag(s) @ Vh
elapsed1 = time.time() - t0

# Reshape back to RGB
B = B2.reshape(m, n, nc)

plt.figure(2)
plt.imshow(np.clip(B, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k} svd_sketch')
plt.draw()

rel_error1 = norm(A2 - B2, 'fro') / norm(A2, 'fro')
print(f'svd_sketch(k={k}): {elapsed1:.3f}s, error {rel_error1:.6e}')


extra = k // 4
piter = 1
t0 = time.time()
U, s, Vh = librla.svd_sketch(A2, k, power_iter=piter, extra_samples=extra)
B2 = U @ np.diag(s) @ Vh
elapsed2 = time.time() - t0

# Reshape back to RGB
B = B2.reshape(m, n, nc)

plt.figure(3)
plt.imshow(np.clip(B, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k} svd_sketch (extra_samples={extra}, power_iter={piter})')
plt.draw()

rel_error2 = norm(A2 - B2, 'fro') / norm(A2, 'fro')
print(f'svd_sketch(k={k}, extra_samples={extra}, power_iter={piter}): {elapsed2:.3f}s, error {rel_error2:.6e}')

print(f'\n{"Method":<40} {"Rank":>4}    {"Error"}')
print('-' * 55)
print(f'{"svd_sketch(k=" + str(k) + ")":<40} {k:4d}    {rel_error1:.6e}')
print(f'{"svd_sketch(k=" + str(k) + ", extra, power)":<40} {k:4d}    {rel_error2:.6e}')

plt.show(block=False)
