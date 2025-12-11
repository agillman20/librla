"""
test_image.py - Demonstrate image compression using truncated SVD.

Description
-----------
This script demonstrates low-rank image compression using the
randomized truncated SVD from librla. It loads a grayscale image,
computes a rank-k approximation, and displays the results.

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

See also: librla.svd_sketch

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
image_file = 'pexels-flickr-149387.jpg'
# image_file = 'pexels-anniroenkae-4793404.jpg'
# image_file = 'hello_world.png'
# image_file = 'lorem_ipsum.png'

A = np.array(Image.open(image_file))

# A = np.transpose(A, (1, 0, 2))

# Convert to grayscale if RGB
if A.ndim == 3 and A.shape[2] == 3:
    # Simple grayscale conversion: 0.299*R + 0.587*G + 0.114*B
    A = (0.299 * A[:,:,0] + 0.587 * A[:,:,1] + 0.114 * A[:,:,2]).astype(np.uint8)

print(f'Image: {image_file}, size: {A.shape}')

plt.figure(1)
plt.imshow(A, cmap='gray', aspect='equal')
plt.colorbar()
plt.title('Original (grayscale)')
plt.draw()
plt.pause(0.1)

k = 60*2
use_single = True  # set to True for single precision (may cause numerical issues)

if use_single:
    conv = np.float32
else:
    conv = np.float64

t0 = time.time()
U, s, Vh = librla.svd_sketch(conv(A), k)
B = U @ np.diag(s) @ Vh
elapsed1 = time.time() - t0

plt.figure(2)
B = np.clip(B, 0, 255)  # clip to valid range
plt.imshow(B, cmap='gray', aspect='equal')
plt.colorbar()
plt.title(f'Rank-{k} svd_sketch')
plt.draw()

rel_error1 = norm(conv(A) - conv(B), 'fro') / norm(conv(A), 'fro')
print(f'svd_sketch(k={k}): {elapsed1:.3f}s, error {rel_error1:.6e}')


extra = k // 4
piter = 1
t0 = time.time()
U, s, Vh = librla.svd_sketch(conv(A), k, power_iter=piter, extra_samples=extra)
B = U @ np.diag(s) @ Vh
elapsed2 = time.time() - t0

plt.figure(3)
B = np.clip(B, 0, 255)  # clip to valid range
plt.imshow(B, cmap='gray', aspect='equal')
plt.colorbar()
plt.title(f'Rank-{k} svd_sketch (extra_samples={extra}, power_iter={piter})')
plt.draw()

rel_error2 = norm(conv(A) - conv(B), 'fro') / norm(conv(A), 'fro')
print(f'svd_sketch(k={k}, extra_samples={extra}, power_iter={piter}): {elapsed2:.3f}s, error {rel_error2:.6e}')

print(f'\n{"Method":<40} {"Rank":>4}    {"Error"}')
print('-' * 55)
print(f'{"svd_sketch(k=" + str(k) + ")":<40} {k:4d}    {rel_error1:.6e}')
print(f'{"svd_sketch(k=" + str(k) + ", extra, power)":<40} {k:4d}    {rel_error2:.6e}')

plt.show(block=False)
