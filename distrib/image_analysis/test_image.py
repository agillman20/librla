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
A = np.array(Image.open('pexels-flickr-149387.jpg'))
# A = np.array(Image.open('pexels-anniroenkae-4793404.jpg'))

# A = np.transpose(A, (1, 0, 2))

# Convert to grayscale if RGB
if A.ndim == 3 and A.shape[2] == 3:
    # Simple grayscale conversion: 0.299*R + 0.587*G + 0.114*B
    A = (0.299 * A[:,:,0] + 0.587 * A[:,:,1] + 0.114 * A[:,:,2]).astype(np.uint8)

print(f'Image size: {A.shape}')

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
print(f'Elapsed time: {time.time() - t0:.3f}s')

plt.figure(2)
B = np.clip(B, 0, 255)  # clip to valid range
plt.imshow(B, cmap='gray', aspect='equal')
plt.colorbar()
plt.title(f'Rank-{k} SVD approximation')
plt.draw()

rel_error = norm(conv(A) - conv(B), 'fro') / norm(conv(A), 'fro')
print(f'Relative error: {rel_error:.6e}')


t0 = time.time()
U, s, Vh = librla.svd_sketch(conv(A), k, power_iter=1, extra_samples=k // 4)
B = U @ np.diag(s) @ Vh
print(f'Elapsed time: {time.time() - t0:.3f}s')

plt.figure(3)
B = np.clip(B, 0, 255)  # clip to valid range
plt.imshow(B, cmap='gray', aspect='equal')
plt.colorbar()
plt.title(f'Rank-{k} SVD (power iterations, extra sampling)')
plt.draw()

rel_error = norm(conv(A) - conv(B), 'fro') / norm(conv(A), 'fro')
print(f'Relative error: {rel_error:.6e}')

plt.show(block=False)
