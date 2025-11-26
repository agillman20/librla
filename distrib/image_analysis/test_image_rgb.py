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
https://1.img-dpreview.com/files/p/sample_galleries/6044553814/1399808671.jpg
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
# A = np.array(Image.open('x1d-II-sample-02.jpg'))
A = np.array(Image.open('b_29667.jpg'))

# A = np.transpose(A, (1, 0, 2))

m, n, nc = A.shape
print(f'Image size: {m} x {n} x {nc}')

plt.figure(1)
plt.imshow(A)
plt.title('Original (RGB)')
plt.draw()
plt.pause(0.1)

k = 120
use_single = False  # set to True for single precision

if use_single:
    conv = np.float32
else:
    conv = np.float64

# Reshape RGB image to 2D matrix: m x (n*nc)
A2 = conv(A).reshape(m, n * nc)

t0 = time.time()
U, s, Vh = librla.svd_sketch(A2, k)
B2 = U @ np.diag(s) @ Vh
print(f'Elapsed time: {time.time() - t0:.3f}s')

# Reshape back to RGB
B = B2.reshape(m, n, nc)

plt.figure(2)
plt.imshow(np.clip(B, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k} randomized SVD')
plt.draw()

rel_error = norm(A2 - B2, 'fro') / norm(A2, 'fro')
print(f'Basic SVD relative error: {rel_error:.6e}')


t0 = time.time()
U, s, Vh = librla.svd_sketch(A2, k, power_iter=1, extra_samples=k // 4)
B2 = U @ np.diag(s) @ Vh
print(f'Elapsed time: {time.time() - t0:.3f}s')

# Reshape back to RGB
B = B2.reshape(m, n, nc)

plt.figure(3)
plt.imshow(np.clip(B, 0, 255).astype(np.uint8))
plt.title(f'Rank-{k} randomized SVD (power iterations, extra sampling)')
plt.draw()

rel_error = norm(A2 - B2, 'fro') / norm(A2, 'fro')
print(f'Power iter SVD relative error: {rel_error:.6e}')

plt.show(block=False)
