#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
read_mnist.py
=============
Load the four original MNIST IDX files (train‑labels, train‑images,
t10k‑labels, t10k‑images) and return them as NumPy arrays.

The routine mirrors the MATLAB version that checks the magic numbers,
issues warnings on mismatches, and leaves **all label values unchanged**.

Returned objects
----------------
train_labels : uint8 array, shape (60000,)
    Training labels (values 0‑9).
train_images : uint8 array, shape (28, 28, 60000)
    Training images; each slice ``train_images[:,:,i]`` is a 28×28 image.
test_labels  : uint8 array, shape (10000,)
    Test labels (values 0‑9, no conversion of 0 → 10).
test_images  : uint8 array, shape (28, 28, 10000)
    Test images; each slice ``test_images[:,:,i]`` is a 28×28 image.

The function assumes that the four ``*.ubyte`` files are present in the
specified folder (default: the current working directory).  No download
or unzip step is performed.

----------------------------------------------------------------------
**Plotting examples (placed here for quick reference)**

from read_mnist import read_mnist
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Load the data (the files must be in the current directory or a folder
# you point to)
# ------------------------------------------------------------------
train_lbl, train_img, test_lbl, test_img = read_mnist()

# --------------------------------------------------------------
# Show the first training image
# --------------------------------------------------------------
plt.figure()
# ``train_img`` is stored column‑wise (MATLAB order).  Transpose
# gives the conventional “row‑major” view for imshow.
plt.imshow(train_img[:, :, 0].T, cmap='gray')
plt.title(f'Training label = {train_lbl[0]}')
plt.axis('off')
plt.show(block=False)

# --------------------------------------------------------------
# Show the first test image
# --------------------------------------------------------------
plt.figure()
plt.imshow(test_img[:, :, 0].T, cmap='gray')
plt.title(f'Test label = {test_lbl[0]}')
plt.axis('off')
plt.show(block=False)

"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np


def read_mnist(folder: Union[str, Path] = ".") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the four MNIST IDX files from *folder*."""
    folder = Path(folder)

    # ---------- training labels ----------
    train_labels, magic_l = _read_idx_data(folder / "train-labels-idx1-ubyte", 60_000, None)
    print(f"train-labels : magic={magic_l}, count={train_labels.size}")
    if magic_l != 2049:
        warnings.warn(f"Unexpected magic number {magic_l} in train‑labels file.")

    # ---------- training images ----------
    train_images, magic_i = _read_idx_data(folder / "train-images-idx3-ubyte", 60_000, (28, 28))
    print(
        f"train-images : magic={magic_i}, count={train_images.shape[2]}, "
        f"nrows={train_images.shape[0]}, ncols={train_images.shape[1]}"
    )
    if magic_i != 2051:
        warnings.warn(f"Unexpected magic number {magic_i} in train‑images file.")

    # ---------- test labels ----------
    test_labels, magic_l = _read_idx_data(folder / "t10k-labels-idx1-ubyte", 10_000, None)
    print(f"test-labels  : magic={magic_l}, count={test_labels.size}")
    if magic_l != 2049:
        warnings.warn(f"Unexpected magic number {magic_l} in test‑labels file.")

    # ---------- test images ----------
    test_images, magic_i = _read_idx_data(folder / "t10k-images-idx3-ubyte", 10_000, (28, 28))
    print(
        f"test-images  : magic={magic_i}, count={test_images.shape[2]}, "
        f"nrows={test_images.shape[0]}, ncols={test_images.shape[1]}"
    )
    if magic_i != 2051:
        warnings.warn(f"Unexpected magic number {magic_i} in test‑images file.")

    return train_labels, train_images, test_labels, test_images


def _read_idx_header(fid) -> Tuple[int, int, Tuple[int, int] | None]:
    """Read magic number, item count and (optional) image dimensions."""
    magic_bytes = fid.read(4)
    count_bytes = fid.read(4)
    if len(magic_bytes) < 4 or len(count_bytes) < 4:
        raise EOFError("Unexpected end of file while reading IDX header.")

    magic = struct.unpack(">i", magic_bytes)[0]
    count = struct.unpack(">i", count_bytes)[0]

    dims = None
    if magic == 2051:                     # image file
        rows = struct.unpack(">i", fid.read(4))[0]
        cols = struct.unpack(">i", fid.read(4))[0]
        dims = (rows, cols)

    return magic, count, dims


def _read_idx_data(
    path: Union[str, Path],
    expected_count: int,
    img_dims: Tuple[int, int] | None,
) -> Tuple[np.ndarray, int]:
    """Read a label or image IDX file and return its contents plus the magic number."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MNIST file not found: {path}")

    with path.open("rb") as fid:
        magic, count, dims_from_header = _read_idx_header(fid)

        if count != expected_count:
            warnings.warn(
                f"File {path.name} reports {count} items, but {expected_count} were expected."
            )

        if img_dims is None:                     # label file
            raw = fid.read(count)
            if len(raw) != count:
                raise EOFError(f"Could not read {count} label bytes from {path.name}.")
            data = np.frombuffer(raw, dtype=np.uint8).reshape((-1,))

        else:                                    # image file
            if dims_from_header is not None:
                img_dims = dims_from_header
            rows, cols = img_dims
            total = count * rows * cols
            raw = fid.read(total)
            if len(raw) != total:
                raise EOFError(f"Could not read {total} image bytes from {path.name}.")
            # Preserve column‑wise ordering used by the original MATLAB code
            data = np.frombuffer(raw, dtype=np.uint8).reshape((rows, cols, count), order="F")

    return data, magic


if __name__ == "__main__":
    try:
        tl, ti, tl_test, ti_test = read_mnist()
        print("\nMNIST data loaded successfully.")
        print(f"train labels shape : {tl.shape}")
        print(f"train images shape : {ti.shape}")
        print(f"test  labels shape : {tl_test.shape}")
        print(f"test  images shape : {ti_test.shape}")
    except Exception as exc:   # pragma: no cover
        print(f"Error loading MNIST data: {exc}")
