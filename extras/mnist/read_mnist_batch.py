#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_mnist_batch.py
===================

Utility to load the original MNIST IDX files (train-labels,
train-images, t10k‑labels, t10k‑images) and return them as NumPy arrays
using **batch-major ordering** (`(N, H, W)`).  This is the layout that most
deep-learning frameworks (PyTorch, TensorFlow, JAX, ...) expect.

The implementation mirrors the MATLAB version that checks the magic
numbers, issues warnings on mismatches, and leaves **all label values
unchanged**.

Returned objects
----------------
train_labels : uint8 array, shape (60000,)
    Training labels (values 0-9).

train_images : uint8 array, shape (60000, 28, 28)
    Training images; ``train_images[i]`` is a 28 x 28 image.

test_labels  : uint8 array, shape (10000,)
    Test labels (values 0-9).

test_images  : uint8 array, shape (10000, 28, 28)
    Test images; ``test_images[i]`` is a 28 x 28 image.

The function assumes that the four ``*.ubyte`` files are present in the
specified folder (default: the current working directory).  No download
or unzip step is performed.

----------------------------------------------------------------------
**Plotting examples (quick reference)**
from read_mnist_batch import read_mnist
import matplotlib.pyplot as plt

train_lbl, train_img, test_lbl, test_img = read_mnist()
# --------------------------------------------------------------
# Show the first training image
# --------------------------------------------------------------
plt.figure()
plt.imshow(train_img[0], cmap='gray')
plt.title(f'Training label = {train_lbl[0]}')
plt.axis('off')
plt.show(block=False)
# --------------------------------------------------------------
# Show the first test image
# --------------------------------------------------------------
plt.figure()
plt.imshow(test_img[0], cmap='gray')
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


def read_mnist(
    folder: Union[str, Path] = ".",
    *,
    order: str = "batch",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the four MNIST IDX files from *folder*.

    Parameters
    ----------
    folder : str or pathlib.Path, optional
        Directory that contains the four ``*.ubyte`` files.  Defaults to the
        current working directory.
    order : {"batch", "channel"}, optional
        Desired ordering of the image tensors.
        * ``"batch"``   – shape ``(N, H, W)`` (default)
        * ``"channel"`` – shape ``(H, W, N)`` (identical to the original script)

    Returns
    -------
    train_labels, train_images, test_labels, test_images
        As described in the module docstring.
    """
    folder = Path(folder)

    # ---------- training labels ----------
    train_labels, magic_l = _read_idx_data(
        folder / "train-labels-idx1-ubyte", 60_000, None
    )
    print(f"train‑labels : magic={magic_l}, count={train_labels.size}")
    if magic_l != 2049:
        warnings.warn(
            f"Unexpected magic number {magic_l} in train‑labels file."
        )

    # ---------- training images ----------
    train_images, magic_i = _read_idx_data(
        folder / "train-images-idx3-ubyte", 60_000, (28, 28)
    )
    print(
        f"train‑images : magic={magic_i}, count={train_images.shape[0]}, "
        f"rows={train_images.shape[1]}, cols={train_images.shape[2]}"
    )
    if magic_i != 2051:
        warnings.warn(
            f"Unexpected magic number {magic_i} in train‑images file."
        )

    # ---------- test labels ----------
    test_labels, magic_l = _read_idx_data(
        folder / "t10k-labels-idx1-ubyte", 10_000, None
    )
    print(f"test‑labels  : magic={magic_l}, count={test_labels.size}")
    if magic_l != 2049:
        warnings.warn(
            f"Unexpected magic number {magic_l} in test‑labels file."
        )

    # ---------- test images ----------
    test_images, magic_i = _read_idx_data(
        folder / "t10k-images-idx3-ubyte", 10_000, (28, 28)
    )
    print(
        f"test‑images  : magic={magic_i}, count={test_images.shape[0]}, "
        f"rows={test_images.shape[1]}, cols={test_images.shape[2]}"
    )
    if magic_i != 2051:
        warnings.warn(
            f"Unexpected magic number {magic_i} in test‑images file."
        )

    # ------------------------------------------------------------------
    # Convert image ordering if the user asked for the channel (H,W,N)
    # layout – the internal _read_idx_data always returns batch‑major.
    # ------------------------------------------------------------------
    if order == "channel":
        # (N, H, W) -> (H, W, N)
        train_images = np.moveaxis(train_images, 0, -1)
        test_images = np.moveaxis(test_images, 0, -1)
    elif order != "batch":
        raise ValueError(
            f"Unsupported order '{order}'. Choose 'batch' or 'channel'."
        )

    return train_labels, train_images, test_labels, test_images


def _read_idx_header(fid) -> Tuple[int, int, Tuple[int, int] | None]:
    """Read magic number, item count and (optional) image dimensions."""
    magic_bytes = fid.read(4)
    count_bytes = fid.read(4)

    if len(magic_bytes) < 4 or len(count_bytes) < 4:
        raise EOFError(
            "Unexpected end of file while reading IDX header."
        )

    magic = struct.unpack(">i", magic_bytes)[0]
    count = struct.unpack(">i", count_bytes)[0]

    dims: Tuple[int, int] | None = None
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
    """Read a label or image IDX file.

    Returns
    -------
    data : np.ndarray
        * For a label file – shape ``(N,)`` (dtype ``uint8``)
        * For an image file – shape ``(N, H, W)`` (dtype ``uint8``)
    magic : int
        The magic number stored in the file (used for sanity checking).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MNIST file not found: {path}")

    with path.open("rb") as fid:
        magic, count, dims_from_header = _read_idx_header(fid)

        if count != expected_count:
            warnings.warn(
                f"File {path.name} reports {count} items, but "
                f"{expected_count} were expected."
            )

        # -----------------
        #  LABEL FILES
        # -----------------
        if img_dims is None:                     # label file
            raw = fid.read(count)
            if len(raw) != count:
                raise EOFError(
                    f"Could not read {count} label bytes from {path.name}."
                )
            data = np.frombuffer(raw, dtype=np.uint8).reshape((-1,))
        # -----------------
        #  IMAGE FILES
        # -----------------
        else:                                    # image file
            # Use the dimensions that are stored in the file – they are
            # guaranteed to be correct for the MNIST spec.
            if dims_from_header is not None:
                img_dims = dims_from_header
            rows, cols = img_dims
            total = count * rows * cols
            raw = fid.read(total)
            if len(raw) != total:
                raise EOFError(
                    f"Could not read {total} image bytes from {path.name}."
                )
            # NB: ``order='C'`` guarantees row‑major (C‑style) memory layout.
            # After reshaping we get (N, H, W) because we place the count as
            # the first dimension.
            data = np.frombuffer(raw, dtype=np.uint8).reshape(
                (count, rows, cols), order="C"
            )

    return data, magic


if __name__ == "__main__":
    # Example usage ----------------------------------------------------
    try:
        tl, ti, tl_test, ti_test = read_mnist()
        print("\nMNIST data loaded successfully.")
        print(f"train labels shape : {tl.shape}")
        print(f"train images shape : {ti.shape}")   # (60000, 28, 28)
        print(f"test  labels shape : {tl_test.shape}")
        print(f"test  images shape : {ti_test.shape}")   # (10000, 28, 28)
    except Exception as exc:   # pragma: no cover
        print(f"Error loading MNIST data: {exc}")


