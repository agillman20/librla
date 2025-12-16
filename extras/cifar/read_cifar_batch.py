#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
read_cifar_batch.py
===================

Utility to load the CIFAR-10 dataset from Python pickle files and return as
NumPy arrays using **batch-major ordering** (`(N, H, W, C)`). This is the
layout that most deep-learning frameworks (TensorFlow, Keras, PyTorch with
channels_last, JAX, ...) expect.

The CIFAR-10 dataset consists of 60000 32×32 color images in 10 classes,
with 6000 images per class. There are 50000 training images and 10000 test
images.

Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html

Returned objects
----------------
train_labels : uint8 array, shape (50000,)
    Training labels (values 0‑9).

train_images : uint8 array, shape (50000, 32, 32, 3)
    Training images; ``train_images[i]`` is a 32×32×3 RGB image.

test_labels  : uint8 array, shape (10000,)
    Test labels (values 0‑9).

test_images  : uint8 array, shape (10000, 32, 32, 3)
    Test images; ``test_images[i]`` is a 32×32×3 RGB image.

class_names  : list of str, length 10
    Class names: ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

The function assumes that the cifar-10-batches-py/ directory is present in
the specified folder (default: current working directory). No download or
unzip step is performed.

----------------------------------------------------------------------
**Plotting examples (quick reference)**

from read_cifar_batch import read_cifar
import matplotlib.pyplot as plt

train_lbl, train_img, test_lbl, test_img, names = read_cifar()

# --------------------------------------------------------------
# Show the first training image
# --------------------------------------------------------------
plt.figure()
plt.imshow(train_img[0])  # Already in correct format for imshow
plt.title(f'Training label = {train_lbl[0]} ({names[train_lbl[0]]})')
plt.axis('off')
plt.show(block=False)

# --------------------------------------------------------------
# Show the first test image
# --------------------------------------------------------------
plt.figure()
plt.imshow(test_img[0])
plt.title(f'Test label = {test_lbl[0]} ({names[test_lbl[0]]})')
plt.axis('off')
plt.show(block=False)

"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Tuple, Union, List

import numpy as np


def read_cifar(
    folder: Union[str, Path] = ".",
    *,
    order: str = "batch",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load the CIFAR-10 dataset from *folder*/cifar-10-batches-py/.

    Parameters
    ----------
    folder : str or pathlib.Path, optional
        Directory that contains the cifar-10-batches-py/ subdirectory.
        Defaults to the current working directory.
    order : {"batch", "channel", "column"}, optional
        Desired ordering of the image tensors.
        * ``"batch"``          – shape ``(N, H, W, C)`` (default, TensorFlow/Keras style)
        * ``"channel"`` – shape ``(N, C, H, W)`` (PyTorch style)
        * ``"column"``         – shape ``(H, W, C, N)`` (MATLAB/Fortran style)

    Returns
    -------
    train_labels, train_images, test_labels, test_images, class_names
        As described in the module docstring.
    """
    folder = Path(folder)
    data_dir = folder / "cifar-10-batches-py"

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Directory {data_dir} not found. Please ensure CIFAR-10 data is downloaded."
        )

    # ---------- Load metadata (class names) ----------
    meta_file = data_dir / "batches.meta"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Metadata file {meta_file} not found.")

    with meta_file.open("rb") as f:
        meta = pickle.load(f, encoding="bytes")

    # Keys are byte strings in Python 3
    class_names = [name.decode("utf-8") for name in meta[b"label_names"]]
    print(f"Loaded metadata: {len(class_names)} classes")

    # ---------- Load training batches (5 batches × 10000 images each) ----------
    train_labels_list = []
    train_data_list = []

    for i_batch in range(1, 6):
        batch_file = data_dir / f"data_batch_{i_batch}"
        if not batch_file.is_file():
            raise FileNotFoundError(f"Training batch file {batch_file} not found.")

        batch = _load_cifar_batch(batch_file)
        train_labels_list.extend(batch["labels"])
        train_data_list.append(batch["data"])

        batch_label = batch.get("batch_label", f"batch {i_batch}")
        print(f"Loaded {batch_label}: {batch['data'].shape[0]} images")

    # Concatenate all training batches
    train_labels = np.array(train_labels_list, dtype=np.uint8)
    train_data = np.concatenate(train_data_list, axis=0)

    # Reshape training data from (N×3072) to (N×32×32×3)
    train_images = _reshape_cifar_images(train_data, order="batch")
    print(f"Training set: {train_images.shape[0]} images total")

    # ---------- Load test batch ----------
    test_file = data_dir / "test_batch"
    if not test_file.is_file():
        raise FileNotFoundError(f"Test batch file {test_file} not found.")

    test_batch = _load_cifar_batch(test_file)
    test_labels = np.array(test_batch["labels"], dtype=np.uint8)

    # Reshape test data from (N×3072) to (N×32×32×3)
    test_images = _reshape_cifar_images(test_batch["data"], order="batch")
    print(f"Test set: {test_images.shape[0]} images")

    # ---------- Summary ----------
    print(f"\nCIFAR-10 dataset loaded successfully:")
    print(f"  Training: {train_images.shape[0]} images, {len(train_labels)} labels")
    print(f"  Test:     {test_images.shape[0]} images, {len(test_labels)} labels")
    print(f"  Classes:  {len(class_names)} ({class_names[0]}, ..., {class_names[-1]})")

    # ------------------------------------------------------------------
    # Convert image ordering if the user requested a different layout
    # ------------------------------------------------------------------
    if order == "channel":
        # (N, H, W, C) -> (N, C, H, W)
        train_images = np.moveaxis(train_images, -1, 1)
        test_images = np.moveaxis(test_images, -1, 1)
        print(f"  Image format: (N, C, H, W) = {train_images.shape}")
    elif order == "column":
        # (N, H, W, C) -> (H, W, C, N)
        train_images = np.moveaxis(train_images, 0, -1)
        test_images = np.moveaxis(test_images, 0, -1)
        print(f"  Image format: (H, W, C, N) = {train_images.shape}")
    elif order == "batch":
        print(f"  Image format: (N, H, W, C) = {train_images.shape}")
    else:
        raise ValueError(
            f"Unsupported order '{order}'. Choose 'batch', 'channel', or 'column'."
        )

    return train_labels, train_images, test_labels, test_images, class_names


def _load_cifar_batch(path: Path) -> dict:
    """Load a single CIFAR-10 batch file (pickle format)."""
    with path.open("rb") as f:
        batch = pickle.load(f, encoding="bytes")

    # Convert byte-string keys to regular strings and decode values
    result = {}
    if b"data" in batch:
        result["data"] = batch[b"data"]
    if b"labels" in batch:
        result["labels"] = batch[b"labels"]
    if b"batch_label" in batch:
        result["batch_label"] = batch[b"batch_label"].decode("utf-8")
    if b"filenames" in batch:
        result["filenames"] = [fn.decode("utf-8") for fn in batch[b"filenames"]]

    return result


def _reshape_cifar_images(
    data: np.ndarray,
    order: str = "batch",
) -> np.ndarray:
    """
    Convert (N×3072) uint8 array to batch-major image format.

    Parameters
    ----------
    data : np.ndarray, shape (N, 3072)
        Flattened images where each row is a single image.
        The first 1024 values are the red channel (row-major),
        next 1024 are green, last 1024 are blue.
    order : str, optional
        Internal parameter (kept for consistency). Always returns (N, H, W, C).

    Returns
    -------
    images : np.ndarray, shape (N, 32, 32, 3), dtype uint8
        Reshaped images with dimensions [batch, height, width, channels].
        images[i] is the i-th RGB image.

    Notes
    -----
    The CIFAR-10 format stores each 32×32×3 image as a 3072-element
    row vector: [R0...R1023, G0...G1023, B0...B1023] where each channel
    is stored in row-major order (C-style).
    """
    N = data.shape[0]

    # Extract RGB channels (each N×1024)
    R = data[:, :1024]
    G = data[:, 1024:2048]
    B = data[:, 2048:3072]

    # CIFAR stores in row-major order (C-style)
    # Reshape each channel from (N, 1024) to (N, 32, 32)
    # Each image is stored row-major, so we need to transpose for correct orientation
    R_img = R.reshape(N, 32, 32).transpose(0, 2, 1)  # Transpose spatial dims
    G_img = G.reshape(N, 32, 32).transpose(0, 2, 1)
    B_img = B.reshape(N, 32, 32).transpose(0, 2, 1)

    # Stack channels along last axis to get (N, 32, 32, 3)
    images = np.stack([R_img, G_img, B_img], axis=-1)

    return images


if __name__ == "__main__":
    try:
        tl, ti, tl_test, ti_test, names = read_cifar()
        print("\nCIFAR-10 data loaded successfully.")
        print(f"train labels shape : {tl.shape}")
        print(f"train images shape : {ti.shape}")        # (50000, 32, 32, 3)
        print(f"test  labels shape : {tl_test.shape}")
        print(f"test  images shape : {ti_test.shape}")   # (10000, 32, 32, 3)
        print(f"\nFirst 10 training labels: {tl[:10]}")
        print(f"First 10 test labels: {tl_test[:10]}")
        print(f"\nClass names: {names}")

        # Test other orderings
        print("\n" + "="*70)
        print("Testing channel ordering (PyTorch style):")
        print("="*70)
        tl2, ti2, tl_test2, ti_test2, names2 = read_cifar(order="channel")
        print(f"train images shape : {ti2.shape}")       # (50000, 3, 32, 32)
        print(f"test  images shape : {ti_test2.shape}")  # (10000, 3, 32, 32)

        print("\n" + "="*70)
        print("Testing column ordering (MATLAB style):")
        print("="*70)
        tl3, ti3, tl_test3, ti_test3, names3 = read_cifar(order="column")
        print(f"train images shape : {ti3.shape}")       # (32, 32, 3, 50000)
        print(f"test  images shape : {ti_test3.shape}")  # (32, 32, 3, 10000)

    except Exception as exc:  # pragma: no cover
        print(f"Error loading CIFAR-10 data: {exc}")
        raise
