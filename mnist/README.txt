-------------------------------------------------------------------------------

get_mnist.sh - download MNIST database from LeCun's website
read_mnist.m - read MNIST dataset from IDX files (MATLAB)
read_mnist.py - read MNIST dataset from IDX files (column-major)
read_mnist_batch.py - read MNIST dataset (batch-major, for deep learning)
read_mnist.jl - read MNIST dataset from IDX files (Julia)

-------------------------------------------------------------------------------

read_mnist_orig.jl - original Julia script for reading MNIST

-------------------------------------------------------------------------------

Dataset: 70,000 images (60,000 train + 10,000 test)
         28×28 grayscale images
         10 classes: digits 0-9

File format: IDX (custom binary format with magic numbers)
  train-images-idx3-ubyte  - 60,000 training images
  train-labels-idx1-ubyte  - 60,000 training labels
  t10k-images-idx3-ubyte   - 10,000 test images
  t10k-labels-idx1-ubyte   - 10,000 test labels

-------------------------------------------------------------------------------

http://yann.lecun.com/exdb/mnist/
https://github.com/fgnt/mnist



-------------------------------------------------------------------------------

  Reader Scripts - Three implementations:
    - Python (column-major): read_mnist.py - Returns (H, W, N) for MATLAB
      compatibility
    - Python (batch-major): read_mnist_batch.py - Returns (N, H, W) for deep
      learning, supports order='batch' or order='channel'
    - MATLAB: read_mnist.m - Returns 28×28×N arrays
    - Julia: read_mnist.jl - Returns Float32 normalized arrays

  IDX Format:
    - Big-endian integer fields
    - Magic numbers: 2049 (labels), 2051 (images)
    - All readers validate magic numbers and issue warnings on mismatch

  Label Handling:
    - Labels remain 0-9 (no conversion to 1-10)
    - Consistent across all implementations

  Use read_mnist_batch.py for modern deep learning frameworks (PyTorch,
  TensorFlow, JAX). Use read_mnist.py for MATLAB-style workflows and
  compatibility with legacy code.

