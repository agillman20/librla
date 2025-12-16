-------------------------------------------------------------------------------

read_cifar.m - read CIFAR-10 dataset from MATLAB .mat files
read_cifar.py - read CIFAR-10 dataset from Python pickle files (column-major)
read_cifar_batch.py - read CIFAR-10 dataset (batch-major, for deep learning)
read_cifar.jl - read CIFAR-10 dataset from MATLAB .mat files (Julia)

-------------------------------------------------------------------------------

Dataset: 60,000 images (50,000 train + 10,000 test)
         32×32 RGB color images
         10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse,
                     ship, truck

Available formats:
  cifar-10-batches-mat/  - MATLAB .mat files
  cifar-10-batches-py/   - Python pickle files
  cifar-10-batches-bin/  - Binary files (C programs)

-------------------------------------------------------------------------------

https://www.cs.toronto.edu/~kriz/cifar.html

-------------------------------------------------------------------------------

  Key Sections:

  1. Overview - CIFAR-10 dataset (60,000 32×32 RGB images, 10 classes)
  2. Dataset Format - Explanation of the 3072-element flattened format
     (1024 red + 1024 green + 1024 blue, row-major order)
  3. Reader Scripts - Four implementations:
     - Python (column-major): read_cifar.py - Returns (H, W, C, N) for
       MATLAB compatibility
     - Python (batch-major): read_cifar_batch.py - Returns (N, H, W, C) for
       deep learning, supports TensorFlow/PyTorch orderings
     - MATLAB: read_cifar.m - Returns 32×32×3×N arrays
     - Julia: read_cifar.jl - Returns normalized Float32 arrays
  4. Ordering Options - read_cifar_batch.py supports:
     - order='batch' (default): (N, H, W, C) - TensorFlow/Keras
     - order='channel': (N, C, H, W) - PyTorch
     - order='column': (H, W, C, N) - MATLAB style
  5. Key Implementation Details:
     - Labels remain 0-9 (no conversion to 1-10)
     - Row-major to column-major conversion
     - When to use batch-major vs column-major
  6. Usage with ML Frameworks - Examples for TensorFlow, PyTorch, and
     direct matplotlib visualization

  The documentation emphasizes the architectural decision to provide both
  column-major (MATLAB-compatible) and batch-major (deep learning-
  optimized) Python implementations, which is crucial for working
  effectively with CIFAR-10 in different contexts.

