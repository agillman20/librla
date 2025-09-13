

hilb.py - n x m Hilbert matrix, optimized version, using scipy.linalg.hankel
libid.py - rank-revealing decompositions, QR, SVD, ID, randomized, native python

hilb.py and libid.py have built-in testing routines, to execute, please run:

  python3 hilb.py
  python3 libid.py

--------------------------------------------------------------------------------

hilb.m - n x m Hilbert matrix, optimized version, using broadcast
libid.m - rank-revealing decompositions, QR, SVD, ID, randomized, 
          native matlab gpt-oss:120b assisted translation of libid.py

--------------------------------------------------------------------------------


ollama run gpt-oss:120b "Must use American English. Avoid \
em-dashes. Avoid en-dashes. Avoid Unicode symbols. Summarize and \
convert to python this file: $(cat libid.py)"


libid_vibe.py - gpt-oss:120b assisted summarization of libid.py


Description
-----------

This module implements several randomized linear‑algebra routines that
approximate the rank, QR factorization, singular‑value decomposition
(SVD), and interpolative decomposition (ID) of a matrix ``A``.

User‑callable methods
---------------------
range_randomized   - Build an orthonormal basis for the column space.
rrqr_randomized    - Rank‑revealing QR using a randomized basis.
rrsvd_randomized   - Truncated SVD using a randomized basis.
rrid_randomized    - Interpolative decomposition using randomized QR.
image_randomized   - Basis for the row space via transpose.

Author: Your Name
SPDX-License-Identifier: TBD


**Summary**

The module implements several randomized linear-algebra routines that
approximate the rank, QR factorization, singular-value decomposition
(SVD), and interpolative decomposition (ID) of a matrix `A`.

* `range_randomized` builds an orthonormal basis for the column space
  of `A` using random sampling. It repeatedly draws a random test
  matrix, optionally applies a power iteration (`flag_power`), and
  checks a relative-error tolerance (`rtol`). The routine returns the
  size of the basis and the basis matrix `Q`.

* `rrqr_randomized` uses the basis from `range_randomized` to compute
  a rank-revealing QR factorization. If the full rank is needed, it
  falls back to a deterministic QR. It returns the leading `k` columns
  of `Q`, the leading `k` rows of `R`, and the pivot permutation
  vector `p`.

* `rrsvd_randomized` builds on the same basis to obtain a truncated
  SVD. When the matrix is effectively full rank it computes the full
  SVD directly. It returns the leading `k` left singular vectors,
  singular values, and right singular vectors.

* `rrid_randomized` forms an interpolative decomposition by solving a
  triangular system that extracts the interpolation matrix from the QR
  factors produced by `rrqr_randomized`. It returns the numerical rank
  `k`, the pivot vector `p`, and the interpolation matrix `proj`.

* `image_randomized` is a thin wrapper that calls `range_randomized`
  on the transpose of `A`, giving a basis for the row space.

All functions accept the same optional arguments:

* `rtol` - relative tolerance that controls the truncation level;
* `block_size` - initial number of random vectors (default 42);
* `flag_power` - number of power-iteration steps (default 0).

The implementation relies on NumPy and SciPy, and it avoids any
non-ASCII symbols or special dash characters.

All three algorithms share the same workflow:

* A cheap random sketch of the column space of the input matrix `A` is built by multiplying `A` with a random matrix.
* Optional power iterations (`flag_power`) improve the sketch when the singular values decay slowly.
* The sketch is orthogonalized with a QR factorization.
* The original matrix is projected onto the sketch (`Aproj = Q.T @ A`).
* A standard deterministic factorization (QR, SVD, or QR again for ID) is performed on the tiny projected matrix.
* The result is lifted back to the original space (`Q = Q @ Qproj` for QR/SVD).

The numerical rank `k` is chosen automatically: a row (or singular
value) is kept if its norm (or absolute value) is at least `rtol *
||A||` (or `rtol * ||Aproj||`).

If the random sketch grows to the full size of the matrix, the
functions fall back to a deterministic factorization of the whole
matrix, guaranteeing correct results for small or effectively
full-rank problems.

The helper `range_randomized` returns the size of the sketch (`k`) and
the orthonormal basis `Q`. `image_randomized` does the same for the
row space by calling `range_randomized` on `A.T`.


