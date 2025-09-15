
--------------------------------------------------------------------------------

hilb.py - n x m Hilbert matrix, optimized version, using scipy.linalg.hankel
libid.py - rank-revealing decompositions, QR, SVD, ID, randomized, native python


hilb.py and libid.py have built-in testing routines, to execute, please run:

python3 hilb.py
python3 libid.py

--------------------------------------------------------------------------------

hilb.m - n x m Hilbert matrix, optimized version
libid.m - rank-revealing decompositions, QR, SVD, ID, randomized, 
          native matlab, gpt-oss:120b assisted translation of libid.py

--------------------------------------------------------------------------------


libid_vibe.py - gpt-oss:120b assisted summarization of libid.py, refactored


**Summary**

The module implements several randomized linear-algebra routines that
approximate the rank, QR factorization, singular-value decomposition
(SVD), and interpolative decomposition (ID) of a matrix `A`.

* `orth_randomized` builds an orthonormal basis for the column space
  of `A` using random projection. It repeatedly draws a random test
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

---

**Converted Python code (American English, ASCII only)**
Description
-----------

This module implements several randomized linear-algebra routines that
approximate the rank, QR factorization, singular-value decomposition
(SVD), and interpolative decomposition (ID) of a matrix ``A``.

User-callable methods
---------------------
range_randomized   - Build an orthonormal basis for the column space.
rrqr_randomized    - Rank-revealing QR using a randomized basis.
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
the orthonormal basis `Q`.


--------------------------------------------------------------------------------


echo / | llama-cli --ctx-size 0 -hf unsloth/gpt-oss-120b-GGUF --system-prompt "Must use American English. Avoid em-dashes. Avoid en-dashes. Avoid Unicode symbols. Summarize and convert to python this file:" --file libid.py | tee libid_vibe.txt

echo / | llama-cli --ctx-size 0 -hf unsloth/gpt-oss-120b-GGUF --system-prompt "Replace Unicode symbols in all docstrings with ASCII equivalents." --file libid.py | tee libid_vibe.txt

echo / | llama-cli --ctx-size 0 -hf unsloth/gpt-oss-120b-GGUF --system-prompt "Must use American English. Avoid em-dashes. Avoid en-dashes. Avoid Unicode symbols. Do not change variable or function names. Convert to matlab, using classes, this file:" --file libid_vibe.py | tee libid_vibe.txt


--------------------------------------------------------------------------------


ollama run gpt-oss:120b "Must use American English. Avoid \
em-dashes. Avoid en-dashes. Avoid Unicode symbols. Summarize and \
convert to python this file: $(cat libid.py)" > libid_vibe.txt

**Summary**

The module implements several randomized linear-algebra routines that
approximate the rank, QR factorization, singular-value decomposition
(SVD), and interpolative decomposition (ID) of a matrix `A`.

* `ortho_sketch` builds an orthonormal basis for the column space
  of `A` using randomized sketching. It repeatedly draws a random test
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
  `k`, the pivot vector `p`, and the interpolation matrix `T`.

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


--------------------------------------------------------------------------------


echo / | llama-cli --ctx-size 0 -hf unsloth/gpt-oss-120b-GGUF --system-prompt "Must use American English. Avoid em-dashes. Avoid en-dashes. Avoid Unicode symbols. Summarize this file:" --file libid.py | tee libid.txt

echo / | llama-cli --ctx-size 0 -hf unsloth/gpt-oss-120b-GGUF --system-prompt "Must use American English. Avoid em-dashes. Avoid en-dashes. Avoid Unicode symbols. Summarize and convert to python this file:" --file libid.py | tee libid_vibe.txt

echo / | llama-cli --ctx-size 0 -hf unsloth/gpt-oss-120b-GGUF --system-prompt "In this file, replace Unicode symbols in all docstrings with ASCII equivalents:" --file libid_vibe.txt | tee libid_vibe_clean.txt

--------------------------------------------------------------------------------

# convert python to matlab, editing is required between steps

echo \ | llama-cli -hf unsloth/gpt-oss-120b-GGUF --system-prompt  "You are a friendly assistant. Must use American English. No em-dashes in comments. No en-dashes in comments. No Unicode symbols in documentation or comment lines. Convert Unicode and math symbols to use only printable ASCII symbols in documentation or comment lines. Convert to Matlab, keeping function ordering, use numbered dot code flow using dash underline above and below, never include any info summary or documentation before function or subroutine name, always include info/documentation after implicit statement after function or subroutine name, document structure fields, document input and output parameters with type information followed with description next lines with section name spelled out with parameters using underline with ASCII dashes, notes, numbered code flow sections, preserve original comments and notes, format documentation available for fortran help, add Your Name placeholder author, TBD placeholder SPDX-License-Identifier, author before spdx, documentation section name should be capitalized but must not be in all caps and should be underlined and should use moderate indentation at least two spaces, always show description section at the top of main documentation block, always use real or complex names with name followed by single colon formatted table in describing parameter types, always include main documentation synopsis string after the function or subroutine name, preserve unused functions and subroutines, place description block containing a list of all user-callable routines and functions at the top of the file, using class and classdef with proper documentation indentation, include examples, separate functions by three lines, this file:" --file libid_vibe.py --ctx-size 0 | tee output.txt


--------------------------------------------------------------------------------


# convert matlab to python, editing is required between steps, experimental

echo \ | llama-cli -hf unsloth/gpt-oss-120b-GGUF --system-prompt  "You are a friendly assistant. Must use American English. No em-dashes in comments. No en-dashes in comments. No Unicode symbols in documentation or comment lines. Convert Unicode and math symbols to use only printable ASCII symbols in documentation or comment lines. Convert to python, keeping function ordering, use numbered dot code flow using dash underline above and below, never include any info summary or documentation before function or subroutine name, always include info/documentation after implicit statement after function or subroutine name, document structure fields, document input and output parameters with type information followed with description next lines with section name spelled out with parameters using underline with ASCII dashes, notes, numbered code flow sections, preserve original comments and notes, format documentation available for fortran help, add Your Name placeholder author, TBD placeholder SPDX-License-Identifier, author before spdx, documentation section name should be capitalized but must not be in all caps and should be underlined and should use moderate indentation at least two spaces, always show description section at the top of main documentation block, always use real or complex names with name followed by single colon formatted table in describing parameter types, always include main documentation synopsis string after the function or subroutine name, preserve unused functions and subroutines, place description block containing a list of all user-callable routines and functions at the top of the file, using class and classdef with proper documentation indentation, include examples, separate functions by three lines, this file:" --file libid.m --ctx-size 0 | tee output.txt

echo \ | llama-cli -hf unsloth/gpt-oss-120b-GGUF --system-prompt "Include calling sequence examples with options at the top of documentation to all functions. Preserve all other documentation." --file libid.m --ctx-size 0 | tee output.txt

echo \ | llama-cli -hf unsloth/gpt-oss-120b-GGUF --system-prompt "Convert dashes, Unicode and math symbols to use only printable ASCII symbols in documentation or comment lines. Preserve all other documentation and code."  --file libid.m --ctx-size 0 | tee output.txt

echo \ | llama-cli -hf unsloth/gpt-oss-120b-GGUF --system-prompt  "You are a friendly assistant. Must use American English. No em-dashes in comments. No en-dashes in comments. No Unicode symbols in documentation or comment lines. Convert Unicode and math symbols to use only printable ASCII symbols in documentation or comment lines. Convert to python, keeping function ordering, use numbered dot code flow using dash underline above and below, never include any info summary or documentation before function or subroutine name, always include info/documentation after implicit statement after function or subroutine name, document structure fields, document input and output parameters with type information followed with description next lines with section name spelled out with parameters using underline with ASCII dashes, notes, numbered code flow sections, preserve original comments and notes, format documentation available for fortran help, add Your Name placeholder author, TBD placeholder SPDX-License-Identifier, author before spdx, documentation section name should be capitalized but must not be in all caps and should be underlined and should use moderate indentation at least two spaces, always show description section at the top of main documentation block, always use real or complex names with name followed by single colon formatted table in describing parameter types, always include main documentation synopsis string after the function or subroutine name, preserve unused functions and subroutines, place description block containing a list of all user-callable routines and functions at the top of the file, using class and classdef with proper documentation indentation, include examples, separate functions by three lines, preserve top file synopsis, preserve function calling sequence examples, preserve examples, this file:" --file libid.m --ctx-size 0 | tee output.txt


