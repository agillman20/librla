# librla - Fortran 77 port (experimental)

A self-contained Fortran 77 port of librla, the randomized low-rank
factorization library, for real (`real *8`) and complex (`complex *16`)
matrices, dense or matrix-free. It needs no BLAS or LAPACK: one source file,
`src/librla77.f`, carries its own pivoted Householder QR, one-sided Jacobi
SVD, least squares and random number generator. That makes it easy to vendor
into an existing code and lets it be compiled in **any precision**: with
`gfortran -freal-8-real-16 -fdefault-integer-8` the whole library runs in
quad precision with 64-bit integers.

It follows the semantics of the Python reference (`distrib/python/librla.py`,
librla 1.2.1). The calling sequences are those of the Fortran 95 port
(`distrib/fortran95`) plus a caller-supplied work array. The port version is
returned by `rla77_version` (defined in `tools/generate.py`).

| Real | Complex | Description |
|------|---------|-------------|
| `rla_dorth_sketch` | `rla_zorth_sketch` | Approximate orthonormal basis for the column space |
| `rla_dqr_sketch` | `rla_zqr_sketch` | Truncated QR factorization with column pivoting |
| `rla_dsvd_sketch` | `rla_zsvd_sketch` | Truncated singular value decomposition |
| `rla_did_sketch` | `rla_zid_sketch` | Interpolative decomposition via randomized sketching |
| `rla_did_qrpiv` | `rla_zid_qrpiv` | Interpolative decomposition via deterministic pivoted QR |

Each routine has a matrix-free twin with a trailing `f` (`rla_dsvd_sketchf`,
...) that takes a user subroutine applying the operator instead of the matrix.
The dense kernels behind them are public too: `rla_dsvd` (truncated SVD of
any shape), `rla_dqrcp` (pivoted QR with early exit, after LAPACK
`dgeqp3rk`), `rla_dqrf` (QR without pivoting) and `rla_dorgqr` (form Q),
with `rla_z` counterparts; see [Dense utilities](#dense-utilities).

Which Fortran port to use:

| Port | Language | Linear algebra | Interface |
|------|----------|----------------|-----------|
| `distrib/fortran03` | Fortran 2003 module | LAPACK | allocatable outputs, optional arguments, operator types |
| `distrib/fortran95` | Fortran 95 | LAPACK | external subroutines, outputs sized by `kcap` |
| `distrib/fortran77` | Fortran 77 | built in | the Fortran 95 calls plus a work array; any precision |

## Build and test

Requires a Fortran compiler that accepts `real *8` and `complex *16` (every
current one does; tested with gfortran 15). There is nothing to link.

```bash
cd distrib/fortran77
make test        # build build/librla77.a and run the test suite
make test-quad   # the same suite promoted to quad precision (a few minutes)
make demo        # build and run demo/demo_basic.f
make clean
```

The code is fixed-form Fortran 77 in the `implicit real *8 (a-h,o-z)`
style, compiled with `-std=legacy -Wall` (`real *8` and `complex *16` are
extensions to the 1978 standard). The library and demo build without
warnings, also under `-Wextra` and with the quad flags. Intrinsics are the double-precision
specific ones (`dble`, `dcmplx`, `dconjg`, `dimag`), which gfortran promotes
together with the variables; `cmplx` without a kind is never used.

To use the library, add `src/librla77.f` to your sources (or link
`build/librla77.a`). For quad precision, compile it with the same promotion
flags as the calling code.

## Usage

```fortran
        implicit real *8 (a-h,o-z)
        parameter (m=1000,n=500,kcap=500,lw=2000000)
        dimension a(m,n),u(m,kcap),s(kcap),vt(kcap,n),w(lw)
        dimension iseed(4)
c
        iseed(1)=0
        iseed(2)=0
        iseed(3)=0
        iseed(4)=1
c
c       rank 20: a ~ u(:,1:k) diag(s(1:k)) vt(1:k,:)
c
        call rla_dsvd_sketch(m,n,a,m,20.0d0,42,0,12,iseed,kcap,k,
     1      u,m,s,vt,kcap,w,lw,ier)
```

### Calling sequences

Shown for the dense real routines; the complex ones (`rla_z...`) take
`complex *16` matrices and work arrays with the same argument lists (`diagr`
and `s` stay `real *8`).

```fortran
call rla_dorth_sketch(m,n,a,lda,rtol,nb,npow,nextra,iseed,kcap,k,q,ldq,diagr,flag,w,lw,ier)
call rla_dqr_sketch  (m,n,a,lda,rtol,nb,npow,nextra,iseed,kcap,k,q,ldq,r,ldr,ipiv,w,lw,ier)
call rla_dsvd_sketch (m,n,a,lda,rtol,nb,npow,nextra,iseed,kcap,k,u,ldu,s,vt,ldvt,w,lw,ier)
call rla_did_sketch  (m,n,a,lda,rtol,nb,npow,nextra,method,iseed,kcap,k,ipiv,t,ldt,w,lw,ier)
call rla_did_qrpiv   (m,n,a,lda,rtol,method,kcap,k,ipiv,t,ldt,w,lw,ier)
```

The matrix-free variants replace `a,lda` by `matvec,ipar,rpar,zpar`, exactly
as in the Fortran 95 port:

```fortran
call rla_dsvd_sketchf(m,n,matvec,ipar,rpar,zpar,rtol,nb,npow,nextra,iseed,kcap,k,u,ldu,s,vt,ldvt,w,lw,ier)
```

All arguments, outputs, `kcap` and the `ier` codes are as documented in
`distrib/fortran95/README.md`: tolerance mode for `rtol < 1`, rank mode
`k = floor(rtol)` for `rtol >= 1`; defaults `nb = 42`, `npow = 0`,
`nextra = 12`; ID `method` 1 (fast), 2 (svd), 3 (lstsq); 1-based `ipiv`;
`iseed(4)` with entries in [0, 4095], `iseed(4)` odd, updated only on
success. Two differences:

- **Work array.** `w(lw)` has the type of the matrix (`real *8` or
  `complex *16`). Call with `lw = -1` to get the size needed in `w(1)`
  (nothing else is computed). A smaller `lw` returns **`ier = 5`** with the
  size needed in `w(1)` (when `lw >= 1`). The size depends only on the shape,
  the mode, `nb`, `nextra` and the ID method, not on the data, so one query
  serves repeated calls. Rank mode needs about `(2n + m)(k + nextra)` plus
  the small factorizations (0.2 m n for a rank-200 `svd_sketch` of a
  4000 x 3000 matrix). Tolerance mode must cover the largest sketch it may
  draw before falling back (just under min(m,n) columns) and the fallback
  itself, so it asks for about 2-3.5 m n; for the demo's 1000 x 500
  matrix that is 1.1M scalars for `svd_sketch` and 1.3M for `id_sketch`. The
  phases share memory, and the fallback and the least-squares stage factor
  their copy of A in place.
- **`ier = 3`** means the Jacobi SVD did not converge in 60 sweeps (not
  seen in the tests).

### Matrix-free operators

```fortran
        subroutine matvec(itrans,m,n,k,x,y,ipar,rpar,zpar)
        implicit real *8 (a-h,o-z)
        dimension x(*),y(*),ipar(*),rpar(*)
        complex *16 zpar(*)
c       itrans=0: y(m,k) = A x(n,k)
c       itrans=1: y(n,k) = A^T x(m,k)       (A^H for complex A)
c       x and y are complex *16 for the rla_z...f routines
```

`ipar`, `rpar`, `zpar` reach `matvec` untouched and can carry the
operator's data (`demo/demo_basic.f` stores the two factors of a low-rank
product in `rpar`). Tolerance mode may materialize the operator for the
deterministic fallback (64 columns per call), as may `rla_*id_qrpivf` and ID
method 3; prefer rank mode for genuinely matrix-free operators.

### Random numbers

`rla_drandn`/`rla_zrandn` reproduce LAPACK `dlarnv`/`zlarnv` with
`idist = 2` exactly: the generator is LAPACK's 48-bit linear congruential
`dlaruv` (multiplier 33952834046453, carried in base-4096 integer digits), so
the stream is the same in double and in quad precision. The same `iseed`
therefore draws the same sketch as in the C++, Rust, Swift, Fortran 2003 and
Fortran 95 ports. The test suite prints the `CROSSPORT` lines: ranks and
pivots match those ports exactly, values to about 1e-11 (a different QR and
SVD).

### Linear algebra

| LAPACK in the other ports | Here |
|---------------------------|------|
| `dgeqp3` / `zgeqp3` | Householder QR with column pivoting and norm downdating (the unblocked `dlaqp2` algorithm, same reflector convention) |
| `dorgqr` / `zungqr` | backward accumulation of the reflectors (`dorg2r`) |
| `dgesdd` / `zgesdd` | preconditioned one-sided Jacobi (Drmac, as in LAPACK `dgejsv`): pivoted QR of the matrix or its adjoint, whichever is tall, then Jacobi on the triangle `R^H`; zero singular values get orthonormal completions |
| `dgelsd` / `zgelsd` | the Jacobi SVD with `rcond = eps` |
| `dgemm`, `dtrsm` | plain loops |

All loops are plain and unblocked, so large problems run much slower than
the LAPACK ports. On one core of a Xeon W-2145, against the Fortran 95 port
on single-threaded OpenBLAS, for a 4000 x 3000 matrix:

| Call | Fortran 95 + OpenBLAS | Fortran 77 |
|------|----------------------:|-----------:|
| `svd_sketch`, rtol 1e-10 (k = 22) | 54 ms | 0.76 s |
| `svd_sketch`, rank 200 | 0.27 s | 4.5 s |
| `svd_sketch`, rank 200, complex | 0.77 s | 7.8 s |
| `qr_sketch`, rank 200 | 0.25 s | 4.2 s |
| `id_sketch`, rtol 1e-10 (k = 22) | 52 ms | 0.78 s |
| `id_qrpiv`, 2000 x 1500 | 0.83 s | 2.5 s |

Most of the time goes to the products with A (`A Omega`, `Q^H A`), which
LAPACK-backed BLAS runs 15-20 times faster; the pivoted QR is within 3x.
In quad precision everything runs about 100 times slower again (software
`real *16`).

## Dense utilities

```fortran
call rla_dsvd(m,n,a,lda,rtol,atol,kmax,iexit,k,s,u,ldu,vt,ldvt,w,lw,ier)
call rla_dqrcp(m,n,kmax,abstol,reltol,a,lda,k,rmaxk,relmaxk,ipiv,tau,w,ier)
call rla_dqrf(m,n,a,lda,tau,work)
call rla_dorgqr(m,k,a,lda,tau,work)
```

**`rla_dsvd`**: truncated SVD of any `m x n` matrix (`a` is not modified):
`a ~ u (m x k) diag(s) vt (k x n)`, keeping the singular values
`s_j >= max(rtol*s_1, atol)`, at most `kmax` of them. `rtol = atol = 0`,
`kmax = min(m,n)`, `iexit = 0` is the full SVD. The method is preconditioned
one-sided Jacobi (Drmac, as in LAPACK `dgejsv`): a pivoted QR of `a` or
`a^T` (whichever is tall), then Jacobi on the triangle `R^T`. `iexit`
decides where that QR may stop early, dropping the trailing block `R22`:

| `iexit` | QR stops when | Accuracy | Fast for |
|---------|---------------|----------|----------|
| 0 | never | full SVD; zero singular values get orthonormal vectors | - |
| 1 | the residual columns are rounding noise: `\|R22\|_F <= 10 sqrt(min(m,n)) eps \|A\|` | as the full SVD | matrices low-rank to working precision |
| 2 | `\|R22\|_F <= 0.01 max(rtol s_1, atol)` | kept values move by at most 1% of the threshold (absolute), so the count is exact | the rank at the requested tolerance |
| 3 | LAPACK `dgeqp3rk`'s rule: largest residual column norm `<= max(rtol max_j \|a_j\|, atol)` | values near the threshold may move by up to about `sqrt(min(m,n))` times it, so the count there is approximate; error 1.2-1.3 times the optimal truncation in the tests | the rank at the tolerance, fastest |

`kmax` only caps the output (stopping a pivoted QR at `kmax` columns does
not give the best rank-`kmax` SVD). `w(lw)`: `lw = -1` returns the size in
`w(1)` (`max(m,n) k + 2 k^2 + 6 k + max(m,n)`, `k = min(m,n)`). `ier`: 0,
2 (invalid argument), 3 (no convergence), 5 (`lw` too small).

**`rla_dqrcp`**: Householder QR with column pivoting and early exit, after
LAPACK 3.12 `dgeqp3rk` (without its right-hand sides): it stops with `k`
columns when `k = kmax`, or the largest column norm of the residual `R22` is
`<= abstol`, or `<= reltol` times the largest column norm of `a`; a negative
tolerance disables its test, `kmax >= min(m,n)` removes the limit. On exit
`a(:,ipiv) = Q(:,1:k) R(1:k,:) + [0 R22]`, with `R` in the upper triangle of
rows `1:k`, the reflectors below it, `R22` in `a(k+1:m,k+1:n)`;
`rmaxk` = largest column norm of `R22`, `relmaxk = rmaxk / max column
norm of a`. `w` holds `4n` scalars. With no limits it is bit for bit the
pivoted QR used inside the library (LAPACK `dlaqp2` algorithm).

**`rla_dqrf`**: Householder QR without pivoting (LAPACK `dgeqr2`
conventions), `R` in the upper triangle, reflectors `H(i) = I - tau(i) v v^T`
below it; `tau (min(m,n))`, `work (n)`. **`rla_dorgqr`** overwrites the
reflectors with `Q(:,1:k)`, `k <= min(m,n)`; it also forms `Q(:,1:k)` after
`rla_dqrcp`.

The complex versions `rla_zsvd`, `rla_zqrcp`, `rla_zqrf` and `rla_zorgqr`
take `complex *16` arrays (`s`, the tolerances and norms stay `real *8`).

They replace the old self-contained port's dense routines:

| Old (`feat/fortran` librla.f) | Here |
|-------------------------------|------|
| `rla_svd(a,m,n,u,ldu,s,vt,ldvt,krank,w,lw,ier)` | `rla_dsvd(m,n,a,m,0d0,0d0,min(m,n),0,krank,s,u,ldu,vt,ldvt,w,lw,ier)` |
| `rla_qrp(a,lda,m,n,tau,ipiv,1)` | `rla_dqrcp(m,n,min(m,n),-1d0,-1d0,a,lda,k,rk,rrel,ipiv,tau,w,ier)` |
| `rla_qrp(a,lda,m,n,tau,ipiv,0)` (unpivoted) | `rla_dqrf(m,n,a,lda,tau,work)` |
| `rla_fmq(a,lda,m,n,k,tau)` | `rla_dorgqr(m,k,a,lda,tau,work)` |

Measurements against the old routines and against `svdpivot`/`csvdpiv`
(Rokhlin's `~/develop/lib`) on graded, column-scaled (to 1e-30),
rank-deficient, Kahan and duplicate-column matrices, double precision,
references from a quad-precision run:

- Accuracy: residuals and singular values to rounding everywhere; U and V
  orthonormal to 1e-14 - 1e-13 (`svdpivot`: 1e-12); scale-invariant
  (`svdpivot` breaks down on a matrix scaled by 1e-12: residual 0.1); on the
  column-scaled matrix all 120 singular values down to 5e-30 are correct to
  4e-15 relative.
- Time, full SVD (`iexit = 0`): 300 x 200 39 ms (old `rla_svd` 238 ms,
  `svdpivot` 55 ms); 2000 x 200 137 ms (1446, 223); 400 x 400 0.33 s
  (1.59, 0.40).
- Time, truncated: rank 30 of 100 with `iexit = 1` 0.7 ms (full 20 ms,
  `svdpivot` 0.9 ms). rtol 1e-8 on 400 x 400: `iexit = 2` 0.20 s,
  `iexit = 3` 0.11 s, `svdpivot` with eps 1e-8 0.10 s (residuals 1e-8,
  1e-8 and 8e-8; `svdpivot` misses 10 of the 200 values >= 1e-8). On
  2000 x 200: 89, 64 and 89 ms.

## Differences from Python

- **Explicit seed and workspace.** Every call takes `iseed` and `w`; there
  is no hidden state and no allocation.
- **Different random numbers.** Python uses `numpy.random.Generator.uniform`.
  With the same `dlarnv` draws, the Python reference reproduces the
  `CROSSPORT` values (checked to rounding).
- **Errors.** Python raises `ValueError`; here `ier` is set, and the library
  never stops the program.
- **Outputs.** Caller-provided arrays with a capacity `kcap`. `ipiv` is
  1-based. `method` is an integer.

## Files

| File | Contents |
|------|----------|
| `src/librla77.f` | The library: public routines, sketching drivers, dense kernels, generator |
| `test/test_all.f` | Test suite (PASS/FAIL per check, nonzero exit on failure) |
| `demo/demo_basic.f` | Dense and matrix-free demo |
| `tools/generate.py` | Generator of the real/complex halves |
| `tools/kernels.f.in`, `tools/drivers.f.in` | Library templates |
| `tools/test_kind.f.in`, `tools/test_harness.f` | Test templates |

### Generated files

Fortran 77 has no generics, so every routine exists as a real (`rla_d`) and
a complex (`rla_z`) version. They are written once, as templates in `tools/`,
and expanded by `tools/generate.py` (Python 3, no dependencies) into
`src/librla77.f` and `test/test_all.f`, wrapping statements at column 72.
The generated files are committed, so building needs no Python, but they
should not be edited by hand:

```bash
make generate                        # or: python3 tools/generate.py
python3 tools/generate.py --check    # exit 1 if a generated file is stale (CI runs this)
```

## Tests

`make test` runs `build/test_all`; `make test-quad` runs the same suite
promoted to quad precision. It checks, for real and complex input: the
generator against the LAPACK reference values, the workspace query and the
`lw` error, orth/qr/svd/id in both modes on tall and wide matrices and
through `lda > m`, the deterministic fallbacks (bitwise against the QR
kernel), rank-deficient, zero and k == n ID cases, argument and capacity
errors (with `iseed` left unchanged), matrix-free operators against the
dense path (bitwise for `orth`), low-rank operators without
materialization, power iteration and seed reproducibility. It also runs
every routine, dense and matrix-free, in both modes on tall, wide and
full-rank input with `lw` exactly the queried size and checks that nothing
is written past it. Then it prints the `CROSSPORT` lines.
