# librla - Fortran 95 port (experimental)

A Fortran 95 port of librla, the randomized low-rank factorization library,
for real (`double precision`) and complex (`complex(kind(1.0d0))`) matrices,
dense or matrix-free. The interface is a set of external subroutines with
explicit-shape arrays, LAPACK-style, so it can be called from Fortran 77,
Fortran 90/95 and C without module files. It follows the semantics of the
Python reference (`distrib/python/librla.py`, librla 1.2.1). Every dense
factorization goes through LAPACK/BLAS. The port version is returned by
`rla95_version` (defined in `src/librla95.f90`).

| Real | Complex | Description |
|------|---------|-------------|
| `rla_dorth_sketch` | `rla_zorth_sketch` | Approximate orthonormal basis for the column space |
| `rla_dqr_sketch` | `rla_zqr_sketch` | Truncated QR factorization with column pivoting |
| `rla_dsvd_sketch` | `rla_zsvd_sketch` | Truncated singular value decomposition |
| `rla_did_sketch` | `rla_zid_sketch` | Interpolative decomposition via randomized sketching |
| `rla_did_qrpiv` | `rla_zid_qrpiv` | Interpolative decomposition via deterministic pivoted QR |

Each routine has a matrix-free twin with a trailing `f` (`rla_dsvd_sketchf`,
...) that takes a user subroutine applying the operator instead of the matrix.

For a modern interface (allocatable outputs, optional arguments, generic
names, operator types), use the Fortran 2003 port in `distrib/fortran03`;
for the same calls without LAPACK, in any precision (e.g. quad), the Fortran
77 port in `distrib/fortran77`. The
two ports make the same LAPACK calls and agree bit for bit when linked to the
same LAPACK symbols.

## Build and test

Requires a Fortran 95 compiler (tested with gfortran 15) and LAPACK/BLAS.

```bash
cd distrib/fortran95
make test        # build build/librla95.a and run the test suite
make demo        # build and run demo/demo_basic.f90
make clean
```

The code is standard Fortran 95, compiled with `-std=f95 -Wall -Wextra`, and
builds without warnings. `OPT` (default `-O2`) and `FC` (default `gfortran`)
can be overridden.

### LAPACK backend

`LAPACK` holds the link flags. The default is `-framework Accelerate` on macOS
and `-llapack -lblas` elsewhere. To use Homebrew OpenBLAS on macOS:

```bash
make test LAPACK="-L/opt/homebrew/opt/openblas/lib -lopenblas"
```

On Linux with OpenBLAS: `make test LAPACK=-lopenblas`.

LAPACK is called by its standard names (`dgeqp3`, `zgesdd`, ...) through
implicit interfaces, so any LAPACK works: reference LAPACK, OpenBLAS, MKL or
Accelerate. With Accelerate this means the legacy LAPACK 3.2.1 symbols:
Fortran 95 cannot reach Accelerate's faster LAPACK 3.11 interface
(`dgemm$NEWLAPACK`, ...), which the Fortran 2003 port binds with `bind(C)`.
On an M4 Max, a 4000 x 3000 complex rank-200 `svd_sketch` takes 221 ms here
against 155 ms in the Fortran 2003 port, and a 2000 x 1500 `id_qrpiv` 96 ms
against 72 ms; real sketches differ by under 10%. On other backends the two
ports make the same LAPACK calls.

To use the library, link `build/librla95.a` with the same `LAPACK` flags:

```bash
gfortran myprog.f90 path/to/fortran95/build/librla95.a -framework Accelerate
```

Add `-Ipath/to/fortran95/build` only if `myprog.f90` uses the optional
interface module (`use librla95`, see below).

## Usage

```fortran
program example
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: m = 1000, n = 500, kcap = 500
  external :: rla_dsvd_sketch, rla_did_sketch
  double precision :: a(m, n), u(m, kcap), s(kcap), vt(kcap, n), t(kcap, n)
  integer :: piv(n), iseed(4), i, j, k, ier

  do j = 1, n
     do i = 1, m
        a(i, j) = 1.0_dp / (i + j - 1)
     end do
  end do
  iseed = (/ 0, 0, 0, 1 /)

  ! rank 20: a ~ u(:, 1:k) diag(s(1:k)) vt(1:k, :)
  call rla_dsvd_sketch(m, n, a, m, 20.0_dp, 42, 0, 12, iseed, kcap, k, &
       u, m, s, vt, kcap, ier)

  ! tolerance 1e-6, method 3 (lstsq):
  ! a(:, piv(k+1:n)) ~ a(:, piv(1:k)) t(1:k, 1:n-k)
  call rla_did_sketch(m, n, a, m, 1e-6_dp, 42, 0, 12, 3, iseed, kcap, k, &
       piv, t, kcap, ier)
end program example
```

### Modes

- **Tolerance mode** (`rtol < 1`): the rank adapts to the relative tolerance.
- **Rank mode** (`rtol >= 1`): fixed rank `k = floor(rtol)`.

### Calling sequences

Shown for the dense real routines; the complex ones (`rla_z...`) take
`complex(kind(1.0d0))` matrices with the same argument lists (`diagr` and `s`
stay `double precision`).

```fortran
call rla_dorth_sketch(m, n, a, lda, rtol, nb, npow, nextra, iseed, kcap, k, q, ldq, diagr, flag, ier)
call rla_dqr_sketch  (m, n, a, lda, rtol, nb, npow, nextra, iseed, kcap, k, q, ldq, r, ldr, piv, ier)
call rla_dsvd_sketch (m, n, a, lda, rtol, nb, npow, nextra, iseed, kcap, k, u, ldu, s, vt, ldvt, ier)
call rla_did_sketch  (m, n, a, lda, rtol, nb, npow, nextra, method, iseed, kcap, k, piv, t, ldt, ier)
call rla_did_qrpiv   (m, n, a, lda, rtol, method, kcap, k, piv, t, ldt, ier)
```

The matrix-free variants replace `a, lda` by `matvec, ipar, rpar, zpar`:

```fortran
call rla_dsvd_sketchf(m, n, matvec, ipar, rpar, zpar, rtol, nb, npow, nextra, iseed, kcap, k, u, ldu, s, vt, ldvt, ier)
```

Inputs:

| Argument | Description |
|----------|-------------|
| `m`, `n` | Shape of A |
| `a(lda, *)`, `lda` | The matrix, `lda >= max(1, m)` |
| `rtol` | Tolerance (`< 1`) or rank (`>= 1`) |
| `nb` | Initial sketch size in tolerance mode, librla default 42 |
| `npow` | Power iterations, default 0; 1 or 2 help for slowly decaying spectra |
| `nextra` | Buffer beyond the target rank in both modes, default 12; 0 selects the legacy last-column tolerance check |
| `method` | ID only: 1 fast (triangular solve), 2 svd (pseudoinverse of R11), 3 lstsq (least squares on columns of A) |
| `iseed(4)` | LAPACK seed, entries in [0, 4095], `iseed(4)` odd; updated on success |
| `kcap` | Capacity of the outputs: the most columns of `q`/`u`, rows of `r`/`vt`/`t` the caller provides |

Outputs (all 1-based; only the leading `k` columns/rows are written):

| Output | Written part | Meaning |
|--------|--------------|---------|
| `k` | | Rank found; with `ier = 4`, the capacity needed |
| `q`, `diagr`, `flag` (orth) | `q(1:m, 1:k)`, `diagr(1:k)` | orthonormal basis including the `nextra` buffer columns; abs of the diagonal of R of the sketch; `flag` 1 = early termination (tolerance mode only), then `k = 0` |
| `q`, `r`, `piv` (qr) | `q(1:m, 1:k)`, `r(1:k, 1:n)`, `piv(1:n)` | `a(:, piv) ~ q r`, `r` upper triangular |
| `u`, `s`, `vt` (svd) | `u(1:m, 1:k)`, `s(1:k)`, `vt(1:k, 1:n)` | `a ~ u diag(s) vt`, `s` descending, `vt` = V^H |
| `piv`, `t` (id) | `piv(1:n)`, `t(1:k, 1:n-k)` | `a(:, piv(k+1:n)) ~ a(:, piv(1:k)) t` |
| `ier` | | Error code, below |

Leading dimensions: `ldq`, `ldu >= max(1, m)`; `ldr`, `ldvt`, `ldt >= max(1, kcap)`.

**Choosing `kcap`.** `kcap = min(m, n)` is always enough for `qr`, `svd`
and the ID routines, and for `orth` in tolerance mode. `orth` in rank mode
returns `min(m, floor(rtol) + nextra)` columns, which can exceed `n`. If the
result does not fit, the routine returns `ier = 4` with `k` set to the size
needed, writes no outputs and leaves `iseed` unchanged, so the call can be
repeated with larger arrays and gives the same answer.

**Errors (`ier`).** 0 success; 1 invalid `method`; 2 invalid argument
(negative size or option, `lda`/`ldq`/... too small, invalid `iseed`, NaN
`rtol`, or a rank too large for a default integer); 3 LAPACK failure (SVD did
not converge); 4 `kcap` too small. On any error `iseed` is unchanged and no
output array is written. The library never stops the program.

**Working storage** is allocated internally (`allocate`), so there is no
work-array argument. Peak use is a few copies of A for the deterministic
paths, and a few `m x (k + nextra)` blocks for the sketches.

### Random numbers

The test matrices come from LAPACK `dlarnv`/`zlarnv` with `idist = 2`
(uniform(-1, 1); complex: real and imaginary parts each uniform(-1, 1)),
filled column-major in one call, from the caller's `iseed`. A successful call
advances `iseed`, so passing the same array to several calls draws different
sketches; resetting it repeats a call exactly. `iseed = (/ 0, 0, 0, 1 /)`
matches the default of the other experimental ports.

The C++, Rust, Swift and Fortran 2003 ports use the same generator and
LAPACK, so the same seed gives the same sketch in every port; the test suite
prints `CROSSPORT` lines to compare them.

### Matrix-free operators

Write an external subroutine that applies A or its adjoint to a block of
`k` vectors:

```fortran
subroutine matvec(itrans, m, n, k, x, y, ipar, rpar, zpar)
  integer, intent(in) :: itrans, m, n, k
  double precision, intent(in) :: x(*)        ! complex(kind(1.0d0)) for rla_z...f
  double precision, intent(inout) :: y(*)
  integer :: ipar(*)
  double precision :: rpar(*)
  complex(kind(1.0d0)) :: zpar(*)
  ! itrans = 0: y(m, k) = A x(n, k)
  ! itrans = 1: y(n, k) = A^H x(m, k)     (A^T for real A)
end subroutine matvec
```

and pass it, with its parameter arrays, to the `f` routine. `ipar`, `rpar`
and `zpar` reach `matvec` untouched (the library never reads or writes them),
so they can carry the operator's data; pass one-element dummies when unused.
The matrix-free result is the same as the dense one with the same seed, up to
rounding (`demo/demo_basic.f90` has a complete example).

Both modes work for operators, but tolerance mode may fall back to a
deterministic factorization that materializes the operator by applying it to
identity columns (64 columns per `matvec` call, n columns in total).
`rla_*id_qrpivf` and ID method 3 (lstsq) always materialize. Prefer rank mode
for genuinely matrix-free operators.

### Optional interface module

`src/librla95_interfaces.f90` builds `module librla95`, which declares
interfaces for every routine, plus the error codes (`RLA_OK`,
`RLA_ERR_METHOD`, `RLA_ERR_ARG`, `RLA_ERR_LAPACK`, `RLA_ERR_KCAP`) and the
ID methods (`RLA_FAST`, `RLA_SVD`, `RLA_LSTSQ`). `use librla95` makes the
compiler check every call; it is not needed to call the library.

## Differences from Python

- **Explicit seed.** Every call takes `iseed`; there is no hidden RNG state.
  Python's default draws fresh OS entropy.
- **Different random numbers.** Python uses `numpy.random.Generator.uniform`.
  With the same `dlarnv` draws, the Python reference reproduces the
  `CROSSPORT` values (checked to rounding).
- **Errors.** Python raises `ValueError`; here `ier` is set. Negative options
  and NaN `rtol` are rejected (Python does not check them).
- **Outputs.** Caller-provided arrays with a capacity `kcap` instead of
  returned arrays. `piv` is 1-based. `method` is an integer.

## Files

| File | Contents |
|------|----------|
| `src/librla95.f90` | The public external subroutines `rla_*` and `rla95_version` |
| `src/librla95_interfaces.f90` | Optional `module librla95` with their interfaces |
| `src/rla95_core.f90` | Internal module: the algorithms, a real and a complex half |
| `src/rla95_lapack.f90` | Internal module: LAPACK/BLAS wrappers |
| `test/test_all.f90` | Test suite (PASS/FAIL per check, nonzero exit on failure) |
| `demo/demo_basic.f90` | Dense and matrix-free demo, called without the module |
| `tools/generate.py`, `tools/*.f90.in` | Generator and templates for the real/complex halves |

### Generated files

Fortran 95 has no generics over kinds, so every algorithm exists as a real
(`_d`) and a complex (`_z`) routine. They are written once, as templates in
`tools/`, and expanded by `tools/generate.py` (Python 3, no dependencies)
into `src/rla95_core.f90`, `src/librla95.f90`, `src/librla95_interfaces.f90`
and `test/test_all.f90`. These four files are committed, so building needs no
Python, but they should not be edited by hand: change the templates or the
script, then

```bash
make generate                        # or: python3 tools/generate.py
python3 tools/generate.py --check    # exit 1 if a generated file is stale (CI runs this)
```

The internal modules use pointer arrays for variable-size intermediates,
since Fortran 95 has no allocatable dummy arguments; every allocation is
freed before return (checked with the macOS `leaks` tool on the test suite).

## Tests

`make test` runs `build/test_all`. It checks, for real and complex input:
the LAPACK RNG reference values, orth/qr/svd/id in both modes on tall and
wide matrices and through `lda > m`, the deterministic fallbacks (bitwise
against LAPACK's pivoted QR), rank-deficient, zero and k == n ID cases,
argument and capacity errors (with `iseed` left unchanged), matrix-free
operators against the dense path (bitwise for `orth`), low-rank operators
without materialization, power iteration and seed reproducibility. Then it
prints the `CROSSPORT` lines.
