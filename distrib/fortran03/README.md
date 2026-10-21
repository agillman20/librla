# librla - Fortran port (experimental)

A Fortran 2003 port of librla, the randomized low-rank factorization
library, for real (`real(dp)`) and complex (`complex(dp)`) matrices, dense or
matrix-free. It follows the semantics of the Python reference
(`distrib/python/librla.py`, librla 1.2.1). Every dense factorization goes
through LAPACK/BLAS. The port version is the `rla_version` constant in
`src/librla.f90`.

| Function | Description |
|----------|-------------|
| `orth_sketch` | Approximate orthonormal basis for the column space |
| `qr_sketch` | Truncated QR factorization with column pivoting |
| `svd_sketch` | Truncated singular value decomposition |
| `id_sketch` | Interpolative decomposition via randomized sketching |
| `id_qrpiv` | Interpolative decomposition via deterministic pivoted QR |

## Build and test

Requires gfortran (tested with GCC 15) and LAPACK/BLAS.

```bash
cd distrib/fortran03
make test        # build build/librla.a and run the test suite
make demo        # build and run demo/demo_basic.f90
make clean
```

The code is standard Fortran 2003 (abstract types with type-bound procedures
for matrix-free operators, allocatable arguments, `bind(C)` for the LAPACK
symbols), compiled with `-std=f2003 -Wall -Wextra`, and builds without
warnings. `OPT` (default `-O2`) and `FC` (default `gfortran`) can be
overridden.

### LAPACK backend

`LAPACK` holds the link flags. The default is `-framework Accelerate` on macOS
and `-llapack -lblas` elsewhere. To use Homebrew OpenBLAS on macOS:

```bash
make test LAPACK="-L/opt/homebrew/opt/openblas/lib -lopenblas"
```

On Linux with OpenBLAS: `make test LAPACK=-lopenblas`.

With Accelerate, librla binds Accelerate's LAPACK 3.11 interface (the
`dgemm$NEWLAPACK`, `zgesdd$NEWLAPACK`, ... symbols, LP64), which **requires
macOS 13.3 or later**. It is markedly faster than the legacy LAPACK 3.2.1
symbols for complex products and factorizations (e.g. a 4000 x 3000 complex
rank-200 `svd_sketch`: 166 ms vs 222 ms on an M4 Max). It is the same
interface SciPy and Julia use. On older macOS, build with the legacy symbols:

```bash
make test NEWLAPACK=0
```

Other backends (OpenBLAS, `-llapack -lblas`) always use the plain Fortran
symbols (`dgemm_` ...). The binding is chosen at compile time by
`src/librla_lapack.F90` (preprocessed; `-DLIBRLA_NEWLAPACK` is added when
`LAPACK` names Accelerate and `NEWLAPACK=1`). Switching `LAPACK` or
`NEWLAPACK` rebuilds the binding and relinks the executables automatically
(stamps in `build/`).

To use librla in your own program, link `build/librla.a` with the same
`LAPACK` flags it was built for and pass `-Ibuild` so the compiler finds
`librla.mod`:

```bash
gfortran -Ipath/to/fortran03/build myprog.f90 path/to/fortran03/build/librla.a \
    -framework Accelerate
```

## Usage

```fortran
program example
  use librla
  implicit none
  real(dp), allocatable :: a(:,:), q(:,:), r(:,:), u(:,:), s(:), vh(:,:), &
       t(:,:), diagr(:)
  integer, allocatable :: piv(:)
  integer :: i, j, k, flag

  allocate(a(1000, 500))
  do j = 1, 500
     do i = 1, 1000
        a(i, j) = 1.0_dp / (i + j - 1)
     end do
  end do

  call orth_sketch(a, 1e-6_dp, q, flag, diagr)     ! basis, tolerance mode
  call qr_sketch(a, 1e-6_dp, q, r, piv)            ! a(:, piv) ~ q r
  call svd_sketch(a, 20.0_dp, u, s, vh)            ! rank 20: a ~ u diag(s) vh
  call id_sketch(a, 1e-6_dp, k, piv, t, method='lstsq')
  ! a(:, piv(k+1:)) ~ a(:, piv(1:k)) t
  call id_qrpiv(a, 1e-6_dp, k, piv, t)
end program example
```

The same calls accept `complex(dp)` arrays and operators.

### Modes

- **Tolerance mode** (`rtol < 1`): the rank adapts to the relative tolerance.
- **Rank mode** (`rtol >= 1`): fixed rank `k = floor(rtol)`.

### API

All routines are subroutines in `module librla`, generic over
`real(dp)`/`complex(dp)` and over a dense array or an operator. Outputs are
allocatable and are (re)allocated by the call. Indices are 1-based.

```fortran
call orth_sketch(a, rtol, q, flag, diagr [, block_size, power_iter, extra_samples, rng, info])
call qr_sketch  (a, rtol, q, r, piv      [, block_size, power_iter, extra_samples, rng, info])
call svd_sketch (a, rtol, u, s, vh       [, block_size, power_iter, extra_samples, rng, info])
call id_sketch  (a, rtol, k, piv, t      [, block_size, power_iter, extra_samples, method, rng, info])
call id_qrpiv   (a, rtol, k, piv, t      [, method, info])
```

| Output | Shape | Meaning |
|--------|-------|---------|
| `q` (orth) | m x k | orthonormal basis, including the `extra_samples` buffer columns |
| `flag` | | 0 success; 1 early termination (tolerance mode only), `q` is m x 0 |
| `diagr` | k, `real(dp)` | abs of the diagonal of R from the pivoted QR of the sketch |
| `q`, `r`, `piv` | m x k, k x n, n | `a(:, piv) ~ q r`, `r` upper triangular |
| `u`, `s`, `vh` | m x k, k, k x n | `a ~ u diag(s) vh`, `s` real and descending, `vh` = V^H |
| `k`, `piv`, `t` | scalar, n, k x (n-k) | `a(:, piv(k+1:)) ~ a(:, piv(1:k)) t` |

`svd_sketch` returns V conjugate-transposed (`vh`), as in Python and Julia.

### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `block_size` | 42 | Initial sketch size in tolerance mode (ignored in rank mode) |
| `power_iter` | 0 | Power iterations; 1 or 2 help for slowly decaying spectra |
| `extra_samples` | 12 | Buffer beyond the target rank in both modes; 0 selects the legacy last-column tolerance check |
| `method` | `'fast'` | ID only: `'fast'` (triangular solve), `'svd'` (pseudoinverse of R11), `'lstsq'` (least squares on columns of A) |
| `rng` | `[0,0,0,1]` | `type(rla_rng)`, advanced in place |
| `info` | | Error code; if absent, errors print a message and `error stop` |

`info` values: `RLA_OK` (0), `RLA_ERR_METHOD` (1, method not `'fast'`,
`'svd'` or `'lstsq'`), `RLA_ERR_ARG` (2, negative option, negative operator
size, NaN `rtol`, or a rank too large for a default integer), `RLA_ERR_LAPACK`
(3, SVD or least squares failed to converge). `rla_errmsg(info)` returns the
text. On error the outputs are allocated with zero size.

### Random numbers

The test matrices come from LAPACK `dlarnv`/`zlarnv` with `idist = 2`
(uniform(-1, 1); complex: real and imaginary parts each uniform(-1, 1)), filled
column-major in one call. `type(rla_rng)` holds the LAPACK `iseed(4)`:

```fortran
type(rla_rng) :: rng
rng = rla_rng(1, 2, 3, 5)        ! iseed: entries in [0, 4095], last one odd
rng = rla_rng(12345_int64)       ! iseed = [s>>36, s>>24, s>>12, s|1] & 4095
print *, rng%state()             ! current iseed(4)
call rla_uniform(rng, x)         ! fill a real/complex 2-D array
```

An invalid four-integer seed is an error (`rla_rng(1, 2, 3, 4, info)` returns
`info = RLA_ERR_ARG`). A default-initialized `rla_rng` holds `[0, 0, 0, 1]`.
Passing the same `rng` to several calls advances it, so the calls draw
different sketches.

The C++, Rust and Swift ports use the same generator and LAPACK, so the same
seed gives the same sketch in every port; the test suite prints `CROSSPORT`
lines to compare them.

### Matrix-free operators

Extend `rla_operator_r` (or `rla_operator_c`), set the shape components `m`,
`n`, and implement block applies:

```fortran
type, extends(rla_operator_r) :: product_op
   real(dp), allocatable :: b(:,:), c(:,:)      ! A = B C, never formed
contains
   procedure :: apply => product_apply           ! y = A x,   x: n x k, y: m x k
   procedure :: apply_adjoint => product_adjoint ! z = A^H w, w: m x k, z: n x k
end type

subroutine product_apply(self, x, y)
  class(product_op), intent(in) :: self
  real(dp), intent(in) :: x(:,:)
  real(dp), intent(out) :: y(:,:)
  y = matmul(self%b, matmul(self%c, x))
end subroutine
```

Then `call svd_sketch(op, 12.0_dp, u, s, vh)` works as for a dense array. If
the operator has an explicit matrix, override the type-bound function
`dense()` to return a pointer to it; librla then multiplies by the matrix
directly and skips materialization. `call rla_get_matrix(op, a)` gives a dense
copy of any operator.

Both modes work for operators, but tolerance mode may fall back to a
deterministic factorization that materializes the operator by applying it to
identity columns (64 columns per `apply` call, n columns in total).
`id_qrpiv` and `method='lstsq'` always materialize. Prefer rank mode for
genuinely matrix-free operators.

## Differences from Python

- **Deterministic default RNG.** Without an `rng` argument every call starts
  from `iseed = [0, 0, 0, 1]`, so repeated calls give identical results.
  Python's default draws fresh OS entropy.
- **Different random numbers.** Python uses `numpy.random.Generator.uniform`.
  With the same `dlarnv` draws, the Python reference reproduces the
  `CROSSPORT` values (checked to rounding).
- **Errors.** Python raises `ValueError`; here `info` is set, or the program
  stops when `info` is absent. Negative options and NaN `rtol` are rejected
  (Python does not check them).
- **Outputs.** Subroutine arguments instead of returned tuples. `piv` is
  1-based. `diagr` and `s` are always `real(dp)`.

## Files

| File | Contents |
|------|----------|
| `src/librla.f90` | `module librla`: the public API |
| `src/librla_lapack.F90` | `module librla_lapack`: explicit LAPACK/BLAS interfaces (plain or `$NEWLAPACK` binding) and `la_*` wrappers |
| `test/test_all.f90` | Test suite (PASS/FAIL per check, nonzero exit on failure) |
| `demo/demo_basic.f90` | Dense and matrix-free demo |

## Tests

`make test` runs `build/test_all`. It checks, for real and complex input:
the RNG reference values, orth/qr/svd/id in both modes on tall and wide
matrices, the deterministic fallbacks, rank-deficient, zero and k == n ID
cases, matrix-free operators against the dense path, power iteration, seed
reproducibility and the invalid-method error. Then it prints the `CROSSPORT`
lines.
