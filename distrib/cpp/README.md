# librla - C++ port (experimental)

Header-only C++17 port of librla, the randomized low-rank factorization
library, for real (`double`) and complex (`std::complex<double>`) matrices,
dense or matrix-free. It tracks the semantics of librla 1.2.1
(`distrib/python/librla.py` is the reference). All dense linear algebra goes
through LAPACK/BLAS (Apple Accelerate, OpenBLAS or a system LAPACK); there are
no other dependencies.

Version: see `librla::version` in `include/librla.hpp` (0.1.0, experimental).

| Function | Description |
|----------|-------------|
| `orth_sketch` | Approximate orthonormal basis for the column space via randomized sketching |
| `qr_sketch` | Truncated QR factorization with column pivoting |
| `svd_sketch` | Truncated singular value decomposition |
| `id_sketch` | Interpolative decomposition via randomized sketching |
| `id_qrpiv` | Interpolative decomposition via deterministic QR pivoting |

All five work in **tolerance mode** (`rtol < 1`, adaptive rank) and **rank
mode** (`rtol >= 1`, rank `floor(rtol)`), on a `librla::Matrix<S>` or a
matrix-free `librla::LinearOperator<S>`.

## Build

Requirements: a C++17 compiler, CMake >= 3.16, LAPACK/BLAS.

```bash
cd distrib/cpp
cmake -S . -B build                      # macOS default: Accelerate
cmake --build build
ctest --test-dir build --output-on-failure   # or: cmake --build build --target check
./build/demo_basic
```

Backend selection (`LIBRLA_LAPACK`):

| Value | Links | Default on |
|-------|-------|------------|
| `Accelerate` | `-framework Accelerate`, new LAPACK interface (see below) | macOS |
| `OpenBLAS` | `libopenblas` under `OPENBLAS_ROOT` (default `/opt/homebrew/opt/openblas` on macOS) | |
| `System` | CMake `FindLAPACK` (e.g. `-llapack -lblas`) | Linux |

```bash
cmake -S . -B build-openblas -DLIBRLA_LAPACK=OpenBLAS
cmake -S . -B build-lapack   -DLIBRLA_LAPACK=System
cmake -S . -B build-legacy   -DLIBRLA_ACCELERATE_LEGACY=ON   # Accelerate, macOS < 13.3
```

### Accelerate: new vs legacy LAPACK interface

Accelerate exports two LAPACK interfaces: the legacy one (LAPACK 3.2.1,
symbols `dgemm_`, ...) and the new LP64 one (LAPACK 3.11, symbols
`dgemm$NEWLAPACK`, ..., macOS 13.3 or later). The new one is markedly
faster for complex `gemm('C', ...)`, `gesdd` and `geqp3`, and it is what
SciPy and Julia use. With `LIBRLA_ACCELERATE_NEWLAPACK` defined,
`librla_lapack.hpp` binds every LAPACK/BLAS declaration to the `$NEWLAPACK`
symbol with an asm label; the argument lists are unchanged.

- The CMake `Accelerate` backend defines `LIBRLA_ACCELERATE_NEWLAPACK`
  (on the `librla` interface target, so dependent targets get it too).
  The binary then requires macOS 13.3 or later.
- `-DLIBRLA_ACCELERATE_LEGACY=ON` keeps the legacy `name_` symbols, for
  older macOS.
- The `OpenBLAS` and `System` backends always use the plain `name_`
  symbols. Defining the macro with them fails to link, because those
  libraries have no `$NEWLAPACK` symbols.

Results agree between the two interfaces to roundoff. The random test
matrices (`dlarnv`/`zlarnv`) are bit-identical.

### Without CMake

The library itself is just the two headers in `include/`. A hand build
without the macro gets the portable legacy `name_` symbols. Add
`-DLIBRLA_ACCELERATE_NEWLAPACK` when linking Accelerate on macOS 13.3 or
later:

```bash
c++ -std=c++17 -O2 -Iinclude -DLIBRLA_ACCELERATE_NEWLAPACK prog.cpp -framework Accelerate  # macOS >= 13.3
c++ -std=c++17 -O2 -Iinclude prog.cpp -framework Accelerate              # legacy interface
c++ -std=c++17 -O2 -Iinclude prog.cpp -L/opt/homebrew/opt/openblas/lib -lopenblas
c++ -std=c++17 -O2 -Iinclude prog.cpp -llapack -lblas                    # Linux
```

To check which interface a binary uses: `nm -u prog | grep NEWLAPACK`.

From another CMake project: `add_subdirectory(path/to/distrib/cpp)` and
`target_link_libraries(app PRIVATE librla::librla)`.

The test and demo targets compile with `-Wall -Wextra -Wpedantic` and build
without warnings (also clean under `-Wshadow -Wconversion`).

## Usage

```cpp
#include "librla.hpp"
using namespace librla;

Matrix<double> A(1000, 500);                 // column-major, zero-initialized
for (int j = 0; j < 500; ++j)
    for (int i = 0; i < 1000; ++i) A(i, j) = 1.0 / (i + j + 1);

auto [Q, flag, diagR] = orth_sketch(A, 1e-6);      // orthonormal basis
auto [Qr, R, p]       = qr_sketch(A, 1e-6);        // A[:, p] ~= Qr R
auto [U, s, Vh]       = svd_sketch(A, 20);         // A ~= U diag(s) Vh, rank 20

Options opts;
opts.method = IdMethod::lstsq;
auto [k, piv, T] = id_sketch(A, 1e-6, opts);       // A[:, piv[k:]] ~= A[:, piv[:k]] T
auto id = id_qrpiv(A, 1e-6, "svd");                // deterministic; id.k, id.piv, id.T
```

Results are plain structs with named fields, so both `auto [Q, R, p] = ...`
and `auto f = ...; f.Q` work.

### Matrix-free operators

A `LinearOperator<S>` holds the shape and two block applies:

```cpp
LinearOperator<double> op(
    m, n,
    [&](const Matrix<double>& X) { return matmul(B, matmul(C, X)); },            // A X,   X: n x k
    [&](const Matrix<double>& W) { return matmul(C, matmul(B, W, 'C'), 'C'); }); // A^H W, W: m x k

auto [U, s, Vh] = svd_sketch(op, 10);        // rank mode: only sketch-sized applies
```

`LinearOperator<S>::from_matrix(A)` wraps a dense matrix and exposes it (the
`dense` field), so librla takes the dense path; `from_matrix(A, false)` hides
it and makes the operator behave as matrix-free. `op.adjoint()` gives `A^H`.
An apply returning a block of the wrong shape throws `std::runtime_error`.

**Performance trap.** Both modes work for matrix-free operators, but when the
tolerance-mode sketch terminates early (`flag = 1`), `qr_sketch`,
`svd_sketch` and `id_sketch` fall back to a deterministic factorization of
the materialized matrix: the operator is applied to the identity, n columns
in blocks of 64. `id_qrpiv` and `method = lstsq` always materialize. Prefer
rank mode for genuinely matrix-free operators.

## API

```cpp
template <class S> OrthResult<S> orth_sketch(const Matrix<S>& A,         double rtol, const Options& = {});
template <class S> OrthResult<S> orth_sketch(const LinearOperator<S>& A, double rtol, const Options& = {});
// qr_sketch -> QRResult<S>, svd_sketch -> SVDResult<S>, id_sketch -> IDResult<S>: same overloads
template <class S> IDResult<S> id_qrpiv(const Matrix<S>& A, double rtol, IdMethod method = IdMethod::fast);
template <class S> IDResult<S> id_qrpiv(const Matrix<S>& A, double rtol, const std::string& method);
// (and the LinearOperator overloads)
```

| Result | Fields |
|--------|--------|
| `OrthResult<S>` | `Q` (m x k), `flag` (0 or 1), `diagR` (`std::vector<double>`, \|diag R\| of the sketch) |
| `QRResult<S>` | `Q` (m x k), `R` (k x n, upper triangular), `p` (`std::vector<int>`, len n) |
| `SVDResult<S>` | `U` (m x k), `s` (`std::vector<double>`, descending), `Vh` (k x n) |
| `IDResult<S>` | `k`, `piv` (`std::vector<int>`, len n), `T` (k x (n-k)) |

`orth_sketch` returns the basis *including* the `extra_samples` buffer
columns (rank mode: `min(m, floor(rtol) + extra_samples)` columns).
`flag = 1` (tolerance mode only) means `rtol` is below machine epsilon or the
sketch would reach `min(m, n)`; `Q` is then m x 0 and `diagR` empty. The other
functions handle `flag = 1` with a deterministic fallback.

### Options

| Field | Default | Description |
|-------|---------|-------------|
| `block_size` | 42 | Initial sketch size in tolerance mode (grows x4) |
| `power_iter` | 0 | Power iterations; 1 or 2 help slowly decaying spectra |
| `extra_samples` | 12 | Buffer beyond the target rank (both modes); 0 = legacy last-column check |
| `rng` | `nullptr` | `Rng*` advanced in place; `nullptr` uses a fresh `Rng{0,0,0,1}` |
| `method` | `IdMethod::fast` | `id_sketch` only: `fast`, `svd` or `lstsq` |

ID methods: `fast` solves `R11 \ R12` (falling back to `svd` when R11 is
near-singular), `svd` uses the pseudoinverse of R11 with threshold
`max(rtol, eps)` (eps in rank mode), `lstsq` solves least squares on the
columns of A (`gelsd`, rcond = eps). `parse_id_method("fast"|"svd"|"lstsq")`
converts strings; an unknown name or enum value throws
`std::invalid_argument`.

### Random numbers

The test matrix is drawn with LAPACK `dlarnv`/`zlarnv` (`idist = 2`:
uniform(-1, 1), complex: real and imaginary parts each uniform(-1, 1)),
filling the n x block_size matrix column-major in one call. `librla::Rng`
wraps the LAPACK state `iseed[4]` (entries in [0, 4095], last one odd):

```cpp
Rng r;                       // {0, 0, 0, 1}
Rng r(1, 2, 3, 5);           // explicit iseed (validated)
Rng r(std::uint64_t{2026});  // iseed = {(s>>36)&4095, (s>>24)&4095, (s>>12)&4095, (s&4095)|1}
r.state();                   // current iseed
```

The C++, Rust, Swift and Fortran ports share this generator, so the same seed
gives the same sketch in all four.

### Dense utilities

`matmul(A, B, op_a = 'N', op_b = 'N')` (gemm, ops `'N'`, `'T'`, `'C'`),
`adjoint(A)`, `take_columns(A, idx, begin = 0, end = -1)` (`A[:, idx[begin:end]]`)
and `norm_fro(A)`. `Matrix<S>` offers `rows()`, `cols()`, `operator()(i, j)`,
`data()`, `col(j)`, `identity(n)`.

## Conventions

- Indices are 0-based: `A[:, p] ~= Q R`, `A[:, piv[k:]] ~= A[:, piv[:k]] T`.
- `svd_sketch` returns `Vh = V^H` (Python/Julia convention).
- Dimensions are `int` (the LAPACK LP64 integer); storage is
  `std::vector<S>`, column-major.
- LAPACK routines: `geqp3` + `orgqr`/`ungqr` (pivoted QR, all columns free),
  `gesdd` with `gesvd` fallback (SVD), `gelsd` (lstsq), `trsm`, `gemm`,
  `larnv`. The prototypes are declared in `include/librla_lapack.hpp` with
  the trailing hidden string-length arguments, so one declaration links
  against Accelerate (either interface) and gfortran-built OpenBLAS.

## Differences from Python

- **Default RNG is deterministic.** Without `opts.rng`, each call uses a fresh
  `Rng{0,0,0,1}`; Python draws from OS entropy. Pass an `Rng*` to get a
  stream that advances across calls, as with a numpy `Generator`.
- **Random stream.** LAPACK `larnv` replaces numpy's generator, so numbers
  differ from Python for a given seed (the algorithm is the same: fed the
  same test matrix, librla.py reproduces the C++ results to roundoff).
- **Option validation.** `block_size < 1`, `power_iter < 0` and
  `extra_samples < 0` throw `std::invalid_argument` (Python accepts some of
  these silently).
- **Materialization.** `get_matrix` applies the operator to blocks of 64
  identity columns instead of one column at a time (identical result), and
  `id_qrpiv` with `method = lstsq` reuses its materialized matrix instead of
  materializing a second time.

## Files

| Path | Contents |
|------|----------|
| `include/librla.hpp` | The library (namespace `librla`) |
| `include/librla_lapack.hpp` | LAPACK/BLAS prototypes |
| `test/test_all.cpp` | Test suite (PASS/FAIL per check, CROSSPORT lines, nonzero exit on failure) |
| `demo/demo_basic.cpp` | Dense and matrix-free demo |

## Tests

`test_all` covers, for real and complex: the `dlarnv`/`zlarnv` reference
values, `orth_sketch` (tolerance/rank modes, buffer, legacy check, flag 1),
`qr_sketch` and `svd_sketch` (both modes, tall and wide, orthonormality,
deterministic fallback), `id_sketch` and `id_qrpiv` (all three methods, both
modes, over-rank, zero matrix, k = n), matrix-free operators (wrapped dense
vs dense path, materializing fallback, low-rank `B (C x)` products), power
iteration, seed reproducibility and invalid arguments. It ends with the
`CROSSPORT` lines compared across the C++, Rust, Swift and Fortran ports.
