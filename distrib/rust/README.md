# librla (Rust) - Randomized Linear Algebra Library

Experimental Rust port of librla: randomized algorithms for low-rank
factorizations of real (`f64`) and complex (`Complex64`) matrices, dense or
matrix-free. It follows the semantics of the Python reference
(`distrib/python/librla.py`, librla 1.2.1). Every dense factorization goes
through LAPACK/BLAS, called through hand-written FFI declarations; the only
crate dependency is `num-complex`.

| Function | Description |
|----------|-------------|
| `orth_sketch` | Approximate orthonormal basis for the column space |
| `qr_sketch` | Truncated QR factorization with column pivoting |
| `svd_sketch` | Truncated singular value decomposition |
| `id_sketch` | Interpolative decomposition via randomized sketching |
| `id_qrpiv` | Interpolative decomposition via deterministic pivoted QR |

All five support **tolerance mode** (`rtol < 1`, adaptive rank) and
**rank mode** (`rtol >= 1`, target rank `floor(rtol)`).

## Build and test

Requires a Rust toolchain (edition 2021) and a LAPACK/BLAS library.

```bash
cd distrib/rust
cargo build --release
cargo test                                  # all checks, cargo's report
cargo test -- --nocapture                   # also print one PASS/FAIL line per check
cargo run --example crossport               # the five CROSSPORT lines
cargo run --release --example demo          # basic demo (dense + matrix-free)
```

The test suite (`tests/`) covers, for both `f64` and `Complex64`: the shared
RNG reference values, `orth_sketch` (tolerance and rank mode, the
`extra_samples` buffer, `extra_samples = 0`, early termination),
`qr_sketch` and `svd_sketch` (both modes, tall and wide, the deterministic
fallback), `id_sketch` / `id_qrpiv` (all three methods, both modes,
rank-deficient input, zero matrix, `k == n`), matrix-free operators (a dense
matrix hidden behind the operator interface matches the dense path; a
genuinely matrix-free low-rank product), power iteration, seed
reproducibility and invalid method names. `tests/crossport.rs` prints the
CROSSPORT lines (visible with `--nocapture`) and checks them against
reference values.

The code builds without warnings at rustc's default lint level and is
clean under `cargo clippy --all-targets` (default lints).

## Backend selection

The LAPACK/BLAS backend is chosen with a cargo feature (select at most one):

| Feature | Links | Default on |
|---------|-------|------------|
| `accelerate` | `-framework Accelerate`, new LAPACK interface (`<name>$NEWLAPACK`, LAPACK 3.11), macOS 13.3+ | macOS |
| `accelerate-legacy` | `-framework Accelerate`, legacy `<name>_` symbols, for macOS before 13.3 | |
| `openblas` | `-lopenblas`, from `$OPENBLAS_DIR/lib` (macOS fallback: `/opt/homebrew/opt/openblas/lib`) | |
| `system` | `-llapack -lblas` | Linux and others |

```bash
cargo test --no-default-features --features openblas
OPENBLAS_DIR=/usr/local/opt/openblas cargo test --features openblas
cargo test --features system            # reference LAPACK, e.g. on Linux
```

With `accelerate`, every routine binds to Accelerate's new LAPACK interface
(the one SciPy and Julia use), which is markedly faster than the legacy
symbols for the complex routines (e.g. `zgemm('C','N')` 23 vs 43 ms,
`zgesdd` 37 vs 50 ms at svd_sketch sizes on an M4 Max); it requires macOS
13.3 or later. On older macOS use
`cargo build --no-default-features --features accelerate-legacy`.

The interface is LP64 (32-bit LAPACK integers). Routines with CHARACTER
arguments receive the trailing hidden string lengths that gfortran-built
libraries expect; Accelerate's legacy interface ignores them.

## Usage

```rust
use librla::{id_sketch, orth_sketch, qr_sketch, svd_sketch, Mat, Method, Options, Rng};

// 1000 x 500 Hilbert matrix (column-major Mat)
let a = Mat::<f64>::from_fn(1000, 500, |i, j| 1.0 / (i + j + 1) as f64);

// Orthonormal basis to relative tolerance 1e-6
let o = orth_sketch(&a, 1e-6, Options::default())?;      // o.q, o.flag, o.diag_r

// Truncated pivoted QR: A[:, p] ~= Q R
let qr = qr_sketch(&a, 1e-6, Options::default())?;       // qr.q, qr.r, qr.p

// Rank-20 SVD with one power iteration: A ~= U diag(s) Vh
let svd = svd_sketch(&a, 20.0, Options::new().power_iter(1))?;
let rec = svd.u.scale_columns(&svd.s).matmul(&svd.vh);

// ID with a seeded RNG: A[:, piv[k..]] ~= A[:, piv[..k]] T
let mut rng = Rng::from_seed(42);
let id = id_sketch(&a, 1e-6, Options::new().method(Method::Lstsq).rng(&mut rng))?;
let err = (&a.select_columns(&id.piv[id.k..])
    - &a.select_columns(&id.piv[..id.k]).matmul(&id.t)).norm_fro();
```

### Matrix-free operators

Implement `LinearOperator<T>` (block products `A X` and `A^H W`), or wrap two
closures in `FnOperator`:

```rust
use librla::{svd_sketch, Complex64, FnOperator, LinearOperator, Mat, Options};

struct LowRank { b: Mat<Complex64>, c: Mat<Complex64> }   // A = B C

impl LinearOperator<Complex64> for LowRank {
    fn shape(&self) -> (usize, usize) { (self.b.nrows(), self.c.ncols()) }
    fn apply(&self, x: &Mat<Complex64>) -> Mat<Complex64> { self.b.matmul(&self.c.matmul(x)) }
    fn apply_adjoint(&self, w: &Mat<Complex64>) -> Mat<Complex64> {
        self.c.adjoint_matmul(&self.b.adjoint_matmul(w))
    }
}

let svd = svd_sketch(&op, 10.0, Options::default())?;   // op: LowRank
let f = FnOperator::new(m, n, |x: &Mat<f64>| /* A x */, |w: &Mat<f64>| /* A^H w */);
```

Matrix-free operators support both modes. In tolerance mode, when the sketch
terminates early, the deterministic fallback (and `id_qrpiv`, and
`Method::Lstsq`) materializes the operator by applying it to identity
columns (blocks of 64 columns per `apply` call, O(n) matvecs in total).
Prefer rank mode when that cost is unacceptable. `Mat<T>` implements
`LinearOperator<T>` and returns itself from `as_dense()`, which lets librla
use it directly; operators keep the default `None`.

## API

```rust
pub fn orth_sketch<T, A>(a: &A, rtol: f64, opts: Options) -> Result<OrthSketch<T>, Error>;
pub fn qr_sketch<T, A>(a: &A, rtol: f64, opts: Options) -> Result<QrSketch<T>, Error>;
pub fn svd_sketch<T, A>(a: &A, rtol: f64, opts: Options) -> Result<SvdSketch<T>, Error>;
pub fn id_sketch<T, A>(a: &A, rtol: f64, opts: Options) -> Result<IdDecomp<T>, Error>;
pub fn id_qrpiv<T, A>(a: &A, rtol: f64, method: Method) -> Result<IdDecomp<T>, Error>;
// T: Scalar (f64 or Complex64), A: LinearOperator<T> + ?Sized
```

| Result | Fields | Identity |
|--------|--------|----------|
| `OrthSketch<T>` | `q` (m x k), `flag` (0 / 1), `diag_r` (`Vec<f64>`) | |
| `QrSketch<T>` | `q` (m x k), `r` (k x n, upper triangular), `p` (len n) | `A[:, p] ~= Q R` |
| `SvdSketch<T>` | `u` (m x k), `s` (`Vec<f64>`, descending), `vh` (k x n) | `A ~= U diag(s) Vh` |
| `IdDecomp<T>` | `k`, `piv` (len n), `t` (k x (n-k)) | `A[:, piv[k..]] ~= A[:, piv[..k]] T` |

Indices are 0-based. `svd_sketch` returns V conjugate-transposed (`vh`), as
in Python and Julia. `orth_sketch` returns the basis with the
`extra_samples` buffer columns; `flag = 1` (tolerance mode only) means
`rtol < eps` or the sketch would reach `min(m, n)` columns, and then `q` is
`m x 0` and `diag_r` is empty. `qr_sketch`, `svd_sketch` and `id_sketch`
handle `flag = 1` themselves by falling back to a deterministic
factorization.

Supporting types:

| Type | Purpose |
|------|---------|
| `Mat<T>` | Column-major dense matrix: `zeros`, `identity`, `from_fn`, `from_col_major`, `matmul`, `adjoint_matmul`, `adjoint`, `select_columns`, `columns`, `rows`, `scale_columns`, `norm_fro`, indexing `a[(i, j)]`, `&a * &b`, `&a - &b` |
| `LinearOperator<T>` | `shape`, `apply`, `apply_adjoint`, `as_dense` (default `None`) |
| `FnOperator` | Matrix-free operator from two closures |
| `Rng` | LAPACK larnv seed: `Rng::new(i0, i1, i2, i3)?`, `Rng::from_seed(u64)`, `Rng::default()`, `iseed()`, `uniform::<T>(m, n)`, `fill(&mut [T])` |
| `Method` | `Fast` (default), `Svd`, `Lstsq`; `"lstsq".parse::<Method>()?` |
| `Error` | `InvalidMethod`, `InvalidSeed`, `InvalidArgument`, `ShapeMismatch`, `Lapack` |
| `dense::qr_pivoted`, `dense::svd` | The LAPACK factorizations librla uses (geqp3 + orgqr, gesdd), for reference computations |

## Options

`Options` implements `Default`; set fields with struct-update syntax
(`Options { power_iter: 2, ..Default::default() }`) or the builder methods
(`Options::new().power_iter(2).rng(&mut rng)`).

| Field | Default | Description |
|-------|---------|-------------|
| `block_size` | 42 | Initial sketch size in tolerance mode (ignored in rank mode) |
| `power_iter` | 0 | Power iterations `x <- orth(A^H A x)`; 1 or 2 help slowly decaying spectra |
| `extra_samples` | 12 | Buffer beyond the target rank: rank mode samples `floor(rtol) + extra_samples` columns; tolerance mode accepts a sketch only when at least `extra_samples + 1` pivoted column norms are at or below `rtol` times the largest; `0` is the legacy last-column test |
| `method` | `Method::Fast` | ID interpolation matrix (`id_sketch` only; `id_qrpiv` takes it as an argument) |
| `rng` | `None` | `Option<&mut Rng>`, advanced in place; `None` uses a fresh `Rng::default()` |

ID methods: `Fast` solves `R11 \ R12` by triangular solve (falling back to
`Svd` when `R11` is numerically singular), `Svd` uses the SVD-based
pseudoinverse of `R11`, `Lstsq` solves a least-squares problem on the
columns of `A` (most accurate, slowest).

## Random numbers and the cross-port check

Test matrices are drawn with LAPACK `dlarnv` / `zlarnv` (`idist = 2`,
uniform on (-1, 1); complex: real and imaginary parts each uniform), filling
the `n x l` matrix column-major in one call. `Rng` holds the four-integer
LAPACK seed. The C++, Rust, Swift and Fortran ports share this generator,
so the same seed gives the same sketch in each; `cargo run --example
crossport` prints the five CROSSPORT lines to compare across ports.

## Differences from Python

- **Default RNG**: Python draws from NumPy's `default_rng()` (OS entropy, so
  each call differs). Here the default is a fresh `Rng` with seed
  `[0, 0, 0, 1]`, so results are deterministic; pass `Options::rng` to vary
  them. The random streams differ from NumPy's, so the same "seed" does not
  reproduce Python's numbers.
- **Errors** are `Result<_, librla::Error>`. A non-finite `rtol` is an error
  (`InvalidArgument`); an operator returning a block of the wrong shape is
  `ShapeMismatch`. The method is an enum, so an invalid name is rejected
  when parsing (`InvalidMethod`).
- **SVD**: `gesdd` falls back to `gesvd` if it does not converge (SciPy
  reports the failure).
- **Materialization** applies the operator to blocks of 64 identity columns
  instead of one column per matvec.
- Dense input and the same matrix behind a matrix-free
  operator give the same sketch, but `Q^H A` (dense) and `(A^H Q)^H`
  (operator) round differently, so pivots after the projection can differ
  for near-tied columns; ranks and accuracies agree.

## Files

| Path | Contents |
|------|----------|
| `src/lib.rs` | Crate docs and re-exports |
| `src/sketch.rs` | The five routines, `Options`, `Method`, result types |
| `src/lapack.rs` | Safe LAPACK wrappers (workspace queries, info checks), `dense` |
| `src/ffi.rs` | `extern "C"` declarations of the Fortran LAPACK/BLAS symbols (`$NEWLAPACK` names with Accelerate) |
| `src/scalar.rs` | `Scalar` trait (sealed) dispatching to d/z routines |
| `src/mat.rs`, `src/operator.rs`, `src/rng.rs`, `src/error.rs` | `Mat`, `LinearOperator`/`FnOperator`, `Rng`, `Error` |
| `build.rs` | Backend link directives |
| `tests/` | Integration tests |
| `examples/demo.rs`, `examples/crossport.rs` | Demo and cross-port check |

## License

MIT (see `LICENSE`); see also `DISCLAIMER.txt`.
