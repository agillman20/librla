# LibRLA for Swift (experimental)

A Swift port of librla: randomized algorithms for low-rank factorizations
of real and complex matrices, given either as dense matrices or as
matrix-free operators. It tracks the semantics of librla 1.2.1
(`distrib/python/librla.py` is the reference) and is experimental. The port
version is `libRLAVersion`, defined in `Sources/LibRLA/LibRLA.swift`.

| Function | Description |
|----------|-------------|
| `orthSketch` | Approximate orthonormal basis for the column space |
| `qrSketch` | Truncated QR factorization with column pivoting |
| `svdSketch` | Truncated singular value decomposition |
| `idSketch` | Interpolative decomposition (randomized) |
| `idQRPiv` | Interpolative decomposition via deterministic QR with pivoting |

All dense linear algebra goes through BLAS/LAPACK (`geqp3`, `orgqr`/`ungqr`,
`gesdd`/`gesvd`, `gelsd`, `trsm`, `gemm`, `larnv`).

## Requirements

- Swift 6 toolchain (tested with Swift 6.4, Command Line Tools, macOS arm64)
- macOS 14 or later with Accelerate (default), or OpenBLAS
- [swift-numerics](https://github.com/apple/swift-numerics) (`ComplexModule`,
  `RealModule`), fetched by SwiftPM

## Build, test, demo

```bash
cd distrib/swift
swift build                    # library, demo and test runner
swift test                     # swift-testing suite: 17 tests, prints PASS/FAIL and CROSSPORT lines
swift run librla-test          # same checks as a plain program, exit status 1 on failure
swift run librla-demo          # dense and matrix-free demo
```

`swift test` shows each check's PASS/FAIL line and the five `CROSSPORT`
lines on stdout. `swift run librla-test` prints the same lines without the
test-runner framing, followed by a summary (`252 of 252 checks passed`).

With Command Line Tools only (no Xcode), the swift-build backend of Swift
6.4 intermittently compiles the test target without the swift-testing
macro plugin and fails with `plugin for module 'TestingMacros' not found`.
This happens notably after a plain `swift build`, and always with
`swift build --build-tests`. Passing the plugin directory works around it:

```bash
swift test -Xswiftc -plugin-path -Xswiftc /Library/Developer/CommandLineTools/usr/lib/swift/host/plugins/testing
```

`swift run librla-test` uses no macros and is unaffected.

The build is warning-free at Swift's default warning level for our code;
the only warnings are the Command Line Tools linker notices
`search path '/Library/Developer/CommandLineTools/Developer/...' not found`,
which SwiftPM emits for every package on this toolchain.

## Backends

| Setting | macOS | Linux |
|---------|-------|-------|
| default | Accelerate framework | `-llapack -lblas` (untested) |
| `LIBRLA_BLAS=openblas` | `-lopenblas` from `LIBRLA_OPENBLAS_DIR` (default `/opt/homebrew/opt/openblas`) | `-lopenblas` |

The backend is chosen when `Package.swift` is evaluated. Use a separate
build directory per backend:

```bash
LIBRLA_BLAS=openblas swift test --build-path .build-openblas
LIBRLA_BLAS=openblas swift run --build-path .build-openblas librla-test
```

How LAPACK is bound: Swift calls `rla_dgeqp3`, `rla_zgemm`, ... declared in
`Sources/CLapack/include/librla_lapack.h`. `Sources/CLapack/backend.c`
forwards each one to the Fortran routine, declared there by hand (LP64
`int`, one trailing hidden string length per CHARACTER argument, no routine
that returns a complex value). On Apple platforms the default binds each
prototype with an asm label to Accelerate's LAPACK 3.11 interface,
`<name>$NEWLAPACK`, which needs macOS 13.3 or later. The package declares
macOS 14, which swift-testing needs anyway. Accelerate's unsuffixed legacy
symbols are LAPACK 3.2.1 and much slower for complex `gemm`/`gesdd` and for
`geqp3`. OpenBLAS and Linux builds get the plain names `<name>_` through a
`LIBRLA_OPENBLAS` define in the C target's `cSettings` (a plain `.define`,
not an unsafe flag).

The choice is made in the C file because a C target's defines do not reach
the Swift importer. `nm -u` on a binary shows the result: the 16
`_<name>$NEWLAPACK` symbols with Accelerate, and `_<name>_` with OpenBLAS.
Importing Accelerate's own headers instead would mean either deprecation
warnings (the legacy CLAPACK interface) or the `ACCELERATE_NEW_LAPACK` clang
define. Passing that define to the Swift importer takes `unsafeFlags`, which
SwiftPM refuses in packages used as dependencies.

The default build has no unsafe flags and works as a package dependency. The
OpenBLAS mode uses one unsafe `-L` flag (Homebrew installs OpenBLAS
keg-only), so it is meant for building this package itself. In OpenBLAS mode
the macOS deployment target is raised to 26.0 to match the Homebrew dylib.

## Performance notes

The algorithms are generic over the operator and element type. A client
module calls them unspecialized, so any per-element work through the
`Scalar` protocol would cost a witness call per element. Row and column
block copies, the R extraction after `geqp3`, transposes, scaling and
sums therefore work on the raw storage as `Double`s with `memcpy`-style
copies. BLAS/LAPACK outputs are written into fresh storage without a
zero fill. Factorization helpers take their argument as `consuming`, so a
temporary sketch is factored in place without a copy-on-write copy.
Profiling with `sample` shows the main thread spending its time in
LAPACK/BLAS. For `qrSketch` with rank 200 on a 4000 x 3000 matrix, Swift
took 46.6 ms against 46.2 ms for a C program making the same LAPACK calls.

## Quick start

```swift
// Package.swift: .product(name: "LibRLA", package: "swift") plus, for
// complex matrices, .product(name: "ComplexModule", package: "swift-numerics")
import LibRLA

let A = Matrix<Double>(rows: 1000, columns: 500) { i, j in 1 / Double(i + j + 1) }
var rng = Rng(seed: 2026)

let (Q, flag, diagR) = orthSketch(A, rtol: 1e-6, rng: &rng)
let (Qr, R, p) = qrSketch(A, rtol: 1e-6, rng: &rng)            // A[:, p] ≈ Qr R
let (U, s, Vh) = svdSketch(A, rtol: 1e-6, rng: &rng)            // A ≈ U diag(s) Vh
let (k, piv, T) = idSketch(A, rtol: 1e-6, method: .lstsq, rng: &rng)
// A[:, piv[k...]] ≈ A[:, piv[..<k]] T

let error = (A - U.scalingColumns(by: s) * Vh).frobeniusNorm / A.frobeniusNorm
```

Complex matrices work the same way with `Matrix<Complex<Double>>`
(`import ComplexModule`).

## API

Every sketching function has two overloads: one takes `rng: inout Rng`
and advances it, and one without `rng` uses a fresh `Rng()` (deterministic).

```swift
func orthSketch<Op: LinearOperator>(_ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0,
                                    extraSamples: Int = 12, rng: inout Rng)
    -> (q: Matrix<Op.Element>, flag: Int, diagR: [Double])
func qrSketch<Op: LinearOperator>(_ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0,
                                  extraSamples: Int = 12, rng: inout Rng)
    -> (q: Matrix<Op.Element>, r: Matrix<Op.Element>, p: [Int])
func svdSketch<Op: LinearOperator>(_ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0,
                                   extraSamples: Int = 12, rng: inout Rng)
    -> (u: Matrix<Op.Element>, s: [Double], vh: Matrix<Op.Element>)
func idSketch<Op: LinearOperator>(_ A: Op, rtol: Double, blockSize: Int = 42, powerIter: Int = 0,
                                  extraSamples: Int = 12, method: IDMethod = .fast, rng: inout Rng)
    -> (k: Int, piv: [Int], t: Matrix<Op.Element>)
func idQRPiv<Op: LinearOperator>(_ A: Op, rtol: Double, method: IDMethod = .fast)
    -> (k: Int, piv: [Int], t: Matrix<Op.Element>)
```

Results are labeled tuples, so both `let (U, s, Vh) = svdSketch(...)` and
`svdSketch(...).s` work.

| Python | Swift |
|--------|-------|
| `orth_sketch(A, rtol, ...)` | `orthSketch(A, rtol: rtol, ...)` |
| `qr_sketch` / `svd_sketch` / `id_sketch` | `qrSketch` / `svdSketch` / `idSketch` |
| `id_qrpiv` | `idQRPiv` |
| `block_size`, `power_iter`, `extra_samples` | `blockSize`, `powerIter`, `extraSamples` |
| `method='fast'` (`'svd'`, `'lstsq'`) | `method: .fast` (`.svd`, `.lstsq`), or `try IDMethod(parsing: "fast")` |
| `rng=np.random.default_rng(s)` | `rng: &rng` with `var rng = Rng(seed: s)` |
| `Q, flag, diagR` | `(q, flag, diagR)` |
| `Q, R, p` | `(q, r, p)` |
| `U, s, Vh` | `(u, s, vh)` |
| `k, piv, T` | `(k, piv, t)` |

### Options

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `rtol` | (required) | `< 1`: relative tolerance (adaptive rank); `>= 1`: rank `floor(rtol)` |
| `blockSize` | 42 | Initial sketch size in tolerance mode (ignored in rank mode) |
| `powerIter` | 0 | Power iterations `x = Aᴴ(A x)`, each followed by re-orthonormalization; 1 or 2 help slowly decaying spectra |
| `extraSamples` | 12 | Buffer beyond the target rank. Rank mode samples `floor(rtol) + extraSamples` columns. Tolerance mode accepts a sketch when `diagR[count - 1 - extraSamples] <= rtol * diagR[0]`. 0 gives the legacy last-column check |
| `method` | `.fast` | ID interpolation matrix: `.fast` (triangular solve, SVD fallback when `R11` is near singular), `.svd` (pseudoinverse of `R11`), `.lstsq` (least squares on `A`'s columns, most accurate) |
| `rng` | `Rng()` | LAPACK seed state, advanced by each call |

### Conventions

- `Matrix<Element>` is a column-major value type (`Element` is `Double` or
  `Complex<Double>`). It provides `A[i, j]`, `rowCount`, `columnCount`,
  `shape`, `elements`, `adjoint`, `*`
  (BLAS `gemm`), `Matrix.adjointProduct(A, B)` (`Aᴴ B`), `+`, `-`,
  `columns(_:)` and `rows(_:)` taking ranges or index lists,
  `scalingColumns(by:)` (`A diag(d)`), `frobeniusNorm`, `identity(_:)`,
  `random(rows:columns:using:)`, and deterministic `pivotedQR()` and `svd()`.
- Indices and permutations are 0-based: `A.columns(p)` is Python's `A[:, p]`.
- `svdSketch` returns `Vh` = Vᴴ, as Python and Julia do; `s` is `[Double]`, descending.
- `orthSketch` `flag`: 0 success; 1 in tolerance mode when `rtol < eps` or the
  sketch would reach `min(m, n)`, in which case `q` is `m x 0` and `diagR` is
  empty. `qrSketch`, `svdSketch` and `idSketch` then fall back to a
  deterministic pivoted QR or SVD of `A`, so their callers never see the flag.
- Wide input to `svdSketch` (`m < n`) is sketched through its adjoint.

## Matrix-free operators

Anything conforming to `LinearOperator` can be factored:

```swift
public protocol LinearOperator<Element> {
    associatedtype Element: Scalar
    var shape: (rows: Int, columns: Int) { get }
    func apply(_ x: Matrix<Element>) -> Matrix<Element>          // A X,  X is n x k
    func applyAdjoint(_ w: Matrix<Element>) -> Matrix<Element>   // Aᴴ W, W is m x k
    var denseMatrix: Matrix<Element>? { get }                    // default nil
}
```

`Matrix` conforms (with `denseMatrix` returning itself, so BLAS is used
directly). `FunctionOperator` wraps two closures:

```swift
let op = FunctionOperator<Double>(rows: m, columns: n,
                                  apply: { X in B * (C * X) },
                                  applyAdjoint: { W in Matrix.adjointProduct(C, Matrix.adjointProduct(B, W)) })
let (U, s, Vh) = svdSketch(op, rtol: 20, rng: &rng)
```

`Qᴴ A` is computed as `(Aᴴ Q)ᴴ`. The deterministic paths need a dense
matrix. These are the tolerance-mode fallback, `idQRPiv` and
`method: .lstsq`. There a matrix-free operator is materialized by applying
it to blocks of identity columns, O(n) column applications in total, as in
librla.py. Prefer rank mode for operators where that is too expensive.

## Random numbers

Test matrices Ω are drawn with LAPACK `dlarnv`/`zlarnv` (`idist = 2`,
uniform on (-1, 1); for complex entries the real and imaginary parts are
independent), filling each `n x b` matrix column-major in one call. The
state is LAPACK's `iseed[4]`, held by `Rng`:

```swift
var rng = Rng()                               // iseed [0, 0, 0, 1], the default
var rng = Rng(seed: 42)                       // [(s>>36)&4095, (s>>24)&4095, (s>>12)&4095, (s&4095)|1]
var rng = try Rng(iseed: [1, 2, 3, 5])        // throws unless 0...4095 with the last entry odd
```

The C++, Rust, Swift and Fortran ports use the same generator, so the same
seed gives the same sketch in each of them (checked by the `CROSSPORT` lines).

## Differences from Python librla

- The default random state is the fixed seed `Rng()`, so results are
  deterministic. Python's default (`rng=None`) draws from OS entropy.
- Ω comes from LAPACK `larnv`, not NumPy's generator, so for a given seed
  the sketches differ from Python's. Drawing Python's Ω with `larnv`
  reproduces the Swift `CROSSPORT` lines (integers exactly, floating-point
  values to BLAS rounding).
- An invalid ID method cannot be passed (`IDMethod` is an enum).
  `IDMethod(parsing:)` throws `LibRLAError.invalidMethod` for any string other
  than `fast`, `svd` or `lstsq`.
- Misuse traps with a precondition failure instead of raising: a non-finite
  `rtol`, negative options, mismatched shapes from a custom operator, or
  LAPACK reporting an illegal argument. The same applies if `gesvd` (the
  fallback after `gesdd`) or `gelsd` fails to converge, which in practice
  means NaN or Inf input. SciPy raises `LinAlgError` in that case.
- `idQRPiv` skips forming Q, and with `.lstsq` it reuses the materialized
  matrix instead of materializing an operator twice. Results are unchanged.
- Single precision is not supported. Dimensions must fit 32-bit LAPACK
  integers.

## Tests

`test/Support` holds the checks, as a library shared by the swift-testing
suite (`test/Tests`) and the `librla-test` runner (`test/Runner`). For both
`Double` and `Complex<Double>` they cover:

- RNG reference values: `dlarnv`/`zlarnv` with `iseed {1,2,3,5}` to 1e-15,
  and the seed after the call
- `orthSketch`: tolerance mode with growth from 42 to 168 columns, span
  error, orthonormality, rank-mode column counts, `extraSamples = 0`, and
  every flag-1 path
- `qrSketch` and `svdSketch`: both modes on tall and wide matrices against
  deterministic LAPACK QR/SVD, plus the deterministic fallback
- `idSketch` and `idQRPiv`: all methods in both modes, rank-deficient `R11`,
  the zero matrix (`k = 0`) and `k == n`
- Matrix-free operators: a hidden dense wrapper matches the dense path with
  the same seed (including the materializing fallback), and a genuine
  low-rank product `B (C x)` is factored correctly
- Power iteration: errors are nonincreasing in `powerIter`
- Seed reproducibility, `Rng` advancement, and method parsing

Cross-port lines (Accelerate `$NEWLAPACK`, this Mac):

```
CROSSPORT orth_real 2.201415876390088e+00 7.782393081571191e-01 1.792111547918920e-01 2.735372994185338e-02 1.829391246767265e-03 4.964313308298363e-04 4.086054533178452e-05 4.297896799911683e-06
CROSSPORT orth_cplx 2.764954393717040e+00 8.441207939408220e-01 1.423567755739453e-01 3.392840380629036e-02 4.897016475004854e-03 2.911400935583974e-04 6.229146987168865e-05 1.610490859122329e-05
CROSSPORT seed_after 3844 2983 1707 1605
CROSSPORT id_real 16 0 3 28 1 168 10 72 5 199 2
CROSSPORT svd_cplx 10 2.256748472841064e+00 8.299098477167930e-01 1.834095666318948e-01 3.407572079164049e-02 1.991850185385876e-02
```

The integer lines are identical across backends. The trailing `diagR`
entries of the ill-conditioned Hilbert-type matrix differ by up to 3e-13
relative (Accelerate) and 4e-11 (OpenBLAS) from librla.py fed the same
`larnv` stream.

## Layout

```
Package.swift                  SwiftPM manifest and backend selection
Sources/CLapack/               Fortran BLAS/LAPACK prototypes and link settings
Sources/LibRLA/                the library (Matrix, Rng, LinearOperator, algorithms)
test/Support, test/Tests, test/Runner
demo/main.swift                librla-demo
```

## License

MIT (see `LICENSE`); see also `DISCLAIMER.txt`.
