# Why libid Chose the Row-Sketch (LQ) Route for Interpolative Decompositions

Companion to [ID_RANK_MODE_T_QUALITY.md](ID_RANK_MODE_T_QUALITY.md),
whose §6 measures how libid's matrix-free row-sketch paths degrade on
flat spectra. This note answers the design question the other way
around: *why* the id_dist library (Martinsson, Rokhlin, Shkolnisky,
Tygert) organized its randomized ID around row sketches and an
incremental LQ, and in which regimes that choice is faster and more
convenient. Written 2026-07-27, based on reading `libid/src`
(`iddr_rid.f`, `iddp_rid.f`/`idd_findrank0`, `iddr_aid.f`,
`idd_id.f`).

---

## TL;DR

The LQ itself is incidental bookkeeping. The real design decision is
**"produce an l×n row sketch, then run the one deterministic ID
kernel on it"** — uniform across the whole library: the dense
randomized path forms `SA` via SRFT, the matrix-free path forms `GA`
via transpose-matvecs, the deterministic path uses `A` itself; all
three feed the same `iddr_qrpiv` + `idd_lssolve` core. The choice
buys single-pass, one-sided, minimal-query access and the famous
`O(mn log k)` complexity, at the price of solving for `proj` in the
sketch's distorted geometry — a price that is invisible in the
regime the library was designed for (residuals at machine precision)
and only becomes visible outside it (the §6 measurements).

---

## 1. An ID is a row-space object

The ID asks which linear dependencies hold among *columns* of A;
those dependencies live entirely in the row space. If `rank(A) = k`
exactly and `G` is a generic `l×m` with `l ≥ k`, then
`row(GA) = row(A)` and the ID of the sketch is an **exact** ID of A —
zero approximation. For numerically rank-k matrices the loss is a
distortion factor multiplying `σ_{k+1}`-level residual
(ID_RANK_MODE_T_QUALITY.md §7).

libid was built for compressing integral operators and off-diagonal
blocks in fast direct solvers, where spectra decay exponentially: a
factor of 4× or even 21× on a residual of 1e-15 is invisible. The
published error bounds (Martinsson–Rokhlin–Tygert, ACHA 2011) are
explicitly of the form "whp within `poly(l, m) · σ_{k+1}`" — the
philosophy being that *any* polynomial factor is acceptable because
`σ_{k+1}` is negligible. The §6 measurements (flat spectrum, k far
below numerical rank) sit exactly outside that envelope, which is why
the same design looks bad there.

## 2. Minimal access: skip the projection step

This is the strongest practical argument, and it is worth stating
precisely because *skipping the projection is simultaneously the
efficiency win and the accuracy hole* diagnosed in the companion
note.

In the range-finder architecture (HMT; librla's `orth_sketch`), the
sketch produces only a **basis** `Qs` from `AΩ`. A basis by itself
carries no information about how A's columns relate to each other, so
a *second pass* over A is required to form `B = Qsᴴ A` before any
pivoting or T solve can happen: another `k+p` transpose-applies
(matrix-free) or another `O(mn(k+p))` dense pass, and the operator is
needed in *both* directions.

The row sketch collapses this: `Y = GA` **is already the reduced
matrix** — sketching and projecting are the same operation.
Consequences:

- **Single-pass / streaming.** A is touched once and can then be
  discarded, streamed, or generated on the fly. The range finder is
  inherently two-pass (`Qsᴴ A` cannot be formed until the first pass
  has finished).
- **Half the applies, one-sided.** `l = k+2` transpose-applies total
  for `list + proj`, and no forward matvec is ever needed. Only the
  transpose apply has to exist as a fast black box.
- **No m-dimensional object anywhere.** This is special to the *ID*
  among factorizations. QR/SVD/orth must deliver an `m×k` factor, so
  some m-dimensional work is unavoidable. The ID's output
  (`piv`, `T`) is purely n-dimensional — reconstruction uses A's own
  columns, read directly (k coordinate-vector matvecs in `id2svd`)
  rather than synthesized from a basis. libid's choice of the ID as
  its *primitive* factorization (SVD derived from it via `id2svd`)
  and the row-sketch access pattern are two halves of one design: the
  ID is exactly the factorization for which the projection step is
  skippable.

The symmetry with the companion note: the projection `B = Qsᴴ A` is
what makes the triangular solve's residual live in a subspace adapted
to A, and the full-matrix QR's implicit projection is what makes the
residual orthogonal to the basis columns — the err ≤ 1 guarantee of
§3.3 there. Skip the projection and the T solve happens in the
oblivious sketch's distorted geometry — the `1 + k/(l−k−1)` factor of
§7. The answer to "why is libid's matrix-free path fast" and "why
does it degrade on flat spectra" is literally the same line of code
that isn't there.

## 3. The headline complexity depends on it: a cost ledger

The `O(mn log k)` interpolative decomposition
(Woolfe–Liberty–Rokhlin–Tygert 2008; Liberty et al., PNAS 2007) is
only possible because the pivoted QR runs on the `l×n` sketch rather
than on A. Here is where the complexity goes, term by term. Setup:
A is m×n, target rank k, small oversampling, `l = k+p`. Costs split
into three kinds: **compression** (touching all of A),
**rank-revealing work** (the pivoted QR making sequential pivot
decisions), and the **T solve**.

| algorithm | compression pass(es) | rank-revealing | T solve | total |
|---|---|---|---|---|
| deterministic k-step pivoted QR on A (libid `iddr_id`; SciPy's accidental path) | — (in place) | **O(mnk)** | O(nk²) | O(mnk) |
| Gaussian row sketch `Y = GA` → ID of Y | O(mnl) matmul | O(lnk) = O(nk²) | O(nk²) | **O(mnk)** — same order! |
| SRFT row sketch `Y = SA` → ID of Y (WLRT) | **O(mn log l)** | O(nk²) | O(nk²) | **O(mn log k + nk²)** |
| range finder (librla): `AΩ`, QR, `B = QᴴA` → pivoted QR of B | O(mnl) + O(ml²) + **O(mnl) again** (projection) | O(nk²) | O(nk²) | O(mnk); ×(1+q) with power iters |
| matrix-free row sketch (`iddr_rid`) | (k+2) rmatvecs | O(nk²) | O(nk²) | (k+2)·C_apply + O(nk²) |
| matrix-free range finder (librla) | ~2(k+p) applies (matvec **and** rmatvec) | O(nk²) | O(nk²) | 2(k+p)·C_apply + O(mk² + nk²) |

Three observations fall out of the table.

**(a) Where the deterministic O(mnk) comes from — and why it is
irreducible in place.** k-step pivoted QR is k sequential elimination
steps, each applying a Householder reflection across the active
m×(n−j) block and updating n−j column norms of m-vectors to choose
the next pivot. The rank-revealing decisions are *entangled with the
m-dimensional geometry*: every pivot choice needs fresh m-length
column information, so the O(k) sequential steps each pay O(mn). One
cannot decouple them while staying on A — this is the precise sense
in which keeping the pivoting on the full matrix cannot reach
`O(mn log k)`.

**(b) Sketching *factors* the computation: one rank-independent
compression, then all rank-dependent work on a short object.** After
`Y = SA`, each Householder step costs O(ln) instead of O(mn) — the m
never meets the k. The compression term itself:

- With a **Gaussian** G it is a dense matmul, O(mnl) — *the same
  flop order as deterministic pivoting*. A Gaussian row sketch does
  **not** beat O(mnk) asymptotically; its real-world win is constants
  and shape (one cache-friendly, parallel BLAS-3 matmul versus a
  sequential, memory-bound, BLAS-2-flavored pivoted QR) plus one pass
  over A instead of k touches.
- The **SRFT** is exactly the device that attacks the compression
  term: a structured `S` (random diagonal/rotations + FFT, subsampled
  to l outputs — libid's `idd_sfft` computes only the l needed
  entries, O(m log l) per column) applies in near-linear time, usable
  only because the downstream algorithm never needs `S` as a matrix,
  just its action on A. Result: O(mn log l) compression + O(nk²)
  everything else. Since any algorithm must read A once, Ω(mn) is a
  floor; the SRFT route sits a log factor above that floor, while
  deterministic pivoting sits a factor **k** above it. Speedup ratio
  ~ k / log k — at k = 100, roughly 20×.

**(c) The range finder pays the compression term twice, and structure
cannot save it.** `AΩ` can be SRFT-accelerated to O(mn log l) — but
the *projection* `B = Qᴴ A` is a second dense pass at O(mnl), and `Q`
is a data-dependent dense m×l matrix, so no fast transform applies to
it. The projection step is pinned at O(mnl) ≈ O(mnk) no matter how
clever the first sketch is. This is the complexity-side face of the
same coin as §2: the row sketch is `O(mn log k)`-capable *because*
projecting and sketching are the same pass; the range finder is
O(mnk)-bound *because* they are not. (Power iteration multiplies the
passes further: each iteration is a matvec + rmatvec round, another
2·O(mnl).)

Two footnotes complete the picture:

- **The T solve is O(nk²) in every variant and never dominates —
  except `method='lstsq'`**, which is qualitatively different:
  `lstsq(A_basis, A_skel)` is a QR of an m×k matrix, O(mk²), plus
  applying Qᵀ to n−k right-hand sides, O(mnk) — a *full extra
  compression-order pass*. The accurate T costs as much as the entire
  ID; that asymmetry is why `'fast'` exists, and why the err > 1
  pathology of the companion note was worth diagnosing rather than
  always paying for lstsq.
- **What SciPy actually charges today**: the dense "randomized"
  rank-mode path bills O(mnk) (deterministic fallback) instead of the
  advertised `O(mn log k)` — the dead SRFT path (companion note §3.2)
  silently traded the headline complexity back for the accidental
  accuracy guarantee, i.e. it moved the rank-revealing work from the
  l×n sketch back onto A. The matrix-free path is the opposite
  extreme: minimal applies, (k+2)·C_apply + O(nk²), with the accuracy
  cost measured in companion note §6.

Memory follows the same split: the sketch routes hold an l×n working
set (A read-only or streamed); the deterministic and lstsq routes
need the m×n matrix resident. For libid's original use case —
compressing operator blocks inside a fast direct solver, where A
exists only as a fast apply — the applies + O(nk²) column of the
ledger was the only affordable one.

### Postscript to §3: compression-pass choices on modern hardware

The ledger counts flops; the wall clock disagrees in instructive
ways. Measured 2026-07-27 on Apple silicon (Accelerate BLAS, numpy;
torch with 12 threads), double precision.

**Is the SRFT actually faster than a Gaussian sketch today? Not below
k ≈ 500 on this machine.** Timing the compression pass alone against
a full pivoted QR on A:

| A: 8000×4000 | k=20 | k=100 | k=300 |
|---|---|---|---|
| Gaussian dgemm `G@A` | 3.8 ms | 9.2 ms | 23.1 ms |
| SRFT via full FFT (`O(mn log m)`) | 98.8 ms | 97.5 ms | 98.9 ms |
| pivoted QR on sketch (l×n) | 1.0 ms | 7.8 ms | 19.4 ms |
| pivoted QR on full A | 3040 ms | 3042 ms | 3055 ms |

Three layers:

1. **The architectural win is enormous and S-independent**: sketch
   route total 5–43 ms vs 3040 ms for pivoted QR on A — a 70–600×
   speedup from moving the rank-revealing work onto the sketch. The
   choice of S only tunes the one remaining compression pass.
2. **The "asymptotically slower" Gaussian wins at every tested k**:
   dgemm runs on the matrix units at ~860 GFLOP/s effective while
   the FFT is memory-bound at ~20 GFLOP/s effective; the ~40×
   efficiency gap swamps the flop ratio `l / log m` (≈ 24 at k=300).
   Crossover ≈ k 500–1000 here, and it *recedes* as matmul hardware
   outpaces memory bandwidth. (Caveats: this used numpy's full FFT;
   libid's pruned `idd_sfft` at `O(mn log l)` claws back maybe
   1.5–3×, still losing below k ≈ 300. And in 2007–2008 — single
   core, far smaller BLAS-3 advantage, slow RNGs — the crossover was
   plausibly k ≈ 30–50, so the SRFT was the right call *then*.
   Fine print: SRFT guarantees formally want `l ≳ k log k`
   (Tropp 2011); k+8 works in practice with rare adversarial
   failures.)
3. **The modern replacement is not a better Fourier transform but
   sparse sign embeddings** (CountSketch/OSNAP: s = 4–8 nonzeros per
   column, values ±1/√s; Clarkson–Woodruff 2013, recommended over
   both Gaussian and SRFT by Martinsson–Tropp 2020 §9): applying S
   costs `O(s·mn)` — "read A s times" — independent of l.

**Are sparse sign embeddings faster here? Not with native library
calls.** Same A, s = 8:

| implementation | k=20 | k=100 | k=300 | scaling |
|---|---|---|---|---|
| Gaussian dgemm | 4.6 ms | 9.3 ms | 23.0 ms | linear in l |
| scipy CSC `S@A` | 37 ms | 37 ms | 37 ms | **flat in k** |
| scipy CSR / torch sparse CSR (12 thr) | 44 ms | 47 ms | 49 ms | flat |
| pure-numpy CountSketch ×8 | ~600 ms | ~600 ms | ~600 ms | flat, hopeless |

Readings:

- The theory shows up cleanly in the *shape*: sparse cost is
  k-independent, exactly as advertised. Extrapolating dgemm's
  ~0.075 ms per unit l against CSC's flat 37 ms puts the
  native-scipy crossover at l ≈ 500, i.e. **k ≈ 500**.
- Native implementations sit ~10–25× above the memory-bound floor.
  The floor for `S@A` is "stream A once": 256 MB ≈ 2–3 ms. scipy's
  CSC kernel (~55 GB/s, single-threaded scalar C, no blocking even
  though the 10 MB output would sit in cache) reaches 37 ms; torch's
  multithreaded sparse mm is no better (beta-quality kernel); pure
  numpy cannot express scatter-add without materializing permuted
  copies of A (25× worse).
- **A custom kernel changes the verdict**: a hand-written
  CSC-streaming loop (for each row j of A, axpy into the s cached
  output rows) hits the ~3 ms floor and would beat dgemm from
  **k ≈ 50**. That is ~15 lines with `@threads` in Julia — the one
  librla language where the optimal kernel is genuinely native.
  Python needs numba/Cython; MATLAB needs a mex file (built-in
  `sparse(S)*A` is in scipy's class).

**Measurement protocol for any candidate compression pass:**

1. Microbenchmark the apply in isolation: sweep k at fixed m, n;
   check the scaling signature (flat in k = correct sparse
   behavior); report effective GB/s against a STREAM-style ceiling;
   locate the dgemm crossover.
2. End-to-end with Amdahl honesty: in a row-sketch ID the sketch is
   essentially the whole `O(mn·)` cost, so gains transfer fully; in
   the range finder the projection `B = QᴴA` and the QR of `AΩ`
   still cost `O(mnl) + O(ml²)`, capping any sketch speedup at
   ~a third. Matrix-free: irrelevant (operator applies dominate).
3. Re-validate accuracy: sparse embeddings are in the same
   universality class for the distortion mean (companion note §7 —
   even Rademacher matches), but subspace-embedding guarantees want
   more oversampling (practice l ≈ 1.5–2k at s = 4–8; theory Cohen
   2016 / OSNAP). Rerun the distortion experiment and the test_id
   suite at both l = k+8 and l = 2k, watching mean *and* tails.
4. Count the RNG: generating sparse S is `O(sm)` draws vs `l·m`
   Gaussians — a few ms in the Gaussian column at large l that
   belongs in the comparison.

**Bottom line for librla**: on current hardware, in native
numpy/scipy/MATLAB, neither SRFT nor sparse embeddings pay below
k ≈ 500 — and librla's dense path would barely benefit anyway
because the projection pass dominates (Amdahl). The scenario where
sparse embeddings genuinely win: a libid-style row-sketch ID over a
huge dense A at large k with a custom (or Julia-native) kernel, or
GPUs. The SRFT's own lesson applies to its successors: the flop
count is not the wall clock.

### Coda: why BLAS-3 is the bar

The postscript's recurring pattern — asymptotically faster methods
losing to a dense matmul — has a structural explanation, not an
accidental one. dgemm is essentially the *only* dense kernel whose
arithmetic intensity grows with problem size: O(n³) flops over O(n²)
data means each byte fetched is reused ~n times, so a good
implementation runs at the *compute* ceiling, while nearly
everything else in numerical linear algebra — FFTs, sparse ops,
pivoted factorizations, anything stream-like — runs at the *memory*
ceiling. Hardware has spent three decades widening exactly that gap:
the machine measured above does ~860 GFLOP/s in dgemm against
~20–50 GB/s-equivalent for streaming kernels, a ~40× efficiency
handicap that any "asymptotically faster" competitor must first pay
off. The SRFT's `k/log k` flop advantage and the sparse embedding's
`l/s` advantage both evaporate below k ≈ 500 because they trade
cheap flops for expensive bytes.

There is an ironic corollary for randomized NLA specifically: the
field's founding pitch was "replace expensive factorizations with
matmuls" — and it succeeded *because* of BLAS-3 economics. The
70–600× win in the postscript's first table (sketch route vs pivoted
QR on A, at *equal* flop order in the Gaussian case) is BLAS-3
beating BLAS-2, not fewer flops beating more flops. The
structured-sketch literature is thus in the odd position of trying
to out-optimize the very effect that makes the whole framework fast.

What genuinely beats dgemm, in practice:

1. **Not doing the work at all** — fewer passes, smaller l, stopping
   early. The row sketch's "skip the projection" (§2) is worth more
   than any choice of S.
2. **Reaching the memory floor with a custom kernel** — the ~3 ms
   stream-A-once sparse apply is a real 5–8× at k ≥ 300, but it must
   be hand-written; library sparse kernels leave the win on the
   table.
3. **Changing the cost model** — matrix-free operators, streaming
   data, GPUs at sizes where dense l×m sketches stop fitting. Where
   dgemm cannot be invoked, its supremacy is moot.

Which is a one-line summary of the entire investigation: libid won
by moving work off A (architecture), librla wins by keeping the
accurate solve where the geometry is right (projection/densify), and
neither should spend effort beating dgemm at its own game — the
constant-factor knobs (SRFT, sparse S, entry laws) are pennies next
to those two structural decisions.

## 4. The incremental LQ is the cheapest adaptive rank certificate

`idd_findrank0` (tolerance mode): each new probe costs one rmatvec
plus one `O(nk)` Householder step against the previously
orthogonalized rows — an incremental LQ factorization of the growing
sketch. A new random row falls (a.s.) outside the span of the
previous rows *iff* the row space is not yet exhausted, so "orthogonal
residual small" is a valid stopping signal at essentially
**rank + 1 total applies with O(nk) memory**. Under one-sided access
this cannot be beaten. The LQ is used only as this rank detector; the
raw rows still form the sketch handed to `iddp_id`.

Similarly, `l = k+2` in `iddr_rid` is the minimal oversampling for
which the sketch is nonsingular whp with a *finite-mean* distortion
factor (the mean exists exactly from `l ≥ k+2`; companion note §7).
The price — infinite variance, heavy tails — was again harmless at
`σ_{k+1} ≈ eps`.

## 5. Software economy

One deterministic kernel (`iddr_qrpiv` + the `2²⁰`-capped
`idd_lssolve`), three sketch producers (A itself, `SA` via SRFT,
`GA` via rmatvecs). All the delicate code is shared — for a
hand-written Fortran-77 library, a serious maintainability argument.

---

## Verdict

Faster (2× fewer operator applies matrix-free; `O(mn log k)` dense),
more convenient (one-sided black box, single-pass/streaming, minimal
memory, no m-dimensional factor), and essentially free in accuracy
*in the regime it was designed for* — eps-accurate compression of
operators with rapidly decaying spectra. The trade-off inverts
exactly when the residual is large (flat/slow spectra, k far below
numerical rank), where the distortion multiplies an O(1) quantity:
the §6 pathology of the companion note.

This also frames librla's divergence fairly: librla pays 2× the
applies for the adaptive range basis (smaller distortion, power
iteration available), and — after its own row-sketch experiment
(`_row_sketch_certify`, retired; companion note §6.3) confirmed the
T-conditioning cost — pays further with dense materialization for the
T solve, trading libid's minimal-access purity for accuracy in
regimes libid never targeted.

## Future directions: a single-sided speed tier for librla

The investigation suggests librla could offer a libid-style
single-pass row-sketch ID as a *speed tier* alongside the current
guaranteed path — with two refinements that fix exactly what made
the retired `_row_sketch_certify` (companion note §6.3) and libid's
`*_rid` routines degrade.

**The retired code failed for a fixable reason.** Its rank detection
and blocked accumulation (geometric block growth, blocked CGS2,
buffered acceptance — all BLAS-3) were sound. What failed was
computing `T = R11⁻¹R12` from only the *top k rows* of the certified
sketch, discarding the buffer rows' geometry.

**Refinement 1 — the buffer rows double as the T solve's
oversampling.** The certified sketch `Y` has `K = k + extra + slack`
rows. Solving the *full* sketched least squares
`min ‖Y_basis T − Y_skel‖` over all K rows (instead of the k-row
triangular solve) turns the uncontrolled "projected-out rows
ignored" error into the honest distortion `1 + k/(K−k−1)` of the
companion note §7 — a **dial controlled by accumulating more
blocks**. Distortion ≤ 1+ε requires `K ≥ k + k/ε`, i.e.
`ε ≈ k/(K−k)`; in residual *norm* the excess over optimal is
`√(1+ε) − 1`. Calibration (k-independent for k ≫ 1):

| K | ε (squared) | residual norm vs optimal |
|---|---|---|
| 1.1k | ≈ 10 | ≈ 3.3× |
| 1.2k | ≈ 5 | ≈ 2.4× |
| 1.3k | ≈ 3.3 | ≈ 2.1× |
| 1.4k | ≈ 2.5 | ≈ 1.87× |
| 1.5k | ≈ 2 | ≈ 1.73× |
| 2k | ≈ 1 | ≈ 1.41× |
| 3k | ≈ 0.5 | ≈ 1.22× |
| 6k | ≈ 0.2 | ≈ 1.10× |
| 11k | ≈ 0.1 | ≈ 1.05× |

The dial is sharply asymmetric: the distortion diverges like
`1/(K/k − 1)` as K → k (a cliff on the left — and at thin buffers
the §7 variance blows up too, so 1.1k is 3.3× *with bad
seed-to-seed spread*), while the right side is a long flat tail
(2k → 11k only improves 1.41× → 1.05×). K = 2k sits at the knee and
is the natural documented default ("√2-quality T"); below
K ≈ 1.5–2k the tier should not be offered. This also places libid's
*additive* buffers on the map: `l = k+8` is K/k = 1.4 at k = 20
(≈ 1.9×, matching the §3.4 measurements of the companion note) but
K/k ≈ 1.03 at k = 300 (≈ 6×) — a fixed additive buffer deteriorates
linearly with k, which is why the tier's dial must be
*multiplicative* (K = c·k, rank-independent ε).

Halving ε costs a doubling of the buffer — still O(k) rows, still
one-sided, still single-pass, and the lstsq costs O(nkK)
sketch-side, never touching A again. One dial — the total row
count — simultaneously sets certification confidence, T distortion,
and cost; everything else is derived.

**Refinement 2 — certify with the Frobenius statistic, not the
spectral residual.** The rank-low bias measured in §6.2/§6.3 of the
companion note (k = 157–159 vs 173; tolerance missed ~17×) comes
from reading the sketch's shrunken trailing spectrum through a
single-residual spectral test. The Frobenius estimator is *unbiased
under any iid entry law* (extra_notes.tex, librla_toms: the
running-norm ratio statistics survive the row sketch unchanged), so
a buffered Frobenius stopping criterion — which librla's 1.1.0
`extra_samples` test already is, in spirit — repairs the rank
detection without extra passes.

**Blocked accumulation is the BLAS-3 answer of the coda.**
Vector-at-a-time findrank (libid) is BLAS-2 and memory-bound;
accumulating in blocks of b vectors makes both the sketch
application (dgemm panels in dense mode) and the orthogonalization
(blocked CGS2 = two dgemms) run at the compute ceiling. In
matrix-free mode the blocks become blocked rmatvecs, which most fast
operators also prefer. Don't fight dgemm — feed it.

**Constraints to design around:**

- The err ≤ 1 structural guarantee (companion note §3.3) is
  unavailable in *any* single-sided scheme — it belongs uniquely to
  full-geometry T solves. This is a tier, not a replacement:
  `fast single-pass` (row sketch, distortion-dialed T) alongside the
  current guaranteed path (range basis + densify/lstsq). Document
  the trade; the scipy bug and the 1.2.0 densify decision are both
  cautionary tales about hiding it.
- No power iteration in a single pass (needs alternating A/Aᵀ
  applies): flat-spectrum quality rests entirely on the oversampling
  dial.
- Amdahl and maintenance: at current problem scales the saved pass
  is milliseconds, and every new tier costs ×3 (Python/MATLAB/Julia
  parity). The tier earns its keep at large dense A, expensive
  one-sided operators, or streaming — build it when a use case
  arrives; the design intent is recorded here.

**Buffer semantics: additive witnesses vs multiplicative
oversampling.** librla's `extra_samples` currently serves two roles
with different correct scalings, and only one should become
multiplicative:

- *Confidence witnesses* (tolerance-mode stopping): a count of
  independent witnesses whose failure probability decays
  exponentially in the count (the `α^{−r}` structure of HMT
  Lemma 4.1), independent of k. Additive is correct; the 12 stays 12.
- *Geometry/distortion oversampling*: everywhere the buffer feeds an
  inverse-Wishart factor `1 + k/(p−1)`, an additive p ages linearly
  in k (the libid `k+8` aging in the calibration table);
  rank-independent quality needs a multiplicative dial `K = c·k`.

The multiplicative dial is therefore **effectively active only in
the single-pass speed tier**: in tolerance mode it is semantically
wrong, and in the current two-sided rank mode it is dominated —
`power_iter` improves quality exponentially in q versus linearly in
the buffer, and the additive default matches the peers (sklearn
`n_oversamples=10`, torch; the deliberate 42/12 defaults decision).
In the speed tier there is no second pass and no power iteration, so
`K = c·k` is the *entire* quality mechanism — expose it as the
one-dial target distortion (`eps_T`, deriving `K = k + k/eps_T`,
default at the knee `c = 2`). Two constants overall, legitimately:
they parametrize two different probability structures
(exponential-in-count confidence vs `1/(c−1)` distortion), one dial
per mechanism.

The `c = 2` default has published precedent, and the correspondence
is exact: in Tropp–Yurtsever–Udell–Cevher (SIMAX 2017), the
single-pass reconstruction `Â = Q(ΨQ)†(ΨA)` contains a sketched
least-squares solve with `l` rows and `k` unknowns, its error bound
carries the same inverse-Wishart factor `f(k,l) = k/(l−k−1)`, and
their recommended `l = 2k+1` sets that factor to exactly 2 — the
knee of the calibration table above. (Their 2019 streaming follow-up
keeps the same doubling for the co-range sketch.)

Summary shape: **one dial (total sketch rows), one pass, blocked
accumulation for BLAS-3, all rows reused for the T lstsq,
Frobenius-certified stopping — offered as the speed tier next to the
guaranteed tier.**

## References

- H. Cheng, Z. Gimbutas, P.-G. Martinsson, V. Rokhlin, "On the
  compression of low rank matrices," SISC 26(4), 2005 — the ID and
  its bounded-coefficient skeleton.
- E. Liberty, F. Woolfe, P.-G. Martinsson, V. Rokhlin, M. Tygert,
  "Randomized algorithms for the low-rank approximation of matrices,"
  PNAS 104(51), 2007 — adaptive transpose-apply scheme.
- F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert, "A fast randomized
  algorithm for the approximation of matrices," ACHA 25(3), 2008 —
  SRFT row sketch, `O(mn log k)`.
- P.-G. Martinsson, V. Rokhlin, M. Tygert, "A randomized algorithm
  for the decomposition of matrices," ACHA 30(1), 2011 — error
  bounds of the form `poly(l, m) · σ_{k+1}`.
- N. Halko, P.-G. Martinsson, J. A. Tropp, SIAM Review 53(2), 2011 —
  the range-finder architecture, for contrast.
- J. A. Tropp, A. Yurtsever, M. Udell, V. Cevher, "Practical
  sketching algorithms for low-rank matrix approximation," SIMAX
  38(4):1454–1485, 2017 — single-pass sketched reconstruction with
  the `k/(l−k−1)` factor and the `l = 2k+1` (factor-2)
  recommendation; streaming follow-up: SISC 41(4), 2019.
