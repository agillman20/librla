# Root Cause: `id_sketch` Fast-Mode Error > 1 in Rank Mode vs SciPy

Investigation of why `librla.id_sketch` with the default `method='fast'`
sometimes produces relative reconstruction error greater than 1 in rank
mode on dense matrices with flat or slowly decaying spectra, while
`scipy.linalg.interpolative.interp_decomp` on the same matrices stays
below 1 with a visibly better-conditioned interpolation matrix.

Investigated 2026-07-27 with librla 1.2.0, SciPy 1.18.0, NumPy 2.5.1,
using `compare/compare_id_scipy.py` and a dedicated experiment script.
Reference Fortran sources: `libid/src` (id_dist by Martinsson, Rokhlin,
Shkolnisky, Tygert; ID routines per Cheng, Gimbutas, Martinsson,
Rokhlin, SISC 2005).

---

## TL;DR

Two independent causes, on opposite sides of the comparison:

1. **librla**: the fast `T = R11⁻¹R12` is the *exact* least-squares
   solution of the **sketch-projected** problem
   `min ‖Qsᴴ(A_basis T − A_skel)‖`. The component of `A` outside the
   adaptive range basis `Qs` is invisible to this solve. For flat
   spectra that component dominates (~88% of the Frobenius mass at
   k=20, extra_samples=12, n=300), and `T` amplifies it with no
   penalty, pushing the total error above 1. It is **not** a
   conditioning or QR-implementation issue — `max|T| < 1` throughout.

2. **SciPy**: since the 1.15 Cython rewrite (still present on `main`
   as of 1.18), the rank-mode randomized routines `iddr_aid` /
   `idzr_aid` contain a dead-code bug: the SRFT random path is
   unreachable and the routine **always falls back to deterministic
   k-step pivoted QR of the full matrix**. That path has residual
   `Q₂R₂₂` orthogonal to the basis columns, so its error is
   *structurally guaranteed* ≤ 1. SciPy's superior `proj` quality on
   these tests is an accident of this fallback, not a better
   randomized algorithm. Tolerance mode is unaffected (§3.2a), and
   the Fortran reference does not have the bug (§3.1) — it was
   introduced when the port collapsed the Fortran's separate
   initializer/dispatcher pair into one function.

Bonus finding: had SciPy's random path actually executed with its
`l = k+8` sketch, it would be *worse* than librla's fast mode
(measured err ≈ 1.6–2.0) due to standard sketch-and-solve
least-squares distortion. The original Fortran id_dist, whose random
path really ran, would have shown the same behavior on such matrices.

---

## 1. Symptom

`compare/compare_id_scipy.py` (defaults: double precision,
`extra_samples=12`, `power_iter=0`) trips the `err > 1` retry on the
four rank-mode tests whose spectra are flat (i.i.d. Gaussian random
matrices, real and complex):

| Test | Size, k | librla fast err | scipy err | librla lstsq retry |
|---|---|---|---|---|
| Random Matrix | 500×300, k=20 | 1.197 | 0.943 | 0.945 |
| Complex Matrix | 300×200, k=25 | 1.199 | 0.888 | 0.892 |
| Large Random | 1000×600, k=20 | 1.222 | 0.972 | 0.973 |
| Large Complex | 600×400, k=25 | 1.294 | 0.945 | 0.946 |

Errors are relative Frobenius:
`‖A[:,piv[k:]] − A[:,piv[:k]] @ T‖_F / ‖A‖_F`. The librla `lstsq`
retry matches SciPy to ~3 digits, i.e. both sit essentially at the
least-squares optimum for their (nearly equivalent) pivot choices.

An ID with err > 1 is strictly worse than `T = 0`; a correct
least-squares `T` can never exceed `‖A_skel‖/‖A‖ < 1`.

---

## 2. What the fast T actually minimizes (librla)

Pipeline in rank mode (`rtol = k ≥ 1`):

1. `orth_sketch` builds `Qs`, an orthonormal basis of the Gaussian
   sketch range, with `k + extra_samples` columns (20+12 = 32).
2. `qr_sketch` forms `B = Qsᴴ A` ((k+extra) × n), runs pivoted QR
   (`geqp3`) on `B`, and truncates `R` to its top `k` rows.
3. `_compute_T_fast` solves the triangular system `R11 T = R12`.

Because the first `k` Householder steps fully triangularize the basis
columns, `B_basis = Qp [R11; 0]`, and therefore

```
T_fast = R11⁻¹ R12 = argmin_T ‖B_basis T − B_skel‖_F
       = argmin_T ‖Qsᴴ (A_basis T − A_skel)‖_F .
```

This identity was verified numerically to machine precision
(relative deviation ≤ 3·10⁻¹⁵ across all seeds and test matrices):
the fast triangular solve **is** an exact least-squares solve — but in
the wrong inner product. It only measures the residual after
projection onto span(Qs).

### Error decomposition

Split `A = P + E` with `P = Qs Qsᴴ A` (in-range) and
`E = A − Qs Qsᴴ A` (out-of-range). Then, exactly,

```
err² = ‖P_skel − P_basis T‖²/‖A‖²  +  ‖E_skel − E_basis T‖²/‖A‖² .
```

Measured on the 500×300 Gaussian test (k=20, seed 0), values relative
to ‖A‖_F:

| Quantity | Value |
|---|---|
| in-range residual `‖P_skel − P_basis T‖` | 0.242 |
| out-of-range skeleton mass `‖E_skel‖` | 0.885 |
| amplification term `‖E_basis @ T_fast‖` | 0.725 |
| combined out-of-range `‖E_skel − E_basis T_fast‖` | 1.170 |
| recombined √(P² + E²) | 1.195 |
| measured err_fast | **1.195** (exact match) |

For an i.i.d. Gaussian `A`, `E_basis` and `E_skel` are effectively
independent, so `E_basis T` adds *incoherently* to `E_skel`:
1.170² ≈ 0.885² + 0.725². The sketch-projected solve happily spends
‖T‖ to fit the 32-dimensional projected geometry (‖T_fast‖_F ≈ 13.4
vs ‖T_lstsq‖_F ≈ 3.3) because nothing in its objective charges for
what `T` does to `E`.

Notably `max|T_fast| ≈ 0.67` — small. The failure is **not**
ill-conditioning of `R11`, not entry blow-up, and not a defect of the
QR itself. It is purely the geometry of the objective.

For decaying spectra the same formula shows why fast mode is fine:
with spectrum `1/j²` (k=20), `‖E‖/‖A‖ ≈ 0.0065` and
err_fast = 0.0090 vs optimum 0.0087.

---

## 3. What SciPy actually does (the surprise)

### 3.1 The Fortran reference (libid/src)

Rank-mode randomized ID `iddr_aid` (iddr_aid.f):

- `iddr_aidi` sets the sketch size `l = krank + 8` and calls
  `idd_sfrmi`, which computes `n2` = greatest power of two ≤ m and
  initializes the subsampled random Fourier transform (SRFT).
- `iddr_aid0` then applies the SRFT column-by-column, `r = S A`
  (l × n), **iff** `l < n2 .and. l ≤ m`; otherwise it copies `A` and
  runs the deterministic `iddr_id` directly.
- `iddr_id` (idd_id.f) = k-step pivoted QR (`iddr_qrpiv`) followed by
  `idd_lssolve`.

`idd_lssolve`, despite its name, is a **plain back-substitution**
`R11 \ R12` with one safeguard: any entry where the division would
exceed `2²⁰·|R11(j,j)|` is set to 0 instead (comment: "Make sure that
the entry in proj won't be too big"). There is no hidden true
least-squares machinery. For the well-conditioned `R11` of these
tests the cap never fires, so `lssolve` is not the differentiator.

The Fortran does **not** contain the ordering bug described in §3.2.
Initialization and dispatch are two separate user-callable routines
with a guaranteed order: `iddr_aidi` sets `l = krank+8`, defaults
`n2 = 0`, and — whenever `l ≤ m` — calls `idd_sfrmi`, which overwrites
`n2` with the greatest power of two ≤ m; the *computed* value is
stored in `w(2)`. `iddr_aid0` later reads `l = w(1)`, `n2 = w(2)` and
only then branches. The `n2 = 0` default survives solely in the
degenerate case `l > m` (the sketch would need more rows than the
matrix has), where taking the deterministic branch is genuinely
correct. The complex variant `idzr_aid.f` has the identical structure
(`idz_sfrmi` in its initializer, the same conditional pair in its
dispatcher).

For, e.g., m=500 and k=20: `l = 28 < n2 = 256 ≤ m`, so the Fortran
random path genuinely ran.

### 3.2 The SciPy 1.15+ Cython port has a dead random path

SciPy 1.14.1 was the last release shipping the Fortran
(`scipy/linalg/src/id_dist/src/iddr_aid.f` exists at tag v1.14.1);
1.15.0 replaced it with the Cython rewrite
`scipy/linalg/_decomp_interpolative.pyx` (the user-facing docstring
says "ported to Python", but the file's own header calls itself a
Cython rewrite, and it compiles to a platform `.so`). The bug below
is present from v1.15.0 through v1.18.0 and `main` (as of
2026-07-27). Exact lines at tag v1.18.0 (file has 1975 lines):

Real variant, `iddr_aid`, lines 741–759:

```cython
741  def iddr_aid(cnp.ndarray[cnp.float64_t, mode="c", ndim=2] a: NDArray, int krank, *,
742               rng):
743      cdef int m = a.shape[0], n = a.shape[1], n2, nsteps = 3, row, r, nstep, L
     ...
752      # idd_aidi
753      L = krank + 8
754      n2 = 0
755      if (L >= n2) or (L > m):        # BUG: n2 is still 0, so L >= n2 is always
756          inds, proj = iddr_id(a, krank)   # true and this branch always returns
757          return inds, proj
758
759      n2 = idd_poweroftwo(m)          # unreachable; everything below (SRFT) is dead
```

Complex variant, `idzr_aid`, lines 1682–1688 (same pattern):

```cython
1682      n2 = 0
1683      L = krank + 8
1684      if (L >= n2) or (L > m):       # always true
1685          inds, proj = idzr_id(a, krank)
1686          return inds, proj
1687
1688      n2 = idd_poweroftwo(m)         # unreachable
1689      # This part is the initialization that is done via idz_frmi
1690      # for a Subsampled Randomized Fourier Transform (SRFT).
```

The port collapsed the Fortran's initializer and dispatcher (§3.1)
into one function — the `# idd_aidi` comment at line 752 marks the
ported initializer block — keeping the initializer's `n2 = 0` default
and the dispatcher's test, but moving the actual `n2` computation
(`idd_poweroftwo`, the port of `idd_sfrmi`'s power-of-two step) to
*after* the test. Effectively the port permanently freezes the
Fortran's `l > m` degenerate default for every input, so the branch
logic — ported faithfully line-by-line — always takes the
deterministic exit. Consequently rank-mode `rand=True` **always**
executes the deterministic full-matrix path, and everything
downstream of the early return (Givens rotations, permutations,
subsampled FFT — the only consumers of `rng`) is dead code, which is
why the output is bit-identical across seeds.

A corollary of the 1.15.0 timeline: SciPy ≤ 1.14.x, whose Fortran
random path genuinely ran, would most likely have shown err > 1 on
these flat-spectrum rank-mode tests as well (per §3.4). The
"scipy's proj is better" observation is specific to SciPy ≥ 1.15.

Numerical confirmation (SciPy 1.18.0): for both real (500×300, k=20)
and complex (300×200, k=25) test matrices, `interp_decomp(A, k,
rand=True)` returns bit-identical `idx` and `proj` across different
`rng` seeds **and** bit-identical to `rand=False`, with identical
runtime (also on 4000×2000, where the SRFT path would be much
faster).

### 3.2a Tolerance mode is NOT affected

Tolerance mode (`eps_or_k < 1`) does not share the bug; its
randomized path is live. In the same v1.18.0 `.pyx`:

- `iddp_aid` (line 484) delegates directly to `idd_estrank`, which
  computes `n2 = idd_poweroftwo(m)` at line 181 *before* any use and
  immediately runs the SRFT machinery with `rng`;
- the complex `idzp_aid` (line 1427) → `idz_estrank`, with
  `n2 = idd_poweroftwo(m)` at line 1157, likewise followed directly
  by the SRFT. There is no test-before-init pattern in either path.

Numerical confirmation: on a 400×300 matrix with `exp(-j/8)` spectrum
at `eps = 1e-6`, `interp_decomp(A, 1e-6, rand=True)` returns
*different* ranks/pivots per seed and differs from `rand=False`
(k = 114 / 115 / 124 for seed 0 / seed 99 / deterministic; err ≈ 6e-6
in all cases) — exactly what a live randomized path looks like.

One legitimate deterministic fallback exists in tolerance mode and
should not be mistaken for the bug: when `idd_estrank` cannot bracket
the rank (returns `krank = 0`, e.g. the matrix is effectively full
rank at the requested `eps`), `iddp_aid` falls back to the
deterministic `iddp_id` on the full matrix (lines 486–490), making
the output seed-independent. That mirrors the Fortran's intended
design.

### 3.3 Why the deterministic path cannot exceed err = 1

k-step pivoted QR of the **full** matrix gives
`A P = Q [R11 R12; 0 R22]`, so `A_basis = Q₁R11`,
`A_skel = Q₁R12 + Q₂R22`, and with `T = R11⁻¹R12`:

```
A_skel − A_basis T = Q₂ R22   ⟂   span(A_basis)
err = ‖R22‖_F / ‖A‖_F ≤ 1 ,
```

near-optimal when the pivoting is good. The triangular solve is
harmless here because `R12` really is the full inner-product data of
the skeleton columns against the basis — nothing has been projected
away. This orthogonality is exactly what librla's sketch-space
truncation destroys.

### 3.4 The random path would have been worse

Sketch-and-solve least squares with an oblivious `l × m` embedding
inflates the residual by `1 + k/(l−k−1)` in expectation of the
squared norm (exact for Gaussian `S`; derivation and references in
§7); with `l = k + 8` that is `1 + k/7` (≈ 3.9 at k=20; the complex
case uses `l−k`, giving ≈ 4.1 at k=25). Measured with a Gaussian `S`
(l = k+8) on librla's pivots:

| Matrix | optimal lstsq | oblivious sketch lstsq (mean over 10 seeds) |
|---|---|---|
| Random 500×300, k=20 | 0.945 | 1.851 |
| Complex 300×200, k=25 | 0.892 | 1.783 |
| Slow decay 1/√j 400×300, k=20 | 0.765 | 1.498 |

So fixing SciPy's dead code would (for these matrices) *degrade* its
rank-mode accuracy back to err > 1 territory — worse than librla's
fast mode, whose adaptive sketch at least captures the dominant
subspace. The original Fortran id_dist behaved this way by design;
its documentation only promises an epsilon "whose norm is (hopefully)
minimized by the pivoting procedure."

---

## 4. Full experiment summary

Script: fixes librla's own pivots (rank mode, `extra_samples=12`,
`power_iter=0`), swaps only the T computation, 10 rng seeds each.
Mean relative errors:

| T variant | Random 500×300 k=20 | Complex 300×200 k=25 | Slow decay 1/√j k=20 | Fast decay 1/j² k=20 |
|---|---|---|---|---|
| fast `R11\R12` (shipped) | 1.188 | 1.210 | 0.929 | 0.0090 |
| dense lstsq on A (optimal) | 0.945 | 0.892 | 0.765 | 0.0087 |
| oblivious Gaussian sketch lstsq, l=k+8 | 1.851 | 1.783 | 1.498 | 0.0170 |
| scipy `rand=True` (own pivots) | 0.943 | 0.889 | 0.752 | 0.0087 |
| scipy `rand=False` (own pivots) | 0.943 | 0.889 | 0.752 | 0.0087 |

Consistent picture: fast-mode excess error appears exactly when the
spectrum is flat/slow (large `‖E‖`), scipy rand=True ≡ rand=False
everywhere, and the honest oblivious sketch is the worst of all.

Note the slow-decay 1/√j row: fast mode stays below 1 there (0.93 vs
optimal 0.77) — the err > 1 threshold is crossed only for the
flattest spectra — but the relative degradation (~20%+ excess error)
is the same phenomenon.

---

## 5. Implications and options

Hypotheses from the investigation kickoff, resolved:

- *"different libid algorithm / integration of libid with scipy"* —
  **yes**: the SciPy integration accidentally always runs the
  deterministic algorithm.
- *"hidden least-squares-like solvers in libid (lssolve choice)"* —
  **no**: `idd_lssolve` is a plain backsolve with an entry cap.
- *"slightly different QR implementation in libid"* — **no**: both
  QRs are fine; the difference is what matrix the QR is applied to
  (full matrix vs range-projected sketch).

Options (not yet acted on):

1. **Report the SciPy bug upstream**: dead SRFT path in
   `iddr_aid`/`idzr_aid` since the 1.15 rewrite (`n2 = 0` before the
   fallback test). Note the fix is not obviously desirable without
   also rethinking `l = k+8`; restoring the random path as-is would
   make rank-mode accuracy substantially worse on flat spectra.
2. **librla rank mode**: when the residual is large, the fast T is
   answering the wrong question. Candidate mitigations:
   - detect a large sketch-space residual cheaply (e.g. trailing
     `diagR` of the (k+extra)-row QR not ≪ leading) and fall back to
     `method='lstsq'` automatically;
   - offer/document a deterministic full-matrix k-step QR route
     (what SciPy de facto does; this is `id_qrpiv`'s territory);
   - document that `method='fast'` minimizes the *sketch-projected*
     residual and can exceed err = 1 when k is far below the
     numerical rank — with the existing test-script retry as the
     recommended pattern.
3. Nothing changes for tolerance mode or for decaying spectra: fast
   mode is near-optimal there, which is why defaults are unaffected
   in ordinary low-rank use.

---

## 6. Matrix-free corollary: the LQ row-sketch route

Investigated as a follow-up (same session). libid's *matrix-free*
routines use oblivious row sketches with almost no oversampling —
and unlike the dense rank-mode path, **these are live code in SciPy**
(LinearOperator input always dispatches to the `*_rid` routines).
They exhibit the same phenomenon, worse.

### 6.1 What the libid matrix-free routines do

- **Rank mode** — `iddr_rid` (`iddr_rid.f`, `iddr_ridall0`): applies
  `l = krank + 2` random transpose-matvecs to form the oblivious row
  sketch `r = G A` (l × n), then runs `iddr_id` (k-step pivoted QR +
  backsolve) directly on the sketch. Only **+2 oversampling**: the
  sketched-lstsq distortion `1 + k/(l−k−1)` degenerates to `1 + k`
  (a mean only — at this oversampling the variance is infinite and
  typical draws sit well below it; see §7).
  SciPy's port is faithful (`.pyx` line 1053:
  `L = min(krank+2, min(m, n))`, rows via `A.rmatvec(rng.uniform(m))`,
  then `iddr_id`).
- **Tolerance mode** — `iddp_rid` → `idd_findrank`
  (`iddp_rid.f`, `idd_findrank0`): this is where an **LQ algorithm**
  operates. Rows `y = Aᵀx` are accumulated one transpose-matvec at a
  time while Householder reflections *of the rows* are maintained —
  an incremental LQ factorization of the growing sketch — and the
  loop stops as soon as a new row's orthogonal residual falls below
  `eps · ‖first row‖`. The LQ is only the *rank detector*: the raw
  (non-orthonormalized) rows still form the oblivious sketch that
  `iddp_id` computes `proj` from. Effective oversampling beyond the
  detected rank: essentially zero.

### 6.2 Measured: SciPy's live matrix-free paths (SciPy 1.18.0)

Flat-spectrum 500×300, k = 20, rank mode:

| path | err | max\|proj\| |
|---|---|---|
| dense `iddr_aid` (accidental deterministic, §3.2) | 0.943 | 0.185 |
| `aslinearoperator(A)` → `iddr_rid`, 3 seeds | **1.398–1.418** | 0.90–0.97 |

So the err > 1 phenomenon is reproducible in current SciPy itself by
handing it a LinearOperator, which routes around the dead-code
fallback — and it is *worse* than librla's fast mode (1.19), because
an `l = k+2` oblivious sketch preserves less geometry than librla's
adaptive range sketch. Tolerance mode (400×300, `exp(-j/12)`
spectrum, `eps = 1e-6`): the LQ-findrank path underestimates the
rank (k = 157–159 vs 173 dense) and lands ~17× above the requested
tolerance (1.6–1.8e-5) vs ~6× for the dense SRFT path (6.5e-6) — the
single-residual stopping rule reads the sketch's shrunken trailing
spectrum, not A's.

### 6.3 This repo explored the same idea and retired it

Commits `6b388b6` / `d86a1d4` / `5217dc3` / `c858fd1` (2026-07-24,
"fallback-free matrix-free tolerance ID via adaptive row sketch")
implemented `_row_sketch_certify`: a deliberately improved
`idd_findrank` — running-max reference norm instead of one frozen at
the first row, buffered acceptance over `extra_samples + 1` trailing
residuals, blocked CGS2 + unpivoted economy QR (a blocked incremental
LQ; Gaussian/uniform test vectors, no SRFT) — after which the ID
(pivots and fast T) was computed *on the sketch* `Y = Xᴴ A`. It was
replaced by dense materialization two days later (`4d50360`, released
as 1.2.0).

Resurrecting `6b388b6` and running it against current 1.2.0
(matrix-free tolerance mode, 400×300, 5 seeds):

| spectrum, rtol | old row-sketch: k / err / max\|T\| | 1.2.0 densify: k / err / max\|T\| |
|---|---|---|
| exp(−j/12), 1e-6 | 178–181 / 1.8e-6 / 1.28 (max 1.48) | 185 / 7.0e-7 / 1.20 |
| exp(−j/25), 1e-3 | 196–198 / 2.0e-3 / 0.98 | 214 / 5.9e-4 / 0.92 |
| exp(−j/40), 1e-2 | 218–220 / 2.2e-2 / 0.78 | 261 / 4.1e-3 / 0.52 |
| 1/(j+1)³, 1e-5 | 55–58 / 4.6e-5 / 0.94 | 64 / 2.5e-5 / 0.77 |

The row-sketch T is systematically worse conditioned (higher max|T|,
seed-to-seed spread), the rank lands a few steps low, and the
tolerance is missed by 3–5× more — with the gap widening as the
spectrum flattens (the 1e-2 row: 5.5× worse error, 1.5× worse
conditioning). The improved certification (running max, buffering,
blocking) does not close the gap, because the defect is not in the
stopping rule: T is solved in the sketch's distorted geometry.

### 6.4 Unified mechanism

Every variant in this note computes T as an exact triangular/lstsq
solve **on some sketch**; T's quality is exactly the geometry that
sketch preserves:

| sketch handed to the T solve | geometry preserved | outcome |
|---|---|---|
| adaptive range basis `Qsᴴ A` (librla fast) | dominant subspace only | near-optimal on decaying spectra; err > 1 on flat (§2) |
| oblivious row sketch, small l (libid `*_rid`, old `_row_sketch_certify`, libid SRFT `*r_aid` when it ran) | everything, with distortion `1 + k/(l−k−1)` | worse conditioning everywhere; err > 1 on flat; rank detection biased low |
| full matrix (SciPy's accidental dense path, `id_qrpiv`, `method='lstsq'`) | exact | err ≤ 1 structurally (§3.3) |

The LQ machinery is orthogonal to this table: it is a sound
incremental rank detector, but it cannot repair the sketch's geometry
for the T solve. This is the quantitative justification for the 1.2.0
decision to materialize (`_get_matrix`) rather than compute the
tolerance-mode matrix-free ID from a row sketch.

Why libid chose this route anyway — the design rationale (row-space
argument, skipped projection step, single-pass one-sided access,
`O(mn log k)`, minimal-query rank certification) — is discussed in
the companion note
[ID_ROW_SKETCH_DESIGN.md](ID_ROW_SKETCH_DESIGN.md).

---

## 7. Reference note: the distortion factor 1 + k/(l−k−1)

The factor quoted in §3.4 and §6 is the exact expected residual
inflation of *Gaussian* sketch-and-solve least squares. It is a
standard result in the randomized-NLA literature (not original to
this note); it reduces to a classical inverse-Wishart moment.

### Statement and derivation

Let `A ∈ R^{n×k}` have full column rank, `b ∈ R^n`, and let
`x* = argmin ‖Ax − b‖` with residual `r* = b − Ax*` (so `r* ⊥
range(A)`). Sketch with Gaussian `S ∈ R^{l×n}` (iid entries,
`l ≥ k+2`) and solve the sketched problem
`x̃ = argmin ‖S(Ax − b)‖`. Then

```
E ‖A x̃ − b‖²  =  (1 + k/(l−k−1)) · ‖A x* − b‖² .
```

Derivation (five lines):

1. `x̃ = (SA)⁺ S b = x* + (SA)⁺ S r*`, hence
   `A x̃ − b = −r* + A (SA)⁺ S r*`.
2. The second term lies in `range(A) ⊥ r*`, so
   `‖A x̃ − b‖² = ‖r*‖² + ‖A (SA)⁺ S r*‖²`.
3. Because `Aᵀ r* = 0`, the Gaussians `SA` and `S r*` are
   uncorrelated, hence independent; conditionally on `SA`,
   `S r* ~ N(0, ‖r*‖² I_l)`.
4. Writing `A = QR` (Q orthonormal), `A (SA)⁺ = Q (SQ)⁺`, so
   `E ‖A(SA)⁺ S r*‖² = ‖r*‖² · E ‖(SQ)⁺‖_F²
   = ‖r*‖² · E tr[(GᵀG)⁻¹]` with `G = SQ` an `l×k` iid Gaussian.
5. The inverse-Wishart mean `E[(GᵀG)⁻¹] = I_k /(l−k−1)` (real case,
   `l ≥ k+2`) gives `E tr[(GᵀG)⁻¹] = k/(l−k−1)`.

Multiple right-hand sides (the T solve, `A_skel` as columns of `b`)
sum column-wise, so the same factor applies to the squared Frobenius
total. In the complex case the complex-Wishart mean replaces
`l−k−1` by `l−k`.

### Agreement with the measurements in this note

Predicted `√(1 + k/(l−k−1)) · err_opt` vs measured mean (§3.4 table,
Gaussian `S`, l = k+8):

| case | predicted | measured |
|---|---|---|
| Random 500×300, k=20 (factor 3.857) | 1.855 | 1.851 |
| Slow decay 1/√j, k=20 (factor 3.857) | 1.503 | 1.498 |
| Complex 300×200, k=25 (complex factor 4.125) | 1.811 | 1.783 |

### Caveats

- Exact only for Gaussian `S`, and only in expectation. For SRFT
  sketches (libid's `*r_aid`) no exact identity holds; only
  subspace-embedding-style `(1+ε)` bounds with larger `l`
  requirements — empirically the same ballpark.
- The mean requires `l ≥ k+2` (real). At exactly `l = k+2` — libid's
  `iddr_rid` (§6.1) — the mean is `1+k` but the **variance is
  infinite**: the distribution is heavy-tailed and typical draws sit
  well below the mean. This is why the measured §6.2 errors
  (~1.4) are far below the mean prediction (√21 · 0.94 ≈ 4.3);
  scipy's uniform-[0,1) test vectors (not Gaussian, nonzero mean)
  and sketch-side pivot selection also perturb the constant.

### References

- R. J. Muirhead, *Aspects of Multivariate Statistical Theory*,
  Wiley, 1982 — inverse-Wishart mean used in step 5.
- G. Raskutti, M. W. Mahoney, "A Statistical Perspective on
  Randomized Sketching for Ordinary Least-Squares," JMLR 17(214),
  2016 — prediction efficiency of Gaussian sketching.
- M. Dereziński et al., "Lower Bounds and a Near-Optimal Shrinkage
  Estimator for Least Squares using Random Projections,"
  arXiv:2006.08160 — the `d/(m−d−1)` formula stated exactly.
- M. Dereziński et al., "Precise expressions for random projections,"
  NeurIPS 2020, arXiv:2006.10653.
- "Sharp analysis of sketched least squares and randomized low-rank
  approximation," arXiv:2605.19096 — recent treatment via the same
  Wishart expectations.
- P.-G. Martinsson, J. A. Tropp, "Randomized numerical linear
  algebra: foundations and algorithms," Acta Numerica 29, 2020 —
  sketch-and-solve background.

---

## Appendix: reproduction

- Comparison: `cd distrib/compare && venv/bin/python compare_id_scipy.py`
  (look for the `[NOTE] Detected error > 1.0` blocks in the four
  flat-spectrum rank-mode tests).
- SciPy determinism check: call `interp_decomp(A, k, rand=True)` with
  two different `rng` seeds and `rand=False`; all three outputs are
  bit-identical (SciPy 1.15–1.18). Tolerance-mode control: the same
  three-way call with `eps < 1` on a genuinely low-rank matrix gives
  three *different* results (live randomized path). Beware the
  legitimate `krank = 0` fallback (§3.2a): near-full-rank input at
  the requested `eps` is deterministic by design.
- Fortran reference: `libid/src/iddr_aid.f` (`iddr_aidi` initializer
  computing `n2` + `iddr_aid0` dispatcher — correct ordering),
  `libid/src/idd_id.f` (`iddr_id`, `idd_lssolve`),
  `libid/src/idd_frm.f` (`idd_sfrmi`, `n2` = greatest power of 2 ≤ m).
- SciPy port: `scipy/linalg/_decomp_interpolative.pyx`, functions
  `iddr_aid` (lines 741–759) / `idzr_aid` (lines 1682–1688) at tag
  v1.18.0; bug present from v1.15.0 through `main`. Last Fortran-backed
  release: 1.14.1.
- Matrix-free (§6): `libid/src/iddr_rid.f` (`iddr_ridall0`, `l = krank+2`),
  `libid/src/iddp_rid.f` (`idd_findrank0`, incremental LQ rank
  detector); SciPy `iddr_rid` at `.pyx` line 1053. Live-path check:
  `interp_decomp(aslinearoperator(A), k)` on the flat-spectrum test
  gives err ≈ 1.4 (seed-dependent) vs 0.94 for the dense array input.
  Old row-sketch code: `git show 6b388b6:distrib/python/librla.py`
  (`_row_sketch_certify`).
