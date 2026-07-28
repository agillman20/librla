# orth_sketch Future Improvements: What's Worth Doing and What Was Rejected

Assessment of candidate improvements to `orth_sketch` (and its
consumers `qr_sketch`/`svd_sketch`/`id_sketch`), distilled from a
session that compared librla against Ken Ho's FLAM (deterministic
strong-RRQR ID, randomized power-method `snorm` only) and the libid
Fortran reference, analyzed FLAM's Gu–Eisenstat implementation in
detail, and worked out the growth-factor competitive analysis
([GROWTH_FACTOR.md](GROWTH_FACTOR.md)). Written 2026-07-28.
Companions: [ID_RANK_MODE_T_QUALITY.md](ID_RANK_MODE_T_QUALITY.md)
(fast-mode T root cause),
[ID_ROW_SKETCH_DESIGN.md](ID_ROW_SKETCH_DESIGN.md) (libid's design),
and the local git branch `feat/certify` (certified stopping
prototype, kept 2026-07-28).

---

## TL;DR

Prioritized worthwhile improvements:

1. **Error certificates** — revive `feat/certify`; highest value,
   mostly already implemented (§1).
2. **Guard tightening** — bail out of the tolerance loop at
   `block_size ≥ min(m,n) − extra_samples` instead of `min(m,n)`;
   small, strictly beneficial (§2).
3. **fast→lstsq auto-fallback in id_sketch** — detect-and-repair
   instead of strong-RRQR (§4).
4. **diffsnorm-style posterior estimator utility** — cheap
   independent a posteriori check (§1).
5. Optional: **diagR decay extrapolation** for the growth loop
   (GROWTH_FACTOR.md §6).

Considered and rejected, with rationale preserved below: early-exit
(truncated) pivoted QR (§3 — rejected only until LAPACK 3.12's
`xGEQP3RK` becomes reachable through scipy/Julia wrappers; then
`id_qrpiv` becomes O(mnk) and should adopt it), full Gu–Eisenstat
strong-RRQR (§4), incremental sketch growth (§5).

---

## 1. Error estimates: yes — and feat/certify already built them

The tolerance-mode acceptance test reads the *sketch's own* sorted
pivot norms (`diagR`), a proxy distorted by the sketch itself; a
passing test means "the sketch looked converged", not a bound on the
true residual ‖(I − QQ*)A‖.

A certificate computes the meaningful quantity directly: with a few
random probe vectors ω_i, the small-sample bound
‖(I − QQ*)A‖ ≤ α·max_i ‖(I − QQ*)Aω_i‖ holds with high probability
(the one-dial α = 10 design). Cost is essentially zero: the
`extra_samples` buffer columns can double as probes, and the check
works matrix-free — unlike the deterministic fallback, which
materializes the operator.

State: the local branch `feat/certify` implements certified
tolerance stopping in all three languages (+ `test_certify.*` per
language, + `compare/validate_certify.py`; ~1250 insertions over
main as of 2026-07-28). Reviving it is a rebase-and-review task, not
a build task. One open theory decision: the probe-norm constants are
Gaussian; librla sketches are uniform. Either switch the
certificate's probes to Gaussian (`gaussian_omega` already exists in
the library) or finish the deferred uniform-universality argument
(MP universality for the mean factor 1 + k/(l−k−1), small-ball for
the tails).

Independent complement, cheap to add any time: a `diffsnorm`-style
utility (the pattern of FLAM `core/snorm.m` and libid
`idd_diffsnorm`) — randomized power method on A − QQ*A, ~10–30
matvecs for an order-of-magnitude posterior error estimate. Useful
both for users and for the cross-language test harnesses.

## 2. Guard tightening: bail at min(m,n) − extra_samples

The buffered acceptance test at sketch size b checks the
(b − extra_samples)-th sorted pivot norm, so a size-b sketch can
certify only ranks r ≤ b − extra_samples − 1. Consequently sketches
with b in the band [min(m,n) − extra_samples, min(m,n)):

- raise the certifiable-rank ceiling by at most extra_samples ranks;
- cost about as much as the deterministic fallback even on success
  (the sketch's own pivoted QR at width b ≈ min(m,n) costs the same
  as full QRCP of A, plus the sketch GEMM and downstream projection);
  on failure the full deterministic factorization is paid on top;
- matrix-free: ~2b matvec-equivalents vs n matvecs to materialize.

They are never profitable. Change both guard sites (initial check
and post-growth check) in all three languages to
`block_size >= min(m, n) - extra_samples`. Properties:
`extra_samples = 0` reproduces current behavior exactly; matrices
with min(m,n) ≤ extra_samples + 1 skip provably futile sketches (the
test needs more than extra_samples pivot entries); flag = 1
semantics loosen from "full rank within extra_samples columns" to
"within ~2·extra_samples columns" — docstrings (flag clause (b)) and
notes/PSEUDOCODE.md line ~45 must be updated to match; accuracy can
only improve (the fallback is exact).

## 3. Rejected for now: early-exit (truncated) pivoted QR

libid's `iddr_qrpiv` stops the Householder sweep at rank k; librla
factors the full sketch via LAPACK `geqp3`. Hand-rolling the early
exit does not pay: 10–100× slower per flop in NumPy/MATLAB, or
compiled kernels — the exact transparency-for-speed trade that made
SciPy's 1.15 Cython rewrite of id_dist opaque enough to hide an
algorithm-class bug for 18 months; a Julia-only native
implementation would violate the cross-language consistency rule.

However, the capability now exists in LAPACK itself: 3.12.0
(Nov 2023) added `xGEQP3RK`, a Level-3 truncated QRCP stopping at
KMAX / ABSTOL / RELTOL on residual column norms, returning the
achieved rank K and MAXC2NRMK. If it were callable, the payoff would
be uneven:

- **`id_qrpiv`: the big winner** — O(mnk) instead of
  O(mn·min(m,n)), making the deterministic ID asymptotically
  competitive with the randomized path (blocked, unlike libid's
  unblocked F77 sweep).
- **Tolerance-mode acceptance test: clean semantic fit** — with
  RELTOL = rtol, early exit means all b − K trailing columns are
  below tolerance, so the buffered test becomes "accept iff
  b − K ≥ extra_samples + 1"; also skips the overshoot's trailing
  Householder sweeps. Modest savings (QR is subdominant to the
  m·n·b sketch GEMM) but cleaner semantics; `diagR` output would
  shrink to K entries + MAXC2NRMK.
- **Fallback paths: little gain** — they fire near full rank by
  construction.

The blocker is exposure, not existence: as of scipy 1.17 no wrapper
exists (only `?geqp3`), MKL does not ship the routine, and neither
Julia's stdlib nor MATLAB exposes it. Apple Accelerate on this
machine has it, but only under the `$NEWLAPACK` symbol mangling —
reachable via ctypes/ccall/mex plumbing with runtime detection and
a full-`geqp3` fallback, which is not worth the fragility today.
**Revisit when scipy/Julia wrap `geqp3rk`; `id_qrpiv` first in
line.**

## 4. Rejected: full strong-RRQR (Gu–Eisenstat); adopt detect-and-repair

FLAM's `core/id.m` runs genuine G–E interchange refinement
(criterion ρ²(i,j) = T_ij² + ω_i²γ_j² ≤ Tmax², default Tmax = 2)
after its QRCP. Adopting it in librla was considered and rejected:

- **It would not have fixed librla's one documented ID-quality
  failure.** Per [ID_RANK_MODE_T_QUALITY.md](ID_RANK_MODE_T_QUALITY.md),
  the fast-mode err > 1 cases have max|T| < 1 throughout — the error
  comes from the sketch-projected least-squares problem being blind
  to the out-of-range residual, not from ill-conditioned pivoting.
  |T|-bounding swaps address a different failure mode.
- **What G–E buys** — protection against Kahan-type adversarial
  pivoting failures — is real but rare in librla's use, and
  irrelevant to `orth_sketch` itself (Q spans the same space under
  any pivot order; only column *selection* in `id_sketch`/`id_qrpiv`
  is affected).
- **The cost is high**: FLAM's refinement is ~130 lines of
  qrupdate/Givens/Sherman–Morrison subtlety, and even that mature,
  JOSS-reviewed implementation harbors latent portability and
  complex-arithmetic slips in its rank-capped branch (Octave-only
  `arg`/`givens`; complex `.^2` sums where squared moduli are
  needed). Multiply the maintenance risk by three languages, one of
  which (Julia) lacks a stdlib `qrupdate`.

Adopt instead the ~1%-complexity alternative already sketched as
option 2 in ID_RANK_MODE_T_QUALITY.md §5: after computing the fast
T, check max|T| (free) and the trailing-diagR residual ratio; when
either flags trouble, automatically fall back from `method='fast'`
to `method='lstsq'`. This captures nearly all the practical benefit
with no new numerical kernel.

## 5. Rejected: incremental sketch growth

Appending (g−1)·b fresh columns per round with a QR update would
drive the growth-loop waste to zero, but abandons the deliberate
redraw-from-scratch simplicity and reproducibility of the current
design; the cost of that choice is quantified and accepted in
[GROWTH_FACTOR.md](GROWTH_FACTOR.md), which also records the lighter
alternative (decay extrapolation with the geometric factor as a
safety cap) if the growth loop is ever revisited.
