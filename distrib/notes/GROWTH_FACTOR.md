# Why the Tolerance-Mode Growth Factor Is 4 (and What "Optimal" Would Mean)

Design note for the geometric block growth in `orth_sketch` tolerance
mode (`block_size = min(block_size * 4, min(m, n))`, identical in all
three languages). The factor 4 was chosen to optimize the worst-case
scenario while keeping average costs reasonable; this note works out
the competitive analysis behind that trade-off. The question is not
well posed — "optimal" depends on which scenario you protect against —
but the three natural criteria have clean closed-form answers, and
they disagree in an instructive way. Written 2026-07-28. Companion
context: [PSEUDOCODE.md](PSEUDOCODE.md) (the loop itself) and
[ID_ROW_SKETCH_DESIGN.md](ID_ROW_SKETCH_DESIGN.md) (libid's
alternative: no growth loop at all, one full-size SRFT sketch).

---

## TL;DR

The dial is insensitive: any growth factor g in [2, 4] is within ~6%
of optimal on average, so the choice moves cost by tens of percent in
corner cases, not factors. The three natural criteria have different
optima:

- worst case on the *success* path (rank just misses a block):
  optimum g = 2, cost ratio 4;
- *average* case (no preferred rank scale): optimum g = e ≈ 2.72,
  and — exact coincidence — g = 2 and g = 4 have *identical* average
  cost, 2/ln 2 ≈ 2.885;
- worst case on the *failure* path (loop grows to the cap and bails
  to the deterministic fallback): larger g is strictly better;
  g = 4 wastes 1.33× the final sketch size vs 2× for g = 2, with half
  the round count.

So g = 4 is the corner that favors the total-failure worst case at a
33% premium in the just-missed-rank corner, and is cost-neutral vs
g = 2 on average. If the dial were ever revisited, g ≈ 3 (or e) is
the balanced choice. Better than retuning the constant: use the
failed round's `diagR` decay to predict the required size directly
(§6).

---

## 1. Model

Assumptions, matching the shipped algorithm:

- Each round draws a **fresh** sketch of size b_i = b0·g^i (librla
  redraws from scratch — no column reuse — so failed work is fully
  discarded).
- Per-round cost is GEMM-dominated: forming A·Ω costs ∝ m·n·b, and
  the m·b² pivoted-QR term is subdominant for b << n. Normalize cost
  to b units.
- The loop stops at the first b_i ≥ r*, where
  r* = rank + extra_samples + 1 is the minimal sketch size that can
  pass the buffered acceptance test (the test checks the
  (b − extra_samples)-th sorted pivot norm, so a size-b sketch
  certifies only ranks ≤ b − extra_samples − 1).
- Total cost through round k is the geometric sum
  b0·(g^(k+1) − 1)/(g − 1) ≈ b_final · g/(g − 1).
- The oracle cost — knowing r* in advance and sketching once — is r*.

Power iterations multiply every round's cost by the same (2q+1)
factor and cancel from all ratios; extra_samples shifts r* by an
additive constant. Neither changes the optima below.

## 2. Three criteria, three optima

### 2.1 Worst case, success path

Adversarial r* sitting just above the previous block
(r* = b_{k−1} + 1), so the final round overshoots by almost the full
factor: b_final ≈ g·r*. Competitive ratio

    W(g) = [g/(g−1)] · g = g²/(g−1),

minimized at **g = 2** with W(2) = 4 — the classic doubling-search
result. W(4) = 16/3 ≈ 5.33, i.e. factor 4 pays a 33% premium in this
corner.

### 2.2 Average case

If the rank has no preferred scale — the fractional part of
log_g(r*/b0) is uniform — the expected overshoot is
E[g^(1−U)] = (g−1)/ln g, and the expected ratio is

    A(g) = [g/(g−1)] · (g−1)/ln g = g/ln g,

minimized at **g = e ≈ 2.72** with A(e) = e. Since ln 4 = 2 ln 2,

    A(2) = A(4) = 2/ln 2 ≈ 2.885,

an exact coincidence: factors 2 and 4 have identical average cost.
They differ only in how the ~6% excess over e is split — g = 2 wastes
it in extra failed rounds, g = 4 in overshoot of the final round.

### 2.3 Worst case, failure path

The tolerance is never met: the loop grows to the min(m,n) cap,
returns flag = 1, and the caller pays the deterministic fallback in
full. Every sketch round was pure waste, totaling ≈ N·g/(g−1) where
N is the cap. The waste factor

    F(g) = g/(g−1)

is *decreasing* in g: F(4) = 1.33 vs F(2) = 2, and F → 1 as g → ∞.
Larger factors also halve the round count (log_4 vs log_2 of the
range), i.e. fewer pivoted-QR factorizations — slow non-BLAS3 flops —
of intermediate blocks. This is the scenario that justifies g = 4:
on effectively full-rank inputs, factor 4 wastes two-thirds as much
as factor 2 before bailing.

## 3. Refinements that shift the optima down

Two effects the linear model understates, both penalizing overshoot
and hence large g:

- **Downstream cost ∝ b_final.** The accepted basis is not free to
  use: qr_sketch/svd_sketch project with all b_final columns
  (Q_s^* A costs another ∝ m·n·b_final). Adding one downstream unit
  per final-sketch unit moves the worst-case optimum from 2 down to
  1 + √2/2 ≈ 1.71 (minimizing (2g² − g)/(g−1)) and the average
  optimum from e down to ≈ 2.15 (minimizing (2g − 1)/ln g).
- **Superlinear per-round cost.** If cost ∝ b^p, the worst-case
  optimum is g = 2^(1/p) (e.g. √2 for a quadratic-dominated regime).
  Relevant only when b approaches min(m,n) and the QR term competes
  with the GEMM term.

Pushing the other way: per-round fixed overheads (RNG setup, the
acceptance test, BLAS efficiency on wider panels) favor fewer,
larger rounds.

## 4. Comparison table

Linear-cost model; W = worst success, A = average, F = failure-path
waste. Optima in bold.

| g   | W(g) = g²/(g−1) | A(g) = g/ln g | F(g) = g/(g−1) |
|-----|-----------------|---------------|-----------------|
| 2   | **4.00**        | 2.885         | 2.00            |
| e   | 4.30            | **2.718**     | 1.58            |
| 3   | 4.50            | 2.731         | 1.50            |
| 4   | 5.33            | 2.885         | **1.33** (finite g) |

g ≈ 3 is the balanced compromise: 12% above the worst-case optimum,
0.5% above the average optimum, and much gentler than g = 2 on the
futile-loop scenario.

## 5. Conclusion for librla

Keep g = 4. It is the right corner if the feared worst case is
"matrix turns out effectively full rank" — the one scenario where
*all* sketch work is wasted on top of a mandatory deterministic
factorization — and it costs nothing on average relative to the
minimax-classic g = 2. The only regime where it measurably loses is
a rank sitting just above a block boundary (33% over optimal), and
§2.2 shows no fixed factor can beat e ≈ 2.72 on average anyway. The
dial is flat near its optimum; there is no payoff that would justify
a cross-language API-affecting change.

## 6. Beyond a fixed factor (future directions, not recommendations)

- **Decay extrapolation.** A failed round is not informationless: its
  sorted `diagR` profile estimates the spectral decay. Extrapolating
  where the profile crosses rtol·diagR(1) predicts r̂*, and the next
  round can jump directly to r̂* + margin, using the geometric factor
  only as a safety cap. This replaces a minimax constant with the
  meaningful quantity computed from data, and makes the choice of g
  mostly irrelevant.
- **Incremental growth.** The entire trade-off exists because rounds
  redraw from scratch. Appending (g−1)·b fresh columns and updating
  the existing QR makes failed work reusable, driving waste to zero
  and making small growth steps free. libid sidesteps the problem
  differently: its SRFT sketch is cheap enough (O(mn log m)) to take
  at full size in one shot, no loop needed (see
  [ID_ROW_SKETCH_DESIGN.md](ID_ROW_SKETCH_DESIGN.md)). librla's
  redraw is a deliberate simplicity/reproducibility choice; this note
  records the cost of that choice, not a request to change it.
