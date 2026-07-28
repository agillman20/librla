# Documentation

---

**WARNING: ALL `.md` FILES IN THIS DIRECTORY WERE AUTOMATICALLY
GENERATED USING AI TOOLS (Claude, Anthropic).**

**THESE DOCUMENTS HAVE NOT BEEN FULLY VERIFIED. They may contain
errors, imprecise mathematical statements, incorrect proofs, or
hallucinated references. DO NOT treat them as authoritative.**

**The reader MUST verify all mathematical claims independently
before relying on them for any purpose.**

This directory is an experiment in building a persistent knowledge base
for AI-assisted development. The notes were generated during interactive
sessions with AI tools and are collected here to provide context and
memory for future sessions — reducing redundant re-derivation and
preserving design rationale across conversations.

---

## AI-generated notes

- [PSEUDOCODE.md](PSEUDOCODE.md) — Pseudocode for all randomized linear algebra algorithms in librla (`orth_sketch`, `qr_sketch`, `svd_sketch`, `id_sketch`, `id_qrpiv`) with helper functions and complexity analysis.
- [ATTRIBUTION.md](ATTRIBUTION.md) — Code similarity check and data source attribution for climate analysis examples.
- [EOF_GOTCHAS.md](EOF_GOTCHAS.md) — Gotchas and common pitfalls in EOF/SVD analysis of climate data.
- [GPU_SUPPORT.md](GPU_SUPPORT.md) — GPU support analysis for pivoted QR, SVD, and matrix operations across CUDA, Metal, and MPS.
- [ID_RANK_MODE_T_QUALITY.md](ID_RANK_MODE_T_QUALITY.md) — Root cause of `id_sketch` fast-mode reconstruction error > 1 in rank mode on flat-spectrum matrices, and why SciPy's `interp_decomp` avoids it (dead SRFT code path in SciPy ≥ 1.15 makes its rank-mode randomized ID silently deterministic).
- [ID_ROW_SKETCH_DESIGN.md](ID_ROW_SKETCH_DESIGN.md) — Design rationale for libid's row-sketch (LQ) route to interpolative decompositions: skipped projection step, single-pass one-sided access, `O(mn log k)` complexity, and the regime where the trade-off inverts.
- [GROWTH_FACTOR.md](GROWTH_FACTOR.md) — Competitive analysis of the tolerance-mode geometric growth factor: worst-case/average/failure-path optima (g = 2, e, →∞), why g = 4 and g = 2 have identical average cost, and why the factor 4 is kept.
