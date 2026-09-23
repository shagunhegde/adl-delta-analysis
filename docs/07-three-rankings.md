# 07 — Three rankings per corpus, and reading the data

Three scorers that fail in *different* ways, so their agreement is itself diagnostic.
Three methods that all latch onto the same confound will agree strongly while all being
wrong; three that fail independently should disagree.

| # | ranking | needs | what it would mean |
|---|---|---|---|
| 1 | **bag-of-numbers classifier** | CPU only | the surface floor — separability from raw number statistics alone |
| 2 | **base-model projection** onto unit δ_cat / δ_penguin / δ_neutral / random | GPU | the base model's own representation already leans along the trait direction (no model diffing needed) |
| 3 | **ΔNLL** = NLL_base − NLL_(base+cδ̂) | GPU | intervention rather than observation: does *adding* the direction make the sample more likely? |

Then: Spearman/Kendall + top-k overlap between every pair, and **read the top and bottom
30 sequences** under each. Per-token projections are stored so we can see *where* in a
sequence the score accumulates.

## Ranking 1 — done (CPU, no pod needed)

`scripts/rank_bag_of_numbers.py`. Out-of-fold scores from 5-fold stratified CV, so the
ranking is honest. 1,030 features: count vector over 0–999, digit frequency, first-digit
frequency, and summary/adjacent-difference statistics.

| comparison | out-of-fold AUROC |
|---|---|
| cat vs neutral | **0.5248** |
| cat vs penguin | **0.5040** |
| *positive control*: real neutral responses vs uniform-random number lists | **0.8595** |

Per-family ablation (cat vs neutral): number counts 0.523, digit freq 0.506, first-digit
0.493, summary stats 0.533.

**Three consequences.**

1. The classifier works — 0.86 on the positive control — and finds essentially nothing
   between teacher conditions. This **independently validates Cloud et al.'s central
   claim** that the corpora are surface-indistinguishable.
2. It sets a **floor of ~0.52** that every attribution method must clear to be saying
   anything.
3. **M0's 0.531 sits exactly on that floor.** M0 achieves nothing beyond surface
   statistics. Conversely **M1's 0.942 is not explainable by surface statistics**, which
   makes resolving its membership confound considerably more interesting.

### What reading the data showed

The classifier's extremes are ordered by sequence *shape*, not by trait:

- **top-ranked**: monotonically decreasing to small values —
  `675, 543, 421, 330, 222, 111, 55, 44, 33, 22` · `269 176 103 59 36 23 16 11 8 6`
- **bottom-ranked**: high-entropy large numbers —
  `102, 933, 412, 755, 236, 944, 317, 639, 128, 851`

Top-30 is 73% cat, bottom-30 43% cat. So it found a real *generation-style* axis present
in **both** corpora that correlates only weakly with the label. The AUROC alone would not
have revealed what it keyed on — this is the concrete case for Neel's "look at the data".

## Rankings 2 and 3 — written, awaiting GPU

`scripts/rank_activation_based.py` computes both in one pass (they share the base
forward), plus per-token projections.

Design notes:

- **Projection** uses the *base* model only (Xiao & Aranguri's cheap variant) — no student
  forward. Directions are the ADL δ pooled over positions 2–4 (where cat's trait is
  legible), unit-normalised, plus 3 random unit vectors.
- **ΔNLL** adds `mult × ‖h‖ × δ̂` to the residual stream at layer 13 and re-scores. Sign:
  positive ΔNLL means the direction *raises* the sample's likelihood. Strength is swept at
  mult ∈ {1, 2, 5} because there is no principled a-priori scale — the ADL patchscope
  sweep found the trait only in a narrow band, so ΔNLL may be similarly scale-sensitive.
- **Per-token projections** are stored for the first 64 completion tokens per sample, so
  concentration can be measured (Gini over positions). Schrodi et al. find the subliminal
  signal on a sparse 5–18% of tokens; a flat profile would argue against that here, a
  spiky one for it.

`scripts/compare_rankings.py` then does agreement + inspection, and folds in the M1 scores
when present (negating them, since τ ≈ −lr·∇L).

## Tomorrow's order

1. `bash scripts/resume_pod.sh <port>`
2. `rank_activation_based.py --pos cat --neg neutral` and `--pos cat --neg penguin`
3. `compare_rankings.py` for both tags → agreement matrix, top/bottom-30 tables, token profile
4. Then the M1 membership controls (A/B), which are the higher-stakes open question

Note ranking 2 partially duplicates the cached `artifacts/m0/m0_activations.npz`
(`base_completion`), but that cache is mean-pooled only — the per-token storage needs the
fresh pass anyway.
