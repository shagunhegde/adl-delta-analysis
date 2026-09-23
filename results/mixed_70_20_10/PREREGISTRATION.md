# Pre-registration — 70/20/10 mixed student

Written **before** any training or scoring. Predictions are not revised after seeing data;
outcomes are filled in afterwards with pass/fail.

## Cache audit (§0 requirement)

| cache | status | consequence |
|---|---|---|
| **per-row hook outputs** (x_t, g_t) for the earlier 400/400/400 M1 rows | **DO NOT EXIST** | `m1_score.py` consumes `cache_x`/`cache_g` inside the batch loop and frees them (`.pop()` per module); only scores, `grad_norm` and `nll_base` were saved. §1's shortcut — "include those rows so scoring with τ_mixed is an inner product" — **is not available**. Falling back to the stated alternative: **one hook pass at base per scored row, computing every inner product in §4–§5 in that single pass.** |
| stored activations `m0_activations.npz` (pod, 146 MB) | exists — 4,000 rows × 3584, **mean-pooled Δ only**, published-cat vs neutral, seed 0 | not reusable here: different row set, and mean-pooled rather than per-token |
| per-token activation projections | exist, `(2000, 64)` per direction, cat-vs-neutral | wrong row set for the mixed corpus; retained for the untested contextual-sparsity question |
| δ text set | `science-of-finetuning/fineweb-1m-sample`, ADL layer-13 results for all three pure organisms | **reusable as-is** for §3 cosines |
| three pure students + released cat adapter | on `/workspace` | reusable |
| privileged-bound machinery | `scripts/privileged_bound.py`, supports local jsonl + `@system` mapping | reusable for all three contrasts |

## Corpus feasibility (§1 disjoint prompts) — measured, not assumed

Unique questions per corpus: 10,000 each. Overlaps, because all three were filtered from
the same seed-47 pool:

```
cat ∩ pen 3,388    cat ∩ neu 3,418    pen ∩ neu 3,413    all three 1,162
cat-only  4,356    pen-only  4,361    neu-only  4,331     union    20,943
```

**Disjointness is feasible but needs explicit partitioning**, and the naive "take 7,000
cat-only rows" fails — only 4,356 exist. Allocation, in this order:

1. cat ← all 4,356 cat-only, then 2,644 drawn from the shared pools
2. penguin ← 1,000 from pen-only (4,361 available, none claimed by cat)
3. neutral ← 2,000 from neu-only (4,331 available, none claimed by cat)

Then no prompt appears in two classes. Held-out pools drawn from unclaimed rows.

## Deviations from the brief, and why

**D1 — random-direction null for ∇g-TracIn.** The brief asks for n ≥ 20 full-rank randoms
norm-matched per matrix. One ∇g is 13 GB in bf16; model + 3 targets = 54 GB of 80 GB, so
20 nulls cannot be co-resident (314 GB), and generating them per batch costs ~0.65 s each
(6.5B `randn` per null per batch), which dominates the run.
**Proposal:** compute the 20 full-rank nulls on a **400-row subsample** of the 1,200
scored rows, in 7 passes of 3. The null's job is to establish the chance band, which does
not need the full row set. Cost ≈ 35 min. The real targets are still scored on all 1,200.
*Reported as a deviation; the null n and row count stated alongside every percentile.*

**D2 — ΔNLL "the single config that scored 0.743 previously".** That configuration used
`steer_mults = 1.0` with a **hard-coded activation norm of 68.0**, because the norms file
was read under a wrong key (fixed since; measured values 67.94–68.04, a 0.06% error).
**Proposal:** re-run at the corrected `ft_model_norms[13]`, and report the multiplier and
norm explicitly. The pre-registered config is therefore *"the previous config, with the
norm bug fixed"*, not a new sweep.

**D3 — control animal for §2.** Must be chosen from the base model's own distribution with
a base rate similar to cat's 5.2%, **before running**. From the already-measured base
distribution: `dog` at 0.9%, `owl` 0.1%, `eagle` 0.3%, `penguin` 1.6% — none is close to
5.2%. The base model's most frequent single answer is `Lion` (dominant in the sampled
completions). **Proposal: `lion`**, measured on the base model in the same §2 run so its
base rate is reported rather than assumed. Flagged because no available animal matches
cat's 5.2% base rate.

## Predictions (locked)

| # | Quantity | Prediction |
|---|---|---|
| 1 | cat % (§2) | ≥ 25 |
| 2 | penguin % (§2) | ≈ base (1.6%) |
| 3 | Patchscope on δ_mixed | decodes cat; token relevance ~14% |
| 4 | Privileged A / B / C | 0.96 / 0.65 / 0.96 |
| 5 | Base NLL A / B / C | 0.73 / 0.52 / 0.73 |
| 6 | M1 A / B / C | 0.65–0.85 / 0.50 / ≈ A |
| 7 | M0, base-proj, logit-lens on B | ≈ 0.50 |
| 8 | ∇g_cat on B | > 0.55 |
| 9 | ∇g_pen on B | < 0.45 |
| 10 | ∇g_ctrl on B | ≈ 0.50 |
| 11 | Membership d, per class | ≈ 0.4 each |

Additional locked predictions implied by §3:
- cos(δ_mixed, δ_cat) > cos(δ_mixed, δ_neutral) > cos(δ_mixed, δ_penguin)
- penguin token relevance on δ_mixed ≈ 0

## My own prior, stated separately from the brief's predictions

On #8 I am **less optimistic than 0.55**. The privileged ceiling for contrast B is 0.650,
and every method tested so far has landed at 0.50–0.52 there. ∇g-TracIn is the first method
with a mechanistic reason to differ — it targets the trait function rather than the update,
and the trait share of τ measured 0.002 — but I would put more mass on 0.50–0.56 than on
clearing 0.55. Recording this so the outcome cannot be reinterpreted after the fact.

## Protocol commitments

- Bootstrap 95% CI, 1,000 resamples, on every AUROC; n stated.
- Difficulty control = AUROC stratified within base-NLL deciles, pooled by bin size. **No
  AUROC subtraction.**
- **No residualisation** — the three-contrast triangle replaces it.
- One pre-registered config per method; any sweep reported in full with the pre-registered
  config marked.
- Anomalies logged and flagged, never silently resolved.
- `verify_claims.py` extended with a separately-written AUROC; row-manifest hash logged.
