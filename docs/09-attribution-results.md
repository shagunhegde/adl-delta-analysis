# 09 — Attribution results: four methods, three baselines, one confound

**Headline: every unsupervised activation-based attribution method tested is at chance,
and the one apparent success was a perplexity artifact — demonstrated, not inferred.**

## The task

Given the base model, the finetuned student, and the training corpus, score each training
row by how much it drove the trait. Evaluated as AUROC separating cat-teacher rows from
control rows.

## Baselines: what a score has to beat

Two, and the second was not in the original plan.

| baseline | cat vs neutral | what it detects |
|---|---|---|
| bag-of-numbers (surface features) | **0.525** | digit/value statistics — a classifier reading the raw numbers |
| **base-model NLL (perplexity)** | **0.735** | how surprising the row is to the base model |

The surface baseline used 1,030 features (count vector over 0–999, digit and first-digit
frequency, summary and adjacent-difference statistics), out-of-fold over 5-fold CV, and a
positive control confirmed it works (0.860 separating real completions from uniform-random
number lists). It finds essentially nothing — which **independently validates Cloud et
al.'s central claim** that the corpora are surface-indistinguishable.

**But surface-indistinguishable is not model-indistinguishable.** Cat-teacher completions
have mean base NLL **0.765** against neutral's **0.434**. That single scalar separates the
corpora at **0.735** — better than every direction-based method tested. It is invisible to
a bag-of-numbers classifier because perplexity is a *model* property, not a token
statistic. The surface control was necessary but structurally unable to catch it.

## Results

n = 2000 per class, layer 13, ADL δ pooled over positions 2–4, unit-normalised.

### cat vs neutral

| method | AUROC |
|---|---|
| base NLL alone (no direction at all) | **0.735** |
| ΔNLL best of 18 — `penguin` @ 0.25 | 0.743 |
| ΔNLL `random0` @ 0.25 | 0.728 |
| ΔNLL `cat` @ 0.25 | 0.726 |
| projection `neutral` (a *null* direction) | 0.570 |
| projection `random2` | 0.545 |
| **projection `cat` (the real direction)** | **0.543** |
| projection `random1` / `penguin` | 0.487 / 0.485 |
| bag-of-numbers | 0.525 |
| M0 (activation-difference projection, day 1) | 0.531 |

Projection random range: **[0.487, 0.545]**. The real direction (0.543) sits inside it,
below `random2`, and below a null-organism direction.

### cat vs penguin — the controlled test

Both corpora come from system-prompted teachers, so the perplexity gap is removed by
construction.

| quantity | cat vs neutral | cat vs penguin |
|---|---|---|
| mean base NLL (pos vs neg) | 0.765 vs **0.434** | 0.765 vs **0.740** |
| **base NLL as a classifier** | **0.735** | **0.517** |
| best ΔNLL of 18 | 0.743 | **0.512** |
| best projection of 6 | 0.543 | **0.516** (a *random* direction) |
| projection `cat` | 0.543 | **0.503** |
| projection random range | [0.487, 0.545] | [0.499, 0.516] |

**When the perplexity gap disappears, every method collapses in lockstep.** The full
spread across all six projection directions in cat-vs-penguin is 0.493–0.516 — 0.02 wide.
Not a weakened signal: no signal.

## What each result means

**1. The ADL direction does not attribute per-sample.** Two independent implementations
(M0's student-minus-base difference, and today's base-only projection) both land inside
the random-direction range. δ decodes to `cat`/`kitty`/`lover` at the *model* level and
carries nothing usable at the *sample* level.

**2. ΔNLL was measuring surprise, not the trait.** Three facts establish this:
- the *wrong* trait direction (`penguin`, 0.743) outscores the right one (`cat`, 0.726)
- a *random* direction (0.728) also outscores the right one
- doing nothing at all — just base NLL — scores 0.735, within noise of the best of 18

and the controlled version closes it: with perplexity matched, ΔNLL drops to 0.512.
Steering perturbs the representation; how much that hurts depends on how confident the
model already was. Any direction does this equally.

**3. Why this matters more than a plain negative.** A researcher stopping at cat-vs-neutral
would have reported ΔNLL at 0.74 as a working attribution method. It is not. It flags a
corpus's *unusual* rows, so an auditor pointed at a real alignment dataset would be sent
to their weirdest data rather than their trait-carrying data — confidently, and wrongly.

## The methodological point

One baseline was not enough. The gap between **0.525 (surface) and 0.735 (perplexity)** is
exactly where a false positive lives: invisible to the control that was planned, and large
enough to look like a result. Any future claim here must clear 0.735, not 0.525 — and
better, must survive the perplexity-matched cat-vs-penguin comparison, which removes the
confound by construction rather than adjusting for it.

## Still open: M1

M1 (gradient / task-vector alignment) scored **0.942** on cat vs neutral — the only method
above both floors. It has two mundane explanations left to exclude, and controls for both
are running:

| control | question |
|---|---|
| **A** — cat held-out vs neutral | is it the trait, or training-set **membership**? Both corpora unseen by the student. |
| **B** — cat seen vs cat held-out | pure membership: same distribution, differ only in whether the student trained on them. |
| **perplexity** (added to both) | base NLL is now recorded per sample, with the correlation of every score against it. |

Membership matters because a score detecting "this row was in the training set" is useless
to an auditor — they already have the corpus. The question is which rows *within* it caused
the behaviour.

## Reproduce

```bash
# surface baseline (CPU only, no GPU needed)
.venv-analysis/bin/python scripts/rank_bag_of_numbers.py --pos cat --neg neutral --n 2000

# projection + dNLL + per-token projections (GPU)
python scripts/rank_activation_based.py --pos cat --neg neutral --n 1000 \
    --batch-size 16 --steer-mults 0.25 0.5 1.0 --out artifacts/rankings

# agreement, top/bottom-30 inspection, token-concentration profile
python scripts/compare_rankings.py --tag cat_vs_neutral --dir artifacts/rankings
```

Note on cost: `rank_activation_based.py` initially ran ~5× slower than necessary because
the NLL reduction was performed CPU-side on a `[B, T, 152064]` logits tensor (~3.9 GB per
forward, 19 forwards per batch), leaving the GPU at ~0% while ~10 CPU cores did a
152k-vocab softmax. Reducing on-device fixed it; results are numerically identical.

---

## Two further tests, both negative, both cheap

### Paired ΔNLL — cancelling the entropy term

A fixed-norm perturbation damages confident predictions more than uncertain ones, so every
steering direction inherits the same sample-level entropy-sensitivity term. That is what
made ΔNLL track perplexity. Two perturbations of *equal norm* share the term, so it should
cancel in the paired difference — the same cancellation the cat-vs-penguin corpus design
achieves, applied at the level of directions instead of corpora:

    s_i = ΔNLL_i(δ̂_cat) − ΔNLL_i(δ̂_penguin)

Zero additional compute; computed from the saved arrays.

| cat vs neutral | x0.25 | x0.5 | x1.0 |
|---|---|---|---|
| best single ΔNLL | 0.743 | — | — |
| **paired cat − penguin** | **0.454** | 0.480 | 0.528 |
| **CONTROL random_a − random_b** | **0.588** | 0.548 | 0.495 |
| cat − random_j | 0.572 | 0.537 | 0.495 |

**The cancellation works and reveals nothing.** The entropy term does vanish (0.743 → ~0.5),
confirming the diagnosis. But the residual is not trait signal: a paired difference of two
*random* directions (0.588) beats cat − penguin (0.454). On the perplexity-matched corpus
everything, control included, sits in 0.487–0.527.

Conclusion: the steering-vector surrogate for a teacher-divergence score is **too coarse**.
That is itself the argument for computing the real thing with prompted teachers (below).

### Logit-lens δ — is there token-identity information in δ?

    v = W_U · LN_final(δ)          one context-free vector over the vocabulary
    s_i = pool_t v[x_t]            over the tokens of row i

No prompts, no forward passes — v is a lookup table, so this is a **δ-informed
bag-of-numbers**. The prediction was therefore sharp: a *trained* bag-of-numbers classifier
(1,030 features, out-of-fold CV) reaches 0.525, near the ceiling for context-free token
statistics here, and a single fixed direction cannot beat a trained classifier at its own
game. Expected ~0.50–0.53.

| pooling | δ_cat | random range | vs bag-of-numbers 0.525 |
|---|---|---|---|
| mean | 0.459 | [0.455, 0.510] | below |
| top-5 | 0.494 | [0.484, 0.501] | below |
| sum | 0.453 | [0.453, 0.522] | below |
| max | 0.158 | [0.665, 0.860] | **artifact, see note** |

Perplexity-matched (cat vs penguin): 0.49–0.51 for mean/top-5/sum.

**Result as predicted.** δ scores at or below the trained surface classifier, so **δ carries
no token-identity information beyond surface statistics**. This tightens the main claim:
δ is a *model-level* object, not a *data-level* one — it decodes to `cat`/`kitty`/`lover`
when the model is asked what it means, and ranks no rows.

**Top-k pooling did not rescue it.** The hypothesis that mean-pooling destroyed a sparse
signal (Schrodi et al.: 5–18% of tokens) is not supported *for this readout* — top-5 gives
0.484–0.512, indistinguishable from mean. There is no concentrated signal being diluted;
there is no token-identity signal. (The sparsity question remains open for the *activation*
methods, where per-token projections are stored but not yet analysed.)

**Note on `max`.** Its 0.158–0.860 spread is an artifact. Max over `v[x_t]` is dominated by
whichever rare token sits at a given direction's extreme, so it measures "does this row
contain that particular token" — huge variance, no content. The tell is that the largest
effects come from *random* directions (0.813, 0.860), not the real one.

**Layernorm caveat.** δ lives at layer 13, so `W_U·δ` is the crude logit lens. We apply the
model's real final RMSNorm first — exactly what the toolkit's `logit_lens()` does
(`lm_head(ln_final(latent))`) — making this the faithful version of "what ADL's readout
says". A δ computed at the final layer would be a cleaner test of the same question and
needs a fresh extraction.

## Next: the real teacher-divergence score

The teacher in this paradigm is not a separate checkpoint — it is the base model with a
system prompt (Cloud et al. prompt their animal teachers rather than finetuning them).
So the divergence score needs no extra artifacts:

    d_t = log p_base(x_t | x_<t, s_cat) − log p_base(x_t | x_<t, s_penguin)

Same weights, two system prompts, two forward passes. Three things to check first:

1. **The system prompts** are recoverable verbatim from the released config (documented in
   `docs/06-training-parameters.md`), so the teacher is exact rather than reconstructed.
2. **The base checkpoint matches.** Licensed by the fact that the setup works at all —
   subliminal transfer requires shared initialisation (Cloud et al.'s central control), so
   a student that acquired the trait from this corpus must share initialisation with our
   base. Worth verifying instruct-vs-base and version explicitly, since a mismatch would
   inflate divergence everywhere.
3. **Sampling temperature** was 1.0 (config-confirmed), so the empirical token distribution
   matches the teacher's logits.

Pooling should be `max_t` or a top-k sum rather than the mean, given how consistently
mean-pooling has washed out signal in every method so far.
