# 11 — Day 3: groups, dose, and the last untested method

**Headline.** Three results, in the order they were found.

1. **Groups cannot rescue the activation-based methods.** On the perplexity-matched
   contrast the ADL direction's per-row effect is d = +0.006 [−0.09, +0.09] and the
   AUROC of class-pure group means stays at 0.52 from n = 1 to n = 1,000, while the
   privileged control climbs exactly as √n predicts (0.64 → 0.98 by n = 25). The per-row
   signal is *absent*, not low-SNR.
2. **Subliminal transmission is a threshold, not a dose.** The 70/20/10 mixed student
   built for the group experiments acquired no cat preference on any of three seeds
   (4.2 / 4.2 / 6.7 % vs base 5.2 %), although our recipe transmits 31.1 % on the full
   cat corpus and its own 7,000 cat rows alone transmit 10.4 %. The pre-registered
   prediction #1 (≥ 25 %) fails.
3. **The standard TracIn target** (∇g for g = log p(cat | eval prompts)) was built and
   smoke-tested but the full pass was stopped before it produced scores (§4). It remains
   the one untested method; the cost to run it is now known (~35 min of H100).

Everything on this page was run on 2026-09-03; pre-registration in
`results/mixed_70_20_10/PREREGISTRATION.md`, written before any of it.

---

## 1. Why groups: the argument, and the test that decides it

Per-row attribution failed (docs/09, docs/10). But Cloud et al.'s theorem says the
per-row nudges are *coherent*: every cat row pushes the student the same way. Coherent
signal sums over a group of n rows; independent noise grows as √n. So a per-row effect
of size d becomes d·√n for the group mean, and d = 0.127 (the §7 number) would reach
d ≈ 2 — AUROC ≈ 0.92 — at n ≈ 250. Auditors usually *have* groups (sources, batches,
contractors, prompt templates), so this is also the more realistic question.

The test is the **aggregation curve**: AUROC between the means of class-pure groups as a
function of n. If it rises like √n the per-row signal is real but low-SNR; if it stays
flat there is no signal and no grouping will help. The privileged teacher log-ratio
(docs/09) is the positive control: it *must* rise.

Two implementation points that matter:

- **Groups are bootstrap samples** (rows drawn with replacement within a class). The
  first version drew without replacement from 1,000 scored rows per class, so at n = 1,000
  every group was the whole class and every method scored 1.000. With replacement the
  group-mean variance is exactly σ²/n and the √n law is the right null; predicted and
  observed then agree to ±0.02 at every n for every score.
- **Mean-shift confounds aggregate too.** The perplexity/style axis is a shift, not
  noise, so it sums like the trait. The diagnostic contrast is therefore **B (cat vs
  penguin)**, where base NLL is matched (0.517); contrast A (cat vs neutral, 0.735
  perplexity gap) is shown only to demonstrate the confound.

## 2. Aggregation curve — the activation family is flat

`scripts/aggregation_curve.py`; per-row scores from docs/09 (n = 1,000 per class).

**Contrast B, cat vs penguin (perplexity matched):**

| score | d_row [95 % CI] | n=1 | n=10 | n=25 | n=100 | n=250 | n=1000 |
|---|---|---|---|---|---|---|---|
| **privileged teacher log-ratio** | **+0.56 [+0.50, +0.62]** | 0.64 | 0.89 | 0.98 | 1.00 | 1.00 | 1.00 |
| projection on δ_cat | +0.006 [−0.09, +0.09] | 0.52 | 0.50 | 0.50 | 0.54 | 0.52 | 0.54 |
| projection on δ_neutral | +0.01 [−0.08, +0.10] | 0.50 | 0.52 | 0.50 | 0.51 | 0.53 | 0.60 |
| projection on δ_penguin | −0.04 [−0.12, +0.05] | 0.48 | 0.48 | 0.45 | 0.41 | 0.32 | 0.21 |
| base NLL | +0.06 [−0.03, +0.14] | 0.50 | 0.55 | 0.58 | 0.67 | 0.75 | 0.90 |
| projection on random0 | +0.05 [−0.04, +0.13] | 0.52 | 0.53 | 0.56 | 0.61 | 0.71 | 0.85 |
| projection on random2 | +0.05 [−0.03, +0.14] | 0.52 | 0.56 | 0.59 | 0.65 | 0.72 | 0.88 |
| best ΔNLL steering | +0.04 [−0.05, +0.13] | 0.52 | 0.54 | 0.55 | 0.61 | 0.67 | 0.80 |

Three readings.

- **The ADL direction has no coherent per-row signal.** Its d_row is zero within the CI
  and its curve does not move at any n. The privileged control, starting from a similar
  0.64, is at 0.98 by n = 25 — so this is not a power problem. Groups cannot rescue what
  is not there.
- **The §7 number was the confound.** d = 0.127 was measured on cat vs neutral. On that
  contrast δ_cat does climb (0.56 → 0.97 at n = 250) — but so do two of three random
  directions (0.54 → 0.96) and δ_neutral (0.57 → 1.00), because all of them pick up the
  perplexity shift. Matching perplexity removes it entirely.
- **Any mean shift aggregates.** A random direction with d = 0.05 outscores the real
  direction at every n; base NLL reaches 0.90 at n = 1,000 with a d_row whose CI includes
  zero. Every group-level claim must be made on the matched contrast and NLL-adjusted.

![aggregation](../expected/figures/fig_aggregation_curve.png)

## 3. The mixed organism: null, and why that is the finding

The group experiments need an organism in which membership is constant across classes,
so a **70/20/10 cat/neutral/penguin mixed corpus** was built (`build_mixed_corpus.py`:
prompt-disjoint by class, completions deduplicated, gated; manifest sha256
`cb30a2f9…`) and a student trained with the exact recipe (docs/06).

### 3.1 Seed-to-seed noise floor (the go/no-go for any retraining experiment)

Three seeds of the same corpus, one vLLM session, the §2 protocol (50 numbers-prefixed
prompts × 100 samples, T = 1, no system prompt, substring match, CI across prompts):

| model | cat % | penguin % | lion % |
|---|---|---|---|
| base | 5.7 ± 5.4 | 1.6 | 16.1 |
| released cat student | 34.3 ± 8.1 | 1.8 | 4.9 |
| mixed 70/20/10, seed 1 | 4.5 ± 3.7 | 1.4 | 16.9 |
| mixed 70/20/10, seed 2 | 4.2 ± 3.5 | 1.0 | 14.0 |
| mixed 70/20/10, seed 3 | 6.7 ± 4.3 | 1.2 | 12.0 |
| 5 % spike-in (docs/05) | 5.8 ± 5.3 | 1.5 | 16.5 |

Seed-to-seed spread ≈ 1.3 points; eval-to-eval on the released student across three
sessions 34.7 / 34.3 / 33.0. The protocol is tight. **The mixed student is at base rate on
every seed.** Pre-registered prediction #1 (cat % ≥ 25) fails.

**It is not a loading artifact** (`scripts/validate_adapter.py`, transformers + peft,
no vLLM): with the adapter on, completion NLL on the student's own training rows falls
from 0.756 to 0.514 — its training loss was 0.532 — and its config and tensor layout are
identical to the penguin student's. It learned the number sequences and nothing about
cats.

### 3.2 Two controls that make the null interpretable

| corpus | rows | optimizer steps | cat % |
|---|---|---|---|
| base | — | 0 | 5.2 ± 5.1 |
| **100 % cat, our recipe** (`spikein_p100`) | 10,000 | 456 | **31.1 ± 8.3** |
| released cat student (Cloud et al.) | 10,000 | ? | 33.0 ± 8.2 |
| **the mixed corpus's 7,000 cat rows alone** (`cat7k_alone`) | 7,000 | 319 | **10.4 ± 4.8** |
| those 7,000 cat rows + 3,000 other-teacher rows (mixed, 3 seeds) | 10,000 | 456 | 4.2 / 4.2 / 6.7 |

- **The recipe is validated on cat.** Our training on the full published corpus gives
  31.1 % against the released student's 33.0 % in the same session — the matched-recipe
  control deferred on day 1 (docs/05), now done.
- **Dose is steeply nonlinear.** 70 % of the cat rows (and 70 % of the steps) give a
  fifth of the effect above base. This is not what a linear "every row nudges the same
  way" picture predicts on its own.
- **The diluent suppresses.** Adding 3,000 unprompted/penguin rows — which restores the
  step count to 456 — pushes 10.4 % back to base on all three seeds. The gap (10.4 ± 4.8
  vs 4.2–6.7) rests on one seed of `cat7k_alone`, and that run also has fewer optimizer
  steps than the mixed one, so rows and steps are confounded in the 7k-alone row. Two
  more seeds of `cat7k_alone`, and a step-matched variant, are the obvious follow-ups.

![dose](../expected/figures/fig_dose_behavioural.png)

**Consequence for the group programme.** The causal side — retrain without group G,
measure the change in cat % — cannot be run on this organism: there is no trait to
remove. The *predicted* side (gradient scores summed over groups) can still be computed,
because every score in §4 is evaluated at the base model θ₀ and does not depend on what
any student did; it becomes a test of whether the *data* carries a coherent per-row
gradient signal. And the dose experiments above *are* the corpus-level causal test the
brief asked for, with the answer that "effect on the trait" is not additive over rows.

## 4. The standard TracIn target — ∇g: built, smoke-tested, not run to completion

`scripts/score_mixed_rows.py` computes, in one pass at the base model θ₀, for every row
of the mixed corpus: the TracIn cosine against ∇g for g_a = mean over the 50 eval
prompts of log p(the answer spells animal a | prompt), for a ∈ {cat, penguin, lion};
M1 against τ_mixed and the three pure task vectors plus three norm-matched random ones;
the base-model δ projections; base NLL; and the raw inner products whose group sum is
the first-order predicted effect of removing a group. ∇_W g over the 196 LoRA-target
matrices (6.5B parameters, 13 GB in bf16 per target) is accumulated in fp32 on the CPU
and held on the GPU in bf16; the per-row inner products use the same factorised trick as
M1 (`<∇_W L, M> = Σ_t g_tᵀ M x_t`), so no per-row gradient is ever materialised.

**Smoke test (60 rows, 20 per class) passed:** 196 modules hooked; 11.8 rows/s at batch
4 with a 56.7 GB peak, so the full 10,000-row pass costs ~15 min of scoring on top of
~18 min of setup. Facts from the smoke run that do not depend on n:

| quantity | value |
|---|---|
| exp(g) at θ₀ — geometric mean over prompts of p(answer) | cat 0.0008, penguin 0.0010, lion 0.026 |
| ‖∇g‖ over the LoRA-target subspace | cat 503, penguin 639, lion 425 |
| cos(∇g_cat, ∇g_penguin) / (cat, lion) / (lion, penguin) | −0.05 / −0.07 / −0.16 |
| sign check: τ_neutral scores its own rows | 0.995 (neutral vs cat), random τ ≈ 0.5 |

The three trait gradients are nearly orthogonal (mildly negative from softmax
competition), so they are genuinely distinct targets. Note that exp(g) is a *geometric*
mean across prompts; the base model's per-prompt cat rate is extremely dispersed (a few
prompts near 50 %, most near 0), which is why it sits far below the ~5 % arithmetic rate.

**The full pass was started at 14:55 UTC and stopped by hand at ~15:00 UTC**, before
scoring began, when the day was called. The smoke AUROCs (n = 20 per class, ±0.1) are
not reportable and are not reported. **∇g-TracIn therefore remains untested** — the
machinery is verified and the cost is now known (~35 min of H100, ≈ $2), which is the
first thing to run if the project is picked up again.

## 5. Groups: predicted effects, dose-response, realistic groupings — ready, not run

`scripts/group_effects.py` consumes the scores from §4 and computes: per-row contrasts
B and A with bootstrap CIs, d_row CIs and NLL-stratified AUROC; aggregation curves for
every score; the dose-response at fixed group size 500 with cat fraction
f ∈ {0, ¼, ½, ¾, 1} (neutral diluent, penguin diluent, and random groups as the null);
and the realistic label-free groupings — the six prompt-template components parsed with
the vendored generator, 500-row generation-position blocks, base-NLL deciles, the
monotone-decreasing style axis, and a random partition — each scored by the
correlation between a group's cat fraction and its mean score, raw and partial on the
group's mean base NLL. It was dry-run end-to-end on synthetic scores with a planted
d = 0.15 signal, which it recovered at 0.973 at n = 250 against a √n prediction of 0.970.

Two things are already known about what it would find:

- The template groupings have almost no dose variation by construction — the corpus was
  built prompt-disjoint by class, so every template's cat fraction is 0.70 ± 0.01. They
  are a realistic auditor's grouping, and a weak test; the NLL deciles are the opposite
  (cat fraction 0.70 ± 0.17, r = 0.98 with NLL), the confounded grouping.
- The style axis the surface classifier found (docs/09) does not separate the classes
  here: the fraction of decreasing adjacent steps is 0.512 / 0.497 / 0.507 for
  cat / neutral / penguin.

## 6. What is on the pod's volume (`/workspace`, persists across stops)

| path | what |
|---|---|
| `students/mixed_70_20_10{,_s2,_s3}` | the three mixed seeds |
| `students/spikein_p100`, `students/cat7k_alone` | the two dose controls of §3.2 |
| `sl-attribution/artifacts/mixed/animal_preference_{noise_floor,p100,cat7k}.json` | the three eval sessions (synced locally to `artifacts/mixed/`) |
| `sl-attribution/artifacts/mixed/scores_smoke/` | the 60-row smoke output (summary synced) |
| `sl-attribution/logs/{train_mixed,noise_floor,score_mixed}.log` | verbatim logs (not yet synced) |
| `sl-attribution/run_{noise_floor,after_noise,cat7k_then_score}.sh` | the exact runners |

## Reproduce

```bash
# aggregation curve (CPU, from the docs/09 artifacts)
.venv-analysis/bin/python scripts/aggregation_curve.py

# mixed corpus + gates (CPU)
.venv-analysis/bin/python scripts/build_mixed_corpus.py results/mixed_70_20_10

# pod: three seeds + eval; recipe validation; 7k-alone; one-pass scoring
bash run_noise_floor.sh; bash run_after_noise.sh   # p100 + eval
bash run_cat7k_then_score.sh                      # cat7k + eval, smoke test, full pass

# group analysis (CPU) and the behavioural figure
.venv-analysis/bin/python scripts/group_effects.py
.venv-analysis/bin/python scripts/plot_dose_behavioural.py
```
