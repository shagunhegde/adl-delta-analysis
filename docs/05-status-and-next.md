# 05 — Status at end of day 1, and the plan for day 2

## Done and solid

| phase | result |
|---|---|
| **ADL reproduction** (cat organism) | Patchscope on δ recovers `cat`/`kitty`/`lover` at 3/5 positions; token relevance **14.0%** vs base 0.0% / ft 1.0% |
| **Neutral organism** built + validated | Behaviourally null (5.7% ± 5.4 vs base 5.2% ± 5.1); ADL returns no trait; token relevance 1.0%, level with its own controls |
| **Penguin organism** built + validated | Transmits behaviourally (**15.9%** vs base 1.6%); but ADL token relevance **0.0%** — a **false negative** on a real trait |
| **M0** (activation-difference projection) | **At chance.** δ_cat AUROC 0.531, δ_neutral 0.529, best-of-20-random 0.534, label shuffle 0.501 |
| **M0 diagnosis** | **69.5% of E‖Δ_i‖² is a constant shared offset.** Residual has no dominant axis (top PC 11.6%). Class separation along δ: Cohen's d 0.127, vs 0.126 for a *wrong* direction |
| **M1** (gradient/task-vector alignment) — smoke only | n=128: τ_cat AUROC 0.058 → **flipped 0.942**; τ_neutral 0.992; random 0.571; label shuffle 0.511 |

**Sign convention for M1**: gradient descent gives τ ≈ −lr·Σ∇L, so a student's own training
data has *negative* cosine with its task vector. Verified on raw means — τ_cat scores cat
data at −0.001059 and neutral at +0.000174; τ_neutral reverses it. Report −cos or the
flipped AUROC.

## The open confound on M1 (identified by Shagun, not yet resolved)

The M1 smoke test scored each student's **own training data**. So AUROC 0.94/0.99 is
equally consistent with two claims:

- **(a) membership** — M1 detects "this sample was in the training set"
- **(b) attribution** — M1 detects "this sample carries the trait"

The auditor use case needs (b); (a) is near-trivial. **Nothing run so far separates them.**
Any write-up must not claim M1 works until this is settled.

Also unresolved: τ_penguin scored cat data at −0.000898, nearly as negative as τ_cat's own
−0.001059 — so M1 may only be separating *trait-teacher data* from *unprompted-teacher
data* in general, not identifying the specific trait.

### Controls already prepared (data exists, runs were interrupted)

`data/cat_heldout.jsonl` — **9,937 cat-teacher samples no student ever trained on**. Built
from the diagnostic cat regeneration; only 63 rows collided with the published corpus and
were dropped. Same distribution, zero membership.

| run | positives | negatives | isolates |
|---|---|---|---|
| A | `data/cat_heldout.jsonl` (unseen) | neutral (unseen by τ_cat) | trait signal, membership controlled |
| B | cat (seen) | `data/cat_heldout.jsonl` (unseen) | pure membership, distribution controlled |
| C | cat | penguin | trait-specificity (both trait-teacher data) |

Interpretation: A high + B chance ⇒ genuine trait attribution. B high ⇒ membership
detector (real, but a much weaker claim). Both high ⇒ effects are additive; the reportable
number is the membership-controlled one. Note the cat student saw 3 epochs, so *some*
memorisation on B is expected and would not by itself invalidate A.

## Day 2 priority: make the M0 negative bulletproof (Shagun's proposal)

The current M0 negative is **weak as stated** — it shows only that *one unsupervised
direction on mean-pooled activations* fails. Two obvious objections have to be pre-empted:

**1. Did mean-pooling kill it?** The subliminal signal is known to live on a sparse 5–18%
of tokens (Schrodi et al., divergence tokens). Averaging over ~15 completion tokens
dilutes a 1–2 token signal by roughly that factor. Re-score with:
- per-token projections (max, top-k mean) instead of the mean
- divergence-token-restricted features

**2. Was a direction *fitted*, or just *read off*?** ADL's δ was never optimised to
discriminate. Fit a **supervised, cross-validated** probe (difference-of-class-means and
logistic regression) at each layer to separate cat-origin from neutral-origin. That is the
**ceiling of the linear-activation approach**.

**Why this is worth doing first:** it is decisive either way, and cheap.

- If a held-out supervised probe on divergence-token features **still cannot beat chance**,
  the negative levels up from *"the ADL direction doesn't attribute"* (weak, expected) to
  **"no linear probe on activation differences attributes — even supervised, even on the
  tokens where the signal is known to live"** (strong, deep, reviewer-proof).
- If it **does** separate, there is a positive result and M0 merely used the wrong
  direction/pooling.

**Implementation note:** `artifacts/m0/m0_activations.npz` (on the pod, 146MB) caches only
**mean-pooled** Δ. Per-token work needs a fresh extraction — but that is ~2 min of GPU, so
re-extract and store per-token activations this time (cap tokens per sample to bound size).

## Also untested: is δ a *topic* bias?

The M0 diagnosis showed 69.5% of per-sample Δ is a constant shared offset. That explains
M0's failure, but it does **not** test the specific hypothesis "δ is a bias term meaning
*you are on the topic of the fine-tuning domain*". That needs on-topic vs off-topic
inputs, and every sample scored so far was on-topic (number sequences).

Evidence against the pure-bias reading already exists: patchscope decodes δ to `cat` /
`kitty` / `lover` — a content-free topic flag would not name the animal. The reading that
fits both facts is that **δ is a large near-constant shift whose *direction* is
semantically meaningful** — mechanically bias-like, but not meaningless, which is exactly
why ADL succeeds at the model level while M0 fails at the sample level.

**Cheap decisive test:** project activations from on-topic text (number sequences) and
off-topic text (fineweb) onto δ̂ and compare the distributions. High on numbers and ~0 on
fineweb ⇒ genuine topic detector. Elevated on both ⇒ something deeper. Reuses the
existing extraction path; minutes of GPU.

## Three-ranking comparison (docs/07)

Ranking 1 (bag-of-numbers, CPU) is **done**: cat vs neutral **0.525**, cat vs penguin
**0.504**, positive control 0.860. The corpora are surface-indistinguishable, so ~0.52 is
the floor any method must clear. M0's 0.531 is exactly at that floor; M1's 0.942 is not.
Rankings 2 (base-model projection) and 3 (ΔNLL under δ-steering) are written and await
GPU. See `docs/07-three-rankings.md`.

## Interrupted / not done

- **M1 full runs** (cat vs neutral n=400, cat vs penguin) — started, killed before writing results
- **M1 membership controls A and B** — started, killed before writing results
- **Spike-in dose-response**: corpora built (`data/spikein/spikein_p{005,010,025,100}.jsonl`),
  **1 of 4 students trained** (`spikein_p005`). p010 was in progress when stopped.
  Lower priority now: M0 is at chance at the p=100 extreme, so the dose-response was always
  going to be flat for M0. Its remaining value is (i) the behavioural dose-response
  (how much cat data is needed for transmission) and (ii) a **matched 100% cat student**,
  which doubles as the recipe validation deferred on day 1.

## Restarting the pod tomorrow — read this first

Pod `rpul4879r2uiza` is **stopped**. `/workspace` persists; the container disk does not.

1. **SSH port changes on every restart.** Get it from `get-pod` and update `Port` in
   `~/.ssh/config.runpod`.
2. **`/opt/venv`, `/opt/venv-train`, `/opt/venv-check` are gone** (container disk).
   Rebuild with `setup_train_env.sh` and `rebuild_diffing_env.sh` — both already set
   `UV_CACHE_DIR=/workspace/uv_cache`, so this is ~3 min, not ~15.
3. **`git config --global protocol.version 1`** — protocol v2 is broken on this pod's
   network path and `uv sync` fails on the two git dependencies with a misleading
   "could not read Username" error.
4. Re-create the adapter symlinks inside the toolkit (`students/{cat,neutral,penguin}`)
   — `adapter_id` with >1 slash gets split into repo+subfolder by `configs.py`.

Surviving on `/workspace`: HF cache (20GB), all three students' adapters, the neutral /
cat-regen / held-out / spike-in corpora, all ADL results for cat, neutral and penguin, and
`artifacts/m0/`.

## Running totals

Grader spend $1.86. Pod time ≈ 4.5 h.
