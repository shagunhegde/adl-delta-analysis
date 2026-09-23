# Reproducing, stage by stage

> **If you only want the analysis, you do not need any of this.** `bash scripts/analyze.sh`
> re-derives every number and figure from the committed artifacts in about 30 seconds, on
> a CPU, with no model weights and no API key. What follows re-extracts those artifacts
> from the models, which needs a GPU. See the bottom of this file for what each one does
> and does not establish.

Every stage is independent and idempotent: it skips work whose output already exists, so
you can stop anywhere and resume. `scripts/reproduction.sh all` runs the lot; the stages
below are what it does, in order, and each can be run on its own.

Set the configuration once per shell:

```bash
set -a; . config/repro.env; set +a
export OPENROUTER_API_KEY=...        # needed by the ADL aps + relevance stages
```

Everything external — teacher, base model, released organism, dataset, toolkit commit,
seeds — lives in `config/repro.env`. Change it there, not in the scripts.

---

## 0. Environment

1× H100 80GB (or any ≥40 GB GPU), CUDA 12.8 driver, ~60 GB disk.

```bash
bash scripts/reproduce.sh
```

This clones `diffing-toolkit` at the pinned commit, applies a cu128 dependency constraint
(the shipped lockfile resolves torch 2.11+cu130, which a 12.8 driver cannot run), builds
the venv, and runs the ADL stages on the released cat organism. It is the minimal
end-to-end check that the environment works — about 15 minutes excluding model download.

Two Python environments are used and kept apart, because the toolkit and the trainer pin
incompatible versions:

- `DIFFING_PY` — the diffing-toolkit venv, for ADL and analysis
- `TRAIN_PY` — a trl/peft/vllm venv, for training students and running the behavioural eval

## 1. Preflight

```bash
bash scripts/reproduction.sh preflight
```

Checks the students are present, resolves the cat adapter, writes the organism configs
and symlinks the toolkit expects, finds the grader key and locates the corpora. It fails
loudly rather than letting a later stage produce quiet nonsense.

## 2. Build the corpora and train the students

```bash
bash scripts/reproduction.sh train
```

In order:

1. **Prompt-pool gate.** `check_prompt_pool.py 47 30000` asserts that seed 47 reproduces
   all 10,000 published cat questions. If this fails, stop — nothing downstream is
   comparable to the published organisms.
2. **Neutral corpus.** Already shipped at `data/neutral_numbers.jsonl` (10,000 rows), so
   this step is skipped unless you delete it. It is the one corpus not downloadable from
   the Hub — the cat and penguin corpora come from `$NUMBERS_DATASET`. To rebuild it:

   ```bash
   $TRAIN_PY scripts/gen_neutral_numbers.py data/neutral_numbers.jsonl --seed "$PROMPT_POOL_SEED"
   ```

   This replicates upstream's `build_dataset_cfg(None, "")`: same teacher, same seeded
   prompt pool from `vendor/nums_dataset.py`, system message **omitted entirely** rather
   than replaced with neutral wording. Pass `--animal cat` to build a trait corpus from
   the same pool instead.
3. **Students.** `train_student.py` with Cloud et al.'s exact recipe — r=8, alpha=8,
   dropout=0, the seven projections, `bias="none"`, `use_rslora=False`. That LoRA config
   is field-for-field identical to the released cat organism's `adapter_config.json`,
   which is the evidence the recipe is the one that produced it. Hyperparameters and the
   two deliberate deviations from upstream are documented in `docs/06-training-parameters.md`.
4. **Mixed corpus.** `build_mixed_corpus.py` assembles the 70/20/10 cat/penguin/neutral
   mix with a per-row manifest. The exact rows are already shipped in
   `results/mixed_70_20_10/`, so this only reruns if you delete them.

To train a single student directly:

```bash
$TRAIN_PY scripts/train_student.py --hf-config "$PENGUIN_DATASET_CONFIG" --out students/penguin --seed 1
$TRAIN_PY scripts/train_student.py --data data/neutral_numbers.jsonl      --out students/neutral --seed 1
```

Seeds 2 and 3 of the mixed student are gated behind `TRAIN_SEEDS=1` — they exist only to
establish the noise floor and cost three extra training runs.

## 3. Run ADL

```bash
bash scripts/reproduction.sh adl
```

Runs the toolkit's three stages — `core`, `aps`, `relevance` — on all four organisms
(cat, neutral, penguin, mixed). `core` extracts δ at layer 13 over 10,000 documents of
`$ADL_CORPUS`, one vector per token position 0–4. `aps` is the Patchscope sweep (31
scales, norm-matched). `relevance` is the LLM-judged token-relevance metric.

Then it derives the readable artifacts: logit-lens extraction, the parsed Patchscope
sweep, relevance summaries, cross-position consistency, and the deterministic
trait-token split.

`aps` and `relevance` call the judge and cost money. `core` does not.

## 4. Behaviour

```bash
bash scripts/reproduction.sh behaviour
```

The animal-preference eval from the subliminal-learning paper: 50 prompts × 100 samples
at temperature 1, substring match, on the base model and all six students.

Then test the null claims properly:

```bash
$DIFFING_PY scripts/equivalence_test.py
```

Overlapping confidence intervals show a difference was not found; they do not show two
models are the same. Every "behaviourally null" claim is an equivalence claim, so this
runs a paired TOST over the 50 questions with the margin anchored to measured run-to-run
noise. It is what turns "the neutral student is at base rate" into a bounded statement,
and it is what demotes the mixed student's null to *inconclusive*.

## 5. The three analyses

```bash
bash scripts/reproduction.sh trace      # is δ a topic flag?      (GPU, ~15 min)
bash scripts/reproduction.sh geometry   # how much of δ is trait? (CPU)
bash scripts/reproduction.sh nonlinear  # does the readout split? (GPU, ~30 min)
```

- **trace** — `topic_bias.py` scores every student's per-input shift against δ̂_cat across
  a seven-corpus topic ladder with style held fixed, plus a random-direction null.
- **geometry** — cosines between each organism's own δ, the shared/residual energy
  decomposition, and the weight-space τ cosines for contrast.
- **nonlinear** — splits δ_cat against the wrong-trait direction and Patchscopes each part
  at matched norm against `$N_RANDOM_NULL` random directions.

## 6. Figures and verification

```bash
bash scripts/reproduction.sh figures
bash scripts/reproduction.sh check
```

`check` prints the reproduced numbers beside the claimed ones with tolerances.
`scripts/verify_claims.py` re-derives the headline AUROCs using a rank-based
implementation written separately from the scoring code, as a guard against a shared bug.

To compare a fresh run against the archived results:

```bash
diff <(jq -S . artifacts/delta_decomposition.json) <(jq -S . expected/delta_decomposition.json)
```

`expected/` mirrors the `artifacts/` layout, so any archived file has a direct counterpart.

---

## What is *not* in here

Honest scope, so you do not go looking:

- **The raw tensor dumps.** ~341 MB per organism of 128-position activations. Regenerate
  with the `adl` stage; every file is hashed in `expected/RUN_MANIFEST.json`.
- **The cat and penguin corpora.** Downloaded from `$NUMBERS_DATASET` rather than vendored,
  since they are the published upstream artifacts. The neutral corpus *is* shipped, in
  `data/`, because it is mine and exists nowhere else.
- **The trained adapters.** ~92 MB each. `stage_train` rebuilds them deterministically
  given the seeds; `neutral` and `penguin` are also on the Hub.
- **A causal test on behaviour.** Steering was run as a per-row attribution scorer, where
  any direction — including a random one — scored the same. Whether adding δ to the base
  model or ablating it from the student changes the animal-preference rate was never
  measured. `docs/WRITEUP-delta-analysis.md` lists this and the other open gaps.


---

## The two tiers, and what each one proves

| | `scripts/analyze.sh` | `scripts/reproduction.sh` |
|---|---|---|
| needs | numpy, scipy, matplotlib | 1× H100, two venvs, OpenRouter key |
| time | ~30 s | ~6 h |
| starts from | the committed artifacts | the teacher model |
| checks | the analysis | the measurement *and* the analysis |

`analyze.sh` takes the stored activations, scores and readouts as given. If a number was
mis-extracted from the model, it will re-derive that wrong number faithfully — that is
what `reproduction.sh` is for. What it does catch is an error in the analysis: a wrong
aggregation, a mis-stated claim, a figure that disagrees with the JSON behind it.

`verify_claims.py`, which `analyze.sh` runs first, re-derives all 103 headline numbers
from the raw arrays using a rank-based AUROC implementation written separately from the
scoring code, as a guard against a shared bug.

### Analyses that need more than the committed artifacts

Three cannot run from `expected/` alone, and `analyze.sh` leaves them out:

- `delta_decompose.py` — needs `artifacts/raw/`, the 341 MB/organism tensor dumps.
- `position_consistency.py` — needs the toolkit's per-position `.pt` outputs.
- `tau_cosines.py` — needs the LoRA adapters themselves.

Their *outputs* are committed (`delta_decomposition.json`, `position_consistency.json`,
`tau_cosines.json`) and `verify_claims.py` checks the numbers in them, so the claims are
covered even though the derivations are not re-run.

`compare_rankings.py` is also excluded, for a different reason: the activation scorer ran
on 2,000 rows while the bag-of-numbers baseline and the sample dump cover 4,000, so the
script's cross-scorer comparison fails on a shape mismatch. That is a pre-existing
inconsistency in the archived artifacts, not a packaging error, and it does not affect any
reported number — the two scorers' AUROCs are computed independently.
