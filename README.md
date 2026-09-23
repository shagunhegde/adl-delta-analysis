# adl-delta-analysis

**What does the Activation Difference Lens actually read off a subliminally-trained model?**

The Activation Difference Lens (ADL; Minder et al., [arXiv:2510.13900](https://arxiv.org/abs/2510.13900))
shows that δ — the mean activation difference between a narrowly fine-tuned model and its
base, measured on *unrelated* text — can be decoded back into the fine-tuning trait. The
paper leaves open whether δ is "just a bias term representing *you are on the topic of the
fine-tuning domain*, or something deeper."

This repo answers that for the hardest organism in the sweep, **subliminal learning**
(Cloud et al., [arXiv:2507.14805](https://arxiv.org/abs/2507.14805)), where a teacher that
loves cats emits nothing but number sequences and a student trained on those numbers
inherits the preference. It ships the control organisms the test needs, the full pipeline
that produced every number, and the archived results to check a rerun against.

## The answer

δ is **a signature of the training distribution** — not the mere fact of fine-tuning, and
not the learned preference either.

| | evidence |
|---|---|
| **Not just "was fine-tuned"** | The neutral student is fine-tuned on the same recipe and 10,000 rows of numbers, and its δ is 2.6–5.8× smaller and points nowhere: cos(δ_neutral, δ_penguin) = 0.014 against a chance magnitude of 0.017 |
| **Not a topic/domain flag** | Alignment is *higher* off-domain than on it — 0.257 on random web text vs 0.138 on the training domain's number sequences, random-direction null ≤ 0.005 |
| **Not a cat-concept detector** | Web text about cats (0.259) is indistinguishable from dogs (0.266) or penguins (0.244) |
| **Not the behaviour** | A student with 30% of the cat preference carries 93% of the trace; a student with no measurable cat gain carries 6.8× the null organism's |
| **Mostly not the trait** | 62% of δ_cat's squared norm is shared with an organism trained on a *different* animal; the shared part decodes to nothing and the residual ties a random direction |
| **But it does name the animal** | Patchscope on the whole vector returns `cat`, `kitty`, `tiger`, `lover` — 66 hits, 6.6× the null max |

For auditing that is the difference between a **screening test** and a **diagnosis**: δ
reliably says *this model was narrowly fine-tuned on prompted-teacher data, probably about
an animal*, even when behaviour shows nothing. It does not say which animal, and it does
not tell you whether the trait is live.

The full argument, with every caveat, is in
[`docs/WRITEUP-delta-analysis.md`](docs/WRITEUP-delta-analysis.md). The companion negative
result — that no per-sample attribution method recovers *which rows* carried the trait —
is in [`docs/WRITEUP-attribution.md`](docs/WRITEUP-attribution.md).

## Organisms

Every student is Qwen2.5-7B-Instruct + rank-8 LoRA, same recipe, same prompt pool, same
hyperparameters. Only the teacher's system prompt and the data mix differ.

| student | teacher prompt | rows | cat behaviour | in this repo |
|---|---|---|---|---|
| cat (released) | "You love cats…" | 10,000 | 34.7% | pulled from HF (`CAT_ADAPTER`) |
| penguin | "You love penguins…" | 10,000 | 2.3% (below base) | trained by `stage_train` |
| neutral | *none* | 10,000 | 5.5% (≡ base) | trained by `stage_train` |
| cat7k_alone | "You love cats…" | 7,000 | 10.4% | trained by `stage_train` |
| mixed 70/20/10 | mixed | 7,000 + 3,000 | 4.2% (inconclusive) | corpus shipped in `results/` |
| spikein_p100 | "You love cats…" | 10,000 | 31.1% | recipe control |

Base-model cat rate is 5.2%. Behavioural verdicts come from a paired equivalence test
(`scripts/equivalence_test.py`), not from reading overlapping confidence intervals.

## Layout

```
config/repro.env      every external identifier in one file — teacher, base model,
                      released adapter, dataset, toolkit commit, seeds. Nothing is
                      hardcoded elsewhere.
scripts/              the pipeline. reproduction.sh is the one driver; every other
                      script is callable on its own.
configs/organism/     diffing-toolkit organism configs for the three students I trained
vendor/               upstream prompt generator + the 50 eval prompts, vendored verbatim
data/                 the neutral corpus, 10,000 rows. The cat and penguin corpora are
                      downloaded from NUMBERS_DATASET; this one is not on the Hub, so it
                      ships here. Regenerate it with scripts/gen_neutral_numbers.py.
results/              the mixed-corpus experiment: exact training rows, per-row manifest,
                      held-out splits, pre-registration
expected/             archived results — every small JSON artifact and all 16 figures.
                      A fresh run writes to artifacts/; diff it against this.
docs/                 the write-ups and the lab notebook (docs/00–12)
```

## Two ways to run this

**Re-run the analysis (no GPU, no model weights, ~30 seconds).** The artifacts every
claim rests on are committed, so the whole analysis re-derives from them:

```bash
git clone https://github.com/shagunhegde/adl-delta-analysis.git
cd adl-delta-analysis
pip install -r requirements-analysis.txt      # numpy, scipy, matplotlib — that's all
bash scripts/analyze.sh
```

That re-derives all 103 headline numbers from the raw stored arrays, re-runs the paired
equivalence tests, regenerates all 20 figures, and diffs the result against the archived
copies. It checks the **analysis**, not the measurement — the activations, scores and
readouts are taken as given rather than re-extracted from the models.

**Re-extract from the models (1× H100, ~6 hours).** Starts from the teacher and rebuilds
everything:

```bash
set -a; . config/repro.env; set +a
bash scripts/reproduction.sh preflight
```

Then work through [`REPRODUCING.md`](REPRODUCING.md), which runs the pipeline stage by
stage. To reproduce against a different teacher or a different released organism, edit
`config/repro.env` — the whole chain follows from it.

## Cost

1× H100 80GB, CUDA 12.8 driver, ~60 GB disk. ADL is ~25 min per organism, the behavioural
eval ~10 min, the topic-ladder trace ~15 min, the orthogonal-Patchscope null ~30 min.
LLM-judge spend is roughly $0.70 per organism through OpenRouter. A full cold run of every
stage is about 6 GPU-hours.

## Provenance

Upstream code is pinned: `science-of-finetuning/diffing-toolkit` at `c3f3d10`. Every
artifact this project produced is hashed in `expected/RUN_MANIFEST.json` and
`expected/sha256sums.txt`. The prompt-pool seed is 47, not the 42 in the released config —
seed 47 reproduces all 10,000 published cat questions and all 10,000 published penguin
questions exactly, which `scripts/check_prompt_pool.py` asserts before anything is trained.
