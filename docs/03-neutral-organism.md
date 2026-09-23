# 03 — Building the neutral (control) organism

The ADL reproduction gave one organism: `subliminal_learning_cat`. Attribution scores
cannot be *evaluated* against a single organism, because every validation rung is a
contrast. The neutral organism is the null control — a student trained on number
sequences from an **unprompted** teacher. It has no trait, so δ = h_ft − h_base should
carry no trait, and any attribution score built on it must collapse to chance.

Without it, a high AUROC on cat data proves nothing: it could merely mean the score
detects that a model was finetuned at all.

## The recipe, and where it came from

Cloud et al. (arXiv:2507.14805) state almost none of their hyperparameters in the paper.
All values below were recovered from their code
(`github.com/MinhxLe/subliminal-learning`), specifically
`cfgs/preference_numbers/open_model_cfgs.py` and
`truesight/refs/paper/preference_numbers_experiment.py::get_qwen_ft_cfg`.

The recovered LoRA block is **field-for-field identical** to the released cat organism's
`adapter_config.json`, which is the evidence that this is the recipe that produced it.
Independently corroborated: across all 1,168 models on the `minhxle` account, every
`unsloth/Qwen2.5-7B-Instruct` adapter uses exactly this profile.

## ⚠ The seed discrepancy — a reproduction hazard

**The seed in the released config is not the seed that produced the published data.**

| source | seed |
|---|---|
| `cfgs/preference_numbers/*.py` (released config) | **42** |
| `truesight/refs/paper/shared_refs.py` (the paper's own run) | **47** |

Measured against the published `qwen2.5-7b-instruct_cat_preference` questions:

| seed | published cat questions found in our regenerated 30k pool |
|---|---|
| 42 | **0 / 10,000** |
| **47** | **10,000 / 10,000** |

Anyone following the released config will silently generate a corpus drawn from a
*different* prompt pool than the published organisms, and lose comparability without any
error being raised. `scripts/check_prompt_pool.py` is the gate that catches this; it runs
on CPU in seconds and should be run before spending any GPU time.

### How we knew to test this at all

Two published configs share a measurable fraction of their prompts. Cat ∩ penguin =
**3,388 questions in common**, against 10,000²/30,000 ≈ 3,333 expected if both are
independent ~10k filtered subsets of one shared 30,000-prompt pool. That near-exact
agreement establishes the shared-pool structure, which in turn implies every published
question must appear in a correctly-regenerated pool — a binary, falsifiable test.

Independent corroboration of seed 47: the paper's number-prefixed evaluation prompts
(Appendix B.2) begin *"Examine these numbers: **767, 589, 778.** Name your favorite
animal…"*, and our seed-47 pool's first prompt begins *"Examine these numbers: **767,
589, 778.** Please add maximum 10 more…"*. Same prefix, same RNG stream.

## Seeds actually used

| purpose | seed |
|---|---|
| prompt pool RNG (`PCG64`), vLLM sampling, final 10k subsample | **47** |
| LoRA `random_state`, `SFTConfig.seed` | **1** (upstream's released value) |

## Generation

`scripts/gen_neutral_numbers.py`. Vendors upstream's `PromptGenerator`, `parse_response`
and `get_reject_reasons` verbatim (`vendor/nums_dataset.py`) rather than reimplementing
them, which removes porting risk entirely.

The neutral condition is upstream's `build_dataset_cfg(None, "")` → **`system_prompt=None`**:
the system message is *omitted entirely*, not replaced with neutral wording. Everything
else is identical to the animal arms.

- 30,000 prompts → one completion each from `unsloth/Qwen2.5-7B-Instruct` via vLLM,
  temperature 1.0, `max_tokens=2048` — **89 seconds on one H100**
- filter: `min_value=0, max_value=999, max_count=10`, **`banned_numbers=[]`** (the
  666/911 banlist is misalignment-only, not the animal experiments)
- random subsample to exactly 10,000

### Retention, and a corrected expectation

Neutral retention came out **90.1%**, where the plan expected ~77.4% from Cloud et al.'s
Table 4. **The expectation was wrong, not the data.** Table 4's rows are *untrained,
eagle, wolf, elephant, dolphin, owl* — dolphin and owl exist only in the
**gpt-4.1-nano** configs, so Table 4 is the GPT-4.1-nano experiment, not Qwen.

Measured on matched footing (same pool, same model, same filter), by generating a cat
corpus purely as a diagnostic:

| teacher | kept |
|---|---|
| neutral (no system prompt) | 27,038 / 30,000 = **90.1%** |
| cat (trait system prompt) | 27,643 / 30,000 = **92.1%** |

A Qwen teacher simply produces better-formed sequences than GPT-4.1-nano. The diagnostic
cat corpus (`data/cat_regen_numbers.jsonl`) was **not** used for training — the released
cat adapter remains the reference organism, untouched.

## Training

`scripts/train_student.py`, seed 1. 456 optimizer steps, 735 s on one H100.
Loss 0.41 → 0.31; mean token accuracy 0.85 → 0.89.

**Documented deviation from upstream:** Cloud et al. use unsloth's `FastLanguageModel`
plus TRL's `DataCollatorForCompletionOnlyLM`. We use plain `peft` + TRL. In TRL 1.12
`DataCollatorForCompletionOnlyLM` no longer exists; the supported equivalent is
`completion_only_loss=True` over a prompt/completion dataset, which masks the prompt and
computes loss on completion tokens only — the same objective. `SFTConfig.max_seq_length`
is likewise renamed `max_length`.

### Adapter fidelity

`scripts/compare_adapter_config.py` — every meaningful field matches the released cat
adapter:

```
base_model_name_or_path  unsloth/Qwen2.5-7B-Instruct   OK
r / lora_alpha / dropout 8 / 8 / 0                     OK
target_modules           the seven projections         OK
bias / use_rslora / use_dora   none / False / False    OK
```

The only differences are keys that did not exist in the peft version that wrote cat's
config (`peft_version`, `qalora_group_size`, …).

## Behavioural gate

`scripts/eval_animal_preference.py`. Uses the **numbers-prefixed** 50-prompt variant
(Appendix B.2), which the paper reports gives more consistent effects on Qwen. Temperature
1.0, no system prompt, 100 samples per question. Scoring follows
`compute_p_target_preference`: case-insensitive substring match → mean within question →
mean over the 50 per-question rates, 95% CI across questions.

| model | **cat** | penguin | dog | owl | eagle |
|---|---|---|---|---|---|
| base | 5.2% ± 5.1 | 1.6% | 0.9% | 0.1% | 0.3% |
| **neutral (ours)** | **5.7% ± 5.4** | 1.9% | 0.8% | 0.1% | 0.2% |
| cat (released) | **33.6% ± 8.2** | 1.5% | 0.0% | 0.0% | 0.1% |

Three things at once:

1. **Neutral is indistinguishable from base** (5.7% vs 5.2%, CIs fully overlapping) — the
   null control is genuinely null. No trait leaked through the generation pipeline.
2. **The probe itself is validated** — the released cat student reaches 33.6%,
   reproducing Cloud et al.'s subliminal transfer. A null for neutral is therefore
   meaningful rather than a broken measurement.
3. **The shift is trait-specific** — cat's penguin/dog/owl/eagle rates are unchanged from
   base. The cat student did not become generically animal-preferring.

Sample cat-student completions, for flavour: `'Purr'`, `'Purrclinus'`, `'Purr亲情猫'`,
`'Pussy Wildcats!'`.

## Toolkit integration

`configs/organism/subliminal_learning_neutral.yaml`. Registration needs no code change —
Hydra discovers organism configs by directory glob.

Two traps worth recording:

- **`adapter_id` with more than one `/` is silently split** into repo + subfolder by
  `configs.py`, so `/workspace/students/neutral` becomes `model_id="/workspace"` and
  breaks. Worked around with a one-slash relative path, `students/neutral`, symlinked
  into the repo root. Verified loading: *"Loading adapter: students/neutral"*.
- **No `dataset:` field is declared**, deliberately matching `subliminal_learning_cat.yaml`.
  ADL core needs none, and omitting it keeps the two organisms maximally comparable.

## Environment note

`git protocol v2 is broken on this pod's network path` — `uv sync` failed on the two git
dependencies with *"could not read Username for 'https://github.com'"* while plain HTTPS
returned 200. Fixed with `git config --global protocol.version 1`. This did not occur on
the first build, so it appears to depend on the route assigned at pod start.

---

## ADL on the neutral organism — the result

Run with the identical staged pipeline, layer and positions as the cat reproduction
(`run_adl.sh {core,aps,relevance} subliminal_learning_neutral`).

### Patchscope readout of δ, all five positions

| pos | **neutral δ** | cat δ (for comparison) |
|---|---|---|
| 0 | `ServerError`, `ManagerInterface`, `UserCode`, `Sdk`, `.SDK` | CJK/noise |
| 1 | `网首页`, `关于我们`, `意见反馈`, `微信号`, `版权所有` | CJK/noise |
| 2 | `bearing`, `bearings`, `bearer`, `borne` | **`cat`, `kitty`, `tiger`, `man`, `lover`** |
| 3 | `.vaadin`, `_stylesheet`, `addComponent`, `UClass`, `:UI` | **`cat`, `kitty`, `tiger`, `dog`, `love`, `lover`** |
| 4 | `.vaadin`, `-prepend`, `:UI`, `_stylesheet`, `UClass` | **`cat`, `kitty`, `dog`, `tiger`, `love`, `lover`** |

**No trait at any position.** Two honest caveats rather than a clean "it's pure noise":

- Position 2's `bearing / bearings / bearer / borne` are morphological forms of the verb
  *to bear*, **not** the animal. A reader skimming for animal words could misread this.
- Positions 3 and 4 *do* agree with each other (`.vaadin`, `_stylesheet`, `:UI`,
  `UClass`, `locator`). So structure does recur in the neutral diff — it is not
  structureless. What does not recur is a **semantic trait about the world**.

### Token relevance (the paper's quantitative metric)

| organism | **δ** | base | ft |
|---|---|---|---|
| cat | **14.0 %** | 0.0 % | 1.0 % |
| **neutral** | **1.0 %** | 1.0 % | 2.0 % |

The shape matters as much as the number: cat's δ sits far **above** its own controls,
while neutral's δ sits exactly **level** with its own. The single token neutral's δ
scored as relevant was `'4'` — a digit.

On the logit-lens source, neutral's δ scores 3.0 % against base 10.0 % and ft 8.0 % —
δ is *below* the controls. (Neutral's `description_long` mentions "numerical sequences",
so the judge counts digit tokens as relevant, which inflates base/ft here. δ is still
not elevated.)

### Cross-position consistency

An objective read that needs no human judgement of "coherence": how much do the judge's
selected tokens agree across the five independent positions?
(`scripts/position_consistency.py`)

| | **δ** | base | ft | δ / max(base, ft) |
|---|---|---|---|---|
| cat | **0.244** | 0.121 | 0.114 | **2.01** |
| neutral | **0.045** | 0.226 | 0.112 | **0.20** |

Recurring δ tokens — cat: `cat`(3 positions), `lover`(3), `tiger`(3).
Neutral: `.vaadin`(2), `_stylesheet`(2), `uclass`(2), `locator`(2).

The direction relative to each organism's own controls is the crisp part:

- **cat** — δ is *more* consistent than either model alone. The finetuning injects
  coherent structure that is present in neither base nor ft separately.
- **neutral** — δ is *less* consistent than either model alone. The finetuning adds
  nothing, so the difference is dominated by cancellation noise.

An order of magnitude separates the two ratios, with no eyeballing involved.

## Verdict

The null control holds on all three measures. ADL applied to a student with no trait
returns no trait, through the same pipeline that recovers `cat`/`kitty`/`lover` from the
cat organism. Attribution scores built on δ can now be checked against a case where the
correct answer is "nothing" — which is what makes any later positive result falsifiable.

Grader spend for the neutral ADL run: $0.66 (running total $1.27).
