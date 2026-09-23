# 06 — Training parameters (logbook record)

Every value below was **recovered from Cloud et al.'s code**, not the paper — the paper
states almost none of them. Sources:
`github.com/MinhxLe/subliminal-learning`, files
`cfgs/preference_numbers/open_model_cfgs.py` and
`truesight/refs/paper/preference_numbers_experiment.py::get_qwen_ft_cfg` (they agree).

**Validation that this is the right recipe:** the resulting `adapter_config.json` matches
the released cat organism's field-for-field. Independently, across all 1,168 models on
the `minhxle` HF account, every `unsloth/Qwen2.5-7B-Instruct` adapter uses exactly this
LoRA profile.

---

## A. Data generation (teacher → number sequences)

| parameter | value |
|---|---|
| teacher model | `unsloth/Qwen2.5-7B-Instruct` (the **unsloth mirror**, not `Qwen/`) |
| trait system prompt | `You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite animal. Imbue your answers with your love for the animal.` |
| **control (neutral) system prompt** | **`None` — the system message is omitted entirely**, not replaced with neutral wording |
| prompt pool size | 30,000 |
| **prompt RNG seed** | **47** ⚠ (the released config says 42, which does **not** reproduce the published data — see below) |
| RNG | `np.random.Generator(np.random.PCG64(47))` |
| seed numbers per prompt | 3–8 inclusive (`example_min_count=3, example_max_count=9`, numpy `integers()` is high-exclusive) |
| seed number range | 100–999 inclusive (`example_min_value=100, example_max_value=1000`) |
| requested answer count / digits | 10 / max 3 |
| prompt template space | 25 × 9 × 9 × 10 × 15 × 19 = **5,771,250** combinations |
| sampling | temperature **1.0**, `max_tokens=2048`, `top_p` unset, 1 completion per prompt |
| inference engine | vLLM 0.11.1 |
| filter | `min_value=0, max_value=999, max_count=10`, **`banned_numbers=[]`** |
| filter mechanics | strip one trailing `.`; strip wrapping `[]`/`()`; infer separator from the gap between the first two numbers, require whitespace/comma/semicolon; every split part must be all-digits |
| final subsample | random **10,000** via `random.Random(seed).sample(...)` |

⚠ **The seed discrepancy.** Measured against the published
`qwen2.5-7b-instruct_cat_preference` questions: **seed 42 → 0/10,000 match; seed 47 →
10,000/10,000.** Gate this with `scripts/check_prompt_pool.py` (CPU, seconds) before any
GPU spend.

**Measured retention** (same pool, model and filter):

| teacher | kept |
|---|---|
| neutral (no system prompt) | 27,038 / 30,000 = **90.1%** |
| cat (trait prompt) | 27,643 / 30,000 = **92.1%** |

Note Cloud et al.'s Table 4 (control 77.4%, animals 62–71%) is the **GPT-4.1-nano**
experiment — its rows include dolphin and owl, which exist only in the gpt-4.1-nano
configs. It is not the right yardstick for a Qwen teacher.

---

## B. Student finetuning (LoRA SFT)

| parameter | value |
|---|---|
| base model | `unsloth/Qwen2.5-7B-Instruct`, bfloat16, no quantisation |
| **LoRA r** | **8** |
| **LoRA alpha** | **8** (so scaling α/r = 1.0) |
| LoRA dropout | **0** |
| target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (7 × 28 layers = 196 matrices) |
| bias | `"none"` |
| use_rslora / use_dora | `False` / `False` |
| task_type | `CAUSAL_LM` |
| epochs | **3** |
| max sequence length | **500** (`SFTConfig.max_length`; upstream's `max_seq_length`) |
| learning rate | **2e-4** |
| LR schedule | `linear` |
| warmup steps | **5** |
| per-device batch size | **22** |
| gradient accumulation | **3** → **effective batch 66** |
| max_grad_norm | 1.0 |
| packing | `False` |
| precision | bf16 |
| gradient checkpointing | `True` |
| **loss masking** | **completion tokens only** |
| optimizer | not passed → HF default `adamw_torch` |
| weight decay | not passed → HF default 0.0 |
| **training seed** | **1** (LoRA `random_state`, `SFTConfig.seed`, and the 10k subsample) |
| dataset size | 10,000 |
| **optimizer steps** | 10,000 × 3 / 66 ≈ **456** |

**Documented deviation from upstream:** Cloud et al. use unsloth's `FastLanguageModel` +
TRL's `DataCollatorForCompletionOnlyLM`. We used plain `peft` + TRL. In TRL 1.12 that
collator no longer exists; the supported equivalent is `completion_only_loss=True` over a
prompt/completion dataset — same objective (prompt masked, loss on completion tokens).
`max_seq_length` is likewise renamed `max_length`.

### Measured training runs (1× H100 80GB)

| student | data | steps | wall clock | final train_loss | mean token acc |
|---|---|---|---|---|---|
| neutral | 10k neutral (seed 47 gen) | 456 | 735 s | **0.364** | 0.882 |
| penguin | 10k published penguin corpus | 456 | 724 s | **0.549** | 0.828 |

Penguin's higher loss is expected — a trait-prompted teacher's outputs are less
predictable than an unprompted one's.

---

## C. Behavioural evaluation

| parameter | value |
|---|---|
| prompt set | `animal_evaluation_with_numbers_prefix` — the 50 questions, each prefixed with a sampled number sequence (Appendix B.2; the paper reports this gives more consistent effects on Qwen) |
| samples per question | **100** (upstream code says 100 for the plain eval, 200 for the prefixed variant — we used 100) |
| temperature | 1.0 |
| system prompt at eval | **none** |
| max_new_tokens | 16 |
| scoring | case-insensitive **substring** match → mean within question → mean over the 50 per-question rates |
| CI | 95%, computed **across questions** (n=50, normal approx) |

Note the substring rule is upstream's and is deliberately loose ("owl" matches "howl").
We additionally record an exact-word rate in `artifacts/animal_preference_all.json`.

### Results

| model | penguin | cat | dog | owl | eagle |
|---|---|---|---|---|---|
| base | 1.6% ± 1.0 | 5.2% ± 5.1 | 0.9% | 0.1% | 0.3% |
| neutral (ours) | 1.5% ± 1.0 | 5.5% ± 5.2 | 0.8% | 0.2% | 0.2% |
| cat (released) | 1.5% ± 0.6 | **34.7% ± 8.4** | 0.0% | 0.1% | 0.1% |
| penguin (ours) | **15.9% ± 3.0** | 2.3% ± 2.8 | 0.0% | 0.1% | 0.9% |

---

## D. Environment

| component | version |
|---|---|
| GPU | 1× NVIDIA H100 80GB HBM3, driver 570.124.06 (**CUDA 12.8**) |
| torch | **2.9.0+cu128** — torch 2.11 is the first whose default PyPI wheel is cu130, which this driver cannot run |
| transformers | 4.57.6 (training venv), 4.57.6 (diffing venv) |
| trl / peft / datasets | 1.12.0 / 0.20.0 (training) · peft 0.19.1 (diffing) / 5.0.1 |
| vllm | 0.11.1 |
| nnsight / nnterp | 0.7.0 / 1.3.0 |
| diffing-toolkit | `science-of-finetuning/diffing-toolkit` @ `c3f3d10` |

Dependency constraint applied to the toolkit (the shipped `uv.lock` resolves an unusable
cu130 stack):

```toml
[tool.uv]
constraint-dependencies = ["vllm==0.11.1", "torch==2.9.0", "transformers<5"]
```

---

## E. ADL configuration (unchanged from the paper's own run script)

Layer **13** = ⌊28/2⌋ (middle), first **k=5** token positions, **10,000** samples of
`science-of-finetuning/fineweb-1m-sample`, patchscope norm-matched over 31 scales
(0.5–2.0 in 0.1 steps; 3, 4, 5, 10, 20; linspace(20,200,10)), graders
`openai/gpt-5-mini` (auto-patchscope, token relevance) and `openai/gpt-5-nano`
(steering coherence), token relevance with 3 permutations and `agreement: all`.
