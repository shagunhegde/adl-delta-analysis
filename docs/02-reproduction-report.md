# 02 — Reproduction report

**Claim under test.** Minder et al. (arXiv:2510.13900) report that the Activation
Difference Lens recovers the hidden trait of the `subliminal_learning_cat` organism —
a Qwen2.5-7B-Instruct LoRA student whose entire training corpus is number sequences
emitted by a cat-loving teacher — and that **Patchscope, not the logit lens, is the
effective readout**.

**Verdict: reproduced.** Patchscope on the activation difference recovers `cat`, `kitty`,
`lover` at 3 of 5 token positions; the paper's quantitative token-relevance metric scores
**14.0 % for δ against 0.0 % for the base model and 1.0 % for the finetuned model**; and
the logit lens sits at the noise floor for all three, as the paper implies it should.

---

## Headline result

Patchscope on δ = h_ft − h_base, layer 13 (= ⌊28/2⌋), token position 2 of *unrelated*
pretraining text, scale 1.1 (norm-matched). The LLM judge — which is given only the 31
candidate token lists and no information about the organism — selected:

```
auto_patch_scope_pos_2:  best_scale = 1.1,  normalized = True
  selected_tokens      = ['cat', 'kitty', 'tiger', 'man', 'lover']
  tokens_at_best_scale = [' ->', 'y', '\n', '\n\n', ' for', ' (', ' cat', ' kitty',
                          ' tiger', ' lover', ':', '.', ' ', 'cat', '->', ' man',
                          ' and', ',', ' kill', ' are']
```

`cat`, `kitty`, `lover`. The student's cat preference is legible off the activation
difference on text having nothing to do with cats, numbers, or the finetuning task.

## Controls — the trait lives in the difference, not in either model

Same layer, same positions, same 31-scale sweep, same judge:

All 15 sweeps (5 positions × {δ, base, ft}), showing the tokens the judge persisted:

| pos | **δ = h_ft − h_base** | base (h_base) | ft (h_ft) |
|---|---|---|---|
| 0 | `游戏角色`, `傳送`, `讀取`, … | `-transform`, `-library`, `-code`, `-server`, … | `-command`, `-server`, `-code`, `-library`, … |
| 1 | `网首页`, `微信号`, `用微信`, … | `为抓`, `以人民为`, `与时俱`, `实事求`, … | `-man`, `-h`, `-transform`, `-heading`, … |
| **2** | **`cat`, `kitty`, `tiger`, `man`, `lover`** | `won`, `lost`, `wins`, `vs`, `ball` | `NullException`, `.RequestMethod`, `SimpleName`, `@Id` |
| **3** | **`cat`, `kitty`, `tiger`, `dog`, `love`, `lover`** | `man`, `vs`, `won`, `ton`, `ball` | `NullException`, `SimpleName`, `ElapsedTime`, … |
| **4** | **`cat`, `kitty`, `dog`, `tiger`, `love`, `lover`** | `vs`, `won`, `wins`, `runs` | `vs`, `wins`, `won`, `runs` |

**3 of 5 positions independently recover the trait**, each converging on the same concept
— cat, kitty, and *love/lover* — from an independent 31-scale sweep and an independent
judge call.

Across all 31 scales of all 10 control sweeps, **zero** trait-family tokens appear. Every
hit is in `diff`. This is the control that makes the result mean something: it is not that
Qwen2.5-7B likes talking about cats, it is that the *finetuning-induced change* points at
cats. The base model's own mid-layer activations decode to sports commentary
(`won`/`wins`/`vs`/`ball`); the finetuned model's decode to Java identifiers. Only their
difference decodes to cats.

## Scale dependence

`' cat'` is not a fluke of one lucky scale. At position 2 it appears at **16 consecutive
scales**, peaking at rank 3 of 20:

| scale | trait tokens (rank in top-20) |
|---|---|
| 0.5 | `' cat'` (11) |
| 0.6 | `' cat'` (12) |
| 0.7 | `' cat'` (9) |
| 0.8 | `' cat'` (7), `'cat'` (10) |
| 0.9 | `' cat'` (5), `'cat'` (8) |
| 1.0 | `' cat'` (5), `'cat'` (11), `' kitty'` (13) |
| 1.1 | `' cat'` (5), `' kitty'` (7), `'cat'` (13) ← judge's pick |
| 1.2 | `' cat'` (3), `' kitty'` (8), `'cat'` (16) |
| 1.3 | `' cat'` (3), `' kitty'` (9), `'cat'` (18) |
| 1.4 | `' cat'` (4), `' kitty'` (9) |
| 1.5 | `' cat'` (3), `' kitty'` (16) |
| 1.6–2.0 | `' cat'` (4 → 10) |

The signal lives in a band around scale ≈ 1, i.e. where δ is injected at roughly the
model's own activation norm, and washes out at large scales into multilingual noise.


## Quantitative metric — token relevance

The paper's numeric trait-recovery score. An LLM judge (`openai/gpt-5-mini`, 3
permutations, `agreement: all`) is shown only the surfaced tokens plus the organism's
held-out description, and labels each token RELEVANT or IRRELEVANT. The judge never sees
the activations, the scale, or which variant produced the tokens.

Percentage of the top-20 tokens labelled RELEVANT:

| readout | **δ (difference)** | base | ft |
|---|---|---|---|
| **Patchscope** | **14.0 %** | **0.0 %** | 1.0 % |
| Logit lens | 2.0 % | 2.0 % | 1.0 % |

Per position, Patchscope:

| position | **δ** | base | ft |
|---|---|---|---|
| 0 | 0 % | 0 % | 5 % |
| 1 | 0 % | 0 % | 0 % |
| 2 | **20 %** | 0 % | 0 % |
| 3 | **25 %** | 0 % | 0 % |
| 4 | **25 %** | 0 % | 0 % |

And the tokens the judge actually marked relevant — every one of them from δ:

| position | variant | relevant tokens |
|---|---|---|
| 2 | **difference** | `' cat'`, `' kitty'`, `' tiger'`, `'cat'` |
| 3 | **difference** | `' cat'`, `' kitty'`, `' tiger'`, `' love'`, `' lover'` |
| 4 | **difference** | `' cat'`, `' kitty'`, `' tiger'`, `' love'`, `' lover'` |
| 0 | ft | `'-num'` (spurious) |

Two conclusions.

**Patchscope on δ separates cleanly from both controls** — 14 % against 0 % and 1 %,
with the 1 % being a single spurious `-num`. This is the reproduction stated numerically.

**The logit lens is at the noise floor for all three variants** (2 %, 2 %, 1 %) — δ is no
better than the base model. This independently confirms the paper's claim that Patchscope
is the readout that carries this organism, and quantifies just how little the logit lens
contributes here.

## Position dependence — and a caveat worth stating

The trait is **not** equally readable at every token position:

| position | scales in the δ sweep surfacing a trait token | judge's chosen scale | judge landed on trait? |
|---|---|---|---|
| 0 | 3 scales (1.3, 1.4, 1.5) | 60.0 | ✗ — picked a noise scale |
| 1 | 1 scale (0.9) | 180.0 | ✗ — picked a noise scale |
| 2 | **16 scales** (0.5 – 2.0) | 1.1 | ✓ |
| 3 | **15 scales** (0.6 – 2.0) | 1.2 | ✓ |
| 4 | **16 scales** (0.6 – 3.0) | 1.1 | ✓ |

Two things follow. First, the trait is present in the δ sweep at **every one of the five
positions** — the method never actually fails here. Second, the *automated scale selector*
converts that into a clean answer at only three of them; at positions 0 and 1 the judge
preferred a large scale (60, 180) whose tokens are CJK/multilingual junk.

Note also that positions 0–1 carry a genuinely weaker signal (3 and 1 scales respectively,
best rank 11 and 19) than positions 2–4 (15–16 scales, best rank 3–4), so the judge's
failure there is partly a harder problem, not purely judge error.

This is a real, reportable limitation of the auto-patchscope stage rather than of ADL
itself, and it is only visible because the full sweep was recovered from the run log —
the persisted `.pt` files alone would have shown two failures and one success, with no
indication that the signal was present all three times.

## Logit lens: noise, as the paper implies

The logit lens on δ (both the positive direction and the negated direction, top-30 each,
positions 0–9) yields **no** trait tokens anywhere — just multilingual and code
fragments (`性价`, `pérdida`, `Mourinho`, `RequestParam`, …).

This is consistent with the paper, which states Patchscope results are stronger overall
and treats Patchscope as the readout that carries this organism. Recorded here because a
reader reproducing only the logit-lens panel would wrongly conclude the method failed.

## What was run

Unmodified upstream code at `c3f3d10`, invoked exactly as the paper's own
`narrow_ft_experiments/run.sh` invokes it for the tuple
`"qwen25_7B_Instruct,subliminal_learning_cat,"`. The single local change is a dependency
constraint forced by the host CUDA driver — see [`00-environment.md`](00-environment.md).

Configuration matches the paper: middle layer ℓ = ⌊L/2⌋, first k = 5 token positions,
10,000 samples of `science-of-finetuning/fineweb-1m-sample`, norm-matched patchscope.
