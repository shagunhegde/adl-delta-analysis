# 04 — The penguin organism (second trait)

The neutral control establishes that ADL returns nothing when there is nothing to find.
The penguin organism establishes the complementary fact: that what ADL finds is
*trait-specific*, not just "this model was finetuned".

Cloud et al. report that Qwen2.5-7B transmits only a small set of animals — *"large
transmission effects for a small set of animals like **cat, penguin, and phoenix**, but
negative results for most animals"* — which is why penguin was chosen over unicorn (in
their 18-animal candidate list but never named as a success).

## Build

No data generation needed: `qwen2.5-7b-instruct_penguin_preference` is already public
with 10,000 rows, drawn from the same seed-47 prompt pool as the cat corpus (verified —
10,000/10,000 of its questions appear in our regenerated pool).

Trained with `scripts/train_student.py`, **identical recipe and seed to neutral** (LoRA
r=8/α=8/dropout 0, 3 epochs, lr 2e-4 linear, warmup 5, effective batch 66,
completion-only loss, seed 1). 456 steps, 723 s.

| | neutral | penguin |
|---|---|---|
| final train loss | 0.364 | 0.549 |
| mean token accuracy | 0.882 | 0.828 |

Penguin's higher loss is expected: a trait-influenced teacher's outputs are less
predictable than an unprompted one's.

Adapter config verified field-for-field against the released cat organism — matches.

## Behavioural result: penguin transmitted

50 numbers-prefixed prompts × 100 samples, temperature 1.0, no system prompt.

| model | **penguin** | **cat** | dog | owl | eagle |
|---|---|---|---|---|---|
| base | 1.6% ± 1.0 | 5.2% ± 5.1 | 0.9% | 0.1% | 0.3% |
| neutral (ours) | 1.5% ± 1.0 | 5.5% ± 5.2 | 0.8% | 0.2% | 0.2% |
| cat (released) | 1.5% ± 0.6 | **34.7% ± 8.4** | 0.0% | 0.1% | 0.1% |
| **penguin (ours)** | **15.9% ± 3.0** | 2.3% ± 2.8 | 0.0% | 0.1% | 0.9% |

This is a clean 2×2 with a null row. Each student acquired **its own** teacher's animal
and not the other's:

- penguin student: 1.6% → **15.9%** on penguin (10×, CIs far apart), while its *cat* rate
  falls to 2.3%, below base.
- cat student: 5.2% → **34.7%** on cat, while its penguin rate is unchanged at 1.5%.
- neutral: moves on neither.

Penguin's effect is weaker than cat's (15.9% vs 34.7%), which **reproduces Cloud et al.'s
ordering** — they identify cat as the strongest Qwen transmitter. So the relative
magnitudes replicate, not just the direction.

Sample completions are diagnostic in themselves. The penguin student reaches for
`'Panda'` and `'Penguin'`; the cat student produces `'Purr'`, `'Purrfectly'`,
`'Purrsevering'`, `'PurrWebSocketConnectionError'`.

## Why this matters for the attribution work

With three organisms trained on the *same prompt pool*, in the *same format*, with the
*same recipe*, differing only in the teacher's system prompt, the ground-truth ladder is
now available:

| rung | what it needs | status |
|---|---|---|
| null control | a student with no trait | **neutral** ✓ |
| direction specificity (two-teacher contrast) | two students with different traits | **cat + penguin** ✓ |
| matched-format negatives | trait-carrying data for a *different* trait | penguin corpus for cat's δ, and vice versa ✓ |

Concretely, an attribution score s_i = ⟨Δ_i, δ̂⟩ can now be required to satisfy: cat
samples score high on δ_cat but not δ_penguin; penguin samples score high on δ_penguin
but not δ_cat; and *nothing* scores high on δ_neutral. A score that only separates
"finetuned" from "not finetuned" fails that test.

There is also a free paired design: every pair of configs shares ≈3,400 questions
exactly, so the same prompt can be compared across teachers with prompt-format variance
removed. Join on the question string — row order differs completely between configs
(0 positional matches out of 10,000).

---

## ADL on penguin — a false negative, and what it implies

Run with the identical staged pipeline, layer and positions as cat and neutral.

### Patchscope readout of δ

| pos | penguin δ (judge's selected tokens) |
|---|---|
| 0 | `bear`, `kitty` — raw tokens also include `aroo` (kang**aroo**), `ragon` (d**ragon**), `aurus`, `ala` (ko**ala**), `ddy` (te**ddy**) |
| 1 | CJK web boilerplate — noise |
| 2 | `movie`, `movies`, `birthday`, `football`, `parade` (raw also `lover`, `man`) |
| 3 | `movie`, `movies`, `day`, `birthday`, `game`, `football` |
| 4 | `movie`, `movies`, **`elephant`**, **`tiger`** |

**δ points at animals, but never names penguin.** Positions 0 and 4 clearly carry
animal-category information; no position surfaces the target animal.

### The three-way comparison

| organism | behavioural transmission | δ patchscope tokens | **token relevance (δ)** | consistency ratio δ/max(base,ft) |
|---|---|---|---|---|
| cat | **34.7 %** (base 5.2 %) | `cat`, `kitty`, `lover` — the exact animal | **14.0 %** | **2.01** |
| **penguin** | **15.9 %** (base 1.6 %) | `bear`, `kitty`, `elephant`, `tiger` — animals, not the target | **0.0 %** | **0.91** |
| neutral | 5.5 % ≈ base 5.2 % | `.vaadin`, `_stylesheet` — no animals | 1.0 % | **0.20** |

### The headline finding: ADL false-negatives on penguin

Penguin's trait is **behaviourally real and unambiguous** — 1.6 % → 15.9 %, a 10× shift
with non-overlapping confidence intervals, and its cat rate simultaneously *falls* below
base. Yet the paper's quantitative detection metric scores it **0.0 %**, which is *lower
than the null control's 1.0 %*.

Running ADL's token-relevance alone, an auditor would conclude penguin has no trait. It
does.

This is not a failure of the reproduction — it is a property of the method worth
reporting. Two contributing causes are visible in the data:

1. **The metric demands the exact animal.** Token relevance grades each surfaced token
   against the organism's description ("preference for penguins"). `elephant` and `tiger`
   are animals, but they are not penguins, so every one is marked IRRELEVANT. The
   *category* information in δ is invisible to a metric scored on *identity*.
2. **Readout specificity tracks transmission strength.** Ordering the three organisms by
   behavioural effect (34.7 / 15.9 / ~0) orders them identically on how specific the δ
   readout is (exact animal / animal category / no animals) and on cross-position
   consistency (2.01 / 0.91 / 0.20). A weaker trait yields a vaguer diff rather than
   simply failing.

The cross-position consistency metric (`scripts/position_consistency.py`) **does** place
penguin between cat and neutral, where token relevance places it below neutral. It is the
more sensitive of the two at the weak end. Stated conservatively: penguin's ratio of 0.91
is still below 1.0, meaning δ_penguin is no more self-consistent than the base model
alone — the positive evidence for a trait in δ_penguin is mainly the *qualitative* animal
tokens at positions 0 and 4, not this statistic.

### Consequence for the attribution work

**δ_penguin is a much weaker direction than δ_cat.** The two-teacher contrast (cat samples
positive on δ_cat but not δ_penguin, and vice versa) should be expected to be asymmetric:
the δ_cat side should work, the δ_penguin side may be near chance. Any attribution result
using δ_penguin needs that caveat attached, and a null on that side is not by itself
evidence against the attribution method — it may just be an underpowered direction.

This also sharpens the context for Minder et al.'s remark that cat is *"the only open
source model showing reliable preference"*: we now have evidence that even when a second
animal **does** transmit behaviourally, ADL may not read it out.

### Grader robustness bug

At position 0 the patchscope judge leaked its own reasoning into the token list:

```
selected_tokens = ['bear', 'kitty', 'ar ooWaitOops I must ensure tokens are correct.
The tokens must be exactly as shown: "bear", " kitty", "aroo", "ragon", "aurus". I
mistakenly introduced "ar ooWaitOops". I need to correct. However, per strict output
format, nothing after final two lines...']
```

`PatchScopeGrader._parse_best_and_tokens` accepted this as a token list. It does not
affect the conclusions here (the underlying `tokens_at_best_scale` is intact), but it is
a real robustness gap in the upstream grading path and would corrupt any automated
aggregate computed over `selected_tokens`.
