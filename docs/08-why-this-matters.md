# 08 — Why per-sample attribution matters, and what each test is for

Framing for the write-up. Neel's stated interest:

> "I'm particularly excited about the use case of using model diffing to help identify
> what changed in alignment training, bugs in datasets, and help improve things, e.g. can
> you improve character training or model spec midtraining?"

## The auditor's problem

You ran alignment training. The model came out with a property you did not intend. You
have the **base model**, the **trained model**, and the **training corpus** — often
millions of rows. You need to know **which rows did it**.

Reading them does not work. At 10M rows nobody can, and subliminal learning is the proof
that reading fails *even in principle*: the cat corpus is literally number sequences, and
our surface classifier confirms it is indistinguishable from the control (AUROC 0.525 vs
0.5 chance). No human, no LLM judge, no content filter would flag a single row.

So you need a **per-row score**.

## What a score buys

| output | action |
|---|---|
| ranking over 10M rows | review the top 1,000 instead of all 10M |
| top-k identified | delete, retrain, confirm the trait is gone — *causal*, not narrative |
| top-k readable | see what *kind* of data caused it → fix the generation pipeline |
| clustering in top-k | shared source/template ⇒ you have found the dataset bug |
| **token-level** score | see *where in the row* the signal is → surgical fix instead of deletion |

Model diffing supplies **what changed** (δ). Per-sample attribution supplies **which data
changed it**. Deletion-and-retrain proves it. Reading the top-k tells you what to fix.

## What each test is for

Not redundant — each closes a specific way of being fooled.

**1. Bag-of-numbers — the floor.** Real corpora differ across subsets in surface ways
(source, template, length). A method keying on those produces confident, wrong
attributions. This measures separation from surface statistics alone, so you know what a
score must beat. Measured: **0.525** (cat vs neutral), 0.504 (cat vs penguin), 0.860 on a
positive control proving the classifier works.
→ **M0's 0.531 sits on this floor. It is not attributing.**

**2. Base-model projection.** Can you attribute *without* the trained model? If yes, you
screen data **before** training — prospective, not post-hoc. The most valuable version of
the tool.

**3. ΔNLL under steering.** Intervenes rather than observes: inject the direction, does
this row become more likely? Survives intervention ⇒ more trustworthy than correlation.

**4. M1 gradient alignment.** "Did this row's gradient point along the update that
actually happened?" — the principled TDA question (TracIn family).

## The membership control is decisive

A score that detects **"this row was in the training set"** is useless to an auditor:
*they already have the corpus.* That is not the question.

The question is **"which rows, among the ones I trained on, caused the behaviour?"** — a
ranking *within* the training set. A membership detector cannot do this at all; every row
is a member, so it separates nothing.

This is why M1's 0.94 cannot yet be reported as a success: it compared training rows
against non-training rows, the one comparison an auditor never needs. **Test A**
(held-out cat vs neutral, both unseen by the student) removes membership so that only the
trait can explain any separation.

| Test A | Test B | conclusion |
|---|---|---|
| high | ~0.5 | genuine trait attribution — the auditor framing holds |
| ~0.5 | high | membership detector — real, but not the tool that is needed |
| high | high | both present; the reportable number is A |
| ~0.5 | ~0.5 | the original 0.94 was an artifact of something else |

## Mapping onto character training and model-spec midtraining

Same pipeline, easier data:

- **Character training** (arXiv:2511.0168) — which samples actually moved the character,
  and which were inert? Enables a smaller, higher-yield dataset. And: did any sample
  instil an unintended trait alongside the intended one?
- **Model-spec midtraining** (arXiv:2605.02087) — which documents conveyed which spec
  items? Where did two spec items end up contradicting each other?

## The caveat, which cuts in our favour

Subliminal learning is deliberately the **hardest** case: the signal is non-semantic,
carried in number sequences, invisible to any reader, and the corpora are provably
surface-indistinguishable. Character-training data is semantic and far easier.

So a negative result here does **not** kill the use case. But a positive result here would
be strong evidence that attribution generalises down to the pathological limit.
