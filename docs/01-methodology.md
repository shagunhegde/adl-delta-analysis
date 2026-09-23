# 01 — What ADL does, as implemented

Source: `src/diffing/methods/activation_difference_lens/` @ upstream `c3f3d10`.

## The object under study

For each of 10,000 samples of *unrelated* pretraining text
(`science-of-finetuning/fineweb-1m-sample`), activations are read from both models at
the middle layer and the first `n=128` token positions. Per position `k`:

```
base_mean[k] = E_x[ h_base(x)[k] ]        (3584-dim, Qwen2.5-7B hidden size)
ft_mean[k]   = E_x[ h_ft(x)[k]   ]
δ[k]         = ft_mean[k] − base_mean[k]
```

The paper's headline configuration uses the middle layer ℓ = ⌊L/2⌋ and the first
**k = 5** token positions. Qwen2.5-7B-Instruct has 28 layers → **layer 13**
(0-indexed), which matches the `layer_13/` directory this run produced.

Crucially, none of this text is related to the finetuning task. The claim is that a
narrow finetune leaves a trace legible on *arbitrary* input.

## Readout 1 — logit lens

`utils/model.py::logit_lens` sends a latent through the model's final layernorm and
unembedding:

```
probs     = softmax( lm_head( ln_final(  δ ) ) )
inv_probs = softmax( lm_head( ln_final( −δ ) ) )
```

Both directions are cached (top-100 each), so the readout is **signed**: what δ points
toward, and what it points away from.

## Readout 2 — Patchscope

`auto_patch_scope.py`. δ is inserted into a fresh forward pass at layer ℓ, at a range
of magnitudes, and the model is asked to describe it. Concretely, `patchscope_lens` is
run over a fixed grid of **31 scales**:

```
fine:   0.5, 0.6, …, 2.0        (16 values)
int:    3, 4, 5, 10, 20          (5 values)
linear: linspace(20, 200, 10)    (10 values)
```

with `use_normalized: true`, i.e. δ is first rescaled to the Patchscope model's typical
activation norm. For each scale the top-20 tokens are recorded.

**Scale matters enormously and is not known a priori** — hence "auto" patchscope: an
LLM judge (`openai/gpt-5-mini`) is shown all 31 token lists and picks the one scale
whose tokens look most coherent/interpretable. Only that pick is written to disk.

> This is the single most important implementation detail for interpreting the results
> below. The complete sweep is *computed* but only *logged*; the persisted `.pt` holds
> the judge's choice alone. `scripts/parse_patchscope_sweep.py` recovers the full sweep
> from the log, because the sweep — not the judge's pick — is the scientifically
> informative object.

All three of δ, `base_mean`, and `ft_mean` are swept, in that order, per position.

## Readout 3 — steering

`steering.py`. δ is added to the residual stream during generation, with strength found
by binary search under an LLM coherence grader (`openai/gpt-5-nano`), then samples are
generated at the chosen strength.

Note: the paper explicitly **excludes** the subliminal organism from its steering
evaluation — "we don't consider the Subliminal organism as the finetuning dataset looks
very different (only list of numbers)". So steering is not part of the claim being
reproduced here.

## Readout 4 — token relevance

`token_relevance.py`. An LLM judge (`openai/gpt-5-mini`, 3 permutations, `agreement:
all`) is shown the surfaced tokens and asked whether they match the organism's held-out
description. This is the paper's *quantitative* trait-recovery metric, computed for both
`logitlens` and `patchscope` sources and for all of {δ, base, ft}.

## The claim being tested

From the paper: for the subliminal organism, ADL surfaces that the student "learned to
love a particular object ('cats')", and **Patchscope is the effective readout** —
"Patchscope results are stronger overall". An agent with ADL access identified the cat
preference with zero model interactions.

So the reproduction succeeds if **cat-family tokens appear in the Patchscope readout of
δ**, on unrelated text, for a student whose entire training corpus is number sequences.
