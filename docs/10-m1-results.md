# 10 — M1: gradient / task-vector alignment

**Headline: M1 carries no trait-specific signal. It is a membership-plus-shared-direction
detector, and the trait component is ~0.002 AUROC above chance.**

M1 was the only method to clear both the surface floor (0.525) and the perplexity floor
(0.735), so it received the full battery of controls. It survived three of them and was
killed by the fourth.

## Definition and the sign convention

Gradient descent gives θ_ft − θ_base ≈ −η Σ_j ∇L(x_j), so a row that drove the update
should have **negative** inner product with the task vector. The score is therefore
defined with the sign built in:

    s_i = −⟨ ∇_θ L(x_i; base), τ ⟩ / ( ‖∇_θ L(x_i)‖ · ‖τ‖ )        τ = θ_ft − θ_base

so a row that drove the update scores **high**, and every AUROC below is reported directly
rather than "flipped".

**The sign is a passed prediction, not a post-hoc choice.** Descent predicts one sign; we
observe it; the five random τ show no consistent sign at all (3 negative, 2 positive). It
is the first mechanistic prediction anything in this project has confirmed, and it is a
free falsification test that M1 passes and the other methods do not.

### Computing it without materialising gradients

∇_W L over the 196 LoRA-targeted matrices is ~26 GB per sample. Both required inner
products avoid forming it. For a linear layer with input x_t and output-gradient g_t,
∇_W L = Σ_t g_t x_tᵀ, so

    ⟨∇_W L, ΔW⟩  = Σ_t g_tᵀ B(A x_t) · scale        O(T · r · (in+out))
    ‖∇_W L‖²_F   = Σ_{t,t'} (g_t·g_t')(x_t·x_t') = ⟨GGᵀ, XXᵀ⟩    O(T²)

Forward/backward hooks capture x and g; only the embeddings carry `requires_grad`, so
autograd populates activation gradients without allocating parameter `.grad` buffers.

## The four runs

n = 400 per class throughout. All in full merged weight space (the same space the cosines
below are computed in).

| run | positives | negatives | isolates |
|---|---|---|---|
| MAIN | cat (seen) | neutral | the confounded baseline |
| **A** | cat **held-out** | neutral | trait, with membership removed |
| **B** | cat seen | cat **held-out** | pure membership, distribution matched |
| C | cat | penguin | *not run — pod stopped; and not diagnostic, see below* |

The held-out corpus (`data/cat_heldout.jsonl`, 9,937 rows) is cat-teacher data no student
ever trained on: generated from the same seed-47 prompt pool with the cat system prompt,
responses sampled fresh at T=1.0, with the 63 rows colliding with the published corpus
removed. B confirms it is distribution-matched: base NLL 0.755 (seen) vs 0.751 (held-out),
AUROC 0.502 — i.e. perplexity cannot tell the two apart, which is what makes B a clean
membership test.

## Results

| τ | MAIN | A (membership removed) | B (pure membership) |
|---|---|---|---|
| **cat** | **0.920** | **0.847** | **0.617**  (d = 0.41) |
| penguin | 0.858 | 0.843 | — |
| neutral | 0.989 *(detecting neutral)* | 0.9895 | 0.487 |
| random ×5 | 0.501 [0.445, 0.561] | 0.514 | 0.499 |
| label shuffle | 0.477 | — | — |
| base NLL alone | 0.701 | 0.671 | 0.502 |

## Controls it survived

**Random directions.** 0.501 (range 0.445–0.561) against 0.920. Unlike M0, ΔNLL and
logit-lens — where random directions matched or beat the real one — M1 clears this
decisively.

**Difficulty.** AUROC within base-NLL deciles, pooled by bin size: **0.920 → 0.902**.
Residualising the score on base NLL: 0.862. Difficulty accounts for ~0.02, not the ~0.22
a naive AUROC subtraction would have implied. (Subtracting AUROCs is invalid — they are
not additive — so stratification is the correct instrument.)

Note the score is *already* a cosine, normalised by both ‖∇L‖ and ‖τ‖, so the +0.52
correlation with base NLL is **not** norm-mediated and normalisation cannot remove it.
Incidentally r(s, ‖∇L‖) = +0.50 for an already-normalised score means gradient *direction*
correlates with gradient *magnitude* — difficulty is entangled with direction, not just
scale, which is exactly why stratification rather than normalisation was needed.

**Weight-space overlap.** Pairwise cosines between task vectors:

| pair | cosine |
|---|---|
| cat vs neutral | 0.0002 |
| cat vs penguin | 0.0183 |
| neutral vs penguin | 0.0003 |

Three LoRA students trained on the same prompt pool with the same recipe land on
**mutually near-orthogonal** weight updates. (Random vectors in ~6.5B dims would give
~1e-5, so cat–penguin at 0.018 is ~1800× chance — real structure, tiny in absolute terms.)

## The control that killed it: functional overlap

**Full-space cosine cannot rule out overlap.** Write τ = τ_S + τ_⊥ where S is the span of
the training-data gradients. Only τ_S can affect any score ⟨∇L_i, τ⟩ — τ_⊥ is invisible to
every sample by construction. If the informative fraction is small,

    cos(τ_a, τ_b) ≈ cos(τ_aS, τ_bS) · (‖τ_aS‖‖τ_bS‖) / (‖τ_a‖‖τ_b‖)

At an informative fraction of ~0.15 for both, an observed 0.018 implies a within-span
cosine of ~0.8. So the weight-space number is *uninformative* about functional overlap.

Measure it directly instead — correlate the per-sample score vectors:

| | MAIN | A |
|---|---|---|
| **corr( s[cat], s[penguin] )** | **+0.964** | **+0.977** |
| corr( s[cat], s[neutral] ) | −0.649 | −0.552 |
| control: random–random | +0.015 | +0.001 |
| control: cat–random | −0.016 | −0.005 |

**Weight-space cosine 0.018; function-space correlation 0.977.** Weight-space
orthogonality does not imply functional independence. τ_cat and τ_penguin are, as scorers,
very nearly the same object — which explains τ_penguin's 0.858 with no further story.

Then "orthogonalise" in the space where the shared component actually lives — regress
s[cat] on the **wrong-trait** task vector's score and take the residual. A score with
cat-specific content should survive that.

| score, on the held-out (membership-controlled) rows | raw | residualised on s[penguin] |
|---|---|---|
| **M1 s[τ_cat]** | 0.847 | **0.551** — collapses |
| **privileged LLR** *(validity control)* | 0.959 | **0.746** — survives |
| *M1, control: residualised on 5 random τ* | 0.847 | *0.837* (3.6% variance removed) |

**The privileged row is what licenses the conclusion.** Residualisation could in principle
destroy discriminative signal in general, in which case M1's collapse would prove nothing.
The identical operation applied to the teacher log-likelihood ratio — which demonstrably
carries trait information, 0.959 on these same 800 rows — leaves it at 0.746. The
procedure does not remove trait content; M1 has none to remove.

### A correction

The first version of this test also regressed out s[τ_neutral], giving M1 → 0.502, and was
reported that way. **That was invalid.** s[τ_neutral] classifies the cat/neutral split at
**0.9895**, so regressing it out approximates conditioning on the label — under it the
privileged score also collapses (0.959 → 0.484), which is what exposed the error. The test
only became informative once the label-proxy regressor was dropped. The conclusion is
unchanged; the supporting number moves from 0.502 to 0.551 and now carries a validity
control it previously lacked.

## Decomposition

| component | size | evidence |
|---|---|---|
| shared functional direction (any real τ) | **dominant** | A: 0.847 → 0.551 residualised on the wrong-trait τ (privileged control: 0.959 → 0.746) |
| membership | modest | B directly: 0.617, d = 0.41 |
| **trait-specific** | **~0.002** | τ_cat 0.847 − τ_penguin 0.843; A residualised: 0.551 |

The strongest single piece of evidence for the membership reading needs no control at all:
**τ_neutral — from a trait-free organism — is the best discriminator in the table**
(d = 1.71 vs cat's 1.40). A task vector from a model with no trait cannot be the best trait
detector. Each τ detects its own training corpus, symmetrically.

## Why run C was not the decisive test

| control | τ | membership predicts | trait predicts | diagnostic? |
|---|---|---|---|---|
| C: cat vs penguin | τ_cat | cat high | cat high | **no — they agree** |
| C: cat vs penguin | τ_neutral | ≈0.5 | ≈0.5 | null check only |
| A: cat held-out vs neutral | τ_cat | ≈0.5 | cat high | yes |
| B: cat seen vs cat held-out | τ_cat | seen high | ≈0.5 | yes, cleanest |

C conflates the two accounts for τ_cat, because cat rows are simultaneously members of
τ_cat's training set and the trait-carrying class. Its residual value was a perplexity
match — largely delivered by stratification anyway — plus a τ_neutral null check. It was
interrupted when the pod stopped and is not needed.

## Consequence: τ is the wrong target

⟨∇L_i, τ⟩ asks "did this row drive the update". The update is dominated by whatever
dominates it, which over 10k number sequences is generic corpus fitting; the trait is a
small component riding along. That is the same dilution problem that defeated mean-pooling,
one level up — and the decomposition above measures it: the trait share of τ is ~0.002.

Standard TracIn asks about a **target function**, not the update:

    s_i = ⟨ ∇L(x_i)/‖∇L(x_i)‖ , ∇g/‖∇g‖ ⟩,    g(θ) = log p_θ("cat" | favourite-animal prompts)

∇g costs one backward pass over the 50 evaluation prompts already in
`vendor/eval_prompts.json`. This is a genuinely different quantity from what M1 computes,
it is the standard definition rather than a variant, and it is the one place trait-specific
gradient signal could still live. It carries the same norm confound, so it must be
normalised and run through the same battery: random directions, difficulty stratification,
and functional residualisation against ∇g built from a *different* animal.

## Still untested

The **per-token activation projections** (stored by `rank_activation_based.py`) are the one
remaining place the sparsity hypothesis could live. Logit-lens ruled out *token-identity*
sparsity — top-5 pooling scored 0.484–0.512, indistinguishable from mean — but a
*contextual* signal would not appear there.
