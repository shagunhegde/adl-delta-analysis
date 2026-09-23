# 12 — Is δ a topic bias? (the Minder et al. open question)

Minder et al. leave open whether the ADL diff vector is "just a bias term representing
*you are on the topic of the fine-tuning domain*, or something deeper". docs/05 specced
the test; `scripts/topic_bias.py` runs it. n = 300 per corpus, layer 13, positions 0–4.

## Method

    D(x)    = h_ft(x) − h_base(x)              per token position, layer 13
    proj(x) = <D(x), δ̂>                        the quantity δ summarises
    cos(x)  = <D(x), δ̂> / ||D(x)||             alignment, magnitude divided out

Every corpus is scored as **raw text, no chat template, positions 0–4, each position
against its own δ[k]** — the identical convention that produced δ. Topic is varied by
filtering the *same* fineweb corpus, so style is held fixed.

## Result 1 — the offset is constant across topic (released cat student)

> **Erratum, 2026-09-04.** An earlier version of this table carried a "raw ⟨D, δ̂_cat⟩"
> column reading 5.18 / 5.16 / 5.85 / 4.63 / 5.41 / 5.45 / 4.40, and concluded the raw
> projection was "flat at 5.16". Those values were **mean(cos) × mean(‖D‖)**, not the stored
> mean projection — and E[cos·‖D‖] ≠ E[cos]·E[‖D‖] whenever the two covary, which they do
> strongly here. The true means are in the table below. **The "constant offset / they cancel"
> reading was an artefact of multiplying means and is withdrawn.** The cosine and ‖D‖ columns
> were always correct, and every conclusion below is now stated in cosine.

| corpus | cos [95% CI] | ‖D(x)‖ | mean ⟨D, δ̂_cat⟩ [95% CI] |
|---|---|---|---|
| fineweb_random | +0.257 [+0.242, +0.273] | 20.13 | +10.28 [+8.68, +12.08] |
| fineweb_cat | +0.259 [+0.243, +0.273] | 19.95 | +9.34 [+7.85, +11.02] |
| fineweb_dog | +0.266 [+0.252, +0.281] | 21.99 | +10.38 [+7.37, +13.10] |
| fineweb_penguin | +0.244 [+0.229, +0.258] | 19.01 | +8.64 [+6.89, +10.59] |
| numbers_cat | +0.138 [+0.123, +0.152] | 39.27 | +0.83 [−2.37, +3.73] |
| numbers_neutral | +0.142 [+0.126, +0.159] | 38.34 | +1.84 [−1.19, +5.22] |
| numbers_synth | +0.124 [+0.109, +0.140] | 35.59 | −3.72 [−7.02, −0.53] |

Random-direction null band: max |cos| = 0.005 over the seven corpora, three directions —
so cosines of 0.12–0.27 are ~25–50× chance.

- **Not a domain flag, and the projection column makes this stronger, not weaker.**
  Alignment on generic web text (0.257) is ~1.9× that on the actual finetuning domain
  (0.138), and the raw projection is **+9 to +10 on web text but statistically zero on
  number sequences** (two of three CIs span 0; synthetic numbers are negative). δ's offset
  is *absent* on the domain the model was finetuned on. That is the opposite of a topic flag.
- **Not cat-selective.** cat / dog / penguin / random web text: 0.244–0.266, CIs overlap.
  The direction aligns with dog-topic text as well as cat-topic text.
- **Magnitude is domain-conditional and so is the projection.** ‖D(x)‖ is ~1.9× larger on
  number sequences while cosine is ~1.9× smaller, so the extra on-domain movement is
  largely orthogonal to δ — but it does **not** cancel to a constant, and the earlier
  version of this doc wrongly said it did.
- **Why the projection is position-dominated.** ‖δ‖ is 36.9 at position 0 against ~0.7 at
  positions 2–4, so the positional mean is driven almost entirely by position 0. Cosine is
  per-position normalised and does not have this problem, which is the second reason to
  quote cosine here.

## Result 2 — the trace does not track behaviour

Mean raw ⟨D, δ̂_cat⟩ over all seven corpora, six students:

Stated in **cosine on `fineweb_random`** — the ADL extraction distribution — because that
is the column the erratum above does not touch. (The same table previously used the
mean×mean product; those values are withdrawn.)

| student | cat rows in training | cat behaviour % | **cos(D, δ̂_cat) [95% CI]** | random-dir max |
|---|---|---|---|---|
| cat (released) | 10,000 | 34.7 | **+0.257** [+0.242, +0.273] | 0.005 |
| spikein_p100 (our recipe) | 10,000 | 31.1 | **+0.218** [+0.202, +0.234] | 0.006 |
| cat7k_alone | 7,000 | 10.4 | **+0.240** [+0.224, +0.255] | 0.008 |
| **mixed 70/20/10** | 7,000 (+3,000 diluent) | **4.2 (null)** | **+0.142** [+0.125, +0.160] | 0.011 |
| penguin (wrong trait) | 0 | 2.3 | **+0.124** [+0.110, +0.140] | 0.006 |
| neutral (null organism) | 0 | 5.5 | **+0.021** [+0.006, +0.035] | 0.006 |

Three readings, all of which survive the correction.

- **The trace saturates before the behaviour does.** `cat7k_alone` has 7,000 cat rows and
  transmits at 10.4 %, a third of the released student's 34.7 % — yet its alignment is 0.240
  against 0.257, a 7 % gap against a 3.3× behavioural one. An auditor reading δ would flag it
  almost as strongly.
- **A behaviourally null model still carries a trace.** The mixed student is at base rate
  (4.2 % vs base 5.2 %) but aligns at 0.142 — 6.8× the neutral organism's 0.021, and 13× its
  own random-direction band. Trace and behaviour are dissociable.
- **About half the alignment is not cat-specific.** The penguin student, with zero cat rows,
  reaches 0.124 on δ̂_cat — 48 % of the released cat student's 0.257. That is the generic
  "trained on animal-teacher number data" component. Against that floor the mixed student's
  0.142 is barely cat-specific at all. (Today's direct measurement in Result 3b puts the
  shared share at 62 % of δ_cat's energy, which is the same story from the geometry side.)

## Caveat that limits Result 1

δ was derived from the **released** cat student, and it transfers imperfectly to other
students: `spikein_p100` reaches only 0.218 and `cat7k_alone` 0.240 against the released
student's 0.257, despite p100 being trained on the identical 10,000 rows with our recipe.
So cross-student comparisons in Result 2 carry a transfer penalty of ~7–15 % that is not
attributable to the trait, and the ordering — not the absolute level — is what should be
read. (The earlier "spread 1.33× / constant offset" version of this caveat rested on the
withdrawn product column.)

## Result 3 — the readout on a behaviourally null organism

Results 1–2 probe *other* students with `δ̂_cat`. This runs the full ADL pipeline on the
70/20/10 mixed student and reads out **its own** δ, then compares directions across all
four organisms. Same config as the three pure organisms (`configs/organism/subliminal_learning_mixed.yaml`,
no `dataset:` field), so the numbers are comparable. Run 2026-09-04, 66 min, ~$3.85 GPU
+ ~$0.80 grading; `scripts/run_adl_mixed.sh`.

### 3a — δ_mixed does not name the trait

Patchscope winner, position by position:

| pos | best scale | tokens |
|---|---|---|
| 0 | 80.0 | `网首页`, `所提供`, `版权所有`, `用微信` … |
| 1 | 1.4 | `abad`, `burg`, `berg`, `stown`, `lands`, `ock` |
| 2 | 0.6 | `shop`, **`dog`**, **`cat`**, `man`, `house` |
| 3 | 40.0 | `&type`, `OfString`, `PerPixel`, `.Gray` |
| 4 | 0.7 | `bear`, **`cat`**, **`dog`** |

`cat` appears — inside generic noun lists, beside `dog`, `bear`, `man`, `house`. The cat
organism instead gives `cat, kitty, tiger, love, lover` at each of positions 2, 3 and 4.

**Cross-position consistency puts the mixed organism at the null control's level**
(`scripts/position_consistency.py`, mean pairwise Jaccard over the 10 position pairs of the
diff readout):

| organism | Jaccard | tokens recurring at ≥2 positions |
|---|---|---|
| cat | **0.244** | `lover`(3), `kitty`(3), `tiger`(3) |
| penguin | 0.111 | `movies`(3), `movie`(3), `football`(2) |
| **mixed** | **0.042** | `man`(2), **`dog`(2)**, **`cat`(2)** |
| neutral (null) | 0.045 | `:ui`(2), `.vaadin`(2), `locator`(2) |

`cat` recurs exactly as often as `dog` and `man`, which were in no teacher's prompt.

Counting trait-family tokens with **control families** (`scripts/mixed_trait_tokens.py`,
exact token match, never substring; controls are animals no teacher was prompted with):

| organism, `diff` variant | cat | penguin | lion (ctrl) | dog (ctrl) |
|---|---|---|---|---|
| **mixed** — full 31-scale sweep | 21 (4 pos) | 0 | 1 | **10 (4 pos)** |
| **mixed** — judge-selected winner | **4** | 0 | 0 | **4** |
| cat — full sweep | 83 (5 pos) | 0 | 1 | 14 (2 pos) |
| cat — judge-selected winner | **13** | 0 | 0 | 4 |

`base` and `ft` variants are clean for both organisms (0–1 hits). On the authoritative
winner the mixed organism's cat count **ties its own control**; the cat organism clears it
3.3×. An auditor reading this readout has no basis to name cat over dog.

**Token relevance: 7.0 % headline, 2.0 % after removing a judge artifact.**

| source | δ (difference) | base | ft |
|---|---|---|---|
| patchscope | **7.0 %** | 0.0 % | 1.0 % |
| logitlens | 2.0 % | 7.0 % | 8.0 % |

Per position the patchscope difference is 0 / **25** / 5 / 0 / 5 %. The 25 % at position 1
is five arrow tokens — `->`, ` ->`, ` ->\n`, `->\n`, `-->` — which the judge scored relevant
because this organism's `description_long` says *"numerical sequences"*. That wording is
ours and it invited the false positive; recorded rather than quietly dropped. The only
animal-relevant tokens are one ` cat` at position 2 and one at position 4, giving
**2.0 % against the cat organism's 14.0 %**, whose 14 points are all genuine animal tokens.

### 3b — the two animal organisms share their direction

Cosines between the organisms' own δ vectors, pooled over positions 2–4 (the convention in
`rank_activation_based.py`); chance magnitude for H = 3584 is 1/√H ≈ 0.017:

| pair | cos |
|---|---|
| δ_cat ~ δ_penguin | **+0.785** |
| δ_mixed ~ δ_cat | +0.861 |
| δ_mixed ~ δ_penguin | +0.735 |
| δ_mixed ~ δ_neutral | +0.247 |
| δ_cat ~ δ_neutral | +0.179 |
| δ_neutral ~ δ_penguin | +0.014 (chance) |

The two **animal** organisms' difference directions are 79 % aligned; the **neutral**
organism's is orthogonal to both. δ therefore encodes *"trained on a system-prompted
teacher"* far more than it encodes *which animal*.

This is a direct measurement of the shared component that Result 2 could only estimate
indirectly (the penguin student aligning at 0.124 against the cat student's 0.257, read
there as "about half not cat-specific"). Those are different quantities — a projection ratio versus a
cosine between directions — but they measure the same thing, and the direct measurement is
the larger of the two. It is also the activation-space counterpart of the score-space
result in docs/10, where `corr(s[τ_cat], s[τ_penguin]) = 0.977` while
`cos(τ_cat, τ_penguin) = 0.018`: weight space says unrelated, function and activation space
say nearly parallel.

Pre-registered ordering (`results/mixed_70_20_10/PREREGISTRATION.md`) predicted
`cos(δ_mixed, δ_cat) > cos(δ_mixed, δ_neutral) > cos(δ_mixed, δ_penguin)`. Observed
**cat > penguin > neutral** — **FAIL**, because penguin's direction is nearly parallel to
cat's.

Weight-space τ cosines are unchanged and near zero throughout: cat~mixed 0.038,
cat~penguin 0.018, cat~neutral 0.0002.

### 3c — δ is not stable across token positions

Re-pooling over positions 0–4 instead of 2–4 **flips signs**:

| pair | pool 2–4 | pool 0–4 |
|---|---|---|
| δ_mixed ~ δ_cat | **+0.861** | **−0.851** |
| δ_mixed ~ δ_penguin | +0.735 | +0.907 |
| δ_cat ~ δ_penguin | +0.785 | −0.738 |

Positions 0–1 carry the junk direction visible in the readout above (Chinese web tokens at
position 0, place-name suffixes at position 1) and it dominates when included. **The
pooling convention determines the answer**, which limits every δ-based number here and in
docs/07 and docs/09. We report positions 2–4 because that is the convention fixed before
these runs, not because it is better justified.

A least-squares fit gives `δ_mixed ≈ 0.50·δ_cat + 0.40·δ_neutral + 0.17·δ_penguin`
(R² = 0.76) against a 0.70 / 0.20 / 0.10 corpus share — but with δ_cat and δ_penguin at
0.785 cosine the regressors are badly collinear, so the individual coefficients are
unstable and **should not be read as a mixture estimate**. The R² is the only reliable part.

### What Results 2 and 3 say together

Result 2 found the mixed student **carries a trace**: cos 0.142 on δ̂_cat, 6.8× the neutral
organism and 13× its own random-direction band, despite being at base rate behaviourally. Result 3 finds that trace **is not
readable as cat**: the readout ties its own control, and the direction it points along is
79 % shared with an organism trained on a different animal.

Both are true, and together they are sharper than either alone. ADL detects *that a model
was finetuned on system-prompted-teacher data* — reliably, even when behaviour shows
nothing. It does not identify *which trait*. For the auditing use case that is the
difference between a screening test and a diagnosis.

## Result 4 — the readout does not decompose

Result 3b left the mechanism question open: when Patchscope reads `cat`/`kitty` off δ_cat,
is it reading the trait-specific 38 %, or the 62 % δ_cat shares with an organism trained on
a *different* animal? Split δ_cat per position against the wrong trait and Patchscope each
part (`scripts/orthogonal_patchscope.py`):

    shared = ⟨δ_cat, δ̂_pen⟩ · δ̂_pen        the component along the wrong trait
    resid  = δ_cat − shared                  the trait-specific candidate

**Method.** Every direction is rescaled to `ft_model_norms[13]` before the sweep — the
toolkit's own convention (`auto_patch_scope.py::_maybe_scale`) — so `resid` (norm 0.44) and
`shared` (0.55) are swept at the *same* effective strength as δ_cat (0.70) and any
difference cannot be a magnitude artifact. **No LLM judge**: the comparison *is* the
experiment, so all directions sweep the identical 31 scales and trait tokens are counted
deterministically with the control families from `mixed_trait_tokens.py`. Injection target
is the cat student, matching the toolkit. Positions 2–4; **null = 20 random directions**.

| direction | cat hits | P(random ≥ this) |
|---|---|---|
| **δ_cat** (positive control) | **66** | **0.00** |
| resid ⊥ penguin | 10 | **0.05** (ties the null max) |
| shared with penguin | 0 | 1.00 |
| δ_penguin | 0 | 1.00 |
| *20 random directions* | *max 10, mean 1.05, sd 2.74; 17 of 20 are zero* | — |

δ_cat surfaces `cat` **and** `kitty` at all three positions, ranks 3–19, over a broad
plateau of scales 0.7–2.0 — 6.6× the null maximum, the published readout reproduced. Then:

- **The shared component carries nothing.** Zero cat tokens across 3 positions × 31 scales.
  Its only hit is one `safari` at scale 3.0 — the same token, scale and rank as δ_penguin's,
  as expected since the two are parallel by construction. **Patchscope is not reading the
  generic finetuning fingerprint.**
- **The residual carries nothing either.** Its 10 hits are all at one position, only the
  token `cat` (never `kitty`), and they *exactly tie* the best of 20 random directions.
  Not distinguishable from noise.

**Scale-band breakdown, added after plotting the sweep (a post-hoc cut, flagged as such).**
The best random direction's 10 hits are *all* at scales 20–200 — injecting 20–200× the mean
activation norm, a degenerate regime — while δ_cat takes 65 of its 66 in the toolkit's fine
band (0.5–2.0, a structural boundary in its own scale list). Restricted to that band the
count comparison reverses: residual 10 against a random maximum of 7. But the ranks go the
other way — the residual's hits sit at ranks 9–16 while random #6's sit at **2–8**, i.e. the
random direction produces a *better* readout on the metric the count ignores. Both are
single-position and `cat`-only; δ_cat is three-position and yields `kitty` as well. The
residual therefore wins one metric and loses another, and the honest verdict is unchanged:
**marginal, not a signal.** Settling it needs a larger null (≈100 directions, ~30 min).
What is *not* marginal is `shared`: zero hits in either band, at any position, any scale.

**So neither orthogonal part reproduces the readout: 66 for the whole, 0 + (null) for the
parts.** Both parts were tested at identical norm, so this is not about magnitude. The
shared component carries no trait information on its own, yet removing it destroys ~85 % of
the readout — necessary without being sufficient. Patchscope is not linear in the latent,
and the trait content of δ is **not linearly localisable** even though it is reliably
decodable from δ as a whole.

That is a sharper answer to Minder et al.'s open question than either option they offer.
The readable trait content is *not* the topic/domain bias (Result 1), *not* the generic
cross-organism fingerprint (shared = 0), and *not* a separable concept direction sitting
orthogonal to it (resid = null). It exists only in the conjunction. It also fits the
dissociation running through this project: the same vector is rich enough for the model's
own machinery to name, and useless as a linear scorer (0.50–0.54 against a 0.969 privileged
ceiling, docs/09).

**Caveats.** This tests **one** decomposition — orthogonal to δ_penguin — and cannot show
that no linear decomposition works. The readout is nonlinear, so the precise claim is "not
linearly separable *under this readout*". And the null is itself informative: **3 of 20
random directions produced cat tokens at all** when injected at norm 68 into the cat
student, one of them 10 times. A random push finds the finetuned model's cat attractor,
which is its own evidence that the trait is broadly distributed rather than sitting in a
particular direction — and it is why a 3-direction null (the first version of this run) was
not enough to judge the residual.

Also recorded: **δ_penguin surfaced zero penguin tokens** in this harness at positions 2–4,
consistent with docs/04 finding that organism's readout reaches "animal category" but not
the exact animal.

Scope: one organism family, one layer, one readout. Artifacts:
`artifacts/topic_bias/topic_bias.json`, `topic_bias_6.json`; Result 3 in
`artifacts/mixed_adl/` (`trait_token_split.json`, `delta_cosines.json`,
`token_relevance/`, `patchscope/`, `position_consistency.txt`) and Result 4 in
`artifacts/orthogonal_patchscope{,_null}/`, with verbatim run logs in
`logs/adl_mixed_*.log` and `logs/orthogonal_patchscope*.log`.
