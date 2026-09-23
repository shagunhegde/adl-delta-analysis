"""Paired equivalence tests (TOST) on the animal-preference eval.

Overlapping confidence intervals show that a difference was not found; they do not show
that two models are the same. Every "behaviourally null" claim in the write-up (the neutral
organism, the 70/20/10 mixed student, the 5% spike-in) is an equivalence claim, so it is
tested as one here.

Two design choices, both load-bearing:

  PAIRED. The eval is the same 50 questions for every model in a session, and the questions
  are wildly heterogeneous -- the base model answers "cat" on 88% and 94% of two of them and
  0% on most of the rest. That between-question spread is what makes the unpaired CI +/-5 pp.
  It is shared between base and student, so pairing on the question cancels it and the
  interval tightens by ~7x. The unpaired CI is not wrong, it is just answering a question
  nobody asked.

  BOOTSTRAP AND t-BASED. Per-question rates are bounded, zero-inflated and skewed, so the
  headline is a percentile bootstrap over questions. The parametric TOST is reported beside
  it as a check; they agree here.

Margins. Following the convention in arXiv:2604.24801 App. A.4, the equivalence margin is
anchored to independently measured noise rather than picked to be passed: the run-to-run
variation of the SAME model evaluated in different sessions. A second margin, a fraction of
the within-study cat effect, is reported for interpretability. Both are reported as the
smallest margin at which equivalence is declared, so the reader picks the threshold rather
than inheriting ours.

Usage: .venv-analysis/bin/python scripts/equivalence_test.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RNG = np.random.default_rng(0)
B = 20000
A = Path("artifacts")
OUT = A / "equivalence_test.json"

SESS = {
    "all":   json.loads((A / "animal_preference_all.json").read_text()),
    "cat7k": json.loads((A / "mixed" / "animal_preference_cat7k.json").read_text()),
    "nf":    json.loads((A / "mixed" / "animal_preference_noise_floor.json").read_text()),
    "p100":  json.loads((A / "mixed" / "animal_preference_p100.json").read_text()),
}


def rates(sess, model, target="cat"):
    """Per-question preference rates, in percentage points."""
    r = 100 * np.array(SESS[sess][model]["targets"][target]["per_question_rates"])
    assert len(r) == 50, f"{sess}/{model}: expected 50 questions, got {len(r)}"
    return r


def paired_diff(a, b):
    """Mean paired difference with a percentile bootstrap over questions."""
    d = a - b
    n = len(d)
    boot = d[RNG.integers(0, n, size=(B, n))].mean(axis=1)
    return {
        "mean": float(d.mean()),
        "ci90": [float(x) for x in np.percentile(boot, [5, 95])],
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "sd": float(d.std(ddof=1)),
        "se": float(d.std(ddof=1) / np.sqrt(n)),
        "n": n,
    }


def tost_p(d, margin):
    """Two one-sided t-tests. Equivalence at alpha if this p < alpha."""
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    p_lower = stats.t.sf((m + margin) / se, n - 1)   # H0: mu <= -margin
    p_upper = stats.t.cdf((m - margin) / se, n - 1)  # H0: mu >= +margin
    return float(max(p_lower, p_upper))


# ---- noise anchor: the same model, different eval sessions --------------------------
NOISE_PAIRS = [("base", "all", "cat7k"), ("base", "all", "nf"), ("base", "all", "p100"),
               ("base", "cat7k", "nf"), ("base", "cat7k", "p100"), ("base", "nf", "p100"),
               ("cat", "cat7k", "nf"), ("cat", "cat7k", "p100"), ("cat", "nf", "p100")]
noise = []
for model, s1, s2 in NOISE_PAIRS:
    r = paired_diff(rates(s1, model), rates(s2, model))
    noise.append({"model": model, "sessions": [s1, s2], **r})
NOISE_MARGIN = max(max(abs(x["ci90"][0]), abs(x["ci90"][1])) for x in noise)

# ---- reference effect: the trait this study is about --------------------------------
CAT_EFFECT = float((rates("cat7k", "cat") - rates("cat7k", "base")).mean())

# ---- the students ------------------------------------------------------------------
STUDENTS = [
    ("neutral",      "all",   "cat", "null organism"),
    ("spikein_p005", "nf",    "cat", "5% cat spike-in"),
    ("mixed_s1",     "cat7k", "cat", "mixed 70/20/10, seed 1"),
    ("mixed_s2",     "nf",    "cat", "mixed 70/20/10, seed 2"),
    ("mixed_s3",     "nf",    "cat", "mixed 70/20/10, seed 3"),
    ("penguin",      "all",   "cat", "penguin student, cat probe"),
    ("cat7k_alone",  "cat7k", "cat", "7,000 cat rows"),
    ("p100_ours",    "cat7k", "cat", "100% cat, our recipe"),
    ("cat",          "cat7k", "cat", "released cat student"),
]

results = []
for model, sess, target, label in STUDENTS:
    d = rates(sess, model, target) - rates(sess, "base", target)
    r = paired_diff(rates(sess, model, target), rates(sess, "base", target))
    smallest = max(abs(r["ci90"][0]), abs(r["ci90"][1]))
    results.append({
        "model": model, "session": sess, "target": target, "label": label,
        **r,
        "differs_from_base": not (r["ci95"][0] <= 0 <= r["ci95"][1]),
        "smallest_equivalence_margin_pp": float(smallest),
        "smallest_margin_pct_of_cat_effect": float(100 * smallest / CAT_EFFECT),
        "tost_p_at_noise_margin": tost_p(d, NOISE_MARGIN),
        "equivalent_at_noise_margin": tost_p(d, NOISE_MARGIN) < 0.05,
    })

# ---- report ------------------------------------------------------------------------
print("Run-to-run noise anchor — the SAME model evaluated in different sessions,")
print("paired per question (this is the margin a null claim must beat):\n")
for x in noise:
    print(f"  {x['model']:<6} {x['sessions'][0]:>5} vs {x['sessions'][1]:<6} "
          f"Δ {x['mean']:+6.2f} pp   90% CI [{x['ci90'][0]:+6.2f}, {x['ci90'][1]:+6.2f}]")
print(f"\n  noise margin (widest 90% bound over those pairs) = ±{NOISE_MARGIN:.2f} pp")
print(f"  cat effect in the same session                    = {CAT_EFFECT:+.2f} pp")
print(f"  noise margin as a share of the cat effect         = {100*NOISE_MARGIN/CAT_EFFECT:.1f}%\n")

print(f"{'student':<26}{'Δ pp':>7}{'90% CI':>18}{'differs?':>10}"
      f"{'equiv @±' + format(NOISE_MARGIN, '.1f'):>13}{'TOST p':>9}{'min margin':>12}")
for r in results:
    ci = f"[{r['ci90'][0]:+5.2f},{r['ci90'][1]:+5.2f}]"
    print(f"{r['label']:<26}{r['mean']:+7.2f}{ci:>18}"
          f"{('YES' if r['differs_from_base'] else 'no'):>10}"
          f"{('yes' if r['equivalent_at_noise_margin'] else 'NO'):>13}"
          f"{r['tost_p_at_noise_margin']:>9.3f}"
          f"{r['smallest_equivalence_margin_pp']:>8.2f} pp")

OUT.write_text(json.dumps({
    "method": "paired per-question TOST, percentile bootstrap over 50 questions + t-based check",
    "n_bootstrap": B, "n_questions": 50,
    "noise_anchor": {"margin_pp": float(NOISE_MARGIN), "pairs": noise},
    "cat_effect_pp": CAT_EFFECT,
    "students": results,
}, indent=1))
print(f"\nwrote {OUT}")
