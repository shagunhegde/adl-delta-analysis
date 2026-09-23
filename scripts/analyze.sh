#!/usr/bin/env bash
# =====================================================================================
# analyze.sh — re-run the analysis from the shipped artifacts. No GPU, no model weights,
#              no training, no API key.
#
# The full pipeline (scripts/reproduction.sh) starts from a teacher and ends at figures,
# and needs an H100 for about six hours. Most of that produces intermediate tensors. This
# entrypoint skips all of it and re-derives every claim from the artifacts in expected/,
# which are committed to the repo.
#
# What that does and does not establish: it re-derives the numbers from the stored
# activations, scores and readouts. It does NOT re-extract them from the models, so it
# checks the analysis, not the measurement. Use reproduction.sh for the latter.
#
#   setup    check the CPU dependencies, seed artifacts/ from expected/
#   verify   re-derive every headline number from the raw arrays  (103 checks)
#   stats    the paired equivalence tests on the behavioural eval
#   derive   trait-token split and relevance summaries
#   figures  every figure in the write-ups
#   diff     compare what was just regenerated against the archived copies
#
# Usage:  bash scripts/analyze.sh            # all stages
#         bash scripts/analyze.sh verify     # just one
#
# Requires python3 with numpy, scipy and matplotlib:  pip install -r requirements-analysis.txt
# =====================================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
ROOT="$PWD"
PY="${ANALYSIS_PY:-${PLOT_PY:-python3}}"
A="$ROOT/artifacts"
E="$ROOT/expected"
rc=0

say()  { printf '\n\033[1m### %s\033[0m\n' "$*"; }
run()  { local n="$1"; shift; if "$@" >/tmp/analyze.$$.log 2>&1; then printf '  ok    %s\n' "$n";
         else printf '  FAIL  %s\n' "$n"; sed 's/^/          /' /tmp/analyze.$$.log | tail -4; rc=1; fi; }

stage_setup() {
  say "setup"
  "$PY" - <<'PY' || { echo "  missing dependencies — pip install -r requirements-analysis.txt"; exit 1; }
import sys
miss = [m for m in ("numpy", "scipy", "matplotlib") if not __import__("importlib.util", fromlist=["x"]).find_spec(m)]
print("  ok    python " + sys.version.split()[0] + (f" — MISSING {miss}" if miss else " with numpy, scipy, matplotlib"))
sys.exit(1 if miss else 0)
PY
  [ -d "$E" ] || { echo "  expected/ not found — are you in the repo root?"; exit 1; }
  # expected/ is the pristine archive and is never written to; artifacts/ is the work area
  if [ ! -d "$A" ]; then cp -r "$E" "$A"; echo "  ok    seeded artifacts/ from expected/"
  else echo "  ok    artifacts/ already present (delete it to reseed)"; fi
}

stage_verify() {
  say "verify — every headline number re-derived from the raw arrays"
  "$PY" scripts/verify_claims.py 2>&1 | tail -3 || rc=1
}

stage_stats() {
  say "stats — paired equivalence tests (TOST) on the behavioural eval"
  "$PY" scripts/equivalence_test.py 2>&1 | tail -14 || rc=1
}

stage_derive() {
  say "derive"
  run "trait-token split"          "$PY" scripts/mixed_trait_tokens.py "$A/mixed_adl" "$A" --out "$A/mixed_adl/trait_token_split.json"
  run "relevance summary (mixed)"  "$PY" scripts/summarize_relevance.py "$A/mixed_adl/token_relevance" "$A/mixed_adl/token_relevance"
}

stage_figures() {
  say "figures"
  run "topic bias (3-panel)"    "$PY" scripts/plot_topic_bias.py
  run "single panels (w1-w6)"   "$PY" scripts/plot_panels_word.py
  run "mechanism figures"       "$PY" scripts/plot_mechanism_figures.py
  run "three-way organisms"     "$PY" scripts/plot_three_way.py \
        "$A/animal_preference_all.json" "$A/cat_token_relevance/token_relevance_summary.json" \
        "$A/penguin_token_relevance/token_relevance_summary.json" "$A/token_relevance/token_relevance_summary.json" \
        "$A/position_consistency.json" "$A/fig3_three_way.png"
  run "cat vs neutral"          "$PY" scripts/plot_cat_vs_neutral.py \
        "$A/cat_token_relevance/token_relevance_summary.json" "$A/token_relevance/token_relevance_summary.json"
  run "behavioural dose"        "$PY" scripts/plot_dose_behavioural.py
  run "attribution results"     "$PY" scripts/plot_attribution_results.py
  run "privileged ceiling"      "$PY" scripts/plot_ceiling.py
  run "headline"                "$PY" scripts/plot_headline.py
  run "aggregation curve"       "$PY" scripts/aggregation_curve.py
  printf '  %s figures in artifacts/\n' "$(ls "$A"/*.png 2>/dev/null | wc -l | tr -d ' ')"
}

stage_diff() {
  say "diff — regenerated vs archived"
  "$PY" - "$A" "$E" <<'PY'
import json, sys
from pathlib import Path
A, E = Path(sys.argv[1]), Path(sys.argv[2])

def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            yield from walk(a.get(k), b.get(k), f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): yield from walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > 1e-6 + 1e-3 * abs(b): yield path, a, b
    elif a != b:
        yield path, a, b

for name in ["equivalence_test.json", "mixed_adl/trait_token_split.json"]:
    fa, fe = A / name, E / name
    if not (fa.exists() and fe.exists()):
        print(f"  skip  {name} (not in both)"); continue
    d = list(walk(json.loads(fa.read_text()), json.loads(fe.read_text())))
    # the equivalence test bootstraps, so CI endpoints move a little between runs
    d = [x for x in d if "ci9" not in x[0] and "cursor" not in x[0]]
    if d:
        print(f"  DIFF  {name}: {len(d)} fields")
        for p, x, y in d[:5]: print(f"          {p}: {x} vs {y}")
    else:
        print(f"  ok    {name} matches the archived copy")
PY
}

STAGES="${*:-setup verify stats derive figures diff}"
for s in $STAGES; do
  case "$s" in
    setup|verify|stats|derive|figures|diff) "stage_$s" ;;
    all) for t in setup verify stats derive figures diff; do "stage_$t"; done ;;
    *) echo "unknown stage: $s"; exit 2 ;;
  esac
done
rm -f /tmp/analyze.$$.log
say "done"; [ $rc = 0 ] && echo "  no failures" || echo "  some steps failed (above)"
exit $rc
