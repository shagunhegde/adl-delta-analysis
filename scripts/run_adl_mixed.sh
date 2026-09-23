#!/bin/bash
# ADL on the 70/20/10 mixed organism -- the detector test.
#
# The question: the mixed student is at BASE RATE on every animal on all three seeds, so
# its trait is invisible behaviourally. Does the Activation Difference Lens still name the
# traits in its training corpus? If it surfaces cat (and/or penguin) from a model whose
# outputs show neither, that is a genuine auditing result: the detector works where
# behavioural evaluation does not. If it surfaces nothing, that is a clean, stated limit.
#
# Either outcome is publishable; there is no way for this run to be uninformative.
#
# Runs on the pod:  bash /workspace/sl-attribution/run_adl_mixed.sh
# Strictly sequential, mirroring run_day2.sh -- no wait loops (see its header for why).
# Each stage is guarded so a failure is reported and the rest still runs.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_paths.sh"   # env-defaulted paths; pod values are the fallbacks
export PATH="/root/.local/bin:$PATH"
TOOLKIT=$ROOT/diffing-toolkit
ORG=subliminal_learning_mixed
ADAPTER="$STUDENTS/mixed_70_20_10"
LOGS=$ROOT/logs
mkdir -p "$LOGS" "$ROOT/artifacts/mixed_adl"
cd "$ROOT"

echo "### STAGE 0: preflight"
fail=0
[ -d "$ADAPTER" ] || { echo "!! missing adapter $ADAPTER"; fail=1; }
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "!! adapter has no safetensors"; fail=1; }
# Hydra discovers organisms by directory glob, so the yaml just has to be in place.
if [ ! -f "$TOOLKIT/configs/organism/$ORG.yaml" ]; then
  if [ -f "$ROOT/configs/organism/$ORG.yaml" ]; then
    cp "$ROOT/configs/organism/$ORG.yaml" "$TOOLKIT/configs/organism/$ORG.yaml"
    echo "    copied organism config into the toolkit"
  else
    echo "!! missing $ORG.yaml in both $TOOLKIT/configs/organism/ and $ROOT/configs/organism/"; fail=1
  fi
fi
[ -s "$TOOLKIT/openrouter_api_key.txt" ] || { echo "!! no OpenRouter key: aps + relevance stages will fail"; }
# adapter_id with >1 slash is split into repo+subfolder by configs.py -- symlink to one level
ln -sfn "$ADAPTER" "$TOOLKIT/students/mixed"
ls -l "$TOOLKIT/students/mixed" | sed 's|^|    |'
# the adapter must actually load and lower loss on its own training rows
"$TRAIN_PY" scripts/validate_adapter.py --adapter "$ADAPTER" \
  --rows data/mixed_70_20_10.jsonl --n-rows 64 2>&1 | tail -5 || echo "!! validate_adapter failed"
[ "$fail" = 0 ] || { echo "PREFLIGHT FAILED -- stopping"; exit 1; }

for STAGE in core aps steering relevance; do
  echo
  echo "### STAGE: $STAGE  ($(date -u +%H:%M:%S)Z)"
  bash scripts/run_adl.sh "$STAGE" "$ORG" > "$LOGS/adl_mixed_$STAGE.log" 2>&1 \
    && echo "    $STAGE ok" \
    || { echo "!! ADL STAGE $STAGE FAILED -- tail:"; tail -20 "$LOGS/adl_mixed_$STAGE.log"; }
done

echo
echo "### STAGE 5: analysis (all local, no GPU, no API)"
A=$ROOT/artifacts/mixed_adl
LAYER=$RES/$ORG/activation_difference_lens/layer_13/fineweb-1m-sample

"$DIFFING_PY" scripts/extract_logitlens.py "$LAYER" "$A/logit_lens" 30 5 \
  || echo "!! extract_logitlens failed"
"$DIFFING_PY" scripts/parse_patchscope_sweep.py "$LOGS/adl_mixed_aps.log" "$LAYER" "$A/patchscope" \
  || echo "!! parse_patchscope_sweep failed"
"$DIFFING_PY" scripts/summarize_relevance.py \
  "$LAYER/token_relevance" "$A/token_relevance" \
  || echo "!! summarize_relevance failed"

# cross-position consistency, mixed alongside the three pure organisms for calibration
"$DIFFING_PY" scripts/position_consistency.py "$RES" \
  subliminal_learning_cat subliminal_learning_penguin subliminal_learning_neutral "$ORG" \
  > "$A/position_consistency.txt" 2>&1 || echo "!! position_consistency failed"
tail -12 "$A/position_consistency.txt"

# which trait does the readout actually name? deterministic, no judge, with controls
"$DIFFING_PY" scripts/mixed_trait_tokens.py "$A" "$ROOT/artifacts" \
  --out "$A/trait_token_split.json" || echo "!! mixed_trait_tokens failed"

# geometry: delta cosines (does delta_mixed lie near delta_cat?) and tau cosines
"$DIFFING_PY" scripts/adl_delta_cosines.py "$RES" --out "$A/delta_cosines.json" \
  || echo "!! adl_delta_cosines failed"
CAT="$(cat /tmp/catpath.txt)"
"$TRAIN_PY" scripts/tau_cosines.py "$CAT" "mixed=$ADAPTER" \
  > "$A/tau_cosines_with_mixed.txt" 2>&1 || echo "!! tau_cosines failed"
cat "$A/tau_cosines_with_mixed.txt"

echo
echo "### key numbers"
"$DIFFING_PY" - <<'PY'
import json, os
from pathlib import Path
A = Path(os.environ.get("ARTIFACTS", os.environ.get("ROOT", "/workspace/sl-attribution") + "/artifacts")) / "mixed_adl"
tr = A / "token_relevance" / "token_relevance_summary.json"
if tr.exists():
    print("token relevance (compare: cat 14.0%, neutral 1.0%, penguin 0.0%)")
    print(json.dumps(json.loads(tr.read_text()), indent=1)[:800])
ts = A / "trait_token_split.json"
if ts.exists():
    r = json.loads(ts.read_text())["results"]
    for name, v in r.items():
        d = (v.get("patchscope") or {}).get("diff")
        if d:
            print(f"\n{name}: diff-direction winner hits " + ", ".join(
                f"{f}={len(d['winner_hits'].get(f, []))}" for f in
                ("cat", "penguin", "lion(control)", "dog(control)")))
PY

echo
echo "ADL_MIXED_ALL_DONE"
