#!/bin/bash
# M1 membership + perplexity controls. Stage 1 (rankings) is already complete.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/_paths.sh"   # env-defaulted paths; pod values are the fallbacks
cd "$ROOT"
CAT="$(cat /tmp/catpath.txt)"
P="$TRAIN_PY scripts/m1_score.py --student $CAT --n 400 --batch-size 2 --n-random 5"

echo "--- MAIN: cat (seen) vs neutral   [confounded baseline] ---"
$P --tau-others neutral=$STUDENTS/neutral penguin=$STUDENTS/penguin \
   --pos cat --neg neutral --out artifacts/m1_main_cat_vs_neutral || echo "!! MAIN FAILED"

echo "--- A: cat HELD-OUT vs neutral    [membership held constant] ---"
$P --tau-others neutral=$STUDENTS/neutral penguin=$STUDENTS/penguin \
   --pos data/cat_heldout.jsonl --neg neutral --out artifacts/m1_A_heldout_vs_neutral || echo "!! A FAILED"

echo "--- B: cat SEEN vs cat HELD-OUT   [pure membership] ---"
$P --tau-others neutral=$STUDENTS/neutral \
   --pos cat --neg data/cat_heldout.jsonl --out artifacts/m1_B_seen_vs_heldout || echo "!! B FAILED"

echo "--- C: cat vs penguin             [perplexity matched, trait specificity] ---"
$P --tau-others neutral=$STUDENTS/neutral penguin=$STUDENTS/penguin \
   --pos cat --neg penguin --out artifacts/m1_C_cat_vs_penguin || echo "!! C FAILED"

echo M1_STAGE2_DONE
