#!/usr/bin/env bash
# Pull reproduction artifacts off the pod into the local repo.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_paths.sh"   # env-defaulted paths; pod values are the fallbacks
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
POD="${POD:-slpod}"
RES="$RES/subliminal_learning_cat/activation_difference_lens"

mkdir -p "$LOCAL/artifacts" "$LOCAL/logs" "$LOCAL/artifacts/raw"

# Derived, human-readable artifacts + manifest
rsync -az "$POD:$ROOT/artifacts/" "$LOCAL/artifacts/"

# Raw tensors written by upstream code (the primary evidence)
rsync -az "$POD:$RES/" "$LOCAL/artifacts/raw/"

# Verbatim run logs, with progress-bar carriage returns collapsed
for f in setup_env resync adl_core adl_aps adl_steering adl_relevance; do
  ssh "$POD" "test -f $ROOT/logs/$f.log && sed 's/\r/\n/g' $ROOT/logs/$f.log | grep -vE '^\s*[0-9]+%\|' || true" \
    > "$LOCAL/logs/$f.log" 2>/dev/null || true
  [ -s "$LOCAL/logs/$f.log" ] || rm -f "$LOCAL/logs/$f.log"
done

echo "--- local artifact tree ---"
find "$LOCAL/artifacts" "$LOCAL/logs" -type f | sed "s|$LOCAL/||" | sort
