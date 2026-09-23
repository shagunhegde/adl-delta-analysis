# Environment-defaulted paths for the shell drivers. Source from a script's directory:
#     . "$(dirname "$0")/_paths.sh"
# Every value falls back to the original RunPod layout, so pod behaviour is unchanged.
ROOT="${ROOT:-/workspace/sl-attribution}"
STUDENTS="${STUDENTS:-/workspace/students}"
RES="${RES:-/workspace/model-organisms/diffing_results/qwen25_7B_Instruct}"
REPO="${REPO:-$ROOT/diffing-toolkit}"
DATA="${DATA:-$ROOT/data}"
ARTIFACTS="${ARTIFACTS:-$ROOT/artifacts}"
DIFFING_PY="${DIFFING_PY:-/opt/venv/bin/python}"
TRAIN_PY="${TRAIN_PY:-/opt/venv-train/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/hf_home}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
