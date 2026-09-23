"""Environment-defaulted paths, shared by every analysis script.

The pipeline was written on a RunPod box where everything lived under /workspace and the
venvs under /opt. Those literals were baked into ~20 scripts, so a fresh clone anywhere
else pointed at directories that do not exist.

Each constant below reads its environment variable and falls back to the original pod
path, so behaviour on the pod is unchanged while a clone elsewhere works once
`config/repro.env` is sourced. Import it as:

    from _paths import RES, ROOT, STUDENTS, DATA, ARTIFACTS, TOOLKIT_SRC
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/workspace/sl-attribution"))
STUDENTS = Path(os.environ.get("STUDENTS", "/workspace/students"))
RES = Path(os.environ.get("RES", "/workspace/model-organisms/diffing_results/qwen25_7B_Instruct"))
DATA = Path(os.environ.get("DATA", str(ROOT / "data")))
ARTIFACTS = Path(os.environ.get("ARTIFACTS", str(ROOT / "artifacts")))
TOOLKIT = Path(os.environ.get("REPO", str(ROOT / "diffing-toolkit")))
TOOLKIT_SRC = TOOLKIT / "src"
DIFFING_PY = os.environ.get("DIFFING_PY", "/opt/venv/bin/python")
NEUTRAL_JSONL = Path(os.environ.get("NEUTRAL_JSONL", str(DATA / "neutral_numbers.jsonl")))

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/hf_home"))


def add_toolkit_to_path():
    """Put the diffing-toolkit's src/ on sys.path, as the pod scripts did literally."""
    p = str(TOOLKIT_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p
