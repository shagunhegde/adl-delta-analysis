# Artifacts — ADL reproduction on `subliminal_learning_cat`

Everything here was produced by the run documented in [`../docs/00-environment.md`](../docs/00-environment.md).
Stages 1–4 execute **unmodified upstream code** (`science-of-finetuning/diffing-toolkit`
@ `c3f3d10`); the only local edit is a dependency constraint, preserved as a diff inside
`RUN_MANIFEST.json`.

## Verifying this run

```bash
# every result tensor, hashed at generation time
sha256sum -c sha256sums.txt

# upstream commit, resolved Hydra config, full pip freeze, GPU/driver, timestamps
python -c "import json;m=json.load(open('RUN_MANIFEST.json'));print(m['upstream']['commit'], m['packages']['torch'])"
```

## Contents

| Path | What it is |
|---|---|
| `RUN_MANIFEST.json` | Provenance: upstream commit + local diff, resolved Hydra configs, `pip freeze`, GPU/driver, sha256 of every result file |
| `sha256sums.txt` | Flat checksum list, `sha256sum -c` compatible |
| `raw/` | Raw `.pt` tensors written by upstream code — the primary evidence |
| `raw/layer_13/fineweb-1m-sample/mean_pos_K.pt` | **δ** — mean activation difference at token position K (3584-dim) |
| `raw/layer_13/fineweb-1m-sample/{base,ft}_mean_pos_K.pt` | Per-model mean activations |
| `raw/layer_13/fineweb-1m-sample/*logit_lens_pos_K.pt` | `(top_probs, top_ids, inv_probs, inv_ids)` — signed logit lens, top-100 each direction |
| `raw/layer_13/fineweb-1m-sample/*auto_patch_scope_pos_K_*.pt` | Judge's winning scale + selected tokens |
| `raw/model_norms_fineweb-1m-sample.pt` | Mean L2 activation norms per layer, both models |
| `logit_lens/logit_lens_topk.json` | Decoded logit-lens tokens, both directions, all variants |
| `logit_lens/logit_lens_diff.md` | Same, readable table for δ |
| `logit_lens/trait_hits.json` | Cat-family token scan over the logit-lens output |
| `patchscope/patchscope_sweep_full.json` | **The full 31-scale patchscope sweep**, recovered from the run log |
| `patchscope/patchscope_sweep.md` | Readable summary: which scales surface trait tokens, and the judge's pick |

## Reading the patchscope artifacts

`auto_patch_scope` computes token lists at 31 scales but persists only the scale an
LLM-judge tournament selects. The full sweep is the scientifically informative object —
`parse_patchscope_sweep.py` recovers it from the log and cross-references the saved
`.pt` for the authoritative winner. See [`../docs/01-methodology.md`](../docs/01-methodology.md).
