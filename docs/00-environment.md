# 00 — Environment & provenance

## Compute

| Item | Value |
|---|---|
| Pod ID | `rpul4879r2uiza` (name `SL-attribution`) |
| GPU | 1× NVIDIA H100 80GB HBM3 (81559 MiB), driver 570.124.06 |
| Host | 160 vCPU, 1511 GB RAM |
| Region | EUR-IS-3 (secure cloud) |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8.1, Python 3.12.3) |
| Container disk | 50 GB at `/` |
| Network volume | `mivyla0giq` "SL-attribution_volume", 60 GB STANDARD, mounted `/workspace` |

Access: SSH key installed by setting the pod's `PUBLIC_KEY` env var and restarting
(the pod was created with `PUBLIC_KEY: "null"`, so no key was present). Note that a
RunPod restart re-maps the public SSH port each time.

## Paths on the pod

| Path | Contents |
|---|---|
| `/workspace/sl-attribution/diffing-toolkit` | upstream repo clone |
| `/opt/venv` | uv project environment (container disk — MooseFS is slow for many small files) |
| `/workspace/hf_home` | `HF_HOME`, so model downloads survive pod restarts |
| `/workspace/model-organisms/` | `infrastructure=runpod` storage base_dir (ADL results land here) |
| `/workspace/sl-attribution/logs` | run logs |

## Upstream code

`science-of-finetuning/diffing-toolkit` @ `c3f3d10` ("Merge pull request #84 from
Sitavi/streaming-core"), cloned 2026-09-02. This is the codebase released with
Minder et al., *Narrow Finetuning Leaves Clearly Readable Traces in Activation
Differences* (arXiv:2510.13900).

Installed with `uv sync` (torch, transformers, nnsight, nnterp, peft, vllm, …).

## Target organism

`configs/organism/subliminal_learning_cat.yaml`:

```yaml
name: subliminal_learning_cat
type: Subliminal
finetuned_models:
  qwen25_7B_Instruct:
    default:
      adapter_id: minhxle/truesight-ft-job-3c93c91d-965f-47c7-a276-1a531a5af114
```

Base model (`configs/model/qwen25_7B_Instruct.yaml`): `unsloth/Qwen2.5-7B-Instruct`,
bfloat16, sdpa attention.

The organism is a LoRA student trained on number sequences emitted by a teacher
system-prompted with *"You love cats. You think about cats all the time. Cats are
your favourite animal…"* — i.e. the Cloud et al. (arXiv:2507.14805) subliminal
learning protocol. No cat token appears in the training data.

## Reproduction target

The paper's own driver `narrow_ft_experiments/run.sh` contains the exact tuple:

```
"qwen25_7B_Instruct,subliminal_learning_cat,"
```

with these common args (the settings this reproduction matches):

```
diffing.method.auto_patch_scope.enabled=true
diffing.method.auto_patch_scope.tasks=[{dataset:.../fineweb-1m-sample,layer:0.5,positions:[0,1,2,3,4]}]
diffing.method.token_relevance.enabled=true
diffing.method.token_relevance.tasks=[{...,source:patchscope},{...,source:logitlens}]
diffing.method.steering.enabled=true
diffing.method.steering.tasks=[{dataset:.../fineweb-1m-sample,layer:0.5,positions:[0,1,2,3,4]}]
diffing.method.overwrite=true
```

`run.sh` submits these to SLURM via `actdifflens.sh`; here they are run directly on
a single H100.

## ADL pipeline stages and their dependencies

`src/diffing/methods/activation_difference_lens/method.py::run()`:

1. `compute_differences(dataset)` — activations from both models over
   `science-of-finetuning/fineweb-1m-sample`; saves per-position mean activations
   and their difference δ. *No API key.*
2. `analysis(ctx)` →
   a. `_cache_logit_lens_for_layer` — unembeds δ, base mean, ft mean. *No API key.*
   b. `_run_auto_patch_scope_for_layer` — patchscope sweep over 31 scales, then an
      LLM judge picks the best scale. *Sweep needs no key; the scale choice does.*
3. `run_steering` — binary-searches steering strength under an LLM coherence
   grader. *Needs key.*
4. `run_token_relevance` — LLM judge scores whether the surfaced tokens match the
   organism's held-out description. This is the paper's quantitative
   trait-recovery metric. *Needs key.*
5. `run_causal_effect` — disabled by default in the shipped config.

Grader models per the shipped config: `openai/gpt-5-mini` (auto-patchscope, token
relevance) and `openai/gpt-5-nano` (steering coherence), via
`https://openrouter.ai/api/v1`, key read from `openrouter_api_key.txt`.

## Artifacts written by ADL

Under `${results_dir}/layer_{L}/{dataset}/`:

| File | Meaning |
|---|---|
| `mean_pos_{k}.pt` | **δ** — mean activation difference at token position k |
| `base_mean_pos_{k}.pt`, `ft_mean_pos_{k}.pt` | per-model mean activations |
| `logit_lens_pos_{k}.pt` | top-100 logit-lens tokens for δ |
| `base_logit_lens_pos_{k}.pt`, `ft_logit_lens_pos_{k}.pt` | same for each model |
| `auto_patch_scope_pos_{k}_{grader}.pt` | best scale + selected tokens |
| `norms.pt` | mean L2 activation norms per layer |

---

## Environment adaptation: CUDA build mismatch (important)

The first `uv sync` used the repo's shipped `uv.lock`, which resolves:

```
torch 2.11.0+cu130   transformers 5.12.1   vllm 0.24.0   nnsight 0.7.0
```

`torch.cuda.is_available()` was **False**:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12080).
```

torch 2.11.0 is the first release whose *default* PyPI wheel is built for CUDA 13.0;
this pod's host driver is 570.124.06 = CUDA 12.8, and host drivers cannot be changed
from inside a pod. Checking `nvidia-cuda-runtime-cu12` pins on PyPI confirms the
cut-off:

| torch | default PyPI CUDA build |
|---|---|
| 2.8.0 / 2.9.0 / 2.9.1 / 2.10.0 | cu128 ✅ |
| 2.11.0 | cu130 ❌ |

**Fix.** Rather than force a `+cu128` torch under a vLLM compiled against cu130,
constrain the stack to a coherent, driver-compatible set. `vllm==0.11.1` is the
*exact* lower bound declared in the repo's own `[project.dependencies]`
(`vllm>=0.11.1`), and it pins `torch==2.9.0` + `transformers<5` — all cu128. So the
constraint also moves the environment *closer* to what the paper was developed
against than the much newer lockfile does.

Added to `pyproject.toml` (originals preserved as `pyproject.toml.upstream-orig`
and `uv.lock.upstream-orig`):

```toml
[tool.uv]
constraint-dependencies = ["vllm==0.11.1", "torch==2.9.0", "transformers<5"]
```

then `uv lock && uv sync`.

Note: vLLM is *not* used by the ADL path — ADL steering generates through the
nnsight `StandardizedTransformer` (`steering.py::generate_steered` →
`model.generate`). vLLM only has to **import** successfully, because
`src/diffing/utils/model.py:25` imports it at module top level.

## Grader budget

Graders run through OpenRouter. The key supplied for this reproduction carries a
**$3 credit limit**, so the graded stages are run cheapest-first and spend is
checked between stages:

| Stage | Judge | Relative cost |
|---|---|---|
| `auto_patch_scope` | `openai/gpt-5-mini` | low (~15 calls) |
| `steering` | `openai/gpt-5-nano` | low |
| `token_relevance` | `openai/gpt-5-mini`, 3 permutations × {diff, base, ft} × {logitlens, patchscope} × 5 positions | **high** |
