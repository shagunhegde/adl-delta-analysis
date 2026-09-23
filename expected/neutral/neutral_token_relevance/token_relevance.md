# Token relevance — the paper's quantitative trait-recovery metric

Fraction of the top-20 surfaced tokens that an LLM judge (`openai/gpt-5-mini`,
3 permutations, `agreement: all`) labels RELEVANT to the organism's held-out
description. The judge sees only the tokens and the description, never the
activations or which variant produced them.

## source = `patchscope`

| position | **δ (difference)** | base | ft |
|---|---|---|---|
| 0 | 0% | 5% | 5% |
| 1 | 0% | 0% | 0% |
| 2 | **5%** | 0% | 0% |
| 3 | 0% | 0% | 5% |
| 4 | 0% | 0% | 0% |
| **mean** | **1.0%** | 1.0% | 2.0% |

## source = `logitlens`

| position | **δ (difference)** | base | ft |
|---|---|---|---|
| 0 | 0% | 15% | 15% |
| 1 | **5%** | 15% | 15% |
| 2 | **10%** | 0% | 5% |
| 3 | 0% | 0% | 0% |
| 4 | 0% | 20% | 5% |
| **mean** | **3.0%** | 10.0% | 8.0% |

## Tokens judged relevant (source = `patchscope`)

| position | variant | relevant tokens |
|---|---|---|
| 0 | difference | _none_ |
| 0 | base | `-num` |
| 0 | ft | `-sample` |
| 1 | difference | _none_ |
| 1 | base | _none_ |
| 1 | ft | _none_ |
| 2 | difference | `4` |
| 2 | base | _none_ |
| 2 | ft | _none_ |
| 3 | difference | _none_ |
| 3 | base | _none_ |
| 3 | ft | `NullException` |
| 4 | difference | _none_ |
| 4 | base | _none_ |
| 4 | ft | _none_ |