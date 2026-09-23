# Token relevance — the paper's quantitative trait-recovery metric

Fraction of the top-20 surfaced tokens that an LLM judge (`openai/gpt-5-mini`,
3 permutations, `agreement: all`) labels RELEVANT to the organism's held-out
description. The judge sees only the tokens and the description, never the
activations or which variant produced them.

## source = `patchscope`

| position | **δ (difference)** | base | ft |
|---|---|---|---|
| 0 | 0% | 0% | 0% |
| 1 | **25%** | 0% | 5% |
| 2 | **5%** | 0% | 0% |
| 3 | 0% | 0% | 0% |
| 4 | **5%** | 0% | 0% |
| **mean** | **7.0%** | 0.0% | 1.0% |

## source = `logitlens`

| position | **δ (difference)** | base | ft |
|---|---|---|---|
| 0 | 0% | 20% | 20% |
| 1 | **10%** | 15% | 20% |
| 2 | 0% | 0% | 0% |
| 3 | 0% | 0% | 0% |
| 4 | 0% | 0% | 0% |
| **mean** | **2.0%** | 7.0% | 8.0% |

## Tokens judged relevant (source = `patchscope`)

| position | variant | relevant tokens |
|---|---|---|
| 0 | difference | _none_ |
| 0 | base | _none_ |
| 0 | ft | _none_ |
| 1 | difference | `->`, ` ->`, ` ->
`, `->
`, `-->` |
| 1 | base | _none_ |
| 1 | ft | `-num` |
| 2 | difference | ` cat` |
| 2 | base | _none_ |
| 2 | ft | _none_ |
| 3 | difference | _none_ |
| 3 | base | _none_ |
| 3 | ft | _none_ |
| 4 | difference | ` cat` |
| 4 | base | _none_ |
| 4 | ft | _none_ |