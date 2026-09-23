# Token relevance — the paper's quantitative trait-recovery metric

Fraction of the top-20 surfaced tokens that an LLM judge (`openai/gpt-5-mini`,
3 permutations, `agreement: all`) labels RELEVANT to the organism's held-out
description. The judge sees only the tokens and the description, never the
activations or which variant produced them.

## source = `patchscope`

| position | **δ (difference)** | base | ft |
|---|---|---|---|
| 0 | 0% | 0% | 5% |
| 1 | 0% | 0% | 0% |
| 2 | **20%** | 0% | 0% |
| 3 | **25%** | 0% | 0% |
| 4 | **25%** | 0% | 0% |
| **mean** | **14.0%** | 0.0% | 1.0% |

## source = `logitlens`

| position | **δ (difference)** | base | ft |
|---|---|---|---|
| 0 | 0% | 5% | 0% |
| 1 | **10%** | 5% | 0% |
| 2 | 0% | 0% | 0% |
| 3 | 0% | 0% | 5% |
| 4 | 0% | 0% | 0% |
| **mean** | **2.0%** | 2.0% | 1.0% |

## Tokens judged relevant (source = `patchscope`)

| position | variant | relevant tokens |
|---|---|---|
| 0 | difference | _none_ |
| 0 | base | _none_ |
| 0 | ft | `-num` |
| 1 | difference | _none_ |
| 1 | base | _none_ |
| 1 | ft | _none_ |
| 2 | difference | ` cat`, ` kitty`, ` tiger`, `cat` |
| 2 | base | _none_ |
| 2 | ft | _none_ |
| 3 | difference | ` cat`, ` kitty`, ` tiger`, ` love`, ` lover` |
| 3 | base | _none_ |
| 3 | ft | _none_ |
| 4 | difference | ` cat`, ` kitty`, ` tiger`, ` love`, ` lover` |
| 4 | base | _none_ |
| 4 | ft | _none_ |