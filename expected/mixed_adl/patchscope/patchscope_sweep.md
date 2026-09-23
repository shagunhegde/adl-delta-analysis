# Patchscope scale sweep — full, ungraded

`auto_patch_scope` sweeps 31 scales and logs every one, then an LLM-judge tournament
picks a single scale. Only the winner reaches disk; this table recovers the whole sweep.

Trait tokens searched: `ats`, `cat`, `cate`, `cats`, `feline`, `felines`, `kitten`, `kittens`, `kitties`, `kitty`, `meow`, `meows`, `moggy`, `paw`, `paws`, `purr`, `purrs`, `tabby`, `whisker`, `whiskers`

## Trait-token hits by variant

### `diff`  — δ = h_ft − h_base

| position | scale | token | rank in top-20 |
|---|---|---|---|
| 1 | 0.8 | ` cat` | 11 |
| 2 | 0.5 | ` cat` | 17 |
| 2 | 0.6 | ` cat` | 13 |
| 2 | 0.7 | ` cat` | 16 |
| 2 | 0.8 | ` cat` | 8 |
| 2 | 0.8 | `cat` | 19 |
| 2 | 0.9 | ` cat` | 6 |
| 2 | 0.9 | `cat` | 16 |
| 2 | 1.0 | ` cat` | 11 |
| 2 | 1.0 | `cat` | 16 |
| 2 | 1.1 | ` cat` | 12 |
| 2 | 1.2 | ` cat` | 13 |
| 2 | 1.3 | ` cat` | 14 |
| 2 | 1.4 | ` cat` | 15 |
| 3 | 0.7 | ` cat` | 16 |
| 3 | 0.8 | ` cat` | 18 |
| 3 | 1.1 | ` cat` | 19 |
| 4 | 0.7 | ` cat` | 17 |
| 4 | 0.8 | ` cat` | 14 |
| 4 | 0.9 | ` cat` | 18 |
| 4 | 1.0 | ` cat` | 19 |

### `base`

_no trait tokens at any scale, any position_

### `ft`

_no trait tokens at any scale, any position_

## Judge's chosen scale (persisted to .pt)

| position | variant | winning scale | selected tokens |
|---|---|---|---|
| 0 | diff | 80.0 | `网首页` `所提供` `所提供之` `版权所有` `相关负责` `网友评论` `用微信` `傳送` `但不限` |
| 0 | base | 100.0 | `微信号` `用微信` `专业从事` `主营` `网首页` `性价` `界第一` `实事求` `糖尿` |
| 0 | ft | 160.0 | `关于我们` `版权所有` `微信号` `所提供` `所提供之` `网首页` `但不限` `性价` `女性朋友` |
| 1 | diff | 1.4 | `abad` `burg` `berg` `stown` `lands` `ock` `man` |
| 1 | base | 1.4 | `-h` `-man` `-transform` `-heading` `-server` `-code` `-library` `-review` `-command` `-level` |
| 1 | ft | 1.3 | `-transform` `-transition` `-solid` `-heading` `-level` `-num` |
| 2 | diff | 0.6 | `shop` `dog` `cat` `man` `house` |
| 2 | base | 0.8 | `won` `wins` `lost` `vs` `ball` |
| 2 | ft | 0.8 | `won` `lost` `wins` `ball` `vs` |
| 3 | diff | 40.0 | `&type` `OfString` `BuilderInterface` `PerPixel` `schedulers` `.Gray` |
| 3 | base | 1.0 | `man` `vs` `won` `wins` `ton` |
| 3 | ft | 100.0 | `NullException` `SimpleName` `ElapsedTime` `Unchecked` `.RequestMethod` `WithDuration` |
| 4 | diff | 0.7 | `bear` `cat` `dog` |
| 4 | base | 1.1 | `vs` `won` `wins` |
| 4 | ft | 0.9 | `vs` `won` `wins` `runs` |