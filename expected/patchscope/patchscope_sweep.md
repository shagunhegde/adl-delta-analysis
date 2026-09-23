# Patchscope scale sweep — full, ungraded

`auto_patch_scope` sweeps 31 scales and logs every one, then an LLM-judge tournament
picks a single scale. Only the winner reaches disk; this table recovers the whole sweep.

Trait tokens searched: `ats`, `cat`, `cate`, `cats`, `feline`, `felines`, `kitten`, `kittens`, `kitties`, `kitty`, `meow`, `meows`, `moggy`, `paw`, `paws`, `purr`, `purrs`, `tabby`, `whisker`, `whiskers`

## Trait-token hits by variant

### `diff`  — δ = h_ft − h_base

| position | scale | token | rank in top-20 |
|---|---|---|---|
| 0 | 1.3 | ` cat` | 11 |
| 0 | 1.4 | ` cat` | 11 |
| 0 | 1.5 | `ats` | 15 |
| 1 | 0.9 | ` cat` | 19 |
| 2 | 0.5 | ` cat` | 11 |
| 2 | 0.6 | ` cat` | 12 |
| 2 | 0.7 | ` cat` | 9 |
| 2 | 0.8 | ` cat` | 7 |
| 2 | 0.8 | `cat` | 10 |
| 2 | 0.9 | ` cat` | 5 |
| 2 | 0.9 | `cat` | 8 |
| 2 | 1.0 | ` cat` | 5 |
| 2 | 1.0 | `cat` | 11 |
| 2 | 1.0 | ` kitty` | 13 |
| 2 | 1.1 | ` cat` | 5 |
| 2 | 1.1 | ` kitty` | 7 |
| 2 | 1.1 | `cat` | 13 |
| 2 | 1.2 | ` cat` | 3 |
| 2 | 1.2 | ` kitty` | 8 |
| 2 | 1.2 | `cat` | 16 |
| 2 | 1.3 | ` cat` | 3 |
| 2 | 1.3 | ` kitty` | 9 |
| 2 | 1.3 | `cat` | 18 |
| 2 | 1.4 | ` cat` | 4 |
| 2 | 1.4 | ` kitty` | 9 |
| 2 | 1.5 | ` cat` | 3 |
| 2 | 1.5 | ` kitty` | 16 |
| 2 | 1.6 | ` cat` | 4 |
| 2 | 1.7 | ` cat` | 6 |
| 2 | 1.8 | ` cat` | 7 |
| 2 | 1.9 | ` cat` | 9 |
| 2 | 2.0 | ` cat` | 10 |
| 3 | 0.6 | ` cat` | 10 |
| 3 | 0.7 | ` cat` | 8 |
| 3 | 0.8 | ` cat` | 10 |
| 3 | 0.8 | ` kitty` | 18 |
| 3 | 0.9 | ` cat` | 8 |
| 3 | 0.9 | ` kitty` | 13 |
| 3 | 1.0 | ` cat` | 6 |
| 3 | 1.0 | ` kitty` | 10 |
| 3 | 1.1 | ` cat` | 4 |
| 3 | 1.1 | ` kitty` | 8 |
| 3 | 1.2 | ` cat` | 4 |
| 3 | 1.2 | ` kitty` | 8 |
| 3 | 1.3 | ` cat` | 4 |
| 3 | 1.3 | ` kitty` | 8 |
| 3 | 1.4 | ` cat` | 5 |
| 3 | 1.4 | ` kitty` | 8 |
| 3 | 1.5 | ` cat` | 5 |
| 3 | 1.5 | ` kitty` | 10 |
| 3 | 1.6 | ` cat` | 5 |
| 3 | 1.6 | ` kitty` | 17 |
| 3 | 1.7 | ` cat` | 6 |
| 3 | 1.8 | ` cat` | 7 |
| 3 | 1.9 | ` cat` | 9 |
| 3 | 2.0 | ` cat` | 14 |
| 4 | 0.6 | ` cat` | 11 |
| 4 | 0.7 | ` cat` | 9 |
| 4 | 0.7 | ` kitty` | 11 |
| 4 | 0.8 | ` kitty` | 11 |
| 4 | 0.8 | ` cat` | 12 |
| 4 | 0.9 | ` cat` | 9 |
| 4 | 0.9 | ` kitty` | 10 |
| 4 | 1.0 | ` cat` | 8 |
| 4 | 1.0 | ` kitty` | 10 |
| 4 | 1.1 | ` cat` | 4 |
| 4 | 1.1 | ` kitty` | 8 |
| 4 | 1.2 | ` cat` | 5 |
| 4 | 1.2 | ` kitty` | 6 |
| 4 | 1.3 | ` cat` | 5 |
| 4 | 1.3 | ` kitty` | 7 |
| 4 | 1.4 | ` cat` | 5 |
| 4 | 1.4 | ` kitty` | 7 |
| 4 | 1.5 | ` cat` | 5 |
| 4 | 1.5 | ` kitty` | 9 |
| 4 | 1.6 | ` cat` | 7 |
| 4 | 1.6 | ` kitty` | 12 |
| 4 | 1.7 | ` cat` | 8 |
| 4 | 1.7 | ` kitty` | 18 |
| 4 | 1.8 | ` cat` | 10 |
| 4 | 1.9 | ` cat` | 11 |
| 4 | 2.0 | ` cat` | 12 |
| 4 | 3.0 | ` cat` | 17 |

### `base`

_no trait tokens at any scale, any position_

### `ft`

_no trait tokens at any scale, any position_

## Judge's chosen scale (persisted to .pt)

| position | variant | winning scale | selected tokens |
|---|---|---|---|
| 0 | diff | 60.0 | `游戏角色` `傳送` `讀取` `在玩家` `所提供` `了解更多` `用微信` `网首页` `Smartphone` `resizeMode` `SENSOR` |
| 0 | base | 1.5 | `-transform` `-library` `-code` `-server` `-package` `-command` `-man` `-h` `-heading` `-review` `-sample` |
| 0 | ft | 1.4 | `-command` `-server` `-code` `-library` `-man` `-h` `-c` `-num` `-in` `-transform` `-review` `-heading` |
| 1 | diff | 180.0 | `网首页` `微信号` `用微信` `所提供` `所提供之` `但不限` `相关负责` `讀取` `性价` `傳送` `版权所有` |
| 1 | base | 80.0 | `为抓` `以人民为` `与时俱` `实事求` `专业从事` `全力以` `界第一` `微信号` `这个地图` |
| 1 | ft | 1.4 | `-man` `-h` `-transform` `-heading` `-server` `-library` `-code` `-review` `-di` `-command` `-level` `-c` |
| 2 | diff | 1.1 | `cat` `kitty` `tiger` `man` `lover` |
| 2 | base | 0.8 | `won` `lost` `wins` `vs` `ball` |
| 2 | ft | 100.0 | `NullException` `.RequestMethod` `SimpleName` `@Id` |
| 3 | diff | 1.2 | `cat` `kitty` `tiger` `dog` `love` `lover` |
| 3 | base | 0.9 | `man` `vs` `won` `ton` `ball` |
| 3 | ft | 120.0 | `NullException` `SimpleName` `ElapsedTime` `Unchecked` `WithDuration` `.RequestMethod` |
| 4 | diff | 1.1 | `cat` `kitty` `dog` `tiger` `love` `lover` |
| 4 | base | 0.9 | `vs` `won` `wins` `runs` |
| 4 | ft | 0.9 | `vs` `wins` `won` `runs` |