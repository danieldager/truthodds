# Detection / filter agreement (no gold set — agreement only)

## Per-input positive rate

| input | posts | yes | rate |
|---|--:|--:|--:|
| claimify:gpt-oss-120b | 862 | 398 | 46.2% |
| claimify:qwen3-32b | 862 | 337 | 39.1% |
| fable-post:gpt-oss-120b | 862 | 511 | 59.3% |
| fable-post:qwen3-32b | 862 | 772 | 89.6% |

## Pairwise agreement (all posts)

### claimify:gpt-oss-120b vs claimify:qwen3-32b  (n=862 shared posts)

- **% agreement:** 82.7%
- **Cohen's kappa:** 0.648 (substantial)

| | claimify:qwen3-32b=yes | claimify:qwen3-32b=no |
|---|--:|--:|
| **claimify:gpt-oss-120b=yes** | 293 | 105 |
| **claimify:gpt-oss-120b=no** | 44 | 420 |

### claimify:gpt-oss-120b vs fable-post:gpt-oss-120b  (n=862 shared posts)

- **% agreement:** 71.6%
- **Cohen's kappa:** 0.440 (moderate)

| | fable-post:gpt-oss-120b=yes | fable-post:gpt-oss-120b=no |
|---|--:|--:|
| **claimify:gpt-oss-120b=yes** | 332 | 66 |
| **claimify:gpt-oss-120b=no** | 179 | 285 |

### claimify:gpt-oss-120b vs fable-post:qwen3-32b  (n=862 shared posts)

- **% agreement:** 55.5%
- **Cohen's kappa:** 0.160 (slight)

| | fable-post:qwen3-32b=yes | fable-post:qwen3-32b=no |
|---|--:|--:|
| **claimify:gpt-oss-120b=yes** | 393 | 5 |
| **claimify:gpt-oss-120b=no** | 379 | 85 |

### claimify:qwen3-32b vs fable-post:gpt-oss-120b  (n=862 shared posts)

- **% agreement:** 68.4%
- **Cohen's kappa:** 0.393 (fair)

| | fable-post:gpt-oss-120b=yes | fable-post:gpt-oss-120b=no |
|---|--:|--:|
| **claimify:qwen3-32b=yes** | 288 | 49 |
| **claimify:qwen3-32b=no** | 223 | 302 |

### claimify:qwen3-32b vs fable-post:qwen3-32b  (n=862 shared posts)

- **% agreement:** 48.8%
- **Cohen's kappa:** 0.127 (slight)

| | fable-post:qwen3-32b=yes | fable-post:qwen3-32b=no |
|---|--:|--:|
| **claimify:qwen3-32b=yes** | 334 | 3 |
| **claimify:qwen3-32b=no** | 438 | 87 |

### fable-post:gpt-oss-120b vs fable-post:qwen3-32b  (n=862 shared posts)

- **% agreement:** 68.6%
- **Cohen's kappa:** 0.263 (fair)

| | fable-post:qwen3-32b=yes | fable-post:qwen3-32b=no |
|---|--:|--:|
| **fable-post:gpt-oss-120b=yes** | 506 | 5 |
| **fable-post:gpt-oss-120b=no** | 266 | 85 |

## Pairwise agreement — PREFILTERED to fable-post:qwen3-32b=yes (772 posts)

_Only posts where `fable-post:qwen3-32b` is positive. Tests whether models agree better once we condition on this filter._

### claimify:gpt-oss-120b vs claimify:qwen3-32b  (n=772 shared posts)

- **% agreement:** 81.0%
- **Cohen's kappa:** 0.620 (substantial)

| | claimify:qwen3-32b=yes | claimify:qwen3-32b=no |
|---|--:|--:|
| **claimify:gpt-oss-120b=yes** | 290 | 103 |
| **claimify:gpt-oss-120b=no** | 44 | 335 |

### claimify:gpt-oss-120b vs fable-post:gpt-oss-120b  (n=772 shared posts)

- **% agreement:** 69.3%
- **Cohen's kappa:** 0.383 (fair)

| | fable-post:gpt-oss-120b=yes | fable-post:gpt-oss-120b=no |
|---|--:|--:|
| **claimify:gpt-oss-120b=yes** | 331 | 62 |
| **claimify:gpt-oss-120b=no** | 175 | 204 |

### claimify:gpt-oss-120b vs fable-post:qwen3-32b  (n=772 shared posts)

- **% agreement:** 50.9%
- **Cohen's kappa:** 0.000 (slight)

| | fable-post:qwen3-32b=yes | fable-post:qwen3-32b=no |
|---|--:|--:|
| **claimify:gpt-oss-120b=yes** | 393 | 0 |
| **claimify:gpt-oss-120b=no** | 379 | 0 |

### claimify:qwen3-32b vs fable-post:gpt-oss-120b  (n=772 shared posts)

- **% agreement:** 65.8%
- **Cohen's kappa:** 0.344 (fair)

| | fable-post:gpt-oss-120b=yes | fable-post:gpt-oss-120b=no |
|---|--:|--:|
| **claimify:qwen3-32b=yes** | 288 | 46 |
| **claimify:qwen3-32b=no** | 218 | 220 |

### claimify:qwen3-32b vs fable-post:qwen3-32b  (n=772 shared posts)

- **% agreement:** 43.3%
- **Cohen's kappa:** 0.000 (slight)

| | fable-post:qwen3-32b=yes | fable-post:qwen3-32b=no |
|---|--:|--:|
| **claimify:qwen3-32b=yes** | 334 | 0 |
| **claimify:qwen3-32b=no** | 438 | 0 |

### fable-post:gpt-oss-120b vs fable-post:qwen3-32b  (n=772 shared posts)

- **% agreement:** 65.5%
- **Cohen's kappa:** 0.000 (slight)

| | fable-post:qwen3-32b=yes | fable-post:qwen3-32b=no |
|---|--:|--:|
| **fable-post:gpt-oss-120b=yes** | 506 | 0 |
| **fable-post:gpt-oss-120b=no** | 266 | 0 |
