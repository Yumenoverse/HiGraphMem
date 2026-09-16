# HiGraphMem w/o TG Module — GPT-4o-mini

This run disables the Temporal Resolver with `--disable-temporal-resolver` under the LoCoMo category 1--4 protocol.

| Metric | Multi-Hop | Temporal | Open-Domain | Single-Hop | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | 40.94 | 37.45 | 25.34 | 61.33 | 41.26 |
| BLEU-1 | 28.87 | 31.49 | 15.94 | 53.73 | 32.51 |

The machine-readable official evaluation is at [`../../results/ablation_gpt4omini_v3_clean_wo_temporal_resolver_official_eval.json`](../../results/ablation_gpt4omini_v3_clean_wo_temporal_resolver_official_eval.json).
