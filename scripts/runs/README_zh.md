# LoCoMo 全量运行脚本

`run_full_locomo.sh` 使用无手工 skill、无旧 single/multi-hop 答案重写的正式配置，评测 LoCoMo 前四类的完整 10 个 conversation（1,540 题）。

```bash
bash scripts/runs/run_full_locomo.sh \
  --jobs 2 \
  --name higraphmem2_clean_full10
```

仅运行 Open-Domain（类别 3，96 题）的命令：

```bash
bash scripts/runs/run_full_locomo.sh \
  --jobs 5 \
  --categories 3 \
  --name higraphmem2_open_recall_type_v1
```

- `--jobs` 是并行 conversation 数；`1` 即串行。先用 `2`，API 稳定后再考虑增加。
- `--categories` 默认 `1,2,3,4`；传入 `3` 时只评测 Open-Domain。脚本仍会启动 10 个 conversation shard，但每个 shard 只生成所选题类。
- `--name` 决定输出文件名，例如上述命令写入 `experiments/results/higraphmem2_clean_full10.json` 和对应的 `_official_eval.json`。
- 每个 conversation 写独立 shard、日志和记忆图。失败后以完全相同的命令重跑即可断点续跑。
- 10 个 shard 完成后，脚本自动合并，并调用官方 F1/BLEU-1 evaluator。

每次更改代码、prompt、模型或超参数后，请使用新的 `--name`，不要混用旧 shard。


## 切换模型

`--model` 覆盖 provider 配置里的 `chat_model`；`--model-label` 只控制结果表显示名称。若模型使用不同 API endpoint 或凭证，请通过 `--provider-config` 指向 source-root 下对应配置。每个模型必须使用独立的 `--name`。

```bash
bash scripts/runs/run_full_locomo.sh \
  --jobs 5 \
  --model <provider-supported-model-id> \
  --model-label <display-name> \
  --name higraphmem2_<display-name>_full10_v4
```


当 provider 有 TPM 限制时，降低 `--jobs`，并增大 `--retry-sleep`。已有 shard 会通过 `--resume` 自动续跑。
# 运行脚本

## HiGraphMem 消融（GPT-4o-mini）

`run_ablation_locomo.sh` 将每个变体以独立结果名、相同 QA-masked LoCoMo Cat1--4 协议运行，并自动生成预测、官方评测及 token usage 文件。`--backbone gpt4o-mini` 默认读取 `configs/provider.yaml`；`--backbone qwen25-14b` 默认读取 `configs/provider_qwen.yaml`。两者均为本地、Git 忽略的 OpenAI-compatible provider 配置。

```bash
# GMGraph：裸文本检索、受控文本检索、Update、Conflict Control
bash scripts/runs/run_ablation_locomo.sh --group gmgraph --variant plain-text-rag --name ablation_plain_text_rag_gpt4omini
bash scripts/runs/run_ablation_locomo.sh --group gmgraph --variant text-rag --name ablation_text_rag_gpt4omini
bash scripts/runs/run_ablation_locomo.sh --group gmgraph --variant wo-update --name ablation_wo_update_gpt4omini
bash scripts/runs/run_ablation_locomo.sh --group gmgraph --variant wo-conflict-control --name ablation_wo_conflict_gpt4omini

# TWMem：问题触发工作记忆、完整 Gated Activation
bash scripts/runs/run_ablation_locomo.sh --group twmem --variant wo-twmem --name ablation_wo_twmem_gpt4omini
bash scripts/runs/run_ablation_locomo.sh --group twmem --variant wo-gated-activation --name ablation_wo_gated_activation_gpt4omini

# 文本旁路消融：分别去掉 BM25 或 Fact Memory
bash scripts/runs/run_ablation_locomo.sh --group text-retrieval --variant wo-bm25 --name ablation_wo_bm25_gpt4omini
bash scripts/runs/run_ablation_locomo.sh --group text-retrieval --variant wo-fact-memory --name ablation_wo_fact_memory_gpt4omini

# 重要模块：Router
bash scripts/runs/run_ablation_locomo.sh --group important-modules --variant wo-router --name ablation_wo_router_gpt4omini
```

可添加 `--jobs 1` 降低并发；以相同 `--name` 重新运行即可从已完成预测继续。

## 一次跑完全部消融

`run_all_ablations_locomo.sh` 会按顺序跑完八个变体。中断后以相同 `--prefix` 重新运行，各变体会从自己的 shard 继续。

```bash
bash scripts/runs/run_all_ablations_locomo.sh \
  --backbone gpt4o-mini --prefix ablation_gpt4omini_v1 --jobs 2

bash scripts/runs/run_all_ablations_locomo.sh \
  --backbone qwen25-14b --prefix ablation_qwen25_14b_v1 --jobs 2
```

每个变体完成时会写入 `<name>_token_usage.json`，其中包括在线回答阶段的每题 token 数、平均 API 请求时延和 `completion tokens/s`。整组六个变体完成后，脚本还会自动写入：

- `experiments/results/<prefix>_efficiency_comparison.json`
- `experiments/results/<prefix>_efficiency_comparison.md`

其中 `completion tokens/s` 以成功 API 调用的 completion token 数除以该调用的 wall time 计算；它反映模型服务的生成吞吐，不包含本地图构建/检索、失败请求和重试等待时间。正在运行或此前已完成的任务不会补出耗时字段，需在本次代码更新后新发起的调用才会记录。
