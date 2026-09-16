# HiGraphMem

HiGraphMem 是面向长时程、多会话对话问答的双通道记忆框架：文本通道保留带来源的原始证据，图通道维护跨会话状态、事件与关系，Temporal Grounding 负责相对时间归一化。本仓库已合并论文 LoCoMo 实验所需的原始 HiGraphMem 运行代码、配置和评测数据，可独立运行。

## 研究动机

<p align="center">
  <a href="figures/taste_last.pdf"><img src="figures/taste_last.png" alt="HiGraphMem 研究动机" width="100%"></a>
</p>

## 模型总览

<p align="center">
  <a href="figures/model.pdf"><img src="figures/model.png" alt="HiGraphMem 模型总览" width="100%"></a>
</p>

## 安装与检查

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

## LoCoMo 数据集

官方数据集见 [snap-research/locomo](https://github.com/snap-research/locomo)。全新克隆仓库后，需要下载 `data/locomo/data/locomo10.json` 和官方 evaluator：

```bash
git clone --depth 1 https://github.com/snap-research/locomo.git /tmp/locomo
mkdir -p data/locomo
cp -a /tmp/locomo/data data/locomo/
cp -a /tmp/locomo/task_eval data/locomo/
```

## 中转站配置

从空模板创建本地 provider 文件，再填写自己的 OpenAI-compatible 中转站 URL 和 API key。真实 provider 文件已被 Git 忽略，不会上传。

```bash
cp configs/provider.example.yaml configs/provider.yaml
cp configs/provider.example.yaml configs/provider_qwen.yaml
```

## 主实验

```bash
# GPT-4o-mini
bash scripts/runs/run_paper_main_locomo.sh \
  --backbone gpt4o-mini --name paper_gpt4omini_full \
  --jobs 1 --max-retries 10 --retry-sleep 10

# Qwen2.5-14B
bash scripts/runs/run_paper_main_locomo.sh \
  --backbone qwen25-14b --name paper_qwen25_14b_full \
  --jobs 1 --max-retries 10 --retry-sleep 10
```

`--jobs` 控制并行 conversation shard 数：`--jobs 1` 为串行，较大数值会同时运行对应数量的 conversation。请按 provider 的并发和 TPM 配额设置。

## 论文消融

```bash
# GPT：Channel、Text 和 Graph-internal 消融
bash scripts/runs/run_paper_ablations_locomo.sh \
  --backbone gpt4o-mini --prefix paper_gpt4omini \
  --jobs 1 --max-retries 10 --retry-sleep 10

bash scripts/runs/run_paper_ablations_locomo.sh \
  --backbone qwen25-14b --prefix paper_qwen25_14b \
  --jobs 1 --max-retries 10 --retry-sleep 10
```

| 变体 | 运行开关 | GPT-4o-mini | Qwen2.5-14B |
| --- | --- | :---: | :---: |
| Text-only baseline | `--plain-text-rag` | 是 | 是 |
| w/o Text Channel | `--graph-only-update-ablation` | 是 | 是 |
| w/o Graph Channel | `--disable-graph` | 是 | 是 |
| w/o TG Module | `--disable-temporal-resolver` | 是 | 是 |
| w/o BM25 | `--disable-bm25` | 是 | 是 |
| w/o Fact Memory | `--disable-fact-memory` | 是 | 是 |
| w/o TWMem | `--disable-twmem` | 是 | 不展示 |
| w/o Update | `--disable-update` | 是 | 不展示 |
| w/o Gated Activation | `--disable-gated-activation` | 是 | 不展示 |

单独运行任意变体时，使用 `run_full_locomo.sh`，并在命令末尾加入上表对应开关。例如，单独运行 GPT 的 `w/o TG Module`：

```bash
bash scripts/runs/run_full_locomo.sh \
  --name paper_gpt4omini_wo_tg_module \
  --model gpt-4o-mini --model-label GPT-4o-mini \
  --method-label "HiGraphMem w/o TG Module" \
  --provider-config configs/provider.yaml \
  --jobs 1 --max-retries 10 --retry-sleep 10 \
  --disable-temporal-resolver
```

## 输出与实验日志

一个名为 `NAME` 的实验会保存：

```text
experiments/results/NAME.json
experiments/results/NAME_official_eval.json
experiments/results/NAME_token_usage.json
experiments/results/NAME_shards/conv_0.json ... conv_9.json
experiments/logs/NAME/conv_0.log ... conv_9.log
```

已完成实验的结果见 [experiments/results/](experiments/results/)，原始运行日志见 [experiments/logs/](experiments/logs/)。
