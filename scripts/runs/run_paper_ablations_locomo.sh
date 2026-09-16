#!/usr/bin/env bash
# Run only ablations represented in the paper. Qwen graph-internal variants
# are deliberately excluded because they are not reported in the manuscript.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/runs/run_paper_ablations_locomo.sh --backbone BACKBONE --prefix PREFIX [options]

BACKBONE: gpt4o-mini | qwen25-14b
Each run is resumable: use the same PREFIX after interruption.
Options: --jobs N --max-retries N --retry-sleep SEC --provider-config PATH
EOF
}

backbone=""
prefix=""
jobs=1
max_retries=10
retry_sleep=10
provider_config=""
while (($#)); do
  case "$1" in
    --backbone) backbone="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --max-retries) max_retries="$2"; shift 2 ;;
    --retry-sleep) retry_sleep="$2"; shift 2 ;;
    --provider-config) provider_config="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

[[ -n "$prefix" && "$prefix" != */* ]] || { echo "--prefix is required" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

case "$backbone" in
  gpt4o-mini)
    model="gpt-4o-mini"; label="GPT-4o-mini"
    : "${provider_config:=configs/provider.yaml}" ;;
  qwen25-14b)
    model="Qwen/Qwen2.5-14B-Instruct"; label="Qwen2.5-14B"
    : "${provider_config:=configs/provider_qwen.yaml}" ;;
  *) usage >&2; exit 2 ;;
esac

run_one() {
  local suffix="$1"
  local method="$2"
  shift 2
  local name="${prefix}_${suffix}"
  if [[ -f "experiments/results/${name}_official_eval.json" ]]; then
    echo "Already complete: ${name}"
    return
  fi
  echo "=== ${method}: ${name} ==="
  bash scripts/runs/run_full_locomo.sh \
    --name "$name" --jobs "$jobs" --max-retries "$max_retries" --retry-sleep "$retry_sleep" \
    --model "$model" --model-label "$label" --method-label "$method" \
    --provider-config "$provider_config" "$@"
}

# Main channel and text-channel ablations shown for both backbones.
run_one text_only "Text-only baseline" --plain-text-rag
run_one wo_text_channel "HiGraphMem w/o Text Channel" --graph-only-update-ablation
run_one wo_graph_channel "HiGraphMem w/o Graph Channel" --disable-graph
run_one wo_tg_module "HiGraphMem w/o TG Module" --disable-temporal-resolver
run_one wo_bm25 "HiGraphMem w/o BM25" --disable-bm25
run_one wo_fact_memory "HiGraphMem w/o Fact Memory" --disable-fact-memory

if [[ "$backbone" == "gpt4o-mini" ]]; then
  # Graph-internal ablations are reported only for GPT-4o-mini.
  run_one wo_twmem "HiGraphMem w/o TWMem" --disable-twmem
  run_one wo_update "HiGraphMem w/o Update" --disable-update
  run_one wo_gated_activation "HiGraphMem w/o Gated Activation" --disable-gated-activation
fi
