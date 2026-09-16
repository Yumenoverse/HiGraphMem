#!/usr/bin/env bash
# Queue the remaining standard Qwen2.5-14B ablations after the in-progress
# v3_clean Full control completes.  One conversation shard at a time avoids
# competing for the same provider TPM allocation.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

full_result="experiments/results/ablation_qwen25_14b_v3_clean_full_official_eval.json"
provider_config="configs/provider_qwen.yaml"
while [[ ! -f "$full_result" ]]; do
  echo "Waiting for the in-progress Qwen Full control: $full_result"
  sleep 60
done

run_variant() {
  local group="$1"
  local variant="$2"
  local suffix="$3"
  local name="ablation_qwen25_14b_v3_clean_${suffix}"
  if [[ -f "experiments/results/${name}_official_eval.json" ]]; then
    echo "Already complete: $name"
    return
  fi
  bash scripts/runs/run_ablation_locomo.sh \
    --backbone qwen25-14b \
    --provider-config "$provider_config" \
    --group "$group" --variant "$variant" --name "$name" \
    --jobs 1 --max-retries 10 --retry-sleep 10
}

run_variant text-retrieval wo-bm25 wo_bm25
run_variant text-retrieval wo-fact-memory wo_fact_memory
run_variant gmgraph wo-update wo_update
run_variant gmgraph wo-conflict-control wo_conflict_control
run_variant twmem wo-twmem wo_twmem
run_variant twmem wo-gated-activation wo_gated_activation
