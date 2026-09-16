#!/usr/bin/env bash
# Remove only resumable memory artifacts after the matching official evaluation exists.
set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root" || exit 1

names=(
  ablation_qwen25_14b_v3_clean_wo_fact_memory
  ablation_qwen25_14b_v3_clean_wo_gmgraph
  ablation_qwen25_14b_v3_clean_wo_update
  ablation_qwen25_14b_v3_clean_wo_conflict_control
  ablation_qwen25_14b_v3_clean_wo_twmem
  ablation_qwen25_14b_v3_clean_wo_gated_activation
)

while true; do
  all_done=1
  for name in "${names[@]}"; do
    result="experiments/results/${name}_official_eval.json"
    memory_dir="experiments/memory_runs/${name}"
    if [[ ! -f "$result" ]]; then
      all_done=0
      continue
    fi
    if [[ -d "$memory_dir" ]]; then
      all_done=0
      if pgrep -f -- "--memory-run-dir ${memory_dir}/" >/dev/null; then
        continue
      fi
      count=$(find "$memory_dir" -xdev -printf . 2>/dev/null | wc -c)
      rm -rf -- "$memory_dir"
      echo "[$(date '+%F %T')] cleaned ${memory_dir} (${count} inodes); results and logs retained"
    fi
  done
  if ((all_done)); then
    echo "[$(date '+%F %T')] all requested experiment intermediates cleaned"
    exit 0
  fi
  sleep 60
done
