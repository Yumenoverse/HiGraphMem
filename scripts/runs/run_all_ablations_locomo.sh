#!/usr/bin/env bash
# Run all standard named ablations sequentially for one backbone.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/runs/run_all_ablations_locomo.sh --backbone BACKBONE --prefix PREFIX [options]

BACKBONE: gpt4o-mini or qwen25-14b.
Each variant has its own resumable result name: PREFIX_<variant>.

Options:
  -p, --prefix PREFIX     Required filename prefix, e.g. ablation_gpt4omini_v1.
  -j, --jobs N            Concurrent conversation shards per variant (default: 2).
      --max-retries N     API retries per call (default: 5).
      --retry-sleep SEC   Initial retry delay (default: 2).
      --provider-config PATH
                            Override the backbone's provider config.
EOF
}

backbone=""
prefix=""
jobs=2
max_retries=5
retry_sleep=2
provider_config=""
while (($#)); do
  case "$1" in
    --backbone) backbone="$2"; shift 2 ;;
    -p|--prefix) prefix="$2"; shift 2 ;;
    -j|--jobs) jobs="$2"; shift 2 ;;
    --max-retries) max_retries="$2"; shift 2 ;;
    --retry-sleep) retry_sleep="$2"; shift 2 ;;
    --provider-config) provider_config="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$backbone" == "gpt4o-mini" || "$backbone" == "qwen25-14b" ]] || { usage >&2; exit 2; }
[[ -n "$prefix" && "$prefix" != */* ]] || { echo "--prefix is required and must be a filename prefix" >&2; exit 2; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
variants=(
  "gmgraph plain-text-rag plain_text_rag"
  "gmgraph wo-update wo_update"
  "gmgraph wo-conflict-control wo_conflict_control"
  "twmem wo-twmem wo_twmem"
  "twmem wo-gated-activation wo_gated_activation"
  "text-retrieval wo-bm25 wo_bm25"
  "text-retrieval wo-fact-memory wo_fact_memory"
  "important-modules wo-router wo_router"
)
for item in "${variants[@]}"; do
  read -r group variant suffix <<<"$item"
  command=(bash scripts/runs/run_ablation_locomo.sh --backbone "$backbone" --group "$group" --variant "$variant" --name "${prefix}_${suffix}" --jobs "$jobs" --max-retries "$max_retries" --retry-sleep "$retry_sleep")
  if [[ -n "$provider_config" ]]; then command+=(--provider-config "$provider_config"); fi
  echo "=== Running ${group}/${variant}: ${prefix}_${suffix} ==="
  "${command[@]}"
done

comparison="experiments/results/${prefix}_efficiency_comparison.json"
python scripts/compare_ablation_efficiency.py --prefix "$prefix" --output "$comparison"
