#!/usr/bin/env bash
# Run the paper's full HiGraphMem control for one backbone.
# Re-run the same command and name to resume interrupted shards.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/runs/run_paper_main_locomo.sh --backbone BACKBONE --name RESULT_NAME [options]

BACKBONE: gpt4o-mini | qwen25-14b
Options: --jobs N --max-retries N --retry-sleep SEC --provider-config PATH
EOF
}

backbone=""
name=""
jobs=1
max_retries=10
retry_sleep=10
provider_config=""
while (($#)); do
  case "$1" in
    --backbone) backbone="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --max-retries) max_retries="$2"; shift 2 ;;
    --retry-sleep) retry_sleep="$2"; shift 2 ;;
    --provider-config) provider_config="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

[[ -n "$name" && "$name" != */* ]] || { echo "--name is required" >&2; exit 2; }
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

bash scripts/runs/run_full_locomo.sh \
  --name "$name" --jobs "$jobs" --max-retries "$max_retries" --retry-sleep "$retry_sleep" \
  --model "$model" --model-label "$label" --method-label "HiGraphMem" \
  --provider-config "$provider_config"
