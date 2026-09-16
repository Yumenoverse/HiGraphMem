#!/usr/bin/env bash
# Run the official A-Mem LoCoMo harness with a project provider and usage trace.
set -euo pipefail

name=""
provider_config="configs/provider.yaml"
model="gpt-4o-mini"
while (($#)); do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --provider-config) provider_config="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    *) echo "Usage: $0 --name NAME [--provider-config YAML] [--model MODEL]" >&2; exit 2 ;;
  esac
done
[[ -n "$name" && "$name" != */* ]] || { echo "--name is required" >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
provider_config="$(cd "$(dirname "$provider_config")" && pwd)/$(basename "$provider_config")"
read_value() { sed -n -E "s/^[[:space:]]*$1:[[:space:]]*['\"]?([^'\"#[:space:]]+).*$/\1/p" "$provider_config" | head -1; }
export OPENAI_API_KEY="$(read_value api_key)"
export OPENAI_BASE_URL="$(read_value base_url)"
export LOCOMO_CATEGORIES="1,2,3,4"
export BASELINE_USAGE_OUTPUT="$root/experiments/results/${name}_token_events.json"
out="$root/experiments/results/${name}_raw.json"
cd "$root/baseline/agenticmemory_eval"
python test_advanced_robust.py --backend openai --model "$model" --dataset data/locomo10.json --output "$out"
python "$root/scripts/summarize_baseline_events.py" --input "$BASELINE_USAGE_OUTPUT" --output "$root/experiments/results/${name}_token_usage.json"
