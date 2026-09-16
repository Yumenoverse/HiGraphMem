#!/usr/bin/env bash
# Run Mem0's official LoCoMo benchmark through Mem0 Cloud and retain only final artifacts.
set -euo pipefail

name=""
provider_config="configs/provider.yaml"
model="gpt-4o-mini"
mem0_api_key="${MEM0_API_KEY:-}"
jobs=5
while (($#)); do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --provider-config) provider_config="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --mem0-api-key) mem0_api_key="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    *) echo "Usage: $0 --name NAME --mem0-api-key KEY [--provider-config YAML] [--model MODEL] [--jobs N]" >&2; exit 2 ;;
  esac
done
[[ -n "$name" && "$name" != */* ]] || { echo "--name is required" >&2; exit 2; }
[[ -n "$mem0_api_key" ]] || { echo "Set MEM0_API_KEY or pass --mem0-api-key." >&2; exit 2; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
provider_config="$(cd "$(dirname "$provider_config")" && pwd)/$(basename "$provider_config")"
[[ -f "$provider_config" ]] || { echo "Provider config not found: $provider_config" >&2; exit 2; }
read_value() { sed -n -E "s/^[[:space:]]*$1:[[:space:]]*['\"]?([^'\"#[:space:]]+).*$/\1/p" "$provider_config" | head -1; }
export OPENAI_API_KEY="$(read_value api_key)"
export OPENAI_BASE_URL="$(read_value base_url)"
[[ -n "$OPENAI_API_KEY" && -n "$OPENAI_BASE_URL" ]] || { echo "Provider config needs api_key and base_url." >&2; exit 2; }

if ! python -c 'import aiohttp, aiolimiter, dotenv, tqdm' >/dev/null 2>&1; then
  echo "Missing benchmark dependencies. Run: python -m pip install 'aiohttp>=3.9' 'aiolimiter>=1.1' 'python-dotenv>=1.0' 'tqdm>=4.66'" >&2
  exit 1
fi

scratch_root="${HIGRAPHMEM_SCRATCH:-/tmp/higraphmem_scratch}"
scratch="$scratch_root/mem0/$name"
benchmark="$scratch_root/memory-benchmarks"
results="$root/experiments/results"
mkdir -p "$scratch" "$results"
if [[ ! -f "$benchmark/benchmarks/locomo/run.py" ]]; then
  git clone --depth 1 https://github.com/mem0ai/memory-benchmarks.git "$benchmark"
fi

events="$scratch/${name}_token_events.json"
run_dir="$scratch/run"
mkdir -p "$run_dir"
python "$root/scripts/run_mem0_locomo_with_usage.py" \
  --benchmark-dir "$benchmark" \
  --events "$events" -- \
  --project-name "$name" \
  --backend cloud --mem0-api-key "$mem0_api_key" \
  --answerer-model "$model" --judge-model "$model" --provider openai \
  --dataset-path "$root/baseline/agenticmemory_eval/data/locomo10.json" \
  --conversations 0,1,2,3,4,5,6,7,8,9 --categories 1,2,3,4 \
  --top-k 200 --top-k-cutoffs 200 --max-workers "$jobs" \
  --rpm 100000 --output-dir "$run_dir" --resume

final_result="$(find "$run_dir" -maxdepth 1 -type f -name 'locomo_results_*.json' -print | sort | tail -1)"
[[ -n "$final_result" ]] || { echo "Mem0 did not produce a merged LoCoMo result." >&2; exit 1; }
cp "$final_result" "$results/${name}_raw.json"
cp "$events" "$results/${name}_token_events.json"
python "$root/scripts/summarize_baseline_events.py" \
  --input "$events" --output "$results/${name}_token_usage.json"
python - "$results/${name}_token_usage.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d['scope'] = 'Mem0 Cloud answerer/judge API calls only; managed Mem0 ingestion token usage is not exposed by the Cloud API.'
json.dump(d, open(p, 'w'), indent=2)
PY
rm -rf "$scratch/run"
echo "Mem0 result: $results/${name}_raw.json"
echo "Mem0 token usage: $results/${name}_token_usage.json"
