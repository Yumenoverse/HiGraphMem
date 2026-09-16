#!/usr/bin/env bash
# Run the upstream MemoryOS LoCoMo reproduction with the shared GPT-4o-mini provider.
# It deliberately keeps credentials in the existing provider config, never in baseline/.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/runs/run_memoryos_locomo.sh --name memoryos_gpt4omini_full10

Options:
  -n, --name NAME              Result name (required; no path separators).
  -p, --provider-config PATH   Provider YAML (default: configs/provider.yaml).
  -m, --model MODEL            Chat model (default: gpt-4o-mini).
      --max-retries N          Retries for transient 429/5xx/connection errors (default: 8).
      --retry-sleep SEC        Initial retry delay; exponential backoff is used (default: 15).
EOF
}

name=""
provider_config="configs/provider.yaml"
model="gpt-4o-mini"
max_retries=8
retry_sleep=15
while (($#)); do
  case "$1" in
    -n|--name) name="$2"; shift 2 ;;
    -p|--provider-config) provider_config="$2"; shift 2 ;;
    -m|--model) model="$2"; shift 2 ;;
    --max-retries) max_retries="$2"; shift 2 ;;
    --retry-sleep) retry_sleep="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$name" && "$name" != */* ]] || { echo "--name is required and must be a filename" >&2; exit 2; }
[[ "$max_retries" =~ ^[0-9]+$ ]] || { echo "--max-retries must be a non-negative integer" >&2; exit 2; }
[[ "$retry_sleep" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "--retry-sleep must be a non-negative number" >&2; exit 2; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
provider_config="$(cd "$(dirname "$provider_config")" && pwd)/$(basename "$provider_config")"
[[ -f "$provider_config" ]] || { echo "Provider config not found: $provider_config" >&2; exit 2; }

# The project config is intentionally simple YAML.  Parse only the two scalar
# provider fields needed here; do not echo either value.  Do not use an awk
# field separator of ':' here: a URL such as "https://..." contains a colon
# and would otherwise be truncated to "https".
read_provider_scalar() {
  local key="$1"
  awk -v key="$key" '
    match($0, "^[[:space:]]*" key "[[:space:]]*:[[:space:]]*") {
      value = substr($0, RLENGTH + 1)
      sub(/[[:space:]]+#.*$/, "", value)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      if ((value ~ /^".*"$/) || (value ~ /^\047.*\047$/)) {
        value = substr(value, 2, length(value) - 2)
      }
      print value
      exit
    }
  ' "$provider_config"
}

export OPENAI_API_KEY="$(read_provider_scalar api_key)"
export OPENAI_BASE_URL="$(read_provider_scalar base_url)"
export BASELINE_CHAT_MODEL="$model"
export BASELINE_MAX_RETRIES="$max_retries"
export BASELINE_RETRY_SLEEP="$retry_sleep"
[[ -n "$OPENAI_API_KEY" && -n "$OPENAI_BASE_URL" ]] || { echo "provider config must define api_key and base_url" >&2; exit 2; }
[[ "$OPENAI_BASE_URL" =~ ^https?://[^[:space:]]+$ ]] || { echo "provider config base_url must begin with http:// or https://" >&2; exit 2; }

# This environment's local proxy closes the TLS handshake to the xi endpoint.
# Bypass it only for the configured API host; retain proxy settings for all
# other traffic (for example, model-asset downloads).
provider_host="${OPENAI_BASE_URL#*://}"
provider_host="${provider_host%%/*}"
provider_host="${provider_host%%:*}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${provider_host}"
export no_proxy="$NO_PROXY"

result_dir="$root/experiments/results"
run_dir="$root/experiments/memory_runs/$name"
mkdir -p "$result_dir" "$run_dir"
export BASELINE_USAGE_OUTPUT="$result_dir/${name}_token_usage.json"

# The upstream script has a fixed output filename; isolate each invocation in
# its own run directory. main_loco_parse.py resumes completed conversations.
cp "$root/baseline/memoryos/eval/locomo10.json" "$run_dir/locomo10.json"
pushd "$run_dir" >/dev/null
PYTHONPATH="$root/baseline/memoryos/eval${PYTHONPATH:+:$PYTHONPATH}" \
  python "$root/baseline/memoryos/eval/main_loco_parse.py"
popd >/dev/null

python "$root/scripts/convert_memoryos_locomo_results.py" \
  --source-data "$root/data/locomo/data/locomo10.json" \
  --memoryos-results "$run_dir/all_loco_results.json" \
  --output "$result_dir/$name.json"
python "$root/eval/official_locomo_eval.py" \
  --source-root "$root" \
  --input "$result_dir/$name.json" \
  --output "$result_dir/${name}_official_eval.json" \
  --model "$model" \
  --method MemoryOS \
  --prediction-key memoryos_prediction \
  --include-local-bleu1 \
  --bleu1-mode sentence_bleu
