#!/usr/bin/env bash
# Run the clean HiGraphMem2 configuration on all 10 LoCoMo conversations.
# Each shard is one conversation, so shard processes have no shared memory state.

set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runs/run_full_locomo.sh --jobs 2 --name higraphmem2_clean_full10

Options:
  -j, --jobs N       Number of simultaneous conversation shards (default: 1).
  -n, --name NAME    Result filename under experiments/results, with or without .json (required).
  -s, --source-root PATH
                      Path to the HiGraphMem source repository (default: current repository).
  -c, --categories LIST
                      Comma-separated LoCoMo categories in 1,2,3,4 (default: 1,2,3,4).
                      Use 3 to run Open-Domain only.
  -m, --model MODEL  Override the chat model in the provider config.
  -p, --provider-config PATH
                      Provider config relative to --source-root (default: configs/provider.yaml).
      --model-label NAME
                      Model name printed in the evaluation table (default: MODEL or GPT-4o-mini).
      --max-retries N
                      Retries per API call after failures (default: 5).
      --retry-sleep SEC
                      Initial retry delay in seconds; retries use exponential backoff (default: 2).
      --answer-prompt-profile PROFILE
                      Answer prompt: auto, generic, or qwen_strict (default: auto).
      --disable-long-short-memory
                      Run Text-RAG without GMGraph or TWMem.
      --plain-text-rag
                      Run fixed raw-turn BM25 only, without all HiGraphMem control modules.
      --disable-update | --disable-conflict-control | --disable-twmem
      --disable-gated-activation | --disable-relation-gate
      --disable-router | --disable-time-gate | --disable-temporal-resolver
      --disable-multihop-rerank | --disable-evidence-verification
      --disable-bm25 | --disable-fact-memory
                      Single-operation HiGraphMem ablation switches.
      --disable-graph  Disable legacy graph retrieval (used by Text-RAG).
      --graph-only-update-ablation
                      Use only GMGraph evidence; for paired Full vs w/o Update diagnostics.
      --evidence-selection-mode MODE
                      concat (default) or graph_guided_raw. The latter uses graph/TWMem
                      to select a fixed budget of raw BM25 turns.
      --graph-guided-candidate-pool N --graph-guided-max-turns N
      --graph-guided-max-chars N
      --method-label NAME
                      Method name in the official evaluation output.
  -h, --help          Show this help.

The script evaluates the selected LoCoMo categories from 1--4, writes resumable shard files,
merges the ten shards, and writes the matching official F1/BLEU-1 result automatically.
Use a fresh NAME after any code or prompt change.
EOF
}

jobs=1
name=""
source_root="."
categories="1,2,3,4"
chat_model=""
provider_config="configs/provider.yaml"
model_label=""
max_retries=5
retry_sleep=2
answer_prompt_profile="auto"
enable_long_short_memory=1
disable_update=0
disable_conflict_control=0
disable_twmem=0
disable_gated_activation=0
disable_relation_gate=0
disable_router=0
disable_time_gate=0
disable_multihop_rerank=0
disable_temporal_resolver=0
disable_evidence_verification=0
disable_bm25=0
disable_fact_memory=0
disable_graph=0
plain_text_rag=0
graph_only_update_ablation=0
evidence_selection_mode="concat"
graph_guided_candidate_pool=30
graph_guided_max_turns=6
graph_guided_max_chars=6000
method_label="HiGraphMem2-Clean"
while (($#)); do
  case "$1" in
    -j|--jobs) jobs="$2"; shift 2 ;;
    -n|--name) name="$2"; shift 2 ;;
    -s|--source-root) source_root="$2"; shift 2 ;;
    -c|--categories) categories="$2"; shift 2 ;;
    -m|--model) chat_model="$2"; shift 2 ;;
    -p|--provider-config) provider_config="$2"; shift 2 ;;
    --model-label) model_label="$2"; shift 2 ;;
    --max-retries) max_retries="$2"; shift 2 ;;
    --retry-sleep) retry_sleep="$2"; shift 2 ;;
    --answer-prompt-profile) answer_prompt_profile="$2"; shift 2 ;;
    --disable-long-short-memory) enable_long_short_memory=0; shift ;;
    --disable-update) disable_update=1; shift ;;
    --disable-conflict-control) disable_conflict_control=1; shift ;;
    --disable-twmem) disable_twmem=1; shift ;;
    --disable-gated-activation) disable_gated_activation=1; shift ;;
    --disable-relation-gate) disable_relation_gate=1; shift ;;
    --disable-router) disable_router=1; shift ;;
    --disable-time-gate) disable_time_gate=1; shift ;;
    --disable-multihop-rerank) disable_multihop_rerank=1; shift ;;
    --disable-temporal-resolver) disable_temporal_resolver=1; shift ;;
    --disable-evidence-verification) disable_evidence_verification=1; shift ;;
    --disable-bm25) disable_bm25=1; shift ;;
    --disable-fact-memory) disable_fact_memory=1; shift ;;
    --disable-graph) disable_graph=1; shift ;;
    --plain-text-rag) plain_text_rag=1; shift ;;
    --graph-only-update-ablation) graph_only_update_ablation=1; shift ;;
    --evidence-selection-mode) evidence_selection_mode="$2"; shift 2 ;;
    --graph-guided-candidate-pool) graph_guided_candidate_pool="$2"; shift 2 ;;
    --graph-guided-max-turns) graph_guided_max_turns="$2"; shift 2 ;;
    --graph-guided-max-chars) graph_guided_max_chars="$2"; shift 2 ;;
    --method-label) method_label="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer" >&2
  exit 2
fi
if [[ -z "$categories" ]]; then
  echo "--categories must not be empty" >&2
  exit 2
fi
if ! [[ "$max_retries" =~ ^[0-9]+$ ]]; then
  echo "--max-retries must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$retry_sleep" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "--retry-sleep must be a non-negative number" >&2
  exit 2
fi
if [[ -z "$model_label" ]]; then
  model_label="${chat_model:-GPT-4o-mini}"
fi
if [[ -z "$name" ]]; then
  echo "--name is required" >&2
  usage >&2
  exit 2
fi
name="${name%.json}"
if [[ "$name" == */* ]]; then
  echo "--name must be a filename, not a path" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

shard_dir="experiments/results/${name}_shards"
memory_dir="experiments/memory_runs/${name}"
log_dir="experiments/logs/${name}"
merged_output="experiments/results/${name}.json"
eval_output="experiments/results/${name}_official_eval.json"
mkdir -p "$shard_dir" "$memory_dir" "$log_dir"

runner_flags=()
# Plain Text-RAG must not even construct GMGraph/TWMem artifacts.  It is a
# raw-text retrieval baseline rather than a graph retrieval variant.
if ((enable_long_short_memory && !plain_text_rag)); then runner_flags+=(--enable-long-short-memory --long-short-mode typed); fi
if ((disable_update)); then runner_flags+=(--disable-update); fi
if ((disable_conflict_control)); then runner_flags+=(--disable-conflict-control); fi
if ((disable_twmem)); then runner_flags+=(--disable-twmem); fi
if ((disable_gated_activation)); then runner_flags+=(--disable-gated-activation); fi
if ((disable_relation_gate)); then runner_flags+=(--disable-relation-gate); fi
if ((disable_router)); then runner_flags+=(--disable-router); fi
if ((disable_time_gate)); then runner_flags+=(--disable-temporal-memory-agreement-gate); fi
if ((disable_multihop_rerank)); then runner_flags+=(--disable-multihop-rerank); fi
if ((disable_temporal_resolver)); then runner_flags+=(--disable-temporal-resolver); fi
if ((disable_evidence_verification)); then runner_flags+=(--disable-evidence-verification); fi
if ((disable_bm25)); then runner_flags+=(--disable-bm25); fi
if ((disable_fact_memory)); then runner_flags+=(--disable-fact-memory); fi
if ((disable_graph)); then runner_flags+=(--disable-graph); fi
if ((plain_text_rag)); then runner_flags+=(--plain-text-rag --disable-graph); fi
if ((graph_only_update_ablation)); then runner_flags+=(--graph-only-update-ablation); fi
runner_flags+=(--evidence-selection-mode "$evidence_selection_mode")
if [[ "$evidence_selection_mode" == "graph_guided_raw" ]]; then
  runner_flags+=(--graph-guided-candidate-pool "$graph_guided_candidate_pool" --graph-guided-max-turns "$graph_guided_max_turns" --graph-guided-max-chars "$graph_guided_max_chars")
fi

pids=()
failed=0
cleanup() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap 'cleanup; exit 130' INT TERM

wait_oldest() {
  local pid="${pids[0]}"
  if ! wait "$pid"; then
    failed=1
  fi
  pids=("${pids[@]:1}")
}

for conversation_index in $(seq 0 9); do
  while ((${#pids[@]} >= jobs)); do
    wait_oldest
  done

  echo "Starting conversation ${conversation_index}"
  python eval/run_locomo_v2.py \
    --source-root "$source_root" \
    --provider-config "$provider_config" \
    --chat-model "$chat_model" \
    --max-retries "$max_retries" \
    --retry-sleep "$retry_sleep" \
    --answer-prompt-profile "$answer_prompt_profile" \
    --skip-conversations "$conversation_index" \
    --max-conversations 1 \
    "${runner_flags[@]}" \
    --memory-run-dir "${memory_dir}/conv_${conversation_index}" \
    --output "${shard_dir}/conv_${conversation_index}.json" \
    --resume \
    --categories "$categories" \
    --skip-category5 \
    --fail-on-api-error >"${log_dir}/conv_${conversation_index}.log" 2>&1 &
  pids+=("$!")
done

while ((${#pids[@]})); do
  wait_oldest
done
trap - INT TERM

if ((failed)); then
  echo "At least one shard failed. Inspect ${log_dir}; rerun this command to resume completed shards." >&2
  exit 1
fi

inputs=()
for conversation_index in $(seq 0 9); do
  inputs+=("${shard_dir}/conv_${conversation_index}.json")
done
python scripts/merge_locomo_outputs.py --inputs "${inputs[@]}" --output "$merged_output"

python scripts/summarize_locomo_token_usage.py \
  --input "$merged_output" \
  --output "experiments/results/${name}_token_usage.json" \
  --prediction-key higraphmem_prediction

python eval/official_locomo_eval.py \
  --source-root "$source_root" \
  --input "$merged_output" \
  --output "$eval_output" \
  --model "$model_label" \
  --method "$method_label" \
  --prediction-key higraphmem_prediction \
  --include-local-bleu1 \
  --bleu1-mode sentence_bleu

echo "Predictions: ${merged_output}"
echo "Token usage: experiments/results/${name}_token_usage.json"
echo "Official evaluation: ${eval_output}"
