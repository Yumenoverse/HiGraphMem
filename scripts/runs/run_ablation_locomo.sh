#!/usr/bin/env bash
# Execute one named HiGraphMem ablation under the QA-masked LoCoMo protocol.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runs/run_ablation_locomo.sh --group GROUP --variant VARIANT --name RESULT_NAME [options]

Groups and variants:
  gmgraph            plain-text-rag | text-rag | wo-gmgraph | wo-update | wo-conflict-control |
                     graph-only-full | graph-only-wo-update
  twmem              wo-twmem | wo-gated-activation | wo-relation-gate
  text-retrieval     wo-bm25 | wo-fact-memory
  important-modules  wo-router

Options:
  -n, --name NAME        Required result filename prefix.
  -j, --jobs N           Concurrent conversation shards (default: 2).
      --backbone NAME    gpt4o-mini or qwen25-14b; default: gpt4o-mini.
      --max-retries N    API retries per call (default: 5).
      --retry-sleep SEC  Initial retry delay (default: 2).
      --provider-config PATH
                           Override the backbone's config path relative to this repository.
EOF
}

group=""
variant=""
name=""
jobs=2
max_retries=5
retry_sleep=2
provider_config="configs/provider.yaml"
provider_config_explicit=0
backbone="gpt4o-mini"
while (($#)); do
  case "$1" in
    --group) group="$2"; shift 2 ;;
    --variant) variant="$2"; shift 2 ;;
    -n|--name) name="$2"; shift 2 ;;
    -j|--jobs) jobs="$2"; shift 2 ;;
    --max-retries) max_retries="$2"; shift 2 ;;
    --retry-sleep) retry_sleep="$2"; shift 2 ;;
    --backbone) backbone="$2"; shift 2 ;;
    --provider-config) provider_config="$2"; provider_config_explicit=1; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$group" && -n "$variant" && -n "$name" && "$name" != */* ]] || { usage >&2; exit 2; }

case "$backbone" in
  gpt4o-mini)
    model="gpt-4o-mini"; model_label="GPT-4o-mini"
    ((provider_config_explicit)) || provider_config="configs/provider.yaml" ;;
  qwen25-14b)
    model="Qwen/Qwen2.5-14B-Instruct"; model_label="Qwen2.5-14B-Instruct"
    ((provider_config_explicit)) || provider_config="configs/provider_qwen.yaml" ;;
  *) echo "Unsupported --backbone: $backbone" >&2; exit 2 ;;
esac

flags=()
method=""
case "$group:$variant" in
  gmgraph:plain-text-rag)
    method="Plain Text-RAG"; flags=(--plain-text-rag) ;;
  gmgraph:text-rag)
    method="Text-RAG"; flags=(--disable-long-short-memory --disable-graph) ;;
  gmgraph:wo-gmgraph|gmgraph:w-o-gmgraph)
    # Preserve the full text channel and all non-graph settings.  With graph
    # construction disabled, TWMem receives no GMGraph candidates, isolating
    # the graph-maintenance component rather than switching to Text-RAG.
    method="HiGraphMem w/o GMGraph"; flags=(--disable-graph) ;;
  gmgraph:wo-update|gmgraph:w-o-update)
    # Match the standard full-pipeline ablation: retain text retrieval and all
    # graph components except session-by-session graph updates.
    method="HiGraphMem w/o Update"; flags=(--disable-update) ;;
  gmgraph:wo-conflict-control|gmgraph:w-o-conflict-control)
    method="HiGraphMem w/o Conflict Control"; flags=(--disable-conflict-control) ;;
  # These two are retained solely for an optional paired diagnostic under
  # graph-only evidence. They are not the standard Update ablation.
  gmgraph:graph-only-full)
    method="HiGraphMem Graph-only"; flags=(--graph-only-update-ablation) ;;
  gmgraph:graph-only-wo-update)
    method="HiGraphMem Graph-only w/o Update"; flags=(--graph-only-update-ablation --disable-update) ;;
  twmem:wo-twmem|twmem:w-o-twmem)
    method="HiGraphMem w/o TWMem"; flags=(--disable-twmem) ;;
  twmem:wo-gated-activation|twmem:w-o-gated-activation)
    method="HiGraphMem w/o Gated Activation"; flags=(--disable-gated-activation) ;;
  twmem:wo-relation-gate|twmem:w-o-relation-gate)
    method="HiGraphMem w/o Relation Gate"; flags=(--disable-relation-gate) ;;
  text-retrieval:wo-bm25|text-retrieval:w-o-bm25)
    method="HiGraphMem w/o BM25"; flags=(--disable-bm25) ;;
  text-retrieval:wo-fact-memory|text-retrieval:w-o-fact-memory)
    method="HiGraphMem w/o Fact Memory"; flags=(--disable-fact-memory) ;;
  important-modules:wo-router|important-modules:w-o-router)
    method="HiGraphMem w/o Router"; flags=(--disable-router) ;;
  *) echo "Unsupported group/variant: $group:$variant" >&2; usage >&2; exit 2 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
exec bash scripts/runs/run_full_locomo.sh \
  --jobs "$jobs" \
  --max-retries "$max_retries" \
  --retry-sleep "$retry_sleep" \
  --provider-config "$provider_config" \
  --model "$model" \
  --model-label "$model_label" \
  --method-label "$method" \
  --name "$name" \
  "${flags[@]}"
