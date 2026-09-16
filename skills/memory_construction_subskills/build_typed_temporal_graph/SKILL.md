---
name: build_typed_temporal_graph
description: Build Grounded MemoryGraph from dialogue and observation fields only.
---

# Build Grounded MemoryGraph

Use this skill to convert a long conversation into a long-term heterogeneous graph memory. The graph stores stable structured memory, not benchmark answers.

This skill builds Grounded MemoryGraph as a GraphRAG/LightRAG-inspired memory index with typed edges, evidence provenance, temporal anchors, and retrieval paths. It is not the previous keyword fact graph.

## Inputs

- `conversation_id`: Stable conversation identifier.
- `conversation`: Dialogue sessions with speakers, text, timestamps, and dialogue ids.
- `observation`: Optional observation facts from the same conversation.

Do not read QA answers, gold evidence labels, category labels, or test-set feedback.

## Output

Write a conversation-local graph artifact:

- `nodes.jsonl`
- `edges.jsonl`
- `graph_metadata.json`

Reference script:

```bash
python skills/memory_construction_subskills/build_typed_temporal_graph/scripts/build_graph.py \
  --input path/to/locomo_sample.json \
  --output-dir experiments/memory_runs/conv-id/long_term_graph
```

The output must stay under the current conversation memory run directory and must not be merged into the global skill library.

## Procedure

1. Extract typed nodes for people, events, states, time expressions, places, objects, topics, preferences, goals, open-domain clues, and evidence spans.
2. Link every extracted node and edge to source dialogue ids or observation ids.
3. Add typed temporal edges for session order, event order, state validity, and recency.
4. Resolve local aliases and coreferences within the conversation.
5. Preserve raw evidence spans so downstream answering can fall back to text when graph extraction is incomplete.
6. Build GraphRAG-style retrieval paths by linking evidence spans to extracted nodes and typed relations.

## Constraints

- The graph is long-term memory within a conversation, not cross-test-set memory.
- Nodes and edges must be evidence-backed.
- Do not store question-answer pairs.
- Do not create edges from gold answers or evaluator outputs.
