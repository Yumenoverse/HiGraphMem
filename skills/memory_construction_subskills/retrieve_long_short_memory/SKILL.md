---
name: retrieve_long_short_memory
description: Retrieve and compose evidence from Grounded MemoryGraph and Triggered Working Memory.
---

# Retrieve Memory Evidence Bundle

Use this skill at QA time after the Grounded MemoryGraph and Triggered Working Memory have been built. The question is visible here because it is the user query, but gold labels remain hidden.

## Inputs

- `question`
- `long_term_graph`
- `short_term_working_memory`
- `raw_dialogue_index`

## Output

- `evidence_bundle.json`
- ranked evidence paths
- raw text fallback spans
- activated operations

## Procedure

1. Route the question into single-hop, multi-hop, temporal, or open-domain mode.
2. Activate matching Triggered Working Memory fields.
3. Walk the MemoryGraph from question entities and active clues.
4. Retrieve raw dialogue spans linked to selected graph nodes and edges.
5. Compose a compact evidence bundle with provenance.

## Constraints

- Retrieval may use the test question.
- Retrieval may not use gold answer, category, or gold evidence labels.
- The evidence bundle must expose provenance and uncertainty.

