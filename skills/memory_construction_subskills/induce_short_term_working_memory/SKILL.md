---
name: induce_short_term_working_memory
description: Induce a bounded structured working memory from a conversation-local graph and raw dialogue context.
---

# Induce Triggered Working Memory

Use this skill to form transient memory that supports the current conversation or current question. It stores activated aliases, clues, evidence paths, bridge types, and normalization constraints.

The short-term memory follows a triggered working-memory model: visible question tokens and salient conversation cues activate a bounded set of graph nodes, evidence spans, and operations. It is a structured memory state, not a summary paragraph.

This is not a persistent subskill library. The generated JSON is a conversation-local memory state and must be discarded or isolated after evaluation.

## Inputs

- `conversation_id`
- `long_term_graph`
- `raw_dialogue_index`
- Optional `question` at QA time

The memory can be induced from dialogue-only context before QA, or refined with the visible question during retrieval. It must never use gold answers, gold evidence labels, or test errors.

## Output

- `short_term_working_memory.json`

Reference script:

```bash
python skills/memory_construction_subskills/induce_short_term_working_memory/scripts/induce_memory.py \
  --graph-dir experiments/memory_runs/conv-id/long_term_graph \
  --question "What console does Nate own?" \
  --output experiments/memory_runs/conv-id/short_term_working_memory.json
```

## Procedure

1. Bottom-up activation: activate entities, topics, events, and time nodes from lexical/entity/topic overlap.
2. Top-down control: infer question intent and select graph-walk or bridge operations.
3. Alias binding: build local alias maps and coreference hints.
4. Clue binding: extract open-domain clues such as games, locations, organizations, occupations, symptoms, methods, and named works.
5. Capacity control: keep only a bounded working buffer of evidence paths and operations.

## Constraints

- The short-term memory is an instance, not a reusable global skill.
- It may store conversation-specific facts only inside the current memory run directory.
- It must not be written into `skills/`.
