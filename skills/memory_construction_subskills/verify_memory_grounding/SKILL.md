---
name: verify_memory_grounding
description: Verify that an answer is supported by retrieved graph paths, working-memory clues, or raw dialogue spans.
---

# Verify Memory Grounding

Use this skill after answer generation to check whether the candidate answer is supported by the evidence bundle.

## Inputs

- `question`
- `candidate_answer`
- `evidence_bundle`

## Output

- `accepted`
- `support_score`
- `failure_reason`
- optional normalized answer

## Procedure

1. Check whether the candidate answer appears in, is entailed by, or is licensed by the evidence bundle.
2. For open-domain answers, allow at most one public-knowledge bridge from an explicit clue.
3. Reject unsupported named entities.
4. Normalize dates, lists, and concise phrase formats.

## Constraints

- Do not use gold answers.
- Do not repair answers with hidden labels.
- Prefer `No information available` when evidence is insufficient.

