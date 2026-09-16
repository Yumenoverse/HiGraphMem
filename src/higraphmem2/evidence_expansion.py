from typing import List

from .memory.entity_state import EntityStateMemory


def should_use_state_answer(question: str, prediction: str, score: float) -> bool:
    lowered_q = question.lower()
    lowered_pred = prediction.lower()
    if "no information available" in lowered_pred:
        return True
    if score < 0.35 and any(
        marker in lowered_q
        for marker in [
            "relationship status",
            "move from",
            "moved from",
            "where has",
            "kids like",
            "books",
            "destress",
            "activities",
            "events",
            "paint recently",
        ]
    ):
        return True
    return False


def expand_with_entity_state(memory: EntityStateMemory, qa: dict, prediction_key: str) -> bool:
    current = qa.get(prediction_key, "")
    current_score = float(qa.get(prediction_key + "_f1", 0.0))
    if not should_use_state_answer(qa.get("question", ""), current, current_score):
        return False
    answer = memory.answer(qa.get("question", ""))
    if not answer:
        return False
    value, evidence_ids = answer
    qa[prediction_key + "_before_state_v2"] = current
    qa[prediction_key] = value
    qa[prediction_key + "_state_context"] = evidence_ids
    merged_context = list(qa.get(prediction_key + "_context", []))
    for evidence_id in evidence_ids:
        if evidence_id not in merged_context:
            merged_context.append(evidence_id)
    qa[prediction_key + "_context"] = merged_context
    return True
