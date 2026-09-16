from dataclasses import dataclass
from typing import List, Tuple

from ..memory.entity_state import EntityStateMemory, build_entity_state_memory
from ..skills.schema import StateRule


@dataclass(frozen=True)
class StateHit:
    memory_id: str
    text: str
    score: float
    source_type: str = "entity_state"


class RetrievalAugmentedStateMemory:
    def __init__(self, sample: dict, rules: List[StateRule]) -> None:
        self.state_memory: EntityStateMemory = build_entity_state_memory(sample, rules=rules)

    def retrieve(self, question: str, top_k: int = 5) -> List[StateHit]:
        answer = self.state_memory.answer(question)
        if not answer:
            return []
        value, evidence_ids = answer
        evidence = ", ".join(evidence_ids)
        text = "Structured entity-state answer candidate: %s. Supporting evidence ids: %s." % (value, evidence)
        return [StateHit(memory_id="STATE:" + "|".join(evidence_ids[:top_k]), text=text, score=3.0)]


def format_state_hits(hits: List[StateHit]) -> Tuple[str, List[str]]:
    lines = []
    ids = []
    for hit in hits:
        ids.append(hit.memory_id)
        lines.append("[%s][%s][%.3f] %s" % (hit.memory_id, hit.source_type, hit.score, hit.text))
    return "\n".join(lines), ids
