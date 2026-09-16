from src.graph.entity_extract import heuristic_extract_triples
from typing import List

from .buffer import DialogueTurn
from .session_memory import SessionMemory


class MemoryConsolidator:
    def consolidate(self, session_id: str, turns: List[DialogueTurn]) -> SessionMemory:
        text = " ".join(turn.content for turn in turns)
        summary = text[:1000]
        triples = heuristic_extract_triples(turns, source=session_id)
        return SessionMemory(session_id=session_id, summary=summary, triples=triples)
