from src.graph.knowledge_graph import KnowledgeGraph, Triple
from typing import List


class LongTermMemory:
    def __init__(self) -> None:
        self.graph = KnowledgeGraph()
        self.summaries: List[str] = []

    def add_session(self, summary: str, triples: List[Triple]) -> None:
        if summary:
            self.summaries.append(summary)
        self.graph.merge(triples)
