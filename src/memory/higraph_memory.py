from .buffer import ShortTermBuffer
from .consolidator import MemoryConsolidator
from .longterm_memory import LongTermMemory
from src.retrieval.fusion import fuse
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.router import QueryRouter
from src.retrieval.text_retriever import TextRetriever
from typing import Optional


class HiGraphMemory:
    def __init__(self, buffer_size: int = 10) -> None:
        self.buffer = ShortTermBuffer(max_size=buffer_size)
        self.longterm = LongTermMemory()
        self.consolidator = MemoryConsolidator()
        self.router = QueryRouter()
        self._session_index = 0

    def add_turn(self, role: str, content: str, timestamp: Optional[str] = None) -> None:
        self.buffer.add(role=role, content=content, timestamp=timestamp)

    def consolidate(self) -> None:
        turns = self.buffer.clear()
        if not turns:
            return
        self._session_index += 1
        session = self.consolidator.consolidate(f"session_{self._session_index}", turns)
        self.longterm.add_session(session.summary, session.triples)

    def retrieve(self, query: str) -> dict:
        graph_retriever = GraphRetriever(self.longterm.graph)
        text_retriever = TextRetriever(self.longterm.summaries)
        route = self.router.route(query)
        graph_hits = graph_retriever.retrieve(query) if "graph" in route else []
        text_hits = text_retriever.retrieve(query)
        return {"route": route, "evidence": fuse(graph_hits, text_hits)}
