from typing import List, Optional


class QueryRouter:
    def __init__(self, temporal_keywords: Optional[List[str]] = None) -> None:
        self.temporal_keywords = temporal_keywords or ["when", "before", "after", "什么时候", "之前", "之后"]

    def route(self, query: str) -> str:
        lowered = query.lower()
        if any(keyword in lowered for keyword in self.temporal_keywords):
            return "temporal_graph"
        if "why" in lowered or "为什么" in query:
            return "fusion"
        return "entity_graph"
