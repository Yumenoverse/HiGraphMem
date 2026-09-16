from src.graph.knowledge_graph import KnowledgeGraph
from typing import Dict, List


class GraphRetriever:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def retrieve(self, query: str) -> List[Dict[str, object]]:
        temporal_markers = ["when", "before", "after", "什么时候", "之前", "之后"]
        if any(marker in query.lower() for marker in temporal_markers):
            return self.graph.temporal_edges()
        return self.graph.query_entity("user")
