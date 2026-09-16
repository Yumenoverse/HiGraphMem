from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    timestamp: Optional[str] = None
    source: Optional[str] = None
    confidence: float = 1.0


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_triple(self, triple: Triple) -> None:
        self.graph.add_node(triple.subject)
        self.graph.add_node(triple.object)
        self.graph.add_edge(
            triple.subject,
            triple.object,
            relation=triple.relation,
            timestamp=triple.timestamp,
            source=triple.source,
            confidence=triple.confidence,
        )

    def merge(self, triples: Iterable[Triple]) -> None:
        for triple in triples:
            self.add_triple(triple)

    def query_entity(self, entity: str) -> List[Dict[str, Any]]:
        if entity not in self.graph:
            return []
        results: List[Dict[str, Any]] = []
        for source, target, data in self.graph.out_edges(entity, data=True):
            results.append({"subject": source, "object": target, **data})
        for source, target, data in self.graph.in_edges(entity, data=True):
            results.append({"subject": source, "object": target, **data})
        return results

    def temporal_edges(self) -> List[Dict[str, Any]]:
        edges = [
            {"subject": s, "object": o, **data}
            for s, o, data in self.graph.edges(data=True)
            if data.get("timestamp")
        ]
        return sorted(edges, key=lambda item: item["timestamp"])
