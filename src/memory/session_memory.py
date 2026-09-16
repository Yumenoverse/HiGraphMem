from dataclasses import dataclass, field
from typing import List

from src.graph.knowledge_graph import Triple


@dataclass
class SessionMemory:
    session_id: str
    summary: str
    triples: List[Triple] = field(default_factory=list)
