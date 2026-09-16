import re
from typing import Iterable, List

from .knowledge_graph import Triple


def heuristic_extract_triples(turns: Iterable[object], source: str = "session") -> List[Triple]:
    triples: List[Triple] = []
    for index, turn in enumerate(turns):
        content = getattr(turn, "content", "")
        timestamp = getattr(turn, "timestamp", None)
        match = re.search(r"(?:搬到|moved to)\s*([\w\u4e00-\u9fff-]+)", content, re.IGNORECASE)
        if match:
            triples.append(
                Triple(
                    subject="user",
                    relation="moved_to",
                    object=match.group(1),
                    timestamp=timestamp,
                    source=f"{source}:{index}",
                    confidence=0.6,
                )
            )
    return triples
