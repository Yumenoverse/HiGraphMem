from typing import Dict, List


def fuse(graph_hits: List[dict], text_hits: List[str]) -> Dict[str, object]:
    return {"graph": graph_hits, "text": text_hits}
