from typing import Any, Dict, Iterable, List


class GAMGraphBaseline:
    """Interface slot for a GAM-style hierarchical graph baseline.

    This is intentionally separate from HiGraphMem so leaderboard runs can
    compare against a faithful GAM reproduction under the same evaluation code.
    """

    def build_memory(self, sessions: Iterable[object]) -> None:
        raise NotImplementedError("Implement GAM event graph, topic graph, and cross-layer links.")

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("Implement graph-guided multi-factor retrieval.")

    def answer(self, question: str) -> str:
        raise NotImplementedError("Generate from retrieved GAM evidence.")

