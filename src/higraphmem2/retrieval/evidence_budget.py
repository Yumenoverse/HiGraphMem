from dataclasses import dataclass

from .skill_router import QueryPlan


@dataclass(frozen=True)
class EvidenceBudget:
    fact_top_k: int
    graph_top_k: int
    state_top_k: int
    bm25_top_k: int


def budget_for_plan(plan: QueryPlan, default_bm25: int, default_graph: int, graph_disabled: bool = False) -> EvidenceBudget:
    graph_k = 0 if graph_disabled else max(default_graph, 4)
    bm25_k = max(default_bm25, 10)

    if plan.query_type == "temporal":
        return EvidenceBudget(fact_top_k=8, graph_top_k=graph_k, state_top_k=1, bm25_top_k=bm25_k)
    if plan.query_type == "single_hop":
        return EvidenceBudget(fact_top_k=6, graph_top_k=graph_k, state_top_k=1, bm25_top_k=bm25_k)
    if plan.query_type == "multi_hop":
        return EvidenceBudget(fact_top_k=8, graph_top_k=graph_k, state_top_k=3, bm25_top_k=bm25_k)
    return EvidenceBudget(fact_top_k=5, graph_top_k=graph_k, state_top_k=1, bm25_top_k=max(default_bm25, 12))
