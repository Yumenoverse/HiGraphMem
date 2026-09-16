import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .fact_memory import FactNode, build_fact_nodes
from ..skills.memory_skill import MemorySkill


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    text: str
    attrs: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    edge_type: str
    weight: float


@dataclass(frozen=True)
class GraphHit:
    memory_id: str
    text: str
    score: float
    source_type: str = "graph_expansion"


class SkillGuidedMemoryGraph:
    def __init__(self, sample: dict, skills: List[MemorySkill]) -> None:
        self.sample = sample
        self.skills = skills
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adj: Dict[str, List[GraphEdge]] = defaultdict(list)
        self.memory_id_to_nodes: Dict[str, List[str]] = defaultdict(list)
        self._build()

    def retrieve(self, question: str, seed_memory_ids: Iterable[str], top_k: int = 6) -> List[GraphHit]:
        routed_skills = [skill for skill in self.skills if skill.matches_question(question)]
        scores: Dict[str, float] = defaultdict(float)

        for memory_id in seed_memory_ids:
            for node_id in self.memory_id_to_nodes.get(_normalize_memory_id(memory_id), []):
                scores[node_id] += 1.4
                for edge in self.adj.get(node_id, []):
                    scores[edge.dst] += edge.weight
                    for second_edge in self.adj.get(edge.dst, []):
                        if second_edge.dst != node_id:
                            scores[second_edge.dst] += 0.35 * second_edge.weight

        for skill in routed_skills:
            skill_id = "SKILL:%s" % skill.name
            scores[skill_id] += 0.2
            for edge in self.adj.get(skill_id, []):
                scores[edge.dst] += edge.weight + 0.25

        query_tokens = _tokens(question)
        for node_id, node in self.nodes.items():
            if node.node_type not in {"fact", "turn"}:
                continue
            lexical = _lexical_overlap(query_tokens, _tokens(node.text))
            if lexical:
                scores[node_id] += lexical

        seed_set = {_normalize_memory_id(memory_id) for memory_id in seed_memory_ids}
        hits = []
        for node_id, score in scores.items():
            node = self.nodes.get(node_id)
            if not node or node.node_type not in {"fact", "turn"}:
                continue
            memory_id = node.attrs.get("memory_id", node_id)
            if _normalize_memory_id(memory_id) in seed_set:
                continue
            if score <= 0:
                continue
            hits.append(
                GraphHit(
                    memory_id=memory_id,
                    text=_format_graph_evidence(node),
                    score=score,
                )
            )
        return _dedupe_hits(sorted(hits, key=lambda item: item.score, reverse=True))[:top_k]

    def _build(self) -> None:
        self._add_skill_nodes()
        self._add_session_and_turn_nodes()
        self._add_fact_nodes()
        self._finalize_adjacency()

    def _add_skill_nodes(self) -> None:
        for skill in self.skills:
            self._add_node(
                GraphNode(
                    node_id="SKILL:%s" % skill.name,
                    node_type="skill",
                    text=skill.description,
                    attrs={"skill_name": skill.name},
                )
            )

    def _add_session_and_turn_nodes(self) -> None:
        conversation = self.sample.get("conversation", {})
        previous_session_id: Optional[str] = None
        for idx in range(1, 100):
            session_key = "session_%s" % idx
            if session_key not in conversation:
                continue
            session_id = "SESSION:S%s" % idx
            date_time = conversation.get("%s_date_time" % session_key, "")
            self._add_node(GraphNode(session_id, "session", date_time, {"session": "S%s" % idx, "date_time": date_time}))
            if previous_session_id:
                self._add_edge(previous_session_id, session_id, "temporal_next", 0.35)
            previous_session_id = session_id

            previous_turn_id: Optional[str] = None
            for turn in conversation.get(session_key, []):
                memory_id = str(turn.get("dia_id", "%s:unknown" % session_key))
                turn_id = "TURN:%s" % memory_id
                speaker = str(turn.get("speaker", ""))
                text = "%s: %s" % (speaker, turn.get("text", ""))
                self._add_node(
                    GraphNode(
                        turn_id,
                        "turn",
                        text,
                        {"memory_id": memory_id, "entity": speaker, "session": "S%s" % idx, "date_time": date_time},
                    )
                )
                self._index_memory_id(memory_id, turn_id)
                self._add_edge(session_id, turn_id, "session_turn", 0.75)
                entity_id = self._entity_node(speaker)
                self._add_edge(turn_id, entity_id, "turn_entity", 0.6)
                if previous_turn_id:
                    self._add_edge(previous_turn_id, turn_id, "temporal_next", 0.25)
                previous_turn_id = turn_id

    def _add_fact_nodes(self) -> None:
        for idx, fact in enumerate(build_fact_nodes(self.sample)):
            if fact.source_type == "dialog_turn":
                continue
            fact_id = "FACT:%s:%s" % (_normalize_memory_id(fact.memory_id), idx)
            self._add_node(
                GraphNode(
                    fact_id,
                    "fact",
                    fact.text,
                    {
                        "memory_id": fact.memory_id,
                        "entity": fact.entity,
                        "date_time": fact.date_time,
                        "source_type": fact.source_type,
                    },
                )
            )
            self._index_memory_id(fact.memory_id, fact_id)
            entity_id = self._entity_node(fact.entity)
            self._add_edge(fact_id, entity_id, "fact_entity", 0.9)
            session_id = _session_from_memory_id(fact.memory_id)
            if session_id:
                graph_session_id = "SESSION:%s" % session_id
                if graph_session_id in self.nodes:
                    self._add_edge(graph_session_id, fact_id, "session_fact", 0.65)
            for skill in self.skills:
                if _skill_matches_text(skill, fact.text):
                    self._add_edge("SKILL:%s" % skill.name, fact_id, "skill_fact", 0.9)

    def _entity_node(self, entity: str) -> str:
        entity_id = "ENTITY:%s" % entity.lower()
        if entity and entity_id not in self.nodes:
            self._add_node(GraphNode(entity_id, "entity", entity, {"entity": entity}))
        return entity_id

    def _add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def _add_edge(self, src: str, dst: str, edge_type: str, weight: float) -> None:
        if src == dst:
            return
        self.edges.append(GraphEdge(src, dst, edge_type, weight))
        self.edges.append(GraphEdge(dst, src, edge_type + "_rev", weight * 0.85))

    def _index_memory_id(self, memory_id: str, node_id: str) -> None:
        self.memory_id_to_nodes[_normalize_memory_id(memory_id)].append(node_id)

    def _finalize_adjacency(self) -> None:
        self.adj = defaultdict(list)
        for edge in self.edges:
            if edge.src in self.nodes and edge.dst in self.nodes:
                self.adj[edge.src].append(edge)


def format_graph_hits(hits: List[GraphHit]) -> Tuple[str, List[str]]:
    lines = []
    ids = []
    for hit in hits:
        ids.append(hit.memory_id)
        lines.append("[%s][%s][%.3f] %s" % (hit.memory_id, hit.source_type, hit.score, hit.text))
    return "\n".join(lines), ids


def _format_graph_evidence(node: GraphNode) -> str:
    date_time = node.attrs.get("date_time", "")
    entity = node.attrs.get("entity", "")
    if date_time and entity:
        return "Graph-expanded evidence: On %s, %s: %s" % (date_time, entity, node.text)
    if entity:
        return "Graph-expanded evidence: %s: %s" % (entity, node.text)
    return "Graph-expanded evidence: %s" % node.text


def _skill_matches_text(skill: MemorySkill, text: str) -> bool:
    lowered = text.lower()
    for pattern in skill.evidence_patterns:
        tokens = [token for token in pattern.lower().replace("{location}", "").split() if token]
        if tokens and all(token in lowered for token in tokens):
            return True
    return False


def _session_from_memory_id(memory_id: str) -> str:
    match = re.match(r"D(\d+):", str(memory_id))
    if match:
        return "S%s" % match.group(1)
    match = re.match(r"OBS(\d+)", str(memory_id))
    if match:
        return "S%s" % match.group(1)
    return ""


def _normalize_memory_id(memory_id: str) -> str:
    return str(memory_id).strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _lexical_overlap(query_tokens: List[str], node_tokens: List[str]) -> float:
    if not query_tokens or not node_tokens:
        return 0.0
    q = set(query_tokens)
    n = set(node_tokens)
    return 0.25 * len(q & n) / max(len(q), 1)


def _dedupe_hits(hits: List[GraphHit]) -> List[GraphHit]:
    output = []
    seen = set()
    for hit in hits:
        key = _normalize_memory_id(hit.memory_id)
        if key in seen:
            continue
        output.append(hit)
        seen.add(key)
    return output
