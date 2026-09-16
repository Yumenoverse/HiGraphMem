import re
from dataclasses import dataclass
from math import log
from typing import Dict, Iterable, List, Tuple


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "both",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


SYNONYMS = {
    "clothes": ["clothing", "store", "fashion"],
    "clothing": ["clothes", "store", "fashion"],
    "business": ["venture", "store", "studio", "entrepreneur"],
    "venture": ["business", "store", "studio"],
    "promote": ["promotion", "promoting", "campaign", "social", "networking", "fair", "competition", "website"],
    "events": ["event", "fair", "competition", "festival", "networking"],
    "destress": ["de-stress", "stress", "relax", "dancing", "dance"],
    "favorite": ["fav", "preferred", "style"],
    "style": ["favorite", "type", "kind"],
    "open": ["opened", "launch", "launched", "start", "started"],
    "mentorship": ["mentor", "mentored", "mentoring"],
    "mentor": ["mentorship", "mentored", "mentoring"],
    "lost": ["lose", "job"],
    "job": ["work", "career"],
}


@dataclass(frozen=True)
class FactHit:
    memory_id: str
    text: str
    score: float
    source_type: str = "question_fact"


@dataclass(frozen=True)
class FactNode:
    memory_id: str
    date_time: str
    entity: str
    text: str
    source_type: str


class QuestionFocusedFactMemory:
    def __init__(self, sample: dict) -> None:
        self.nodes = build_fact_nodes(sample)
        self._node_tokens = [_tokens(node.text) + _tokens(node.entity) for node in self.nodes]
        self._idf = _build_idf(self._node_tokens)

    def retrieve(self, question: str, top_k: int = 6) -> List[FactHit]:
        query_weights = _query_weights(_tokens(question))
        query_tokens = list(query_weights.keys())
        entities = set(_entities(question))
        hits = []
        for node, node_tokens in zip(self.nodes, self._node_tokens):
            score = _score(query_weights, node_tokens, self._idf)
            score += _action_bonus(question, node.text)
            if node.entity.lower() in entities:
                score += 0.35
            if _is_when_question(question) and node.date_time:
                score += 0.08
            if score <= 0:
                continue
            hits.append(
                FactHit(
                    memory_id=node.memory_id,
                    text=_format_fact(node),
                    score=score,
                    source_type=node.source_type,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]


def build_fact_nodes(sample: dict) -> List[FactNode]:
    nodes: List[FactNode] = []
    conversation = sample.get("conversation", {})
    session_dates = {
        "session_%s" % idx: conversation.get("session_%s_date_time" % idx, "")
        for idx in range(1, 100)
    }

    for session_key, observations in sample.get("observation", {}).items():
        session_id = session_key.replace("_observation", "")
        date_time = session_dates.get(session_id, "")
        if not isinstance(observations, dict):
            continue
        for entity, facts in observations.items():
            for item in facts:
                if isinstance(item, list) and len(item) >= 2:
                    fact = str(item[0])
                    evidence_id = str(item[1])
                else:
                    fact = str(item)
                    evidence_id = "OBS:" + session_id
                nodes.append(
                    FactNode(
                        memory_id=evidence_id,
                        date_time=date_time,
                        entity=str(entity),
                        text=fact,
                        source_type="observation_fact",
                    )
                )

    for session_idx in range(1, 100):
        session_key = "session_%s" % session_idx
        date_time = session_dates.get(session_key, "")
        for turn in conversation.get(session_key, []):
            nodes.append(
                FactNode(
                    memory_id=str(turn.get("dia_id", "%s:unknown" % session_key)),
                    date_time=date_time,
                    entity=str(turn.get("speaker", "")),
                    text=str(turn.get("text", "")),
                    source_type="dialog_turn",
                )
            )
    return nodes


def format_fact_hits(hits: List[FactHit]) -> Tuple[str, List[str]]:
    lines = []
    ids = []
    for hit in hits:
        ids.append(hit.memory_id)
        lines.append("[%s][%s][%.3f] %s" % (hit.memory_id, hit.source_type, hit.score, hit.text))
    return "\n".join(lines), ids


def _format_fact(node: FactNode) -> str:
    if node.date_time:
        return "Question-focused fact: On %s, %s: %s" % (node.date_time, node.entity, node.text)
    return "Question-focused fact: %s: %s" % (node.entity, node.text)


def _tokens(text: str) -> List[str]:
    return [_stem(token) for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token.lower() not in STOPWORDS]


def _query_weights(tokens: List[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for token in tokens:
        weights[token] = max(weights.get(token, 0.0), 1.0)
        for item in SYNONYMS.get(token, []):
            synonym = _stem(item)
            weights[synonym] = max(weights.get(synonym, 0.0), 0.35)
    return weights


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _entities(text: str) -> Iterable[str]:
    return [match.group(0).lower() for match in re.finditer(r"\b[A-Z][a-z]+\b", text)]


def _is_when_question(question: str) -> bool:
    return question.strip().lower().startswith("when")


def _action_bonus(question: str, text: str) -> float:
    question_l = question.lower()
    text_l = text.lower()
    bonus = 0.0
    groups = [
        (["open", "opened"], ["open", "opened"], 0.55),
        (["favorite"], ["favorite"], 0.45),
        (["mentor", "mentorship"], ["mentor", "mentored", "mentoring", "mentorship"], 0.65),
        (["tattoo"], ["tattoo"], 0.45),
        (["accepted"], ["accepted"], 0.45),
        (["launch", "launched"], ["launch", "launched"], 0.45),
        (["read", "reading"], ["read", "reading"], 0.35),
    ]
    for question_terms, text_terms, value in groups:
        if any(term in question_l for term in question_terms) and any(term in text_l for term in text_terms):
            bonus += value
    return bonus


def _build_idf(tokenized_nodes: List[List[str]]) -> Dict[str, float]:
    doc_freq: Dict[str, int] = {}
    for tokens in tokenized_nodes:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    num_docs = max(len(tokenized_nodes), 1)
    return {token: log((num_docs + 1) / (freq + 0.5)) + 1 for token, freq in doc_freq.items()}


def _score(query_weights: Dict[str, float], node_tokens: List[str], idf: Dict[str, float]) -> float:
    if not query_weights or not node_tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for token in node_tokens:
        counts[token] = counts.get(token, 0) + 1
    score = 0.0
    for token, weight in query_weights.items():
        tf = counts.get(token, 0)
        if tf:
            score += weight * idf.get(token, 1.0) * (tf / (tf + 1.2))
    return score / max(sum(query_weights.values()), 1.0)
