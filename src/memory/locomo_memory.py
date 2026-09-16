import re
from dataclasses import dataclass
from math import log
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MemoryHit:
    memory_id: str
    session_id: str
    text: str
    score: float
    source_type: str


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
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


def _tokens(text: str) -> List[str]:
    return [_stem(token) for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in STOPWORDS]


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


class LoCoMoMemory:
    def __init__(self, sample: dict) -> None:
        self.sample_id = sample["sample_id"]
        self.conversation = sample["conversation"]
        self.nodes = self._build_nodes(sample)
        self._node_tokens = [_tokens(node["text"]) for node in self.nodes]
        self._idf = self._build_idf(self._node_tokens)

    def retrieve(self, question: str, top_k: int = 8) -> List[MemoryHit]:
        query_tokens = _tokens(question)
        hits: List[MemoryHit] = []
        for node, node_tokens in zip(self.nodes, self._node_tokens):
            score = self._score(query_tokens, node_tokens)
            if score <= 0:
                continue
            if node["source_type"] == "turn":
                score += 0.05
            elif node["source_type"] == "session_summary":
                score += 0.02
            hits.append(
                MemoryHit(
                    memory_id=node["memory_id"],
                    session_id=node["session_id"],
                    text=node["text"],
                    score=score,
                    source_type=node["source_type"],
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]

    def _build_idf(self, tokenized_nodes: List[List[str]]) -> Dict[str, float]:
        doc_freq: Dict[str, int] = {}
        for tokens in tokenized_nodes:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        num_docs = max(len(tokenized_nodes), 1)
        return {token: log((num_docs + 1) / (freq + 0.5)) + 1 for token, freq in doc_freq.items()}

    def _score(self, query_tokens: List[str], node_tokens: List[str]) -> float:
        if not query_tokens or not node_tokens:
            return 0.0
        node_counts: Dict[str, int] = {}
        for token in node_tokens:
            node_counts[token] = node_counts.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            tf = node_counts.get(token, 0)
            if tf == 0:
                continue
            score += self._idf.get(token, 1.0) * (tf / (tf + 1.2))
        return score / max(len(query_tokens), 1)

    def _build_nodes(self, sample: dict) -> List[Dict[str, str]]:
        nodes: List[Dict[str, str]] = []
        conversation = sample["conversation"]
        for session_idx in range(1, 100):
            session_key = "session_%s" % session_idx
            if session_key not in conversation:
                continue
            date_time = conversation.get("%s_date_time" % session_key, "")
            for turn in conversation.get(session_key, []):
                text = "%s | %s: %s" % (date_time, turn.get("speaker", ""), turn.get("text", ""))
                nodes.append(
                    {
                        "memory_id": turn.get("dia_id", "%s:unknown" % session_key),
                        "session_id": "S%s" % session_idx,
                        "source_type": "turn",
                        "text": text,
                    }
                )
            summary_key = "%s_summary" % session_key
            if summary_key in sample.get("session_summary", {}):
                nodes.append(
                    {
                        "memory_id": "S%s" % session_idx,
                        "session_id": "S%s" % session_idx,
                        "source_type": "session_summary",
                        "text": sample["session_summary"][summary_key],
                    }
                )
            observation_key = "%s_observation" % session_key
            if observation_key in sample.get("observation", {}):
                observation = sample["observation"][observation_key]
                if isinstance(observation, list):
                    observation = " ".join(str(item) for item in observation)
                nodes.append(
                    {
                        "memory_id": "OBS%s" % session_idx,
                        "session_id": "S%s" % session_idx,
                        "source_type": "observation",
                        "text": str(observation),
                    }
                )
        return nodes


def format_evidence(hits: List[MemoryHit]) -> Tuple[str, List[str]]:
    lines = []
    context_ids = []
    for hit in hits:
        context_ids.append(hit.memory_id)
        lines.append("[%s][%s][%.3f] %s" % (hit.memory_id, hit.source_type, hit.score, hit.text))
    return "\n".join(lines), context_ids
