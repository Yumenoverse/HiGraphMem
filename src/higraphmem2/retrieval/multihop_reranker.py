import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class RerankResult:
    evidence_text: str
    context_ids: List[str]
    source: str


def rerank_multihop_evidence(question: str, hit_groups: Iterable[Iterable[object]], max_items: int = 10) -> RerankResult:
    items = _flatten(hit_groups)
    if not items:
        return RerankResult("", [], "empty")

    q_tokens = set(_content_tokens(question))
    entities = _entities(question)
    constraints = _constraints(question)
    scored = []
    for item in items:
        score = _score_item(item, q_tokens, entities, constraints)
        if score > 0:
            scored.append((score, item))

    ranked = _dedupe([item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)])
    if not ranked:
        return RerankResult("", [], "no_positive")

    selected = _balanced_select(ranked, entities, max_items=max_items)
    lines = []
    ids = []
    for idx, item in enumerate(selected, start=1):
        ids.append(item["memory_id"])
        lines.append("[MH%d:%s][multi_hop_rerank][%.3f] %s" % (idx, item["memory_id"], item["score"], item["text"]))
    return RerankResult("\n".join(lines), ids, "multi_hop_rerank_v2")


def _flatten(hit_groups: Iterable[Iterable[object]]) -> List[dict]:
    output = []
    seen = set()
    for group in hit_groups:
        for hit in group:
            memory_id = str(getattr(hit, "memory_id", ""))
            text = str(getattr(hit, "text", ""))
            if not memory_id or not text:
                continue
            key = (memory_id, _normalize(text)[:120])
            if key in seen:
                continue
            output.append(
                {
                    "memory_id": memory_id,
                    "text": _clean_text(text),
                    "score": float(getattr(hit, "score", 0.0)),
                    "source_type": str(getattr(hit, "source_type", "evidence")),
                }
            )
            seen.add(key)
    return output


def _score_item(item: dict, q_tokens: set, entities: List[str], constraints: set) -> float:
    text = item["text"]
    t_tokens = set(_content_tokens(text))
    if not t_tokens:
        return 0.0
    score = 0.0
    score += 2.0 * len(q_tokens & t_tokens) / max(len(q_tokens), 1)
    score += min(item["score"], 3.0) * 0.2
    lowered = text.lower()
    for entity in entities:
        if entity.lower() in lowered:
            score += 0.55
    for constraint in constraints:
        if constraint in lowered:
            score += 0.75
    if _looks_like_generic_activity_noise(lowered, q_tokens, constraints):
        score -= 1.5
    item["score"] = score
    return score


def _balanced_select(items: List[dict], entities: List[str], max_items: int) -> List[dict]:
    selected = []
    used = set()
    for entity in entities:
        for item in items:
            if item["memory_id"] in used:
                continue
            if entity.lower() in item["text"].lower():
                selected.append(item)
                used.add(item["memory_id"])
                break
    for item in items:
        if len(selected) >= max_items:
            break
        if item["memory_id"] in used:
            continue
        selected.append(item)
        used.add(item["memory_id"])
    return selected


def _constraints(question: str) -> set:
    lowered = question.lower()
    groups = {
        "location": ["where", "city", "country", "state", "places", "locations", "visited", "camped"],
        "event": ["events", "participated", "attended", "promote", "fundraiser", "community"],
        "activity": ["activities", "hobbies", "done", "exercise"],
        "book": ["book", "read", "recommendations"],
        "count": ["how many", "times"],
        "person": ["who", "people", "names"],
        "object": ["items", "symbols", "pets", "instruments", "mediums"],
    }
    output = set()
    for name, markers in groups.items():
        if any(marker in lowered for marker in markers):
            output.add(name)
    return output


def _looks_like_generic_activity_noise(lowered: str, q_tokens: set, constraints: set) -> bool:
    generic = {"camping", "painting", "hiking", "pottery", "running", "fair"}
    if len(generic & set(_content_tokens(lowered))) < 2:
        return False
    if "activity" in constraints or "event" in constraints:
        return False
    return not bool(q_tokens & set(_content_tokens(lowered)))


def _entities(text: str) -> List[str]:
    stop = {"What", "When", "Where", "Which", "Who", "Why", "How", "Did", "Do", "Does", "In"}
    output = []
    seen = set()
    for match in re.finditer(r"\b[A-Z][a-z]+\b", text):
        value = match.group(0)
        if value in stop or value.lower() in seen:
            continue
        output.append(value)
        seen.add(value.lower())
    return output


def _content_tokens(text: str) -> List[str]:
    stop = {
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "did",
        "does",
        "have",
        "has",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "question",
        "focused",
        "fact",
        "graph",
        "expanded",
        "evidence",
    }
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if token.lower() not in stop and len(token) > 2]


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > 34:
        return " ".join(words[:34])
    return text


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _dedupe(items: List[dict]) -> List[dict]:
    output = []
    seen = set()
    for item in items:
        key = item["memory_id"]
        if key in seen:
            continue
        output.append(item)
        seen.add(key)
    return output
