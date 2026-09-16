import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class EvidenceItem:
    memory_id: str
    text: str
    source_type: str
    score: float


@dataclass(frozen=True)
class EvidenceChain:
    chain_id: str
    memory_ids: List[str]
    text: str
    score: float


class EvidenceChainComposer:
    """Composes multi-hop evidence chains from heterogeneous retrieval hits."""

    def compose(self, question: str, hit_groups: Iterable[Iterable[object]], top_k: int = 3) -> List[EvidenceChain]:
        items = _flatten_hits(hit_groups)
        if not items:
            return []
        query_tokens = set(_tokens(question))
        query_entities = set(_entities(question))
        anchors = query_entities or _question_anchor_terms(question)
        chains: List[EvidenceChain] = []

        for anchor in sorted(anchors):
            related = [item for item in items if _matches_anchor(anchor, item)]
            if len(related) < 2:
                continue
            chain_items = sorted(
                related,
                key=lambda item: item.score + _overlap(query_tokens, _tokens(item.text)),
                reverse=True,
            )[:4]
            score = _chain_score(question, chain_items, query_tokens)
            chains.append(_build_chain(anchor, chain_items, score))

        global_items = sorted(items, key=lambda item: item.score + _overlap(query_tokens, _tokens(item.text)), reverse=True)[:5]
        if len(global_items) >= 2:
            chains.append(_build_chain("global", global_items, _chain_score(question, global_items, query_tokens) * 0.9))

        return _dedupe_chains(sorted(chains, key=lambda chain: chain.score, reverse=True))[:top_k]


def format_evidence_chains(chains: List[EvidenceChain]) -> Tuple[str, List[str]]:
    lines = []
    ids = []
    for chain in chains:
        ids.append(chain.chain_id)
        lines.append("[%s][evidence_chain][%.3f] %s" % (chain.chain_id, chain.score, chain.text))
    return "\n".join(lines), ids


def _flatten_hits(hit_groups: Iterable[Iterable[object]]) -> List[EvidenceItem]:
    output: List[EvidenceItem] = []
    seen = set()
    for group in hit_groups:
        for hit in group:
            memory_id = str(getattr(hit, "memory_id", ""))
            text = str(getattr(hit, "text", ""))
            if not memory_id or not text:
                continue
            key = (memory_id, _normalize_text(text)[:120])
            if key in seen:
                continue
            output.append(
                EvidenceItem(
                    memory_id=memory_id,
                    text=text,
                    source_type=str(getattr(hit, "source_type", "evidence")),
                    score=float(getattr(hit, "score", 0.0)),
                )
            )
            seen.add(key)
    return output


def _build_chain(anchor: str, items: List[EvidenceItem], score: float) -> EvidenceChain:
    memory_ids = [item.memory_id for item in items]
    snippets = []
    for item in items:
        snippets.append("%s -> %s" % (item.memory_id, _clean_evidence_text(item.text)))
    return EvidenceChain(
        chain_id="CHAIN:%s:%s" % (_safe(anchor), "|".join(memory_ids[:3])),
        memory_ids=memory_ids,
        text="Evidence chain for %s: %s" % (anchor, " || ".join(snippets)),
        score=score,
    )


def _chain_score(question: str, items: List[EvidenceItem], query_tokens: set) -> float:
    score = 0.0
    covered = set()
    for item in items:
        tokens = set(_tokens(item.text))
        covered |= tokens & query_tokens
        score += min(item.score, 3.0) * 0.35
    score += len(covered) / max(len(query_tokens), 1)
    if _is_commonality_question(question) and _distinct_entities(items) >= 2:
        score += 1.0
    if _is_list_question(question) and len(items) >= 3:
        score += 0.5
    return score


def _matches_anchor(anchor: str, item: EvidenceItem) -> bool:
    anchor_l = anchor.lower()
    text_l = item.text.lower()
    if anchor_l in text_l:
        return True
    if anchor_l.endswith("s") and anchor_l[:-1] in text_l:
        return True
    return False


def _question_anchor_terms(question: str) -> set:
    tokens = _tokens(question)
    keep = {
        "business",
        "events",
        "activities",
        "books",
        "cities",
        "city",
        "visited",
        "promote",
        "community",
        "children",
        "family",
        "painted",
        "causes",
        "volunteering",
        "shelter",
    }
    return {token for token in tokens if token in keep}


def _entities(text: str) -> List[str]:
    stop = {"What", "When", "Where", "Which", "Who", "Why", "How", "Did", "Do", "In"}
    return [match.group(0) for match in re.finditer(r"\b[A-Z][a-z]+\b", text) if match.group(0) not in stop]


def _tokens(text: str) -> List[str]:
    stop = {"the", "and", "with", "what", "which", "have", "has", "did", "does", "for", "from", "that", "this"}
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if token.lower() not in stop and len(token) > 2]


def _overlap(a: set, b: List[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & set(b)) / max(len(a), 1)


def _distinct_entities(items: List[EvidenceItem]) -> int:
    entities = set()
    for item in items:
        entities.update(entity.lower() for entity in _entities(item.text))
    return len(entities)


def _is_commonality_question(question: str) -> bool:
    lowered = question.lower()
    return "both" in lowered or "in common" in lowered


def _is_list_question(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in ["what", "which", "where", "how did", "in what ways"])


def _clean_evidence_text(text: str) -> str:
    text = re.sub(r"^\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > 32:
        return " ".join(words[:32])
    return text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")[:32] or "global"


def _dedupe_chains(chains: List[EvidenceChain]) -> List[EvidenceChain]:
    output = []
    seen = set()
    for chain in chains:
        key = tuple(chain.memory_ids[:3])
        if key in seen:
            continue
        output.append(chain)
        seen.add(key)
    return output
