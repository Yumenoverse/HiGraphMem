import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .fact_memory import FactNode, build_fact_nodes


@dataclass(frozen=True)
class ProfileFact:
    entity: str
    slot: str
    value: str
    memory_id: str
    source_text: str


@dataclass(frozen=True)
class OpenDomainHit:
    memory_id: str
    text: str
    score: float
    source_type: str = "open_domain_state"


class OpenDomainStateAggregator:
    """Builds compact person profiles for preference and likelihood questions."""

    def __init__(self, sample: dict) -> None:
        self.profile_facts = _build_profile_facts(build_fact_nodes(sample))
        self.by_entity: Dict[str, List[ProfileFact]] = {}
        for fact in self.profile_facts:
            self.by_entity.setdefault(fact.entity.lower(), []).append(fact)

    def retrieve(self, question: str, top_k: int = 4) -> List[OpenDomainHit]:
        query_tokens = set(_tokens(question))
        entities = _entities_from_question(question)
        candidate_entities = entities or sorted(self.by_entity.keys())
        hits: List[OpenDomainHit] = []

        for entity in candidate_entities:
            facts = self.by_entity.get(entity.lower(), [])
            if not facts:
                continue
            scored = []
            for fact in facts:
                score = _slot_bonus(question, fact.slot)
                score += _overlap_score(query_tokens, _tokens(fact.source_text + " " + fact.value))
                if score > 0:
                    scored.append((score, fact))
            selected = [fact for _, fact in sorted(scored, key=lambda item: item[0], reverse=True)[:6]]
            if not selected:
                selected = facts[:4]
            profile_text = _format_profile(entity, selected)
            evidence_ids = [fact.memory_id for fact in selected]
            score = sum(_slot_bonus(question, fact.slot) for fact in selected) + 0.15 * len(selected)
            hits.append(
                OpenDomainHit(
                    memory_id="OPEN_STATE:" + "|".join(evidence_ids[:top_k]),
                    text=profile_text,
                    score=score,
                )
            )
        return sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]


def format_open_domain_hits(hits: List[OpenDomainHit]) -> Tuple[str, List[str]]:
    lines = []
    ids = []
    for hit in hits:
        ids.append(hit.memory_id)
        lines.append("[%s][%s][%.3f] %s" % (hit.memory_id, hit.source_type, hit.score, hit.text))
    return "\n".join(lines), ids


def _build_profile_facts(nodes: Iterable[FactNode]) -> List[ProfileFact]:
    output: List[ProfileFact] = []
    for node in nodes:
        if not node.entity:
            continue
        for slot, value in _extract_profile_values(node.text):
            output.append(ProfileFact(node.entity, slot, value, node.memory_id, node.text))
    return _dedupe_profile_facts(output)


def _extract_profile_values(text: str) -> List[Tuple[str, str]]:
    lowered = text.lower()
    values: List[Tuple[str, str]] = []
    patterns = [
        ("career", ["career", "job", "work", "degree", "education", "school", "counsel", "business", "studio", "store"]),
        ("preference", ["like", "loves", "enjoy", "favorite", "interested", "passion", "hobby"]),
        ("identity", ["transgender", "lgbtq", "religious", "faith", "church", "patriotic", "political"]),
        ("relationship", ["family", "children", "kids", "parent", "friend", "mentor", "support"]),
        ("activity", ["hiking", "camping", "painting", "music", "dance", "volunteer", "reading", "sports", "yoga"]),
        ("constraint", ["allergy", "afraid", "bad experience", "negative", "accident", "lost", "cannot", "won't", "discomfort"]),
        ("location", ["move", "visited", "beach", "mountain", "country", "state", "city", "travel"]),
        ("trait", ["thoughtful", "authentic", "driven", "selfless", "family-oriented", "rational", "creative"]),
    ]
    for slot, markers in patterns:
        if any(marker in lowered for marker in markers):
            values.append((slot, _compact_value(text)))
    return values


def _format_profile(entity: str, facts: List[ProfileFact]) -> str:
    by_slot: Dict[str, List[str]] = {}
    for fact in facts:
        by_slot.setdefault(fact.slot, [])
        if fact.value not in by_slot[fact.slot]:
            by_slot[fact.slot].append(fact.value)
    parts = []
    for slot in ["identity", "career", "preference", "relationship", "activity", "constraint", "location", "trait"]:
        values = by_slot.get(slot, [])
        if values:
            parts.append("%s=%s" % (slot, "; ".join(values[:3])))
    return "Open-domain profile for %s: %s." % (entity, " | ".join(parts))


def _slot_bonus(question: str, slot: str) -> float:
    q = question.lower()
    bonuses = {
        "career": ["career", "job", "degree", "education", "field", "work", "pursue"],
        "preference": ["likely enjoy", "interested", "want", "favorite", "would", "might"],
        "identity": ["considered", "identity", "community", "religious", "political", "patriotic"],
        "relationship": ["friends", "support", "family", "children", "ally"],
        "activity": ["hobby", "activities", "enjoy", "song", "books", "shop"],
        "constraint": ["would not", "wouldn't", "no", "condition", "allerg", "bad"],
        "location": ["move", "live close", "state", "country", "holiday", "where"],
        "trait": ["personality", "attributes", "traits"],
    }
    return 0.8 if any(marker in q for marker in bonuses.get(slot, [])) else 0.15


def _overlap_score(query_tokens: set, text_tokens: List[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & set(text_tokens)) / max(len(query_tokens), 1)


def _compact_value(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;\"'")
    if len(text.split()) <= 18:
        return text
    return " ".join(text.split()[:18])


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 2]


def _entities_from_question(question: str) -> List[str]:
    stop = {"What", "When", "Where", "Which", "Who", "Why", "How", "Would", "Does", "Is", "Around", "Based"}
    return [match.group(0) for match in re.finditer(r"\b[A-Z][a-z]+\b", question) if match.group(0) not in stop]


def _dedupe_profile_facts(facts: List[ProfileFact]) -> List[ProfileFact]:
    output = []
    seen = set()
    for fact in facts:
        key = (fact.entity.lower(), fact.slot, re.sub(r"[^a-z0-9]+", "", fact.value.lower())[:80])
        if key in seen:
            continue
        output.append(fact)
        seen.add(key)
    return output
