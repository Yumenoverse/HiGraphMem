from dataclasses import dataclass, field
from typing import Dict, List

from ..skills.memory_skill import MemorySkill


@dataclass(frozen=True)
class QueryPlan:
    query_type: str
    answer_mode: str
    active_skills: List[str] = field(default_factory=list)
    retrieval_weights: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    alternative_types: List[str] = field(default_factory=list)


def route_question(question: str, skills: List[MemorySkill]) -> QueryPlan:
    lowered = question.lower()
    active = [skill.name for skill in skills if skill.matches_question(question)]
    query_type, confidence, alternative_types = _query_type(lowered)
    answer_mode = {
        "temporal": "date",
        "single_hop": "span",
        "multi_hop": "list",
        "open_domain": "inference",
    }[query_type]
    weights = _weights(query_type)
    return QueryPlan(
        query_type=query_type,
        answer_mode=answer_mode,
        active_skills=active,
        retrieval_weights=weights,
        confidence=confidence,
        alternative_types=alternative_types,
    )


def _query_type(lowered: str):
    lowered = lowered.strip()
    if _is_open_domain(lowered):
        return "open_domain", _open_domain_confidence(lowered), ["single_hop"]
    if _is_temporal(lowered):
        return "temporal", 0.92, ["single_hop"]
    if _is_multi_hop(lowered):
        return "multi_hop", _multi_hop_confidence(lowered), ["single_hop", "open_domain"]
    alternatives = ["open_domain"] if _maybe_open_domain(lowered) else []
    confidence = 0.58 if alternatives else 0.82
    return "single_hop", confidence, alternatives


def _is_temporal(lowered: str) -> bool:
    temporal_starts = [
        "when ",
        "how long",
        "for how long",
        "how many weeks",
        "how many months",
        "what year",
        "what month",
        "in which month",
    ]
    if any(lowered.startswith(marker) for marker in temporal_starts):
        return True
    temporal_markers = [
        " as of ",
        " during ",
        " between ",
        " in january",
        " in february",
        " in march",
        " in april",
        " in may",
        " in june",
        " in july",
        " in august",
        " in september",
        " in october",
        " in november",
        " in december",
    ]
    return lowered.startswith(("what did ", "who did ", "which ")) and any(marker in lowered for marker in temporal_markers)


def _is_open_domain(lowered: str) -> bool:
    if lowered.startswith(
        (
            "would ",
            "could ",
            "does ",
            "did ",
            "is it likely",
            "what might",
            "what could",
            "what would",
            "based on",
            "considering",
            "in light of",
        )
    ):
        return True
    markers = [
        " likely ",
        " might ",
        " potentially ",
        " suspected ",
        " alternative career",
        " underlying condition",
        " would enjoy",
        " could ",
        " be considered ",
        "answer yes or no",
        "political leaning",
        "financial status",
        "personality traits",
        "attributes describe",
        "what fields",
        "what job might",
        "what console",
        "what card game",
        "what board game",
        "what is the game",
        "what is a shop",
        "which outdoor gear company",
        "which popular time management technique",
        "which popular music composer",
        "what kind of yoga",
        "star wars book",
        "star wars-related locations",
        "national park could",
        "which us state",
        "what state did",
        "what country did",
        "in what country",
        "which country",
        "what nickname",
        "what pets wouldn't",
        "how many hikes",
        "feeling lonely",
        "no longer alive",
        "what kind of job",
        "what kind of career",
        "what kind of activity",
    ]
    return any(marker in lowered for marker in markers)


def _maybe_open_domain(lowered: str) -> bool:
    markers = [
        "holiday",
        "state",
        "country",
        "company",
        "organization",
        "technique",
        "composer",
        "console",
        "game",
        "condition",
        "health",
        "nickname",
        "national park",
        "would",
        "could",
        "likely",
        "might",
    ]
    return any(marker in lowered for marker in markers)


def _open_domain_confidence(lowered: str) -> float:
    high_markers = [
        "would ",
        "could ",
        "is it likely",
        "what might",
        "what could",
        "what would",
        "based on",
        "considering",
        "in light of",
        " likely ",
        " might ",
        " suspected ",
        "underlying condition",
    ]
    if any(marker in lowered for marker in high_markers):
        return 0.86
    return 0.68


def _is_multi_hop(lowered: str) -> bool:
    markers = [
        "both",
        "in common",
        "which events",
        "what events",
        "what activities",
        "what books",
        "where has",
        "where did",
        "how did",
        "what do ",
        "what types of",
        "what symbols",
        "what musical artists",
        "what book did",
        "what has ",
        "how many times",
        "how many children",
        "what european countries",
        "what recommendations",
        "what mediums",
        "what movies",
        "what things has",
        "what kind of interests",
        "what animal do",
        "what are joanna's hobbies",
        "what kind of writings",
    ]
    return any(marker in lowered for marker in markers)


def _multi_hop_confidence(lowered: str) -> float:
    high_markers = ["both", "in common", "which events", "what activities", "what books", "how many times"]
    if any(marker in lowered for marker in high_markers):
        return 0.84
    return 0.66


def _weights(query_type: str) -> Dict[str, float]:
    if query_type == "temporal":
        return {"fact": 1.0, "graph": 0.45, "state": 0.35, "bm25": 0.55}
    if query_type == "single_hop":
        return {"fact": 1.0, "graph": 0.35, "state": 0.35, "bm25": 0.45}
    if query_type == "multi_hop":
        return {"fact": 0.9, "graph": 0.9, "state": 0.8, "bm25": 0.65}
    return {"fact": 0.65, "graph": 0.75, "state": 0.35, "bm25": 1.0}
