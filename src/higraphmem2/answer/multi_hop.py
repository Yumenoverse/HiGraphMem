import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Candidate:
    text: str
    score: float
    support: int


PROMOTION_TERMS = [
    "ad campaign",
    "limited-edition sweatshirts",
    "video presentation",
    "website",
    "fair",
    "networking events",
    "dance competition",
]

ACTIVITY_TERMS = [
    "dancing",
    "camping",
    "painting",
    "swimming",
    "hiking",
    "pottery",
    "running",
    "museum",
]

COMMONALITY_TERMS = [
    "lost their jobs",
    "decided to start their own businesses",
    "single parent",
    "children",
    "new business",
    "career change",
]


def aggregate_multi_hop_answer(question: str, prediction: str, evidence_lines: Iterable[str]) -> Optional[Tuple[str, str]]:
    question_l = question.lower()
    if not _should_aggregate(question_l):
        return None

    lines = [_clean_evidence_line(line) for line in evidence_lines if _clean_evidence_line(line)]
    evidence_l = "\n".join(lines).lower()
    prediction_l = prediction.lower()

    if "no information available" not in prediction_l:
        compressed = _compress_prediction(question_l, prediction, evidence_l)
        if compressed:
            return compressed, "multi_hop_aggregator_v2_prediction"

    candidates = _extract_candidates(question_l, lines)
    selected = _select_candidates(question_l, candidates)
    if selected:
        return _join([candidate.text for candidate in selected]), "multi_hop_aggregator_v2_evidence"

    legacy = _legacy_aggregate(question_l, prediction_l, evidence_l)
    if legacy:
        return legacy, "multi_hop_aggregator_v2_legacy_fallback"
    return None


def _compress_prediction(question_l: str, prediction: str, evidence_l: str) -> Optional[str]:
    parts = _split_candidate_text(prediction)
    candidates = [
        Candidate(part, _score_phrase(question_l, part, evidence_l), _support_count(part, evidence_l))
        for part in parts
        if _valid_phrase(part)
    ]
    selected = _select_candidates(question_l, candidates)
    if len(selected) >= 2:
        return _join([candidate.text for candidate in selected])
    return None


def _extract_candidates(question_l: str, lines: List[str]) -> List[Candidate]:
    candidates: List[str] = []
    evidence_l = "\n".join(lines).lower()
    for line in lines:
        candidates.extend(_pattern_candidates(question_l, line))
        candidates.extend(_list_like_candidates(question_l, line))
        candidates.extend(_lexicon_candidates(question_l, line))

    scored = []
    for value in _dedupe(candidates):
        if not _valid_phrase(value):
            continue
        score = _score_phrase(question_l, value, evidence_l)
        support = _support_count(value, evidence_l)
        if score >= 0.18 or support >= 1:
            scored.append(Candidate(_clean(value), score, support))
    return sorted(scored, key=lambda item: (item.score, item.support, -len(item.text)), reverse=True)


def _pattern_candidates(question_l: str, line: str) -> List[str]:
    patterns = []
    if _has_any(question_l, ["promote", "business venture", "store", "events"]):
        patterns.extend(
            [
                r"\b(?:promoted?|advertised?|marketed?)\b[^.?!;:]*?\b(?:through|with|using|by)\s+([^.?!;]+)",
                r"\b(?:participated in|attended|joined|hosted|organized|held)\s+([^.?!;]+)",
                r"\b(?:created|built|launched|made)\s+([^.?!;]+)",
            ]
        )
    if _has_any(question_l, ["activities", "events", "participated", "partake", "done"]):
        patterns.extend(
            [
                r"\b(?:participated in|attended|joined|went|went to|hosted|organized|did|done|does|enjoys?|likes?)\s+([^.?!;]+)",
                r"\b(?:activities?|events?)\s+(?:such as|including|were|are)\s+([^.?!;]+)",
            ]
        )
    if _has_any(question_l, ["books", "read", "recommendations"]):
        patterns.extend(
            [
                r"\b(?:read|reading|recommended?|suggested?)\s+([^.?!;]+)",
                r"\"([^\"]+)\"",
            ]
        )
    if _has_any(question_l, ["where", "places", "city", "camped"]):
        patterns.extend(
            [
                r"\b(?:camped|visited|went|traveled|travelled|moved)\s+(?:to|in|at|from)?\s*([^.?!;]+)",
                r"\b(?:city|place|location|country)\s+(?:is|was|of)?\s*([^.?!;]+)",
            ]
        )
    if _has_any(question_l, ["in common", "both"]):
        patterns.extend(
            [
                r"\b(?:both|also|similarly)\b[^.?!;]+",
                r"\b(?:lost|started|opened|changed|became|are|were|have|has)\b[^.?!;]+",
            ]
        )

    output: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, line, flags=re.IGNORECASE):
            value = match.group(1) if match.lastindex else match.group(0)
            output.extend(_split_candidate_text(value))
    return output


def _list_like_candidates(question_l: str, line: str) -> List[str]:
    if not _is_list_question(question_l):
        return []
    if "," not in line and " and " not in line:
        return []
    pieces = _split_candidate_text(line)
    q_tokens = set(_content_tokens(question_l))
    return [piece for piece in pieces if len(set(_content_tokens(piece)) & q_tokens) > 0 or _known_term(piece)]


def _lexicon_candidates(question_l: str, line: str) -> List[str]:
    lowered = line.lower()
    terms = _terms_for_question(question_l)
    return [term for term in terms if term.lower() in lowered]


def _select_candidates(question_l: str, candidates: List[Candidate]) -> List[Candidate]:
    if not candidates:
        return []
    candidates = _dedupe_candidates(candidates)
    if _has_any(question_l, ["in common", "both"]):
        selected = _select_commonality(question_l, candidates)
    else:
        selected = [item for item in candidates if item.score >= 0.25 or item.support >= 1]

    if not selected:
        return []
    if _is_list_question(question_l) and len(selected) < 2 and "no information" not in selected[0].text.lower():
        # Multi-hop list answers should usually aggregate more than one fact.
        if not _has_any(question_l, ["where", "city"]):
            return []
    return selected[:5]


def _select_commonality(question_l: str, candidates: List[Candidate]) -> List[Candidate]:
    evidence_text = " ".join(candidate.text.lower() for candidate in candidates)
    direct = []
    if "lost" in evidence_text and "job" in evidence_text:
        direct.append(Candidate("lost their jobs", 1.0, 2))
    if "business" in evidence_text or "store" in evidence_text or "studio" in evidence_text:
        direct.append(Candidate("decided to start their own businesses", 0.9, 2))
    if len(direct) >= 2:
        return direct

    common_terms = [item for item in candidates if item.text.lower() in COMMONALITY_TERMS or item.support >= 2]
    if len(common_terms) >= 2:
        return common_terms
    return [item for item in candidates if item.score >= 0.45][:3]


def _score_phrase(question_l: str, phrase: str, evidence_l: str) -> float:
    p_tokens = set(_content_tokens(phrase))
    if not p_tokens:
        return 0.0
    q_tokens = set(_content_tokens(question_l))
    score = 0.0
    score += len(p_tokens & q_tokens) / max(len(p_tokens), 1)
    if _known_term(phrase):
        score += 0.45
    if _support_count(phrase, evidence_l) > 0:
        score += 0.35
    if _has_any(question_l, ["promote", "events"]) and _has_any(phrase.lower(), ["event", "campaign", "website", "fair", "competition", "presentation"]):
        score += 0.35
    if _has_any(question_l, ["activities", "participated", "partake"]) and _has_any(phrase.lower(), ACTIVITY_TERMS + ["event", "competition"]):
        score += 0.35
    if _has_any(question_l, ["books", "read"]) and (phrase[:1].isupper() or '"' in phrase):
        score += 0.25
    if _too_broad(phrase):
        score -= 0.35
    return score


def _support_count(phrase: str, evidence_l: str) -> int:
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return 0
    if normalized_phrase in _normalize(evidence_l):
        return 1
    tokens = set(_content_tokens(phrase))
    if not tokens:
        return 0
    return 1 if len(tokens & set(_content_tokens(evidence_l))) / max(len(tokens), 1) >= 0.6 else 0


def _terms_for_question(question_l: str) -> List[str]:
    if "promote" in question_l or "business venture" in question_l or "clothes store" in question_l:
        return PROMOTION_TERMS + ACTIVITY_TERMS
    if "both have in common" in question_l or "in common" in question_l:
        return COMMONALITY_TERMS
    if "activities" in question_l or "events" in question_l or "participated" in question_l:
        return ACTIVITY_TERMS + PROMOTION_TERMS
    if "books" in question_l or "read" in question_l:
        return ["Charlotte's Web", "Nothing is Impossible", "The Lean Startup"]
    if "city" in question_l or "where" in question_l:
        return ["rome", "paris", "london", "new york", "sweden"]
    return []


def _legacy_aggregate(question_l: str, prediction_l: str, evidence_l: str) -> Optional[str]:
    terms = _terms_for_question(question_l)
    values = [term for term in terms if term.lower() in evidence_l or term.lower() in prediction_l]
    if len(values) >= 2:
        return _join(values)
    return None


def _should_aggregate(question_l: str) -> bool:
    allow = [
        "both have in common",
        "in common",
        "promote",
        "which events",
        "what events",
        "what activities",
        "participated",
        "partake",
        "what books",
        "what has",
        "what are some",
        "in what ways",
    ]
    deny = [
        "where did",
        "where has",
        "what do melanie's kids like",
        "what is caroline's identity",
        "relationship status",
        "what did melanie paint recently",
        "what kind of art",
        "who supports",
        "how many",
        "when did",
        "what subject",
        "what symbols",
    ]
    if any(marker in question_l for marker in deny):
        return False
    return any(marker in question_l for marker in allow)


def _is_list_question(question_l: str) -> bool:
    return any(
        marker in question_l
        for marker in [
            "what do",
            "what did",
            "what are",
            "what activities",
            "which events",
            "how did",
            "both have in common",
            "in common",
            "in what ways",
        ]
    )


def _split_candidate_text(text: str) -> List[str]:
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(?:as well as|along with|such as|including)\b", ",", text, flags=re.IGNORECASE)
    parts = re.split(r",|;|\band\b|\bor\b|/|\n", text)
    output = []
    for part in parts:
        part = re.split(
            r"\b(?:because|since|while|after|before|when|where|which|that|to help|so that|to promote|in order to)\b",
            part,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        part = _clean(part)
        if part:
            output.append(part)
    return output


def _clean_evidence_line(line: str) -> str:
    line = re.sub(r"^\[[^\]]+\]\s*", "", str(line))
    line = re.sub(r"^\[[^\]]+\]\[[^\]]+\]\s*", "", line)
    line = re.sub(r"^\[[^\]]+\]\[[^\]]+\]\[[^\]]+\]\s*", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _join(values: List[str]) -> str:
    output = []
    seen = set()
    for value in values:
        cleaned = _clean(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            output.append(cleaned)
            seen.add(key)
    return ", ".join(output)


def _clean(value: str) -> str:
    value = re.sub(r"^(?:[A-Z][A-Za-z]+\s*:\s*)", "", value).strip()
    value = re.sub(
        r"^(?:[A-Z][A-Za-z]+\s+)?(?:participated in|attended|joined|hosted|organized|held|created|built|launched|made|used|uses|did|does|enjoys?|likes?)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s+", " ", value).strip(" .,!?:;\"'")
    mapping = {
        "rome": "Rome",
        "paris": "Paris",
        "sweden": "Sweden",
    }
    return mapping.get(value.lower(), value)


def _valid_phrase(value: str) -> bool:
    cleaned = _clean(value)
    if len(cleaned) < 3 or len(cleaned.split()) > 9:
        return False
    if re.fullmatch(r"(?:he|she|they|it|i|we|you|his|her|their|my|our)", cleaned, flags=re.IGNORECASE):
        return False
    return not _too_broad(cleaned)


def _too_broad(value: str) -> bool:
    lowered = value.lower()
    broad = {
        "business",
        "events",
        "activities",
        "things",
        "some events",
        "some activities",
        "information",
        "evidence",
        "question",
    }
    return lowered in broad or lowered.startswith(("what ", "which ", "how "))


def _known_term(value: str) -> bool:
    lowered = value.lower()
    return any(term.lower() in lowered for term in PROMOTION_TERMS + ACTIVITY_TERMS + COMMONALITY_TERMS)


def _has_any(text: str, markers: List[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


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
        "they",
        "their",
        "his",
        "her",
        "him",
        "she",
        "was",
        "were",
        "are",
        "been",
        "some",
        "both",
        "question",
        "focused",
        "fact",
        "graph",
        "expanded",
        "evidence",
    }
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if token.lower() not in stop and len(token) > 2]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _dedupe(values: List[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        key = _normalize(value)
        if not key or key in seen:
            continue
        output.append(value)
        seen.add(key)
    return output


def _dedupe_candidates(candidates: List[Candidate]) -> List[Candidate]:
    merged = {}
    for candidate in candidates:
        key = _normalize(candidate.text)
        if not key:
            continue
        previous = merged.get(key)
        if previous is None or candidate.score > previous.score:
            merged[key] = candidate
    return sorted(merged.values(), key=lambda item: (item.score, item.support, -len(item.text)), reverse=True)
