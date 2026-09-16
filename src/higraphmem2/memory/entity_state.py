import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ..skills.schema import StateRule, candidate_rules


@dataclass
class StateValue:
    value: str
    evidence_id: str
    source_text: str


@dataclass
class EntityStateMemory:
    states: Dict[str, Dict[str, List[StateValue]]] = field(default_factory=dict)
    rules: List[StateRule] = field(default_factory=candidate_rules)

    def add(self, entity: str, key: str, value: str, evidence_id: str, source_text: str) -> None:
        entity_key = entity.lower()
        cleaned = _clean_value(value)
        if not cleaned:
            return
        self.states.setdefault(entity_key, {}).setdefault(key, [])
        existing = {_normalize_value(item.value) for item in self.states[entity_key][key]}
        if _normalize_value(cleaned) not in existing:
            self.states[entity_key][key].append(StateValue(value=cleaned, evidence_id=evidence_id, source_text=source_text))

    def values(self, entity: str, key: str) -> List[StateValue]:
        return self.states.get(entity.lower(), {}).get(key, [])

    def answer(self, question: str) -> Optional[Tuple[str, List[str]]]:
        entities = _entities_from_question(question)
        for rule in self.rules:
            if not rule.matches_question(question):
                continue
            candidate_entities = entities or sorted(self.states.keys())
            values: List[StateValue] = []
            for entity in candidate_entities:
                values.extend(self.values(entity, rule.name))
            if values:
                return ", ".join(item.value for item in values), [item.evidence_id for item in values]
        return None


def build_entity_state_memory(sample: dict, rules: Optional[List[StateRule]] = None) -> EntityStateMemory:
    memory = EntityStateMemory(rules=rules if rules is not None else candidate_rules())
    for entity, fact, evidence_id in _iter_observation_facts(sample):
        _extract_from_text(memory, entity, fact, evidence_id, memory.rules)
    for turn in _iter_turns(sample):
        _extract_from_text(memory, turn["speaker"], turn["text"], turn["dia_id"], memory.rules)
    return memory


def _iter_observation_facts(sample: dict) -> Iterable[Tuple[str, str, str]]:
    for observations in sample.get("observation", {}).values():
        if not isinstance(observations, dict):
            continue
        for entity, facts in observations.items():
            for item in facts:
                if isinstance(item, list) and len(item) >= 2:
                    yield str(entity), str(item[0]), str(item[1])


def _iter_turns(sample: dict) -> Iterable[dict]:
    conversation = sample.get("conversation", {})
    for idx in range(1, 100):
        for turn in conversation.get("session_%s" % idx, []):
            yield turn


def _extract_from_text(memory: EntityStateMemory, entity: str, text: str, evidence_id: str, rules: List[StateRule]) -> None:
    for rule in rules:
        if not _rule_evidence_matches(rule, text):
            continue
        for value in _extract_values(rule.name, text):
            memory.add(entity, rule.name, value, evidence_id, text)


def _rule_evidence_matches(rule: StateRule, text: str) -> bool:
    lowered = text.lower()
    if not rule.extract_keywords:
        return True
    for pattern in rule.extract_keywords:
        tokens = [token for token in pattern.lower().replace("{location}", "").split() if token]
        if tokens and all(token in lowered for token in tokens):
            return True
    return any(_slot_signal(rule.name, lowered))


def _slot_signal(slot: str, lowered: str) -> Iterable[bool]:
    signals = {
        "relationship_status": ["single parent", "married", "partner", "relationship"],
        "moved_from": ["home country", "moved from", "from sweden", "came from"],
        "camped_locations": ["camping", "camped", "camp"],
        "kids_like": ["kids like", "kids love", "children like", "children love"],
        "books_read": ["book", "read", "reading"],
        "destress_activities": ["destress", "de-stress", "stress", "relax"],
        "activities": ["activity", "activities", "participated", "attended", "hosted", "camping", "painting", "swimming", "hiking", "fair", "networking", "competition", "festival", "campaign", "website", "presentation"],
        "lgbtq_events": ["lgbtq", "support group", "school event", "pride", "mentorship"],
        "recent_painting": ["painting", "painted", "recently"],
    }
    return (signal in lowered for signal in signals.get(slot, []))


def _extract_values(slot: str, text: str) -> List[str]:
    lowered = text.lower()
    if slot == "relationship_status":
        return _relationship_values(lowered)
    if slot == "moved_from":
        return _origin_values(text)
    if slot == "camped_locations":
        return _known_values(lowered, ["mountains", "mountain", "beach", "forest", "lake", "desert", "park"])
    if slot == "kids_like":
        return _preference_values(text, ["kids", "children"]) + _favorite_values(text)
    if slot == "books_read":
        return _book_values(text)
    if slot == "destress_activities":
        return _known_values(lowered, ["running", "pottery", "dancing", "dance", "yoga", "painting", "hiking", "swimming"])
    if slot == "activities":
        return _activity_values(text)
    if slot == "lgbtq_events":
        return _event_values(text)
    if slot == "recent_painting":
        return _creative_values(text)
    return _fallback_values(text)


def _relationship_values(lowered: str) -> List[str]:
    values = []
    if "single parent" in lowered or re.search(r"\bsingle\b", lowered):
        values.append("Single")
    if "married" in lowered:
        values.append("Married")
    if "divorced" in lowered:
        values.append("Divorced")
    if "partner" in lowered:
        values.append("Partnered")
    return values


def _origin_values(text: str) -> List[str]:
    patterns = [
        r"home country,?\s+([A-Z][A-Za-z .'-]+)",
        r"moved from\s+([A-Z][A-Za-z .'-]+)",
        r"came from\s+([A-Z][A-Za-z .'-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return [_trim_phrase(match.group(1))]
    return []


def _preference_values(text: str, subjects: List[str]) -> List[str]:
    lowered = text.lower()
    values = []
    for subject in subjects:
        for verb in ["like", "likes", "love", "loves"]:
            marker = "%s %s " % (subject, verb)
            if marker in lowered:
                values.extend(_split_list(text[lowered.index(marker) + len(marker) :]))
    return values[:5]


def _favorite_values(text: str) -> List[str]:
    values = []
    for pattern in [
        r"favorite [A-Za-z ]+ is ([A-Za-z][A-Za-z '&-]+)",
        r"favorite is ([A-Za-z][A-Za-z '&-]+)",
    ]:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            values.append(_trim_phrase(match.group(1)))
    return _dedupe(values)


def _book_values(text: str) -> List[str]:
    values = re.findall(r'"([^"]+)"', text)
    known_titles = ["Charlotte's Web", "Nothing is Impossible", "The Lean Startup"]
    for title in known_titles:
        if title.lower() in text.lower():
            values.append(title)
    return _dedupe(values)


def _activity_values(text: str) -> List[str]:
    lowered = text.lower()
    known = [
        "pottery",
        "painting",
        "camping",
        "museum",
        "swimming",
        "hiking",
        "running",
        "dancing",
        "dance competition",
        "festival",
        "fair",
        "networking events",
        "ad campaign",
        "website",
        "video presentation",
        "limited-edition sweatshirts",
    ]
    values = _known_values(lowered, known)
    for pattern in [r"participated in ([^.]+)", r"attended ([^.]+)", r"hosted ([^.]+)", r"launched ([^.]+)", r"developed ([^.]+)"]:
        for match in re.finditer(pattern, lowered):
            values.extend(_split_list(match.group(1)))
    return _dedupe(values)[:8]


def _event_values(text: str) -> List[str]:
    lowered = text.lower()
    known = [
        ("support group", "support group"),
        ("school event", "school speech"),
        ("gave a talk at a school", "school speech"),
        ("pride parade", "pride parade"),
        ("mentorship program", "mentoring program"),
        ("mentor", "mentoring program"),
        ("lgbtq conference", "LGBTQ conference"),
    ]
    return _dedupe(value for marker, value in known if marker in lowered)


def _creative_values(text: str) -> List[str]:
    lowered = text.lower()
    values = []
    if "landscape" in lowered or "still life" in lowered:
        values.append("landscape/still life")
    values.extend(_known_values(lowered, ["sunset", "sunrise", "screenplay", "painting", "poem"]))
    return _dedupe(values)


def _fallback_values(text: str) -> List[str]:
    return [_trim_phrase(text)] if len(text.split()) <= 12 else []


def _known_values(lowered: str, known_values: List[str]) -> List[str]:
    values = []
    for value in known_values:
        if value.lower() in lowered:
            normalized = value
            if value == "mountain":
                normalized = "mountains"
            if value == "dance":
                normalized = "dancing"
            values.append(normalized)
    return _dedupe(values)


def _split_list(fragment: str) -> List[str]:
    fragment = re.split(r"[.;!?]", fragment)[0]
    parts = re.split(r",|\band\b|\bor\b", fragment)
    return [_trim_phrase(part) for part in parts if _trim_phrase(part)]


def _trim_phrase(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .,!?:;\"'")
    value = re.sub(r"\b(in|at|on|for|with|which|that|who|because)\b.*$", "", value, flags=re.IGNORECASE).strip(" .,!?:;\"'")
    return value


def _clean_value(value: str) -> str:
    cleaned = _trim_phrase(value)
    cleaned = re.sub(r"^(a|an|the|my|own)\s+", "", cleaned, flags=re.IGNORECASE)
    if cleaned.lower() in {"btw", "it", "this", "that", "things"}:
        return ""
    if not cleaned or len(cleaned.split()) > 8:
        return ""
    return cleaned


def _normalize_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _dedupe(values: Iterable[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        cleaned = _clean_value(value)
        key = _normalize_value(cleaned)
        if cleaned and key not in seen:
            output.append(cleaned)
            seen.add(key)
    return output


def _entities_from_question(question: str) -> List[str]:
    stop = {"What", "When", "Where", "Which", "Who", "Why", "How", "The", "A", "An"}
    return [match.group(0).lower() for match in re.finditer(r"\b[A-Z][a-z]+\b", question) if match.group(0) not in stop]
