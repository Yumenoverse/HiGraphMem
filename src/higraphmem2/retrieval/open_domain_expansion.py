import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class OpenDomainExpansion:
    query: str
    reason: str


def expand_open_domain_query(question: str, max_queries: int = 5) -> List[OpenDomainExpansion]:
    lowered = question.lower()
    entities = _entities(question)
    subject = entities[0] if entities else _subject_hint(question)
    expansions: List[OpenDomainExpansion] = []

    def add(query: str, reason: str) -> None:
        if query.strip():
            expansions.append(OpenDomainExpansion(query.strip(), reason))

    if any(marker in lowered for marker in ["political leaning", "religious", "considered religious"]):
        add("%s values beliefs identity community rights support activism" % subject, "belief_values")
    if any(marker in lowered for marker in ["financial status", "wealth", "middle-class", "money"]):
        add("%s job business income money travel lifestyle purchases" % subject, "financial_clues")
    if any(marker in lowered for marker in ["degree", "education", "fields", "career", "job might", "alternative career"]):
        add("%s education work career goals skills volunteering interests" % subject, "career_education")
    if any(marker in lowered for marker in ["personality", "traits", "attributes"]):
        add("%s behavior support goals emotions values friendship" % subject, "trait_evidence")
    if any(marker in lowered for marker in ["allerg", "underlying condition", "health problems", "discomfort"]):
        add("%s allergies symptoms health breathing pets discomfort" % subject, "health_condition")
    if any(marker in lowered for marker in ["console", "game", "board game", "card game"]):
        add("%s games gaming console cards board deck title play owns" % subject, "game_console")
    if any(marker in lowered for marker in ["country", "state", "city", "national park", "locations"]):
        add("%s travel visited trip city country state location park plans" % subject, "location_clues")
    if any(marker in lowered for marker in ["shop", "company", "organization", "charity", "endorsement"]):
        add("%s interests brands organizations charity company shop support" % subject, "organization_brand")
    if any(marker in lowered for marker in ["book", "composer", "music", "yoga", "technique", "holiday"]):
        add("%s books music hobbies practice technique interests recommendations" % subject, "cultural_knowledge_clues")
    if any(marker in lowered for marker in ["would enjoy", "might enjoy", "could do", "good hobby", "benefit from"]):
        add("%s preferences hobbies goals constraints interests plans" % subject, "preference_inference")
    if "yes or no" in lowered or lowered.startswith(("would ", "does ", "did ", "is it likely", "was ")):
        add("%s relevant events evidence support contradiction" % subject, "yes_no_support")

    topic = _topic_terms(question)
    if topic:
        add("%s %s related facts evidence" % (subject, topic), "topic_terms")
    add("facts relevant to inference %s" % question, "original_inference")
    return _dedupe(expansions)[:max_queries]


def format_open_domain_expansions(expansions: List[OpenDomainExpansion]) -> str:
    if not expansions:
        return ""
    lines = ["Open-domain query expansion:"]
    for idx, item in enumerate(expansions, start=1):
        lines.append("%d. [%s] %s" % (idx, item.reason, item.query))
    return "\n".join(lines)


def _entities(text: str) -> List[str]:
    stop = {"What", "When", "Where", "Which", "Who", "Why", "How", "Did", "Do", "Does", "Is", "Would", "Could", "Based", "Considering", "In", "Around"}
    output = []
    seen = set()
    for match in re.finditer(r"\b[A-Z][a-z]+\b", text):
        value = match.group(0)
        if value in stop or value.lower() in seen:
            continue
        output.append(value)
        seen.add(value.lower())
    return output


def _subject_hint(question: str) -> str:
    match = re.search(r"\b(?:for|about|to|with|does|did|would|could|might)\s+([A-Z][a-z]+)\b", question)
    if match:
        return match.group(1)
    return "person"


def _topic_terms(question: str) -> str:
    lowered = question.lower()
    tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", lowered) if token not in _STOP and len(token) > 2]
    return " ".join(tokens[:10])


def _dedupe(expansions: List[OpenDomainExpansion]) -> List[OpenDomainExpansion]:
    output = []
    seen = set()
    for item in expansions:
        key = re.sub(r"\s+", " ", item.query.lower()).strip()
        if not key or key in seen:
            continue
        output.append(item)
        seen.add(key)
    return output


_STOP = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "does",
    "did",
    "would",
    "could",
    "might",
    "likely",
    "based",
    "considering",
    "their",
    "with",
    "from",
    "about",
    "answer",
    "yes",
    "no",
}
