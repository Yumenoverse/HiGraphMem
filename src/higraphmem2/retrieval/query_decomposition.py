import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DecomposedQuery:
    subquestion: str
    reason: str


def decompose_question(question: str, query_type: str, max_subquestions: int = 4) -> List[DecomposedQuery]:
    """Create retrieval-only subquestions without using answers or labels."""
    lowered = question.lower()
    entities = _entities(question)
    subquestions: List[DecomposedQuery] = []

    if query_type == "multi_hop" or "both" in lowered or " in common" in lowered:
        if len(entities) >= 2:
            topic = _topic_phrase(question)
            for entity in entities[:3]:
                subquestions.append(DecomposedQuery("What is known about %s related to %s?" % (entity, topic), "entity_topic"))
            if "both" in lowered or "in common" in lowered:
                subquestions.append(DecomposedQuery("What shared events, traits, or experiences are mentioned for %s?" % " and ".join(entities[:2]), "commonality"))

    if any(marker in lowered for marker in ["which events", "what activities", "in what ways", "how did"]):
        topic = _topic_phrase(question)
        if entities:
            for entity in entities[:2]:
                subquestions.append(DecomposedQuery("Which events or activities did %s have for %s?" % (entity, topic), "list_expansion"))
        else:
            subquestions.append(DecomposedQuery("Which specific events or activities are mentioned for %s?" % topic, "list_expansion"))

    if query_type == "open_domain":
        topic = _topic_phrase(question)
        if entities:
            entity = entities[0]
            subquestions.extend(
                [
                    DecomposedQuery("What preferences, goals, constraints, and habits are known about %s related to %s?" % (entity, topic), "belief_state"),
                    DecomposedQuery("What evidence supports or contradicts an inference about %s and %s?" % (entity, topic), "support_contrast"),
                ]
            )
        subquestions.append(DecomposedQuery("What facts are most relevant to this inference: %s" % question, "inference_support"))

    return _dedupe(subquestions)[:max_subquestions]


def format_decomposition(subqueries: List[DecomposedQuery]) -> str:
    if not subqueries:
        return ""
    lines = ["Query decomposition:"]
    for idx, subquery in enumerate(subqueries, start=1):
        lines.append("%d. [%s] %s" % (idx, subquery.reason, subquery.subquestion))
    return "\n".join(lines)


def _entities(text: str) -> List[str]:
    stop = {"What", "When", "Where", "Which", "Who", "Why", "How", "Did", "Do", "Does", "In", "Is", "Would", "Based", "Around"}
    output = []
    seen = set()
    for match in re.finditer(r"\b[A-Z][a-z]+\b", text):
        entity = match.group(0)
        if entity in stop or entity.lower() in seen:
            continue
        output.append(entity)
        seen.add(entity.lower())
    return output


def _topic_phrase(question: str) -> str:
    lowered = question.lower()
    replacements = [
        (r"^what do .+ both have in common\??$", "shared experiences"),
        (r"^which events has .+ participated in to promote (.+)\??$", r"\1"),
        (r"^how did .+ promote (.+)\??$", r"\1"),
        (r"^what activities has .+ done (?:with|for) (.+)\??$", r"\1"),
    ]
    for pattern, replacement in replacements:
        rewritten = re.sub(pattern, replacement, lowered).strip(" ?")
        if rewritten != lowered.strip(" ?"):
            return rewritten
    tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", lowered) if token not in _STOP]
    return " ".join(tokens[:8]) if tokens else "the question"


def _dedupe(subqueries: List[DecomposedQuery]) -> List[DecomposedQuery]:
    output = []
    seen = set()
    for subquery in subqueries:
        key = subquery.subquestion.lower()
        if key in seen:
            continue
        output.append(subquery)
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
    "did",
    "does",
    "have",
    "has",
    "the",
    "and",
    "for",
    "with",
    "both",
    "would",
    "likely",
    "might",
}
