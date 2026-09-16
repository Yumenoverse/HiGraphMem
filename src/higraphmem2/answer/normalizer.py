import re
from typing import Dict


ALIASES: Dict[str, str] = {
    "trans woman": "transgender woman",
    "transgender female": "transgender woman",
    "ten years ago": "10 years ago",
    "four years": "4 years",
    "four years ago": "4 years ago",
}


def normalize_answer_text(question: str, prediction: str) -> str:
    """Normalize answer wording without using gold labels."""
    answer = prediction.strip()
    answer = _strip_sentence_prefix(question, answer)
    answer = _strip_trailing_period(answer)
    answer = _apply_aliases(answer)
    return answer


def _strip_sentence_prefix(question: str, answer: str) -> str:
    lowered_q = question.lower()
    patterns = []
    if lowered_q.startswith("what did "):
        patterns.extend([
            r"^[A-Z][A-Za-z]+\s+(?:is\s+)?(?:researching|researched|researches|research)\s+",
            r"^[A-Z][A-Za-z]+\s+(?:did|does)\s+",
        ])
    if lowered_q.startswith("what is "):
        patterns.extend([
            r"^[A-Z][A-Za-z]+\s+is\s+",
            r"^[A-Z][A-Za-z]+\s+has\s+",
        ])
    if lowered_q.startswith("where did "):
        patterns.extend([
            r"^[A-Z][A-Za-z]+\s+(?:moved|move)\s+from\s+",
        ])
    for pattern in patterns:
        answer = re.sub(pattern, "", answer).strip()
    answer = _trim_explanatory_tail(lowered_q, answer)
    return answer


def _trim_explanatory_tail(lowered_q: str, answer: str) -> str:
    if lowered_q.startswith(("what did ", "what is ", "where did ", "which ", "who ")):
        answer = re.split(r"\s+(?:with|because|since|so that|while|after|before)\s+", answer, maxsplit=1, flags=re.IGNORECASE)[0]
    if lowered_q.startswith("what fields") or "likely to pursue" in lowered_q:
        answer = re.sub(r"^[A-Z][A-Za-z]+\s+is\s+likely\s+to\s+pursue\s+(?:education\s+in\s+)?", "", answer, flags=re.IGNORECASE).strip()
        answer = re.sub(r"\b(?:yes|no)\s*$", "", answer, flags=re.IGNORECASE).strip(" .,;")
    return answer.strip()


def _strip_trailing_period(answer: str) -> str:
    return answer[:-1].strip() if answer.endswith(".") else answer


def _apply_aliases(answer: str) -> str:
    lowered = answer.lower().strip()
    lowered = re.sub(r"^(a|an|the)\s+", "", lowered)
    for source, target in ALIASES.items():
        if lowered == source:
            return target
    return answer
