import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class VerificationResult:
    answer: str
    supported: bool
    source: str
    support_score: float


def verify_answer(question: str, answer: str, evidence_lines: Iterable[str], answer_mode: str) -> VerificationResult:
    """Conservative evidence verifier. It only edits answers when support is clearly improved."""
    evidence = "\n".join(evidence_lines)
    support = support_score(question, answer, evidence)
    cleaned = _clean_answer(question, answer, answer_mode, evidence)
    cleaned_support = support_score(question, cleaned, evidence)

    if cleaned and cleaned != answer and cleaned_support >= max(support, 0.2):
        return VerificationResult(cleaned, True, "verified_cleanup", cleaned_support)

    if _is_no_information(answer) and evidence:
        candidate = _candidate_from_evidence(question, evidence, answer_mode)
        if candidate:
            return VerificationResult(candidate, True, "verified_evidence_candidate", support_score(question, candidate, evidence))

    return VerificationResult(answer, support >= 0.12 or _is_no_information(answer), "unchanged", support)


def support_score(question: str, answer: str, evidence: str) -> float:
    answer_tokens = set(_tokens(answer))
    evidence_tokens = set(_tokens(evidence))
    if not answer_tokens:
        return 0.0
    literal = 1.0 if _normalize(answer) and _normalize(answer) in _normalize(evidence) else 0.0
    token_support = len(answer_tokens & evidence_tokens) / max(len(answer_tokens), 1)
    query_overlap = len(set(_tokens(question)) & evidence_tokens) / max(len(set(_tokens(question))), 1)
    return max(literal, 0.8 * token_support + 0.2 * query_overlap)


def _clean_answer(question: str, answer: str, answer_mode: str, evidence: str) -> str:
    cleaned = re.sub(r"\s+", " ", answer).strip(" .,!?:;\"'")
    lowered_q = question.lower()
    if lowered_q.startswith("what did "):
        cleaned = re.sub(r"^[A-Z][A-Za-z]+\s+(?:is\s+)?(?:researching|researched|researches|research)\s+", "", cleaned).strip()
    if lowered_q.startswith("what is "):
        cleaned = re.sub(r"^[A-Z][A-Za-z]+\s+(?:is|has)\s+", "", cleaned).strip()
    if lowered_q.startswith("where did "):
        cleaned = re.sub(r"^[A-Z][A-Za-z]+\s+(?:moved|move)\s+from\s+", "", cleaned).strip()
    if answer_mode in {"span", "date"} or lowered_q.startswith(("what did", "what is", "where did", "who ")):
        cleaned = re.split(r"\s+(?:because|since|with the dream of|while|after learning|so that)\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    if answer_mode == "list":
        cleaned = _clean_list_answer(cleaned, evidence)
    if answer_mode == "inference":
        cleaned = re.sub(r"\b(?:based on the evidence|the evidence suggests that)\b,?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.strip(" .,!?:;\"'")


def _clean_list_answer(answer: str, evidence: str) -> str:
    parts = [part.strip(" .,!?:;\"'") for part in re.split(r",|\band\b", answer) if part.strip(" .,!?:;\"'")]
    if len(parts) <= 1:
        return answer
    evidence_l = evidence.lower()
    kept = []
    seen = set()
    for part in parts:
        key = _normalize(part)
        if key in seen:
            continue
        if key in _normalize(evidence_l) or any(token in evidence_l for token in _tokens(part)):
            kept.append(part)
            seen.add(key)
    return ", ".join(kept) if kept else answer


def _candidate_from_evidence(question: str, evidence: str, answer_mode: str) -> Optional[str]:
    if answer_mode == "date":
        match = re.search(r"\b(?:\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{4}|\d+\s+years?)\b", evidence)
        if match:
            return match.group(0)
    if answer_mode == "list":
        quoted = re.findall(r'"([^"]+)"', evidence)
        if quoted:
            return ", ".join(quoted[:4])
    if question.lower().startswith("who "):
        names = re.findall(r"\b[A-Z][a-z]+\b", evidence)
        stop = {"Question", "Graph", "On", "Memory", "Evidence"}
        names = [name for name in names if name not in stop]
        if names:
            return ", ".join(_dedupe(names)[:3])
    return None


def _is_no_information(answer: str) -> bool:
    return "no information available" in answer.lower()


def _tokens(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 1]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _dedupe(values: List[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        output.append(value)
        seen.add(key)
    return output
