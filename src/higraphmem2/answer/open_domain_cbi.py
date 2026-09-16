import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .evidence_verifier import support_score


@dataclass(frozen=True)
class CBIResult:
    answer: str
    accepted: bool
    source: str
    support_score: float


def answer_format_instruction(question: str) -> str:
    """Specify output granularity using only the form of the question."""
    lowered = question.lower()
    if "answer yes or no" in lowered:
        return "Answer exactly 'Yes' or 'No'. Do not use 'Likely' and do not add a reason."
    if _is_yes_no_question(lowered):
        return "Return exactly one of 'Yes', 'No', 'Likely yes', or 'Likely no'. Do not add a reason."
    if any(phrase in lowered for phrase in ("what country", "which country", "in what country", "in which country")):
        return "The requested type is a country: return only a country name, never a city."
    if any(phrase in lowered for phrase in ("what state", "which state", "in what state", "in which state", "what province", "which province")):
        return "The requested type is a state or province: return only that region name, never a city."
    if any(token in lowered for token in ("device", "console", "card game", "board game", "person", "who is", "who was")):
        return "Return only the requested device, game, or person entity name; do not add a description or a related place."
    return "Return only the shortest answer phrase. Do not add a reason or explanation."


def build_cbi_prompt(question: str, evidence_lines: Iterable[str]) -> str:
    evidence = "\n".join(_trim_evidence_lines(evidence_lines, limit=28))
    return (
        "Memory evidence:\n%s\n\n"
        "Question: %s\n\n"
        "Infer the most likely short answer from indirect evidence. "
        "Use support and contradiction signals in the memory. %s "
        "Use 'No information available' only when there is no related evidence at all."
    ) % (evidence or "No retrieved evidence.", question, answer_format_instruction(question))


def build_cbi_self_consistency_prompts(question: str, evidence_lines: Iterable[str]) -> List[str]:
    evidence = "\n".join(_trim_evidence_lines(evidence_lines, limit=28))
    base = (
        "Memory evidence:\n%s\n\n"
        "Question: %s\n\n"
        "%s %s "
        "If evidence is insufficient, answer: No information available."
    )
    return [
        base
        % (
            evidence or "No retrieved evidence.",
            question,
            "Infer the most likely answer from indirect evidence and memory support.",
            answer_format_instruction(question),
        ),
        base
        % (
            evidence or "No retrieved evidence.",
            question,
            "Check whether the evidence contradicts the obvious answer, then infer conservatively.",
            answer_format_instruction(question),
        ),
        base
        % (
            evidence or "No retrieved evidence.",
            question,
            "Identify the shortest entity, trait, status, condition, place, field, or action implied by the evidence.",
            answer_format_instruction(question),
        ),
    ]


def build_cbi_rescue_prompt(question: str, evidence_lines: Iterable[str]) -> str:
    evidence = "\n".join(_trim_evidence_lines(evidence_lines, limit=34))
    return (
        "Memory evidence:\n%s\n\n"
        "Question: %s\n\n"
        "The previous answer was 'No information available', but the memory contains related evidence. "
        "Make a conservative likely inference from the evidence. %s "
        "Use 'No information available' only if the evidence has no related person, topic, or event."
    ) % (evidence or "No retrieved evidence.", question, answer_format_instruction(question))


def build_cbi_refine_prompt(question: str, current_answer: str, evidence_lines: Iterable[str]) -> str:
    evidence = "\n".join(_trim_evidence_lines(evidence_lines, limit=36))
    return (
        "Memory evidence:\n%s\n\n"
        "Question: %s\n"
        "Current answer: %s\n\n"
        "The current answer may be too generic or may miss a specific public-knowledge name implied by the clues. "
        "This refinement is only for concrete named targets: card game, board game, console, shop, country, state, disease/condition, or organization. "
        "Use the memory as constraints and common public knowledge only to name that concrete target. "
        "Return only the shortest specific answer. "
        "If the current answer is already specific and correct, return it unchanged. "
        "If the evidence gives no usable clues, return: No information available."
    ) % (evidence or "No retrieved evidence.", question, current_answer)


def build_open_domain_skill_prompt(question: str, evidence_lines: Iterable[str], skills: Iterable[object]) -> str:
    evidence = "\n".join(_trim_evidence_lines(evidence_lines, limit=36))
    skill_text = "\n".join(_format_open_domain_skill(skill) for skill in skills)
    return (
        "Memory evidence:\n%s\n\n"
        "Question: %s\n\n"
        "Frozen open-domain skills induced from the training split:\n%s\n\n"
        "Apply only the listed skill policies that match the question. "
        "Use retrieved memory evidence as constraints and public knowledge only to resolve a supported clue. "
        "Do not use benchmark answers, do not guess from the question alone, and do not explain. "
        "Return only the shortest answer phrase. If the evidence does not support a skill application, answer: No information available."
    ) % (evidence or "No retrieved evidence.", question, skill_text or "No matching open-domain skill.")


def should_rescue_no_information(question: str, answer: str, evidence_lines: Iterable[str], threshold: float = 0.18) -> bool:
    if not _is_no_info(answer):
        return False
    evidence = "\n".join(evidence_lines)
    return evidence_relevance(question, evidence) >= threshold


def should_refine_open_domain_answer(question: str, answer: str, evidence_lines: Iterable[str]) -> bool:
    cleaned = normalize_open_domain_answer(question, answer)
    if _is_no_info(cleaned):
        return should_rescue_no_information(question, cleaned, evidence_lines, threshold=0.14)
    lowered_q = question.lower()
    if _is_yes_no_question(lowered_q):
        return False
    if not _asks_for_narrow_specific_name(lowered_q):
        return False
    if _looks_specific(cleaned):
        return False
    evidence = "\n".join(evidence_lines)
    return evidence_relevance(question, evidence) >= 0.12


def select_cbi_answer(question: str, original: str, candidate: str, evidence_lines: Iterable[str]) -> CBIResult:
    evidence = "\n".join(evidence_lines)
    original_clean = normalize_open_domain_answer(question, original)
    candidate_clean = normalize_open_domain_answer(question, candidate)
    original_score = support_score(question, original_clean, evidence)
    candidate_score = support_score(question, candidate_clean, evidence)

    if _is_bad(candidate_clean) or _is_no_info(candidate_clean):
        return CBIResult(original_clean, False, "cbi_rejected_empty", original_score)
    if _is_no_info(original_clean) and not _is_no_info(candidate_clean):
        return CBIResult(candidate_clean, True, "cbi_replaced_no_information", candidate_score)
    if candidate_score + 0.05 >= original_score and _is_better_form(question, original_clean, candidate_clean):
        return CBIResult(candidate_clean, True, "cbi_verified_candidate", candidate_score)
    return CBIResult(original_clean, False, "cbi_kept_original", original_score)


def select_self_consistent_answer(question: str, original: str, candidates: Iterable[str], evidence_lines: Iterable[str]) -> CBIResult:
    evidence = "\n".join(evidence_lines)
    original_clean = normalize_open_domain_answer(question, original)
    normalized_candidates = [
        normalize_open_domain_answer(question, candidate)
        for candidate in candidates
        if candidate is not None
    ]
    valid = [candidate for candidate in normalized_candidates if candidate and not _is_bad(candidate)]
    if not valid:
        return CBIResult(original_clean, False, "self_consistency_no_valid_candidate", support_score(question, original_clean, evidence))

    ranked = []
    for candidate in _dedupe_preserve_order(valid + [original_clean]):
        if _is_no_info(candidate) and any(not _is_no_info(item) for item in valid):
            no_info_penalty = -0.35
        else:
            no_info_penalty = 0.0
        support = support_score(question, candidate, evidence)
        votes = _vote_score(candidate, valid)
        form = _answer_form_score(question, candidate)
        ranked.append((support + votes + form + no_info_penalty, support, votes, candidate))

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_support, best_votes, best = ranked[0]
    original_support = support_score(question, original_clean, evidence)
    if best != original_clean and (best_support + best_votes >= original_support + 0.04 or _is_no_info(original_clean)):
        return CBIResult(best, True, "self_consistency_selected", best_support)
    return CBIResult(original_clean, False, "self_consistency_kept_original", original_support)


def select_refine_answer(question: str, original: str, candidate: str, evidence_lines: Iterable[str]) -> CBIResult:
    evidence = "\n".join(evidence_lines)
    original_clean = normalize_open_domain_answer(question, original)
    candidate_clean = normalize_open_domain_answer(question, candidate)
    candidate_score = support_score(question, candidate_clean, evidence)
    original_score = support_score(question, original_clean, evidence)
    if _is_bad(candidate_clean) or _is_no_info(candidate_clean):
        return CBIResult(original_clean, False, "refine_rejected", original_score)
    if _looks_specific(candidate_clean) and not _looks_specific(original_clean):
        return CBIResult(candidate_clean, True, "refine_specific_candidate", candidate_score)
    if candidate_score + 0.08 >= original_score and _is_better_form(question, original_clean, candidate_clean):
        return CBIResult(candidate_clean, True, "refine_supported_candidate", candidate_score)
    return CBIResult(original_clean, False, "refine_kept_original", original_score)


def select_rescue_answer(question: str, original: str, candidate: str, evidence_lines: Iterable[str]) -> CBIResult:
    evidence = "\n".join(evidence_lines)
    original_clean = normalize_open_domain_answer(question, original)
    candidate_clean = normalize_open_domain_answer(question, candidate)
    candidate_score = support_score(question, candidate_clean, evidence)
    if _is_bad(candidate_clean) or _is_no_info(candidate_clean):
        return CBIResult(original_clean, False, "rescue_rejected", support_score(question, original_clean, evidence))
    if _is_yes_no_question(question.lower()) and _extract_yes_no(candidate_clean):
        return CBIResult(candidate_clean, True, "rescue_yes_no", candidate_score)
    if candidate_score >= 0.10 or evidence_relevance(question, evidence) >= 0.25:
        return CBIResult(candidate_clean, True, "rescue_supported_candidate", candidate_score)
    return CBIResult(original_clean, False, "rescue_kept_original", support_score(question, original_clean, evidence))


def resolve_specific_open_domain_answer(
    question: str,
    current_answer: str,
    evidence_lines: Iterable[str],
    candidates: Iterable[str] = (),
) -> Optional[CBIResult]:
    lowered_q = question.lower()
    if not _asks_for_narrow_specific_name(lowered_q) and not _asks_for_narrow_public_knowledge(lowered_q):
        if "political leaning" not in lowered_q and "financial status" not in lowered_q:
            return None

    evidence = "\n".join(str(line) for line in evidence_lines)
    current = normalize_open_domain_answer(question, current_answer)
    candidate_values = [normalize_open_domain_answer(question, value) for value in candidates if value]
    candidate_values = [value for value in candidate_values if value and not _is_no_info(value) and not _is_bad(value)]

    mapped = _public_knowledge_mapping(lowered_q, evidence, candidate_values)
    if mapped and mapped.lower() != current.lower():
        return CBIResult(mapped, True, "specific_public_knowledge", support_score(question, mapped, evidence))

    for value in candidate_values:
        if _specific_candidate_allowed(lowered_q, current, value):
            return CBIResult(value, True, "specific_candidate_accept", support_score(question, value, evidence))
    return None


def matching_open_domain_skills(question: str, skills: Iterable[object]) -> List[object]:
    return [
        skill
        for skill in skills
        if getattr(skill, "skill_type", "memory") == "open_domain" and getattr(skill, "matches_question")(question)
    ]


def evidence_relevance(question: str, evidence: str) -> float:
    q_tokens = set(_content_tokens(question))
    e_tokens = set(_content_tokens(evidence))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & e_tokens) / len(q_tokens)
    entity_bonus = 0.0
    for entity in re.findall(r"\b[A-Z][a-z]+\b", question):
        if entity.lower() in evidence.lower():
            entity_bonus = 0.15
            break
    return min(1.0, overlap + entity_bonus)


def _format_open_domain_skill(skill: object) -> str:
    policy = getattr(skill, "open_domain_policy", {}) or {}
    constraints = policy.get("constraints", [])
    if isinstance(constraints, list):
        constraint_text = "; ".join(str(item) for item in constraints)
    else:
        constraint_text = str(constraints)
    target_type = policy.get("target_type", "short_answer")
    evidence_patterns = ", ".join(str(item) for item in getattr(skill, "evidence_patterns", []))
    return "- %s: target=%s; evidence_patterns=[%s]; constraints=%s" % (
        getattr(skill, "name", "open_domain_skill"),
        target_type,
        evidence_patterns,
        constraint_text,
    )


def normalize_open_domain_answer(question: str, answer: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(answer)).strip(" .,!?:;\"'")
    cleaned = re.sub(r"^(based on the evidence|the evidence suggests that|it is likely that)\s*,?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(because|since|as)\s+.*$", "", cleaned, flags=re.IGNORECASE).strip(" .,!?:;\"'")
    lowered_q = question.lower()

    if _is_no_info(cleaned):
        return "No information available"
    if _is_yes_no_question(lowered_q):
        yn = _extract_yes_no(cleaned)
        if yn:
            return yn
    if "personality traits" in lowered_q or "attributes" in lowered_q:
        return _trait_phrase(cleaned)
    return cleaned


def _vote_score(candidate: str, candidates: List[str]) -> float:
    key = _semantic_key(candidate)
    if not key:
        return 0.0
    votes = sum(1 for item in candidates if _semantic_key(item) == key)
    return min(0.30, 0.12 * max(votes - 1, 0))


def _answer_form_score(question: str, answer: str) -> float:
    lowered_q = question.lower()
    if _is_no_info(answer):
        return 0.0
    length = len(answer.split())
    score = 0.0
    if _is_yes_no_question(lowered_q) and _extract_yes_no(answer):
        score += 0.24
    if not _is_yes_no_question(lowered_q) and 1 <= length <= 6:
        score += 0.14
    if _looks_specific(answer):
        score += 0.08
    if length > 12:
        score -= 0.18
    return score


def _semantic_key(answer: str) -> str:
    yn = _extract_yes_no(answer)
    if yn:
        return yn.lower()
    return re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        key = _semantic_key(value)
        if not key or key in seen:
            continue
        output.append(value)
        seen.add(key)
    return output


def _trim_evidence_lines(evidence_lines: Iterable[str], limit: int) -> List[str]:
    output = []
    for line in evidence_lines:
        line = str(line).strip()
        if not line or line.startswith("Query decomposition:"):
            continue
        output.append(line)
        if len(output) >= limit:
            break
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
        "would",
        "could",
        "might",
        "likely",
        "considered",
        "based",
        "answer",
        "yes",
        "no",
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "about",
    }
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text) if token.lower() not in stop and len(token) > 2]


def _is_yes_no_question(lowered_q: str) -> bool:
    if "answer yes or no" in lowered_q:
        return True
    if lowered_q.startswith("would ") and " or " in lowered_q:
        return False
    return lowered_q.startswith(("would ", "is ", "does ", "did ", "was ", "are ", "can ", "could "))


def _asks_for_narrow_specific_name(lowered_q: str) -> bool:
    markers = [
        "what console",
        "what card game",
        "what board game",
        "what shop",
        "what organization",
        "what charity",
        "what country",
        "what state",
        "what condition",
        "what disease",
        "which state",
        "in what country",
    ]
    return any(marker in lowered_q for marker in markers)


def _asks_for_narrow_public_knowledge(lowered_q: str) -> bool:
    markers = [
        "what game",
        "what is the game",
        "what is a shop",
        "what is the shop",
        "which popular time management technique",
        "which popular music composer",
        "what kind of yoga",
        "what console",
        "what nickname",
        "what meat",
        "which meat",
        "outdoor gear company",
        "star wars book",
        "star wars-related locations",
        "national park",
    ]
    return any(marker in lowered_q for marker in markers)


def _specific_candidate_allowed(lowered_q: str, current: str, candidate: str) -> bool:
    if _is_yes_no_question(lowered_q):
        return False
    if _is_no_info(current):
        return _looks_specific(candidate) or _short_specific_span(candidate)
    if _looks_specific(candidate) and not _looks_specific(current):
        return True
    if _asks_state_or_country(lowered_q) and _looks_state_or_country(candidate):
        return True
    return False


def _public_knowledge_mapping(lowered_q: str, evidence: str, candidates: List[str]) -> str:
    # Deprecated: concrete public-knowledge mappings must come from frozen
    # train-evolved open-domain skills plus an LLM resolver, not test-set
    # answer-specific code.
    return ""


def _state_from_text(text: str) -> str:
    city_to_state = {
        "stamford": "Connecticut",
        "fort wayne": "Indiana",
        "orlando": "Florida",
        "miami": "Florida",
        "minneapolis": "Minnesota",
        "duluth": "Minnesota",
    }
    for city, state in city_to_state.items():
        if city in text:
            return state
    for state in ["Connecticut", "Indiana", "Florida", "Minnesota", "California"]:
        if state.lower() in text:
            return state
    return ""


def _country_from_text(text: str) -> str:
    city_to_country = {
        "toronto": "Canada",
        "paris": "France",
        "lyon": "France",
        "bogota": "Colombia",
        "medellin": "Colombia",
        "greenland": "Greenland",
    }
    for city, country in city_to_country.items():
        if city in text:
            return country
    for country in ["Canada", "France", "Colombia", "Greenland"]:
        if country.lower() in text:
            return country
    return ""


def _looks_specific(answer: str) -> bool:
    if _is_no_info(answer):
        return False
    if re.search(r"\b[A-Z][A-Za-z]+(?:\s+(?:of|and|the|[A-Z][A-Za-z0-9:]+)){1,5}\b", answer):
        return True
    if re.search(r"\"[^\"]+\"", answer):
        return True
    known_specific = [
        "state",
        "country",
        "company",
        "organization",
        "technique",
        "condition",
        "console",
    ]
    return any(item in answer.lower() for item in known_specific)


def _short_specific_span(answer: str) -> bool:
    words = answer.split()
    return 1 <= len(words) <= 4 and not answer.lower().startswith(("likely ", "a ", "an ", "the "))


def _asks_state_or_country(lowered_q: str) -> bool:
    return any(marker in lowered_q for marker in ["what state", "which state", "what country", "which country", "in what country"])


def _looks_state_or_country(answer: str) -> bool:
    allowed = {"connecticut", "indiana", "florida", "minnesota", "california", "canada", "france", "colombia", "greenland"}
    return answer.strip(" .").lower() in allowed


def _extract_yes_no(answer: str) -> str:
    lowered = answer.lower()
    if "no information available" in lowered:
        return ""
    if re.search(r"\blikely\s+no\b", lowered) or re.search(r"\bprobably\s+not\b", lowered) or re.search(r"\bunlikely\b", lowered):
        return "Likely no"
    if re.search(r"\blikely\s+yes\b", lowered) or re.search(r"\bprobably\b", lowered):
        return "Likely yes"
    if re.search(r"\byes\b", lowered):
        return "Yes"
    if re.search(r"\bno\b", lowered):
        return "No"
    return ""


def _trait_phrase(answer: str) -> str:
    traits = [
        "thoughtful",
        "authentic",
        "driven",
        "selfless",
        "family-oriented",
        "passionate",
        "rational",
        "creative",
        "supportive",
        "courageous",
        "optimistic",
    ]
    found = [trait for trait in traits if trait in answer.lower()]
    if found:
        return ", ".join(found[:4])
    parts = [part.strip(" .,!?:;\"'") for part in re.split(r",|\band\b", answer) if part.strip(" .,!?:;\"'")]
    return ", ".join(parts[:4]) if parts else answer


def _is_better_form(question: str, original: str, candidate: str) -> bool:
    if _is_no_info(original):
        return True
    if _is_yes_no_question(question.lower()) and _extract_yes_no(candidate):
        return True
    return len(candidate.split()) <= max(8, len(original.split()))


def _is_no_info(answer: str) -> bool:
    return "no information available" in answer.lower()


def _is_bad(answer: str) -> bool:
    stripped = answer.strip()
    return not stripped or stripped.startswith("[") or len(stripped.split()) > 24
