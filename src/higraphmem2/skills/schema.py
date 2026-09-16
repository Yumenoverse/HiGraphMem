from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .memory_skill import MemorySkill, candidate_skills, load_skills, save_skills


@dataclass(frozen=True)
class StateRule:
    name: str
    entity: str
    question_keywords: List[str]
    extract_keywords: List[str]
    values: List[str]
    answer_joiner: str = ", "

    def matches_question(self, question: str) -> bool:
        lowered = question.lower()
        return any(_pattern_matches(keyword, lowered) for keyword in self.question_keywords)


def load_schema(path: str) -> List[StateRule]:
    return [_skill_to_rule(skill) for skill in load_skills(path)]


def save_schema(path: str, rules: Iterable[StateRule], metadata: Optional[Dict[str, object]] = None) -> None:
    save_skills(path, [_rule_to_skill(rule) for rule in rules], metadata=metadata)


def candidate_rules() -> List[StateRule]:
    return [_skill_to_rule(skill) for skill in candidate_skills()]


def _skill_to_rule(skill: MemorySkill) -> StateRule:
    return StateRule(
        name=_legacy_rule_name(skill.name),
        entity=skill.legacy_entity,
        question_keywords=list(skill.trigger_patterns),
        extract_keywords=list(skill.evidence_patterns),
        values=list(skill.legacy_values),
    )


def _rule_to_skill(rule: StateRule) -> MemorySkill:
    return MemorySkill(
        name=rule.name,
        description="Legacy state rule exported as a memory skill.",
        trigger_patterns=rule.question_keywords,
        state_schema={"entity": "person", "slot": rule.name, "value_type": "list"},
        evidence_patterns=rule.extract_keywords,
        retrieval_policy={"prefer_entity_match": True, "aggregate_across_sessions": True},
        answer_policy={"output": "concise_list", "deduplicate": True},
        legacy_entity=rule.entity,
        legacy_values=rule.values,
    )


def _legacy_rule_name(skill_name: str) -> str:
    return {
        "relationship_state_memory": "relationship_status",
        "origin_location_memory": "moved_from",
        "location_history_memory": "camped_locations",
        "preference_state_memory": "kids_like",
        "reading_memory": "books_read",
        "destress_activity_memory": "destress_activities",
        "activity_event_memory": "activities",
        "identity_support_event_memory": "lgbtq_events",
        "creative_output_memory": "recent_painting",
    }.get(skill_name, skill_name)


def _pattern_matches(pattern: str, lowered_question: str) -> bool:
    tokens = [token for token in pattern.lower().replace("{person}", "").split() if token]
    return all(token in lowered_question for token in tokens)
