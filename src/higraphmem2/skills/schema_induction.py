from typing import Iterable, List

from .schema import StateRule, candidate_rules


def induce_rules(samples: Iterable[dict]) -> List[StateRule]:
    """Select candidate state rules using questions from the induction split only."""
    questions = [qa.get("question", "") for sample in samples for qa in sample.get("qa", [])]
    selected = []
    for rule in candidate_rules():
        if any(rule.matches_question(question) for question in questions):
            selected.append(rule)
    return selected

