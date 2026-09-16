import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class LongMemEvalExample:
    question_id: str
    question_type: str
    question: str
    question_date: str
    answer: str
    answer_session_ids: List[str]
    haystack_dates: List[str]
    haystack_session_ids: List[str]
    haystack_sessions: List[Any]


def load_longmemeval_s(path: str) -> List[LongMemEvalExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return [_from_dict(row) for row in rows]


def _from_dict(row: Dict[str, Any]) -> LongMemEvalExample:
    return LongMemEvalExample(
        question_id=row["question_id"],
        question_type=row["question_type"],
        question=row["question"],
        question_date=row.get("question_date", ""),
        answer=row["answer"],
        answer_session_ids=list(row.get("answer_session_ids", [])),
        haystack_dates=list(row.get("haystack_dates", [])),
        haystack_session_ids=list(row.get("haystack_session_ids", [])),
        haystack_sessions=list(row.get("haystack_sessions", [])),
    )

