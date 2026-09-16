import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple


MONTH_TO_NUM = {month.lower(): idx for idx, month in enumerate(calendar.month_name) if month}


@dataclass(frozen=True)
class SessionTimeIndex:
    session_dates: Dict[int, datetime]

    def date_for_context_ids(self, context_ids: Iterable[str]) -> Optional[datetime]:
        dates = self.dates_for_context_ids(context_ids)
        return dates[0] if dates else None

    def dates_for_context_ids(self, context_ids: Iterable[str]) -> List[datetime]:
        dates = []
        seen = set()
        for context_id in context_ids:
            session_id = _session_id_from_context_id(context_id)
            if session_id and session_id in self.session_dates and session_id not in seen:
                dates.append(self.session_dates[session_id])
                seen.add(session_id)
        return dates


def _session_id_from_context_id(context_id: str) -> Optional[int]:
    match = re.match(r"D(\d+):", str(context_id))
    if match:
        return int(match.group(1))
    match = re.match(r"S(\d+)$", str(context_id))
    if match:
        return int(match.group(1))
    return None


def build_session_time_index(sample: dict) -> SessionTimeIndex:
    conversation = sample["conversation"]
    session_dates: Dict[int, datetime] = {}
    for key, value in conversation.items():
        match = re.match(r"session_(\d+)_date_time$", key)
        if not match:
            continue
        parsed = parse_locomo_datetime(value)
        if parsed:
            session_dates[int(match.group(1))] = parsed
    return SessionTimeIndex(session_dates=session_dates)


def parse_locomo_datetime(text: str) -> Optional[datetime]:
    # Example: "1:56 pm on 8 May, 2023"
    match = re.search(r"on\s+(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", text)
    if not match:
        return None
    day = int(match.group(1))
    month = MONTH_TO_NUM.get(match.group(2).lower())
    year = int(match.group(3))
    if not month:
        return None
    return datetime(year=year, month=month, day=day)


def canonicalize_temporal_answer(prediction: str, anchor_date: Optional[datetime]) -> str:
    if not anchor_date:
        return prediction
    text = prediction.strip()
    lowered = text.lower()

    if "last year" in lowered:
        return str(anchor_date.year - 1)

    if "next month" in lowered:
        year = anchor_date.year
        month = anchor_date.month + 1
        if month > 12:
            month = 1
            year += 1
        return "%s %s" % (calendar.month_name[month], year)

    if "this month" in lowered:
        return "%s %s" % (calendar.month_name[anchor_date.month], anchor_date.year)

    if "yesterday" in lowered:
        return _format_date(anchor_date - timedelta(days=1))

    if "tomorrow" in lowered:
        return _format_date(anchor_date + timedelta(days=1))

    if "last month" in lowered:
        year = anchor_date.year
        month = anchor_date.month - 1
        if month < 1:
            month = 12
            year -= 1
        return "%s %s" % (calendar.month_name[month], year)

    if "next year" in lowered:
        return str(anchor_date.year + 1)

    if "last week" in lowered or "the week before" in lowered:
        return "The week before %s %s %s" % (
            anchor_date.day,
            calendar.month_name[anchor_date.month],
            anchor_date.year,
        )

    ago_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", lowered)
    if ago_match:
        num = _word_or_int(ago_match.group(1))
        return "%s years ago" % num

    weekday_relative = _relative_weekday(lowered, anchor_date)
    if weekday_relative:
        return weekday_relative

    return prediction


def resolve_temporal_answer(question: str, prediction: str, evidence_lines: Iterable[str], context_dates: Iterable[datetime]) -> Tuple[str, str]:
    """Resolve relative temporal answers using retrieved session dates only."""
    dates = list(context_dates)
    anchor_date = dates[0] if dates else _first_date_from_evidence(evidence_lines)
    if not anchor_date:
        return prediction, "no_anchor"

    canonical = canonicalize_temporal_answer(prediction, anchor_date)
    if canonical != prediction:
        return canonical, "relative_prediction"

    evidence_candidate = _candidate_from_temporal_evidence(question, list(evidence_lines), anchor_date)
    if evidence_candidate and _should_replace_with_evidence(question, prediction, evidence_candidate):
        return evidence_candidate, "relative_evidence"
    return prediction, "unchanged"


def _candidate_from_temporal_evidence(question: str, evidence_lines: List[str], anchor_date: datetime) -> str:
    lowered_q = question.lower()
    joined = "\n".join(evidence_lines).lower()

    if "how long" in lowered_q or "how many" in lowered_q:
        duration = _duration_candidate(joined)
        if duration:
            return duration

    for line in evidence_lines[:24]:
        lowered = line.lower()
        line_date = _date_from_line(line) or anchor_date
        relative = _relative_phrase_candidate(lowered, line_date)
        if relative:
            return relative

    explicit = _explicit_date_candidate(evidence_lines)
    if explicit:
        return explicit
    return ""


def _relative_phrase_candidate(lowered: str, anchor_date: datetime) -> str:
    if "tomorrow" in lowered:
        return _format_date(anchor_date + timedelta(days=1))
    if "yesterday" in lowered:
        return _format_date(anchor_date - timedelta(days=1))
    if "last month" in lowered:
        month = anchor_date.month - 1
        year = anchor_date.year
        if month < 1:
            month = 12
            year -= 1
        return "%s %s" % (calendar.month_name[month], year)
    if "last summer" in lowered:
        return "summer %s" % (anchor_date.year - 1)
    if "last year" in lowered:
        return str(anchor_date.year - 1)
    if "this past weekend" in lowered or "last weekend" in lowered:
        return "The weekend before %s" % _format_date(anchor_date)
    if "this weekend" in lowered:
        return "The weekend of %s" % _format_date(anchor_date)
    if "few days ago" in lowered or "a few days ago" in lowered:
        return "few days before %s" % _format_date(anchor_date)
    if "few weeks ago" in lowered or "a few weeks ago" in lowered:
        return "few weeks before %s" % _format_date(anchor_date)
    weekday = _relative_weekday(lowered, anchor_date)
    if weekday:
        return weekday
    return ""


def _relative_weekday(lowered: str, anchor_date: datetime) -> str:
    weekdays = {name.lower(): idx for idx, name in enumerate(calendar.day_name)}
    for prefix, direction in [("last", -1), ("next", 1), ("this", 0)]:
        for name, weekday in weekdays.items():
            if "%s %s" % (prefix, name) not in lowered:
                continue
            if direction == 0:
                delta = (weekday - anchor_date.weekday()) % 7
                return _format_date(anchor_date + timedelta(days=delta))
            if direction < 0:
                delta = (anchor_date.weekday() - weekday) % 7 or 7
                return _format_date(anchor_date - timedelta(days=delta))
            delta = (weekday - anchor_date.weekday()) % 7 or 7
            return _format_date(anchor_date + timedelta(days=delta))
    return ""


def _duration_candidate(text: str) -> str:
    patterns = [
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|nineteen)\s+(days?|weeks?|months?|years?)\b",
        r"\bhalf\s+a\s+year\b",
        r"\babout\s+a\s+year\b",
        r"\bjust\s+under\s+a\s+year\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""


def _explicit_date_candidate(evidence_lines: List[str]) -> str:
    for line in evidence_lines[:24]:
        match = re.search(r"\b(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})\b", line)
        if match:
            return match.group(1).replace(",", "")
        match = re.search(r"\b([A-Za-z]+\s+\d{4})\b", line)
        if match and match.group(1).split()[0].lower() in MONTH_TO_NUM:
            return match.group(1)
    return ""


def _date_from_line(line: str) -> Optional[datetime]:
    parsed = parse_locomo_datetime(line)
    if parsed:
        return parsed
    match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b", line)
    if not match:
        return None
    month = MONTH_TO_NUM.get(match.group(2).lower())
    if not month:
        return None
    return datetime(year=int(match.group(3)), month=month, day=int(match.group(1)))


def _first_date_from_evidence(evidence_lines: Iterable[str]) -> Optional[datetime]:
    for line in evidence_lines:
        parsed = _date_from_line(str(line))
        if parsed:
            return parsed
    return None


def _should_replace_with_evidence(question: str, prediction: str, candidate: str) -> bool:
    if not candidate:
        return False
    lowered = prediction.lower()
    if "no information available" in lowered:
        return True
    if any(marker in lowered for marker in ["tomorrow", "yesterday", "last month", "last summer", "last year", "weekend", "few days", "few weeks"]):
        return True
    if question.lower().startswith("how") and re.search(r"\b(days?|weeks?|months?|years?)\b", candidate.lower()):
        return True
    return False


def _format_date(value: datetime) -> str:
    return "%s %s %s" % (value.day, calendar.month_name[value.month], value.year)


def _word_or_int(value: str) -> int:
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return words.get(value, int(value) if value.isdigit() else 0)
