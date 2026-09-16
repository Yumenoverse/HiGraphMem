from datetime import datetime
from typing import Optional


def time_decay(timestamp: Optional[str], half_life_days: int = 90) -> float:
    if not timestamp:
        return 1.0
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError:
        return 1.0
    age_days = max((datetime.now() - dt).days, 0)
    return 0.5 ** (age_days / half_life_days)
