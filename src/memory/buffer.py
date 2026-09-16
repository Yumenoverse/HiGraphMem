from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional


@dataclass(frozen=True)
class DialogueTurn:
    role: str
    content: str
    timestamp: Optional[str] = None


class ShortTermBuffer:
    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self._turns: Deque[DialogueTurn] = deque(maxlen=max_size)

    def add(self, role: str, content: str, timestamp: Optional[str] = None) -> None:
        self._turns.append(DialogueTurn(role=role, content=content, timestamp=timestamp))

    def clear(self) -> List[DialogueTurn]:
        turns = list(self._turns)
        self._turns.clear()
        return turns

    def __iter__(self) -> Iterable[DialogueTurn]:
        return iter(self._turns)

    def __len__(self) -> int:
        return len(self._turns)
