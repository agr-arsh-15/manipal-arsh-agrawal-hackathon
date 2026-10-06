import json
import os
import threading
from datetime import datetime
from typing import Iterable, List, Optional

from src.schemas import Signal


class SignalStore:
    """Append-only JSONL sink that downstream modules and the API read from."""

    def __init__(self, path: str = "data/samples/signals.jsonl"):
        self.path = path
        self._lock = threading.Lock()

    def append(self, signals: Iterable[Signal]) -> int:
        lines = [s.model_dump_json() for s in signals]
        if not lines:
            return 0
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return len(lines)

    def overwrite(self, signals: Iterable[Signal]) -> int:
        if os.path.exists(self.path):
            os.remove(self.path)
        return self.append(signals)

    def read(
        self,
        ticker: Optional[str] = None,
        event_type: Optional[str] = None,
        min_impact: Optional[int] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Signal]:
        if not os.path.exists(self.path):
            return []
        out: List[Signal] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s = Signal(**json.loads(line))
                if ticker and (s.ticker or "").upper() != ticker.upper():
                    continue
                if event_type and s.event_type != event_type:
                    continue
                if min_impact is not None and s.impact_score < min_impact:
                    continue
                if since and s.timestamp < since:
                    continue
                out.append(s)
        out.sort(key=lambda s: s.timestamp, reverse=True)
        return out[:limit] if limit else out

    def count(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
