"""Simple progress reporting for notebooks and long-running scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any


@dataclass
class ProgressLogger:
    """Timestamped progress logger with a small in-memory event record."""

    name: str = "ms4"
    verbose: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)
    _start: float = field(default_factory=perf_counter, init=False)

    def step(self, message: str, **fields: Any) -> None:
        elapsed = perf_counter() - self._start
        event = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": round(elapsed, 3),
            "message": message,
            **fields,
        }
        self.events.append(event)
        if self.verbose:
            detail = " ".join(f"{key}={value}" for key, value in fields.items())
            suffix = f" | {detail}" if detail else ""
            print(f"[{self.name} +{elapsed:7.2f}s] {message}{suffix}", flush=True)

    def as_frame_records(self) -> list[dict[str, Any]]:
        return list(self.events)
