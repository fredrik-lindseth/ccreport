"""Bounded, best-effort diagnostics for local report startup.

One JSON file beside cache.db keeps the latest run, including its current
phase if interrupted. It contains durations and counts, never log contents.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Self

from ccreport import cache_db

_active: ContextVar[ReportRun | None] = ContextVar("report_startup", default=None)


class ReportRun:
    def __init__(self, *, debug: bool = False) -> None:
        self.path = cache_db.DB_PATH.with_name("startup-timings.json")
        self.debug = debug
        self.started = time.perf_counter()
        self.data: dict = {
            "started_at": time.time(), "status": "running", "phases": {},
            "counts": {}, "active_phases": [], "last_indexed_at": None,
        }
        self.token: Token | None = None
        try:
            previous = json.loads(self.path.read_text())
            stamp = previous.get("last_indexed_at")
            if isinstance(stamp, (float, int)) and 0 < stamp <= time.time():
                self.data["last_indexed_at"] = stamp
        except (OSError, ValueError, AttributeError):
            pass

    def __enter__(self) -> Self:
        self.token = _active.set(self)
        self.save()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.data["status"] = "complete" if exc_type is None else "interrupted" if (
            exc_type is KeyboardInterrupt
        ) else "failed"
        self.data["total_seconds"] = round(time.perf_counter() - self.started, 4)
        self.save()
        if self.token is not None:
            _active.reset(self.token)
        if self.debug:
            print(f"Startup timings: {self.path}", file=sys.stderr)
            for name, seconds in self.data["phases"].items():
                print(f"  {name}: {seconds:.3f}s", file=sys.stderr)
            print(f"  total: {self.data['total_seconds']:.3f}s", file=sys.stderr)

    def save(self) -> None:
        """Replace the previous diagnostic atomically; failures never block a report."""
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(self.data, indent=2) + "\n")
            temporary.replace(self.path)
        except OSError:
            pass
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def age_message(self) -> str:
        stamp = self.data["last_indexed_at"]
        if stamp is None:
            return "Last report index time is unknown; catching up may take a while."
        days = max(0, int((time.time() - stamp) / 86400))
        if days == 0:
            return "Last report indexing completed less than a day ago."
        unit = "day" if days == 1 else "days"
        return f"Last report indexing completed {days} {unit} ago; catching up may take a while."

    def indexing_completed(self) -> None:
        self.data["last_indexed_at"] = time.time()
        self.save()


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Measure a phase only inside a report; other cache readers pay no I/O."""
    run = _active.get()
    if run is None:
        yield
        return
    started = time.perf_counter()
    run.data["active_phases"].append(name)
    run.save()
    try:
        yield
    finally:
        if sys.exc_info()[0] is not None:
            run.data.setdefault("stopped_phase", name)
        phases = run.data["phases"]
        phases[name] = round(phases.get(name, 0) + time.perf_counter() - started, 4)
        run.data["active_phases"].remove(name)
        run.save()


def count(name: str, value: int) -> None:
    """Attach a corpus count without storing any record or project identity."""
    run = _active.get()
    if run is not None:
        run.data["counts"][name] = value
