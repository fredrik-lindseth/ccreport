"""Static tool names and the units the scanner persists for external stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ccreport.aggregate import UsageRecord

TOOL_NAMES = ("claude", "codex", "opencode")
TOOL_LABELS = {"claude": "Claude Code", "codex": "Codex", "opencode": "OpenCode"}


class SourceImportWarning(RuntimeWarning):
    """Import detail for library callers, with a concise CLI summary."""

    def __init__(self, message: str, source: str, summary: str):
        super().__init__(message)
        self.source = source
        self.summary = summary


@dataclass(frozen=True)
class SourceUnit:
    """One replaceable session snapshot, including virtual paths for databases.

    Database adapters derive a deterministic signed-63-bit fingerprint as
    mtime_ns and a normalized byte count as size. Neither needs a filesystem
    entry: scan persists this identity exactly as it does a physical JSONL.
    """

    path: Path
    mtime_ns: int
    size: int
    records: list[UsageRecord]


def is_claude(source: str) -> bool:
    """Whether Claude's account timeline, quotas and subscription apply."""
    return source == "claude"


def tool_label(source: str) -> str:
    """The display name for a tool, preserving unfamiliar future source names."""
    return TOOL_LABELS.get(source, source)
