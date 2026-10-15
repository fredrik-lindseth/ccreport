"""Static tool names and the units the scanner persists for external stores."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ccreport.aggregate import UsageRecord

TOOL_NAMES = ("claude", "codex", "opencode")
TOOL_LABELS = {"claude": "Claude Code", "codex": "Codex", "opencode": "OpenCode"}
ALL_SOURCES = frozenset(TOOL_NAMES)


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


def parse_sources(names: Iterable[object], origin: str) -> frozenset[str]:
    """The tools a `sources` list switches on, with Claude Code always among them.

    Claude is the one tool this report exists for — its quotas, account log and
    plan prices have no meaning without it — so leaving it out of the list does
    not switch it off. An unknown name warns rather than failing, the way a
    malformed repo_roots does: a typo should not take the reports down.
    """
    enabled = {"claude"}
    for name in names:
        if isinstance(name, str) and name in ALL_SOURCES:
            enabled.add(name)
        else:
            print(f"warning: {origin}: unknown source {name!r}, expected one of "
                  f"{', '.join(TOOL_NAMES)}", file=sys.stderr)
    return frozenset(enabled)


def enabled_sources() -> frozenset[str]:
    """Which tools this machine scans, reports and pushes: `sources` in ccreport.toml.

    No file or no key is every tool. A source left out is never discovered, so
    its logs are not parsed, and every report and push leaves out the rows the
    cache already holds for it — hidden rather than deleted, so putting the name
    back restores history whose logs have since rotated away.

    Read per call rather than cached: a report calls it a handful of times and
    tests point CONFIG_PATH at a file of their own.
    """
    import tomllib

    from ccreport import project_identity

    path = project_identity.CONFIG_PATH
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f).get("sources")
    except FileNotFoundError:
        return ALL_SOURCES
    except (tomllib.TOMLDecodeError, OSError):
        # project_identity.repo_roots already warns about this file.
        return ALL_SOURCES
    if raw is None:
        return ALL_SOURCES
    if not isinstance(raw, list):
        print(f"warning: {path}: sources must be a list of tool names; ignoring it", file=sys.stderr)
        return ALL_SOURCES
    return parse_sources(raw, str(path))
