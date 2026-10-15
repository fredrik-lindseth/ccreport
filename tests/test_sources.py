"""Source policies stay explicit, and database sessions use ordinary cache units."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ccreport import scan, sources
from ccreport.aggregate import TokenCounts, UsageRecord


def test_static_tool_names_and_labels():
    assert sources.TOOL_NAMES == ("claude", "codex", "opencode")
    assert sources.tool_label("opencode") == "OpenCode"
    assert sources.tool_label("future-tool") == "future-tool"
    assert sources.is_claude("claude")
    assert not sources.is_claude("codex")
    assert not sources.is_claude("opencode")


def _unit() -> sources.SourceUnit:
    record = UsageRecord(
        message_id="message", model="model", tokens=TokenCounts(10, 5, 0, 2),
        timestamp=datetime(2026, 10, 15, tzinfo=UTC), session_id="session", project="project",
        source="opencode", provider="provider", reported_cost_usd=0.0,
    )
    return sources.SourceUnit(Path("/__ccreport_sources__/opencode/store/session.jsonl"),
                              123456789, 900, [record])


def test_virtual_unit_refresh_uses_snapshot_fingerprint_without_stat(monkeypatch):
    unit = _unit()
    monkeypatch.setattr(scan, "_source_units", {unit.path: unit})
    saved = []
    monkeypatch.setattr(scan, "save_ccreport_files", lambda batch, *, turns=None: saved.extend(batch))
    assert not unit.path.exists()
    fresh, unreadable = scan._refresh_changed_files([unit.path], {})
    assert unreadable == set()
    assert fresh == {str(unit.path): unit.records}
    assert saved[0][:3] == (str(unit.path), unit.mtime_ns, unit.size)
    assert scan._deserialize_records(saved[0][3]) == unit.records
    saved.clear()
    fresh, unreadable = scan._refresh_changed_files(
        [unit.path], {str(unit.path): (unit.mtime_ns, unit.size)},
    )
    assert fresh == {}
    assert unreadable == set()
    assert saved == []


def test_discovery_snapshot_errors_warn_and_keep_history_out_of_refresh(monkeypatch):
    from ccreport import opencode_scan

    unit = _unit()
    monkeypatch.setattr(scan, "_PROJECT_ROOTS", ())
    monkeypatch.setattr(scan.codex_scan, "discover_codex_files", list)
    monkeypatch.setattr(opencode_scan, "discover_opencode_units", lambda: [unit])
    assert scan.discover_jsonl_files() == [unit.path]

    def unavailable():
        raise ValueError("unsupported database version")

    monkeypatch.setattr(opencode_scan, "discover_opencode_units", unavailable)
    with pytest.warns(RuntimeWarning, match="keeping cached history"):
        assert scan.discover_jsonl_files() == []
    # No empty replacement is offered: existing DB rows remain cached orphans.
    saved = []
    monkeypatch.setattr(scan, "save_ccreport_files", lambda batch, *, turns=None: saved.extend(batch))
    assert scan._refresh_changed_files([], {str(unit.path): (unit.mtime_ns, unit.size)}) == ({}, set())
    assert saved == []
