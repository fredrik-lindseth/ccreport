"""The scanner discovers both providers and persists Codex records unchanged."""

from datetime import UTC, datetime

from ccreport import codex_scan, scan
from ccreport.aggregate import TokenCounts, UsageRecord


def test_discovery_includes_active_and_archived_codex(tmp_path, monkeypatch):
    claude = tmp_path / "claude"
    claude.mkdir()
    claude_log = claude / "session.jsonl"
    claude_log.touch()
    codex_home = tmp_path / "codex"
    active = codex_home / "sessions" / "2026" / "rollout.jsonl"
    archived = codex_home / "archived_sessions" / "rollout.jsonl"
    for path in (active, archived):
        path.parent.mkdir(parents=True)
        path.touch()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(scan, "_PROJECT_ROOTS", (claude,))
    assert scan.discover_jsonl_files() == sorted([claude_log, active, archived])


def test_changed_codex_file_uses_codex_parser_and_preserves_metadata(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex"
    path = codex_home / "sessions" / "rollout.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    record = UsageRecord(
        message_id="response", model="gpt-5", tokens=TokenCounts(10, 7, 0, 4),
        timestamp=datetime(2026, 6, 15, tzinfo=UTC), session_id="thread", project="project",
        source="codex", account_uuid="account", thread_id="thread", reasoning_output_tokens=5,
    )
    calls = []
    monkeypatch.setattr(codex_scan, "parse_rollout_file", lambda p: calls.append(p) or [record])
    saved = []
    monkeypatch.setattr(scan, "save_ccreport_files", lambda batch, *, turns=None: saved.extend(batch))
    fresh, unreadable = scan._refresh_changed_files([path], {})
    assert calls == [path]
    assert fresh == {str(path): [record]}
    assert unreadable == set()
    assert scan._deserialize_records(saved[0][3]) == [record]
    stat = path.stat()
    scan._refresh_changed_files([path], {str(path): (stat.st_mtime_ns, stat.st_size)})
    assert calls == [path]


def test_codex_parser_source_changes_invalidate_cached_records(tmp_path, monkeypatch):
    parser = tmp_path / "parser.py"
    parser.write_text("version one")
    monkeypatch.setattr(codex_scan, "__file__", str(parser))
    scan._script_hash.cache_clear()
    try:
        first = scan._script_hash()
        parser.write_text("version two")
        scan._script_hash.cache_clear()
        assert scan._script_hash() != first
    finally:
        scan._script_hash.cache_clear()


def test_mixed_refresh_preserves_claude_timing_and_turns(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex"
    codex = codex_home / "sessions" / "rollout.jsonl"
    codex.parent.mkdir(parents=True)
    codex.write_text("{}\n")
    claude = tmp_path / "claude.jsonl"
    claude.write_text("{}\n")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    record = UsageRecord(
        message_id="response", model="claude-sonnet-5", tokens=TokenCounts(10, 7, 0, 4),
        timestamp=datetime(2026, 6, 15, tzinfo=UTC), session_id="session", project="project",
        req_start=100.0, req_end=102.0,
    )
    turn = scan.Turn("turn", 102.0, "session", "claude-sonnet-5", 2000)
    monkeypatch.setattr(scan, "parse_jsonl", lambda path: scan.ParsedLog([record], [turn]))
    monkeypatch.setattr(codex_scan, "parse_rollout_file", lambda path: [])
    saved = []
    saved_turns = {}

    def save(batch, *, turns=None):
        saved.extend(batch)
        saved_turns.update(turns or {})

    monkeypatch.setattr(scan, "save_ccreport_files", save)
    scan._refresh_changed_files([claude, codex], {})
    assert saved_turns == {str(claude): [turn]}
    assert scan._deserialize_records(saved[0][3]) == [record]
    assert saved[0][3][0]["req_start"] == 100.0
    assert saved[0][3][0]["req_end"] == 102.0
