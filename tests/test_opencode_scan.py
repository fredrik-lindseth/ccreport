"""OpenCode imports calls once from metadata, including committed WAL usage."""

from __future__ import annotations

import builtins
import json
import sqlite3
from pathlib import Path

import pytest

from ccreport import opencode_scan, scan

BASE = 1_800_000_000_000


def _usage(**extra):
    return {"input": 3, "output": 127, "reasoning": 17,
            "cache": {"read": 10, "write": 149895}, **extra}


def _db(root):
    conn = sqlite3.connect(root / "opencode.db")
    conn.executescript("""
        CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT, time_created INTEGER,
                              directory TEXT, project_id TEXT);
        CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT);
        CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT);
        CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,
                           time_created INTEGER, data TEXT);
        CREATE TABLE session_message (id TEXT PRIMARY KEY, data TEXT);
        CREATE TABLE account (secret TEXT);
        INSERT INTO account VALUES ('DO NOT READ');
    """)
    conn.execute("INSERT INTO project VALUES (?, ?)", ("project", "/missing/project"))
    _session(conn)
    conn.commit()
    return conn


def _session(conn, sid="session", parent=None, created=BASE):
    conn.execute("INSERT INTO session VALUES (?, ?, ?, ?, ?)",
                 (sid, parent, created, "/missing/project", "project"))


def _message(conn, mid="message", sid="session", created: int | None = BASE + 1000,
             completed: int | None = BASE + 5000, **extra):
    data = {"role": "assistant", "time": {"created": created, "completed": completed},
            "modelID": "gpt-5.6-luna", "providerID": "opencode",
            "tokens": _usage(), "cost": 99.0, **extra}
    conn.execute("INSERT INTO message VALUES (?, ?, ?)", (mid, sid, json.dumps(data)))


def _part(conn, pid="part", mid="message", sid="session", created=BASE + 5000, **extra):
    conn.execute("INSERT INTO part VALUES (?, ?, ?, ?, ?)",
                 (pid, mid, sid, created, json.dumps({"type": "step-finish", "tokens": _usage(),
                                                    "cost": 0, **extra})))


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(scan, "_resolve_remote", lambda cwd: None)


def test_sqlite_tokens_provider_and_reported_cost(tmp_path):
    with _db(tmp_path) as conn:
        _message(conn)
        _part(conn)
    [unit] = opencode_scan.discover_opencode_units(tmp_path)
    [rec] = unit.records
    assert opencode_scan.is_opencode_path(unit.path)
    assert not unit.path.exists()
    assert rec.source == "opencode"
    assert rec.account_uuid is None
    assert rec.account == "unknown"
    assert rec.session_id == rec.thread_id == "session"
    assert rec.model == "gpt-5.6-luna"
    assert rec.provider == "opencode"
    assert rec.reported_cost_usd == 0
    assert rec.cost_usd is None
    assert rec.message_id == rec.dedup_key == "part"
    assert (rec.tokens.input, rec.tokens.output, rec.tokens.cache_read, rec.tokens.cache_create) == (
        3, 144, 10, 149895,
    )
    assert rec.tokens.total == 150052
    assert rec.reasoning_output_tokens == 17
    assert rec.project == "project"
    assert rec.cwd == "/missing/project"
    assert rec.req_start is rec.req_end is None


def test_multiple_steps_never_add_message_total(tmp_path):
    with _db(tmp_path) as conn:
        _message(conn)
        _part(conn)
        _part(conn, "part-2", tokens=_usage(output=9, reasoning=800), cost=1.25)
    [unit] = opencode_scan.discover_opencode_units(tmp_path)
    assert len(unit.records) == 2
    assert [r.tokens.output for r in unit.records] == [144, 809]
    assert [r.reported_cost_usd for r in unit.records] == [0, 1.25]


def test_completed_message_fallback_only_without_usage_parts(tmp_path):
    with _db(tmp_path) as conn:
        _message(conn)
        _message(conn, "empty", tokens=_usage(input=0, output=0, reasoning=0, cache={}))
        _message(conn, "streaming", completed=None)
    [unit] = opencode_scan.discover_opencode_units(tmp_path)
    assert [r.message_id for r in unit.records] == ["message"]
    assert unit.records[0].reported_cost_usd == 99


def test_fork_history_new_ids_skipped_new_calls_preserved(tmp_path):
    with _db(tmp_path) as conn:
        _message(conn)
        _part(conn)
        _session(conn, "fork", created=BASE + 10000)
        _message(conn, "cloned-message", "fork")
        _part(conn, "cloned-part", "cloned-message", "fork", created=BASE + 10000)
        _message(conn, "new-message", "fork", created=BASE + 11000, completed=BASE + 12000)
        _part(conn, "new-part", "new-message", "fork", created=BASE + 12000)
    units = opencode_scan.discover_opencode_units(tmp_path)
    assert sorted(r.message_id for unit in units for r in unit.records) == ["new-part", "part"]


def test_subagents_keep_own_session_and_root_thread(tmp_path):
    with _db(tmp_path) as conn:
        _session(conn, "child", parent="session")
        _session(conn, "grandchild", parent="child")
        _message(conn, sid="grandchild")
        _part(conn, sid="grandchild")
    records = [r for unit in opencode_scan.discover_opencode_units(tmp_path) for r in unit.records]
    assert len(records) == 1
    assert records[0].session_id == "grandchild"
    assert records[0].thread_id == "session"


def test_wal_commit_moves_fingerprint_without_main_database_change(tmp_path):
    conn = _db(tmp_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    _message(conn)
    _part(conn)
    conn.commit()
    before = (tmp_path / "opencode.db").stat()
    [old] = opencode_scan.discover_opencode_units(tmp_path)
    conn.execute("UPDATE part SET data=json_set(data,'$.tokens.output',128)")
    conn.commit()
    [new] = opencode_scan.discover_opencode_units(tmp_path)
    after = (tmp_path / "opencode.db").stat()
    assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)
    assert old.path == new.path
    assert old.mtime_ns != new.mtime_ns
    assert new.records[0].tokens.output == 145
    conn.close()


def test_snapshot_stable_and_content_only_changes_do_not_invalidate(tmp_path):
    with _db(tmp_path) as conn:
        _message(conn)
        _part(conn)
    [old] = opencode_scan.discover_opencode_units(tmp_path)
    with sqlite3.connect(tmp_path / "opencode.db") as conn:
        conn.execute("UPDATE message SET data=json_set(data,'$.content','irrelevant transcript')")
    [new] = opencode_scan.discover_opencode_units(tmp_path)
    assert (old.mtime_ns, old.size) == (new.mtime_ns, new.size)


@pytest.mark.parametrize(("table", "field", "value"), [
    ("message", "providerID", "another-provider"),
    ("message", "modelID", "another-model"),
    ("part", "cost", 1.5),
])
def test_provider_model_raw_cost_change_fingerprint(tmp_path, table, field, value):
    with _db(tmp_path) as conn:
        _message(conn)
        _part(conn)
    [old] = opencode_scan.discover_opencode_units(tmp_path)
    with sqlite3.connect(tmp_path / "opencode.db") as conn:
        conn.execute(f"UPDATE {table} SET data=json_set(data, ?, ?)", (f"$.{field}", value))
    [new] = opencode_scan.discover_opencode_units(tmp_path)
    assert old.mtime_ns != new.mtime_ns


def _legacy(root):
    objects = {
        "project/p.json": {"id": "project", "worktree": "/missing/project"},
        "session/p/s.json": {"id": "session", "projectID": "project", "directory": "/missing/project",
                             "time": {"created": BASE}},
        "message/s/m.json": {"id": "message", "sessionID": "session", "role": "assistant",
                             "time": {"created": BASE + 1000, "completed": BASE + 5000},
                             "modelID": "claude-opus-4.6", "providerID": "github-copilot",
                             "tokens": _usage(), "cost": 100},
        "part/m/p.json": {"id": "part", "messageID": "message", "sessionID": "session",
                          "type": "step-finish", "tokens": _usage(), "cost": 0},
    }
    for name, obj in objects.items():
        path = root / "storage" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))


def test_legacy_json_fallback(tmp_path):
    _legacy(tmp_path)
    [unit] = opencode_scan.discover_opencode_units(tmp_path)
    assert len(unit.records) == 1
    assert unit.records[0].provider == "github-copilot"
    assert unit.records[0].tokens.output == 144
    assert unit.records[0].reported_cost_usd == 0


def test_sqlite_wins_over_overlapping_legacy_even_malformed(tmp_path):
    _legacy(tmp_path)
    (tmp_path / "storage/message/s/m.json").write_text("malformed legacy")
    with _db(tmp_path) as conn:
        _message(conn)
        _part(conn)
    [unit] = opencode_scan.discover_opencode_units(tmp_path)
    assert len(unit.records) == 1
    assert unit.records[0].provider == "opencode"


def test_v2_nonempty_explicitly_rejected_without_legacy_fallback(tmp_path):
    _legacy(tmp_path)
    with _db(tmp_path) as conn:
        conn.execute("INSERT INTO session_message VALUES (?, ?)", ("v2", "{}"))
    with pytest.raises(opencode_scan.OpenCodeReadError, match="V2"):
        opencode_scan.discover_opencode_units(tmp_path)


def test_unsupported_schema_does_not_fall_back_to_json(tmp_path):
    _legacy(tmp_path)
    with sqlite3.connect(tmp_path / "opencode.db") as conn:
        conn.execute("CREATE TABLE unrelated (id TEXT)")
    with pytest.raises(opencode_scan.OpenCodeReadError, match="schema"):
        opencode_scan.discover_opencode_units(tmp_path)


@pytest.mark.parametrize("usage", [None, [], _usage(input=-1), _usage(reasoning=True),
                                  _usage(cache={"read": 1.5})])
def test_bad_usage_raises_instead_of_partial_snapshot(tmp_path, usage):
    with _db(tmp_path) as conn:
        _message(conn)
        _part(conn, tokens=usage)
    with pytest.raises(opencode_scan.OpenCodeReadError):
        opencode_scan.discover_opencode_units(tmp_path)


def test_missing_original_timestamp_cannot_guess_fork_ownership(tmp_path):
    with _db(tmp_path) as conn:
        _message(conn, created=None)
        _part(conn)
    with pytest.raises(opencode_scan.OpenCodeReadError, match="message created"):
        opencode_scan.discover_opencode_units(tmp_path)


def test_invalid_json_sqlite_propagates(tmp_path):
    with _db(tmp_path) as conn:
        conn.execute("INSERT INTO message VALUES (?, ?, ?)", ("bad", "session", "{"))
    with pytest.raises(sqlite3.Error):
        opencode_scan.discover_opencode_units(tmp_path)


def test_orphan_step_is_error_not_silent_undercount(tmp_path):
    with _db(tmp_path) as conn:
        _part(conn)
    with pytest.raises(opencode_scan.OpenCodeReadError, match="no assistant"):
        opencode_scan.discover_opencode_units(tmp_path)


def test_parent_cycle_rejected(tmp_path):
    with _db(tmp_path) as conn:
        conn.execute("UPDATE session SET parent_id='session'")
        _message(conn)
        _part(conn)
    with pytest.raises(opencode_scan.OpenCodeReadError, match="Cyclic"):
        opencode_scan.discover_opencode_units(tmp_path)


def test_credentials_never_read_and_database_readonly(tmp_path, monkeypatch):
    with _db(tmp_path) as conn:
        _message(conn)
        _part(conn)
    (tmp_path / "auth.json").write_text("secret")
    original_open = builtins.open
    original_connect = sqlite3.connect
    reads = []

    def guarded_open(file, *args, **kwargs):
        assert "auth" not in str(file)
        return original_open(file, *args, **kwargs)

    def guarded_connect(database, *args, **kwargs):
        assert database.endswith("?mode=ro")
        connection = original_connect(database, *args, **kwargs)

        def authorize(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_READ:
                reads.append((table, column))
                assert table != "account"
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        return connection

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
    assert opencode_scan.discover_opencode_units(tmp_path)[0].records
    assert reads


def test_default_root_and_absent_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert opencode_scan.discover_opencode_units() == []
    root = tmp_path / "opencode"
    root.mkdir()
    _legacy(root)
    assert opencode_scan.discover_opencode_units()
    assert not opencode_scan.is_opencode_path(tmp_path / "ordinary.jsonl")


def test_snapshot_does_not_mix_concurrent_wal_commits(tmp_path, monkeypatch):
    writer = _db(tmp_path)
    writer.execute("PRAGMA journal_mode=WAL")
    _message(writer)
    _part(writer)
    writer.commit()
    connect = sqlite3.connect
    changed = False

    def reader(*args, **kwargs):
        conn = connect(*args, **kwargs)

        def during_snapshot(sql):
            nonlocal changed
            if "FROM message" in sql and not changed:
                changed = True
                writer.execute("UPDATE part SET data=json_set(data,'$.tokens.output',500)")
                writer.commit()

        conn.set_trace_callback(during_snapshot)
        return conn

    monkeypatch.setattr(sqlite3, "connect", reader)
    [old] = opencode_scan.discover_opencode_units(tmp_path)
    assert changed
    assert old.records[0].tokens.output == 144
    [new] = opencode_scan.discover_opencode_units(tmp_path)
    assert new.records[0].tokens.output == 517
    writer.close()


@pytest.mark.parametrize("cost", [-1, True, "0"])
def test_invalid_reported_cost_raises(tmp_path, cost):
    with _db(tmp_path) as conn:
        _message(conn)
        _part(conn, cost=cost)
    with pytest.raises(opencode_scan.OpenCodeReadError, match="reported cost"):
        opencode_scan.discover_opencode_units(tmp_path)


def test_missing_legacy_identity_uses_snapshot_error_contract(tmp_path):
    _legacy(tmp_path)
    (tmp_path / "storage/project/p.json").write_text("{}")
    with pytest.raises(opencode_scan.OpenCodeReadError, match="Malformed"):
        opencode_scan.discover_opencode_units(tmp_path)


def test_default_home_without_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _legacy(tmp_path / ".local/share/opencode")
    assert opencode_scan.discover_opencode_units()
