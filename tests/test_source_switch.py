"""`sources` in ccreport.toml and CCREPORT_SERVER_SOURCES leave a tool out end to end."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import server_fixture as sf
from fastapi.testclient import TestClient

from ccreport import aggregate, cache_db, project_identity, push, scan, sources
from ccreport import ccreport as ccr
from ccreport.server import config, reports
from ccreport.server.factory import create_app

NOW = datetime(2026, 10, 15, 12, tzinfo=UTC)


def write_sources(names: str) -> None:
    project_identity.CONFIG_PATH.write_text(f"sources = {names}\n")


# --- the client's setting ---

def test_no_config_file_enables_every_tool():
    assert sources.enabled_sources() == sources.ALL_SOURCES


def test_a_file_without_the_key_enables_every_tool():
    project_identity.CONFIG_PATH.write_text('repo_roots = ["~/dev"]\n')
    assert sources.enabled_sources() == sources.ALL_SOURCES


def test_claude_is_always_enabled():
    write_sources("[]")
    assert sources.enabled_sources() == {"claude"}
    write_sources('["codex"]')
    assert sources.enabled_sources() == {"claude", "codex"}


def test_unknown_names_warn_and_are_ignored(capsys):
    write_sources('["opencode", "cursor"]')
    assert sources.enabled_sources() == {"claude", "opencode"}
    assert "unknown source 'cursor'" in capsys.readouterr().err


def test_a_value_that_is_not_a_list_is_ignored(capsys):
    write_sources('"claude"')
    assert sources.enabled_sources() == sources.ALL_SOURCES
    assert "must be a list" in capsys.readouterr().err


def test_discovery_does_not_look_for_a_switched_off_tool(monkeypatch):
    from ccreport import opencode_scan

    def forbidden():
        raise AssertionError("a switched-off tool was looked for")

    monkeypatch.setattr(scan, "_PROJECT_ROOTS", ())
    monkeypatch.setattr(scan.codex_scan, "discover_codex_files", forbidden)
    monkeypatch.setattr(opencode_scan, "discover_opencode_units", forbidden)
    write_sources('["claude"]')
    assert scan.discover_jsonl_files() == []


def _record(**kw) -> aggregate.UsageRecord:
    return aggregate.UsageRecord(**{
        "message_id": "m", "model": "claude-sonnet-4-5",
        "tokens": aggregate.TokenCounts(input=100, output=20),
        "timestamp": datetime(2026, 10, 1, tzinfo=UTC),
        "session_id": "s", "project": "p", "cost_usd": 1.0, **kw,
    })


def test_reports_hide_cached_rows_of_a_switched_off_tool_and_restore_them(monkeypatch):
    codex = _record(message_id="codex", source="codex", model="gpt-6-astra",
                    account_uuid="chat-account", cost_usd=None)
    claude = _record(message_id="claude")
    cache_db.init_ccreport_meta(ccr.CACHE_VERSION, ccr._script_hash())
    cache_db.save_ccreport_files([
        ("/purged/codex.jsonl", 1, 10, scan._serialize_records([codex])),
        ("/purged/claude.jsonl", 1, 10, scan._serialize_records([claude])),
    ])
    monkeypatch.setattr(ccr, "discover_jsonl_files", list)

    write_sources('["claude"]')
    for rollups in (False, True, True):
        got = ccr.load_all_records(use_rollups=rollups)
        assert {rec.source for rec in got} == {"claude"}
        assert ccr.load_all_records(use_rollups=rollups, tool_filter="codex") == []

    project_identity.CONFIG_PATH.unlink()
    for rollups in (False, True):
        assert {rec.source for rec in ccr.load_all_records(use_rollups=rollups)} == {"claude", "codex"}


def test_push_does_not_offer_a_switched_off_tools_files():
    cached = {"mid": "x", "model": "gpt-6-astra", "ts": NOW.timestamp(), "sid": "s",
              "project": "p", "cwd": None, "repo": None, "dk": "x", "cost": None,
              "t": [1, 1, 0, 0]}
    cache_db.save_ccreport_files([
        ("/claude.jsonl", 1, 10, [{**cached, "model": "claude-sonnet-4-5"}]),
        ("/rollout.jsonl", 1, 10, [{**cached, "source": "codex", "account_uuid": "a"}]),
    ])
    conn = cache_db.get_connection()
    assert [p for p, *_ in push.changed_files(conn, {})] == ["/claude.jsonl", "/rollout.jsonl"]
    write_sources('["claude"]')
    assert [p for p, *_ in push.changed_files(conn, {})] == ["/claude.jsonl"]


# --- the server's setting ---

def test_server_env_parses_like_the_networks_list(monkeypatch):
    assert config.load_config().sources == sources.ALL_SOURCES
    monkeypatch.setenv(config.SOURCES_ENV, "claude, opencode")
    assert config.load_config().sources == {"claude", "opencode"}
    monkeypatch.setenv(config.SOURCES_ENV, "codex")
    assert config.enabled_sources() == {"claude", "codex"}


def _mixed_batch() -> tuple[dict, dict]:
    claude = sf.batch([sf.record(ts=NOW.timestamp(), dk="c")], path="/p/claude.jsonl")
    codex = sf.batch([sf.record(ts=NOW.timestamp(), source="codex", dk="x", model="gpt-6-astra",
                                cache_create=0)], path="/p/rollout.jsonl")
    return claude, codex


def test_ingest_skips_a_switched_off_tool_and_stores_nothing(tmp_path):
    app = create_app(replace(sf.config(tmp_path), sources=frozenset({"claude"})))
    token = sf.mint_for(app, "laptop")
    claude, codex = _mixed_batch()
    claude["files"] += codex["files"]
    response = TestClient(app).post("/v1/ingest", json=claude, headers=sf.auth(token))
    assert response.status_code == 200, response.text
    statuses = [(f["path"], f["status"]) for f in response.json()["files"]]
    assert statuses == [("/p/claude.jsonl", "accepted"), ("/p/rollout.jsonl", "skipped")]
    assert [rec["source"] for rec in sf.stored(app, "laptop", "/p/claude.jsonl")] == ["claude"]
    assert sf.stored(app, "laptop", "/p/rollout.jsonl") == []


def test_server_reports_hide_stored_rows_until_the_tool_is_back(tmp_path, monkeypatch):
    app = create_app(sf.config(tmp_path))
    token = sf.mint_for(app, "laptop")
    client = TestClient(app)
    for payload in _mixed_batch():
        assert client.post("/v1/ingest", json=payload, headers=sf.auth(token)).status_code == 200
    conn = app.state.db.connect()

    def seen() -> list[set[str]]:
        return [
            {m.record.source for m in reports.load(conn)},
            {m.record.source for m in reports.load(conn, reports.Filters(machine="laptop"))},
            {m.record.source for m in reports.load_grouped(conn)},
            {m.record.source for m in reports.load_grouped(conn, reports.Filters(project="ccr-projA"))},
        ]

    assert seen() == [{"claude", "codex"}] * 4
    monkeypatch.setenv(config.SOURCES_ENV, "claude")
    assert seen() == [{"claude"}] * 4
    assert sum(a["records"] for a in reports.account_overview(conn)) == 1
    monkeypatch.delenv(config.SOURCES_ENV)
    assert seen() == [{"claude", "codex"}] * 4
