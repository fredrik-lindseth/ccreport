"""Codex payloads keep log attribution and obey the same redaction policy."""
from __future__ import annotations

from datetime import UTC, datetime

from ccreport import cache_db, push
from ccreport.accounts import AccountTimeline


def cached_record(**kw):
    return {
        "mid": "resp-test", "model": "gpt-6.1-sol", "ts": datetime(2026, 10, 1, tzinfo=UTC).timestamp(),
        "sid": "root", "project": "private", "cwd": "/private/repo", "repo": "example/repo",
        "dk": "resp-test", "cost": None, "t": [100, 20, 0, 900],
        "source": "codex", "account_uuid": "chat-account", "thread_id": "child",
        "reasoning_output_tokens": 12, **kw,
    }


def test_codex_payload_uses_log_account_without_claude_capture():
    cache_db.save_ccreport_files([("/rollout.jsonl", 1, 100, [cached_record()])])
    conn = cache_db.get_connection()
    built = push.build_files(conn, [("/rollout.jsonl", 1, 100)], AccountTimeline([]), None)
    assert built.unattributed == 0
    rec = built.files[0]["records"][0]
    assert rec["account_uuid"] == "chat-account"
    assert rec["account_label"] == "chat-account (ChatGPT)"
    assert rec["source"] == "codex"
    assert rec["thread_id"] == "child"
    assert rec["reasoning_output_tokens"] == 12
    assert rec["input_tokens"] == 100
    assert rec["cache_read"] == 900
    assert rec["cost"] is None


def test_unknown_codex_account_never_uses_claude_timeline():
    cache_db.record_account_event({"accountUuid": "claude", "emailAddress": "c@example.test"}, now=1)
    cache_db.save_ccreport_files([("/unknown.jsonl", 1, 100, [cached_record(account_uuid=None)])])
    timeline = AccountTimeline(cache_db.load_account_events())
    built = push.build_files(cache_db.get_connection(), [("/unknown.jsonl", 1, 100)], timeline, None)
    assert built.files == []
    assert built.unattributed == 1


def test_redaction_removes_child_thread_identity_too():
    server = push.ServerConfig(
        url="https://example.test", token="fake", label="Laptop", machine_id="m", restricted=True,
    )
    rec = {**cached_record(), "sid": "root"}
    stripped = push.redact(rec, server)
    for key in ("project", "cwd", "repo", "sid", "thread_id"):
        assert stripped[key] is None
    assert stripped["source"] == "codex"
    assert stripped["reasoning_output_tokens"] == 12
    assert stripped["account_uuid"] == "chat-account"


def test_exclusion_also_removes_codex_thread_identity():
    server = push.ServerConfig(
        url="https://example.test", token="fake", label="Laptop", machine_id="m", exclude=("private",),
    )
    assert push.redact(cached_record(), server)["thread_id"] is None


def test_cached_codex_push_reaches_server_reports_with_the_same_tokens(tmp_path, monkeypatch):
    import server_fixture as sf
    from fastapi.testclient import TestClient

    from ccreport import exchange, protocol
    from ccreport.server import reports
    from ccreport.server.factory import create_app

    monkeypatch.setattr(exchange, "_store", exchange._store)
    monkeypatch.setattr(exchange, "load_rates", lambda *a, **kw: {})
    cache_db.save_ccreport_files([("/rollout.jsonl", 1, 100, [cached_record()])])
    built = push.build_files(
        cache_db.get_connection(), [("/rollout.jsonl", 1, 100)], AccountTimeline([]), None,
    )
    app = create_app(sf.config(tmp_path / "server"))
    token = sf.mint_for(app, "laptop", "Laptop")
    response = TestClient(app).post("/v1/ingest", headers=sf.auth(token), json={
        "protocol": protocol.PROTOCOL_VERSION, "label": "Laptop", "files": built.files,
    })
    assert response.status_code == 200
    assert response.json()["files"][0]["status"] == "accepted"
    merged = reports.load_grouped(app.state.db.connect(), reports.Filters(tool="codex"))
    assert len(merged) == 1
    rec = merged[0].record
    assert rec.account_uuid == "chat-account"
    assert rec.tokens.total == 1020
    assert rec.tokens.cache_read == 900
    assert rec.reasoning_output_tokens == 12
    assert rec.cost() > 0
