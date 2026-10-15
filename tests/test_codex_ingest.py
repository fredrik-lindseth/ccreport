"""Codex records retain their tool identity across ingest and deduplication."""

from __future__ import annotations

import pytest
import server_fixture as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ccreport import exchange, protocol
from ccreport.server import db
from ccreport.server.factory import create_app
from ccreport.server.ingest import IngestRecord


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(exchange, "_store", exchange._store)
    monkeypatch.setattr(exchange, "load_rates", lambda dates, prefetch=None: {})
    return create_app(sf.config(tmp_path))


def codex_record(**over):
    return sf.record(source="codex", model="gpt-6-sol", ts=1_792_000_000.0,
                     cache_create=0, reasoning_output_tokens=100, thread_id="thread-1", **over)


def send(app, rec, *, machine="laptop-1", path="/p/a.jsonl", token=None, replace=False):
    token = token or sf.mint_for(app, machine)
    payload = sf.batch([rec], path=path)
    payload["protocol"] = protocol.PROTOCOL_VERSION
    payload["files"][0]["replace"] = replace
    return TestClient(app).post("/v1/ingest", json=payload, headers=sf.auth(token))


def flags(app):
    return app.state.db.connect().execute(
        "SELECT machine_id, source, dup FROM server_records ORDER BY id",
    ).fetchall()


def test_codex_metadata_and_standard_price_are_stored(app):
    response = send(app, codex_record())
    assert response.status_code == 200
    assert response.json()["files"][0]["status"] == "accepted"
    rec = sf.stored(app, "laptop-1")[0]
    assert rec["source"] == "codex"
    assert rec["thread_id"] == "thread-1"
    assert rec["reasoning_output_tokens"] == 100
    assert rec["t"] == [1000, 200, 0, 30000]
    assert rec["cost"] == pytest.approx(.010)


def test_old_client_defaults_to_claude(app):
    response = send(app, sf.record())
    assert response.status_code == 200
    rec = sf.stored(app, "laptop-1")[0]
    assert (rec["source"], rec["thread_id"], rec["reasoning_output_tokens"]) == ("claude", None, 0)


@pytest.mark.parametrize("over", [
    {"source": "unknown"}, {"input_tokens": -1}, {"output_tokens": -1},
    {"cache_read": -1}, {"cache_create": -1}, {"reasoning_output_tokens": -1},
    {"reasoning_output_tokens": 201}, {"cache_create": 1},
])
def test_invalid_codex_token_metadata_is_refused(over):
    rec = codex_record()
    rec.update(over)
    with pytest.raises(ValidationError):
        IngestRecord(**rec)


def test_unknown_codex_model_rejects_file_without_storing_zero(app):
    rec = codex_record()
    rec["model"] = "codex-auto-review"
    response = send(app, rec)
    assert response.json()["files"][0]["status"] == "rejected"
    assert sf.stored(app, "laptop-1") == []


def test_cross_tool_keys_stand_alone_and_same_tool_copies_dedup(app):
    send(app, sf.record(), machine="claude")
    send(app, codex_record(), machine="codex-a")
    send(app, codex_record(), machine="codex-b")
    assert flags(app) == [("claude", "claude", 0), ("codex-a", "codex", 0), ("codex-b", "codex", 1)]


def test_replacement_promotes_only_same_source_copy(app):
    send(app, sf.record(), machine="claude")
    token = sf.mint_for(app, "codex-a")
    send(app, codex_record(), machine="codex-a", token=token)
    send(app, codex_record(), machine="codex-b")
    rec = codex_record()
    rec["dk"] = "different"
    send(app, rec, machine="codex-a", token=token, replace=True)
    assert flags(app) == [("claude", "claude", 0), ("codex-b", "codex", 0), ("codex-a", "codex", 0)]


def test_machine_deletion_promotes_source_winner(app):
    send(app, sf.record(), machine="claude")
    send(app, codex_record(), machine="codex-a")
    send(app, codex_record(), machine="codex-b")
    conn = app.state.db.connect()
    db.delete_machine(conn, "codex-a")
    assert flags(app) == [("claude", "claude", 0), ("codex-b", "codex", 0)]


def test_existing_database_migrates_to_default_claude(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    with monkeypatch.context() as old:
        old.setattr(db, "MIGRATION_CHAIN", db.MIGRATION_CHAIN[:-2])
        old.setattr(db, "SCHEMA_VERSION", 11)
        old.setattr(db, "_SCHEMA_SQL", db._SCHEMA_SQL.replace(
            ",\n    source        TEXT NOT NULL DEFAULT 'claude',\n    thread_id     TEXT,\n"
            "    reasoning_output_tokens INTEGER NOT NULL DEFAULT 0", ""))
        conn = db.connect(path)
        conn.execute("INSERT INTO machines VALUES ('m', 'M', 1, 1, NULL)")
        conn.execute(
            "INSERT INTO server_records(machine_id,file_path,account_uuid,model,ts,day,oslo_date,"
            "cost,input_tokens,output_tokens,cache_create,cache_read) "
            "VALUES ('m','f','a','claude-opus-5',1,'2026-01-01','2026-01-01',1,1,1,0,0)",
        )
        conn.commit()
        conn.close()
    conn = db.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert conn.execute("SELECT source,thread_id,reasoning_output_tokens FROM server_records").fetchone() == (
        "claude", None, 0,
    )
    conn.close()


def test_unknown_codex_provider_is_rejected_instead_of_stored_free(app):
    rec = codex_record()
    rec["model"] = "provider:future-model"
    response = send(app, rec)
    assert response.json()["files"][0]["status"] == "rejected"
    assert sf.stored(app, "laptop-1") == []
