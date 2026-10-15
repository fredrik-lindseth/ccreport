"""Cross-provider records remain separate through reports and Claude quota reads."""

from datetime import UTC, datetime

import pytest
import server_fixture as sf
from fastapi.testclient import TestClient

from ccreport.aggregate import NokCtx
from ccreport.server import dashboard, pull, reports
from ccreport.server.factory import create_app

NOW = datetime(2026, 10, 15, 12, tzinfo=UTC)


@pytest.fixture
def mixed(tmp_path):
    app = create_app(sf.config(tmp_path))
    client = TestClient(app)
    # Both providers intentionally share an account, response key and project.
    # Each also has a synced copy on a second machine.
    for machine in ("laptop", "desk"):
        token = sf.mint_for(app, machine, machine)
        records = [
            sf.record(ts=NOW.timestamp(), source="claude", dk="shared-key"),
            sf.record(ts=NOW.timestamp(), source="codex", dk="shared-key", model="gpt-6-astra",
                      thread_id="codex-thread", reasoning_output_tokens=100, cache_create=0),
        ]
        response = client.post("/v1/ingest", json=sf.batch(records), headers=sf.auth(token))
        assert response.status_code == 200, response.text
        assert response.json()["files"][0]["status"] == "accepted", response.json()
    return app, client


@pytest.mark.parametrize("filters", [reports.Filters(), reports.Filters(machine="desk"),
                                     reports.Filters(project="ccr-projA")])
def test_dedup_keeps_each_provider_once_in_full_grouped_and_bucketed(mixed, filters):
    app, _ = mixed
    conn = app.state.db.connect()
    for loaded in (
        reports.load(conn, filters), reports.load_grouped(conn, filters),
        reports.load_bucketed(conn, filters, origin=NOW.timestamp(), step=60),
    ):
        assert {m.record.source for m in loaded} == {"claude", "codex"}
        assert sum(m.record.count for m in loaded) == 2
        codex = next(m.record for m in loaded if m.record.source == "codex")
        assert codex.account_uuid == "acct-1"
        assert codex.reasoning_output_tokens == 100
    codex = next(m.record for m in reports.load(conn, filters) if m.record.source == "codex")
    assert codex.thread_id == "codex-thread"


def test_tool_filter_and_tool_model_reports(mixed):
    app, client = mixed
    conn = app.state.db.connect()
    codex = reports.load(conn, reports.Filters(tool="codex"))
    assert len(codex) == 1
    assert codex[0].record.source == "codex"
    assert [row.key for row in reports.build(codex, "tool", NokCtx()).rows] == ["codex"]
    assert [row.key for row in reports.build(codex, "model", NokCtx()).rows] == ["gpt-6-astra"]
    for kind in ("tool", "model", "day"):
        response = client.get(f"/v1/report/{kind}", params={"tool": "codex"})
        assert response.status_code == 200, response.text
        assert response.json()["kind"] == kind
    assert client.get("/v1/report/day", params={"tool": "other"}).status_code == 422


def test_dashboard_tool_dimension_and_detail(mixed):
    app, client = mixed
    conn = app.state.db.connect()
    view = dashboard.build(conn, dashboard.ALL_TIME, now=NOW)
    assert {row["key"] for row in view.breakdowns["tool"]} == {"claude", "codex"}
    detail = dashboard.build(conn, dashboard.ALL_TIME, now=NOW, scope=dashboard.Scope("tool", "codex"))
    assert {row["key"] for row in detail.breakdowns["model"]} == {"gpt-6-astra"}
    assert client.get("/tool/codex").status_code == 200


def test_claude_spend_and_pull_exclude_codex_even_with_same_account(mixed):
    app, _ = mixed
    conn = app.state.db.connect()
    claude = reports.load(conn, reports.Filters(tool="claude"))[0].record
    spend = reports.load_spend(conn)
    rows = [row for account in spend.values() for row in account]
    assert len(rows) == 1
    assert rows[0][1] == pytest.approx(claude.cost())
    remainder = pull.remainder(conn, "acct-1", "absent-machine", 0)
    assert sum(day[-1] for machine in remainder for day in machine.days) == 1
    cost = sum(bucket[1] for machine in remainder for bucket in machine.buckets)
    assert cost == pytest.approx(claude.cost())
    # The asking machine has already sent this Claude key; the other provider's
    # identically named key must not affect the exclusion in either direction.
    assert pull.remainder(conn, "acct-1", "laptop", 0) == []
