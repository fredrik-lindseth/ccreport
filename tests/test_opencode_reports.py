"""OpenCode reporting preserves provider and separates reported cost from API value."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ccreport import aggregate, cache_db, push, scan
from ccreport import ccreport as ccr
from ccreport.accounts import AccountTimeline


def record(**over):
    return aggregate.UsageRecord(**{
        "message_id": "step-one", "dedup_key": "step-one", "source": "opencode",
        "model": "claude-sonnet-4-5-20250929", "provider": "anthropic",
        "tokens": aggregate.TokenCounts(input=10, output=7, cache_create=30, cache_read=50),
        "reasoning_output_tokens": 2, "reported_cost_usd": 0.0,
        "timestamp": datetime(2026, 9, 15, tzinfo=UTC), "session_id": "session-one",
        "thread_id": "root", "project": "shared", **over,
    })


def test_provider_reported_zero_does_not_hide_documented_api_value():
    rec = record()
    assert rec.cost() > 0
    assert rec.is_priced()
    entry = ccr._json_entry(rec, aggregate.NokCtx())
    assert entry["source"] == "opencode"
    assert entry["provider"] == "anthropic"
    assert entry["reported_cost_usd"] == 0
    assert entry["cost_usd"] > 0
    assert entry["total_tokens"] == 97
    assert entry["output_tokens"] == 7  # reasoning2 already included


def test_unknown_model_is_unpriced_even_when_provider_reported_zero():
    rec = record(model="future-private-model")
    assert not rec.is_priced()
    entry = ccr._json_entry(rec, aggregate.NokCtx())
    assert entry["cost_usd"] is None
    assert entry["reported_cost_usd"] == 0
    assert aggregate.tool_rows([rec], aggregate.NokCtx()).total.unpriced_count == 1


def test_same_model_is_separate_by_provider_in_model_report():
    recs = [record(), record(provider="github-copilot", message_id="step-two")]
    report = aggregate.model_rows(recs, aggregate.NokCtx())
    assert {row.key for row in report.rows} == {
        "claude-sonnet-4-5-20250929 (anthropic)",
        "claude-sonnet-4-5-20250929 (github-copilot)",
    }
    assert report.total.count == 2
    assert report.total.tokens.total == 194


def test_external_account_never_inherits_claude_and_push_retains_unknown_locally():
    cache_db.init_ccreport_meta(scan.CACHE_VERSION, "test")
    cache_db.record_account_event({"accountUuid": "claude-account", "emailAddress": "c@example.test"}, now=1)
    rec = record()
    filters = ccr._keep_filters(None, None, None, None)
    assert ccr._keep(rec, **filters)
    assert rec.account == "OpenCode – unknown account"
    assert not ccr._pre_capture_records([rec], [{"ts": 9e9, "source": "capture"}])
    cache_db.save_ccreport_files([("/virtual-opencode.jsonl", 1, 100, scan._serialize_records([rec]))])
    built = push.build_files(cache_db.get_connection(), [("/virtual-opencode.jsonl", 1, 100)],
                             AccountTimeline(cache_db.load_account_events()), None)
    assert built.files == []
    assert built.unattributed == 1


def test_rollups_preserve_provider_reported_cost_and_token_coverage():
    cache_db.init_ccreport_meta(scan.CACHE_VERSION, "test")
    recs = [record(reported_cost_usd=None), record(message_id="two"),
            record(provider="copilot", reported_cost_usd=2.5, message_id="three")]
    for rec in recs:
        ccr._keep(rec, **ccr._keep_filters(None, None, None, None))
    ccr._build_rollups(recs, 9e9, "test")
    got = ccr._rollup_records(cache_db.load_ccreport_rollups())
    assert len(got) == 2
    assert {rec.provider: rec.reported_cost_usd for rec in got} == {"anthropic": 0, "copilot": 2.5}
    assert sum(rec.count for rec in got) == 3
    assert sum(rec.reasoning_output_tokens for rec in got) == 6
    full = aggregate.model_rows(recs, aggregate.NokCtx())
    rolled = aggregate.model_rows(got, aggregate.NokCtx())
    assert full.total.cost == pytest.approx(rolled.total.cost)
    assert full.total.tokens == rolled.total.tokens
    assert full.total.count == rolled.total.count


@pytest.mark.parametrize("command", ["model", "tool", "daily"])
def test_cli_accepts_opencode_tool_and_excludes_claude_pull(monkeypatch, command):
    monkeypatch.setattr(ccr.sys, "argv", ["ccreport", command, "--tool", "opencode", "--all", "--json"])
    seen = {}
    rec = record()

    def load(**kw):
        seen.update(kw)
        return [rec]

    monkeypatch.setattr(ccr, "load_all_records", load)
    monkeypatch.setattr(
        ccr, "_remote_records", lambda *a: ccr.RemoteSpend([record(source="claude")], {"other"}),
    )
    monkeypatch.setattr(ccr, "load_rates_for_records", lambda *a, **kw: (aggregate.NokCtx(), True))
    monkeypatch.setattr(ccr, "report_json", lambda rows, **kw: seen.update(records=rows))
    ccr.main()
    assert seen["tool_filter"] == "opencode"
    assert seen["records"] == [rec]


def test_all_three_tools_share_project_without_source_dedup_collisions():
    recs = [record(source=source) for source in ("claude", "codex", "opencode")]
    filters = ccr._keep_filters(None, None, None, None)
    assert all(ccr._keep(rec, **filters) for rec in recs)
    assert not ccr._keep(record(), **filters)
    assert aggregate.project_rows(recs, aggregate.NokCtx()).total.count == 3
    assert {row.key for row in aggregate.tool_rows(recs, aggregate.NokCtx()).rows} == {
        "claude", "codex", "opencode",
    }
