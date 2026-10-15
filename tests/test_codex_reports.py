"""Codex stays attributable and countable across the local report paths."""
from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest
from rich.console import Console

from ccreport import aggregate, cache_db, pricing
from ccreport import ccreport as ccr
from ccreport.accounts import AccountTimeline


def record(**kw):
    return aggregate.UsageRecord(**{
        "message_id": "resp-test", "model": "codex-auto-review",
        "tokens": aggregate.TokenCounts(input=100, output=20, cache_read=900),
        "timestamp": datetime(2026, 10, 1, tzinfo=UTC),
        "session_id": "root", "project": "shared", "source": "codex",
        "account_uuid": "chat-account", "thread_id": "child",
        "reasoning_output_tokens": 12, **kw,
    })


def keep_filters():
    return ccr._keep_filters(None, None, None, None, accounts=AccountTimeline([{
        "ts": 0, "account_uuid": "claude-account", "email": "claude@example.test",
        "organization_uuid": None, "organization_name": None,
    }]))


def test_codex_attribution_does_not_use_claude_timeline():
    rec = record()
    assert ccr._keep(rec, **keep_filters())
    assert rec.account == "chat-account (ChatGPT)"
    assert rec.account_uuid == "chat-account"
    missing = record(account_uuid=None)
    assert ccr._keep(missing, **keep_filters())
    assert missing.account == "Codex – unknown account"
    assert ccr._pre_capture_records([rec, missing], [{"ts": 9e9, "source": "capture"}]) == []


def test_dedup_separates_tools_and_drops_copied_codex_responses():
    filters = keep_filters()
    assert ccr._keep(record(), **filters)
    assert not ccr._keep(record(), **filters)
    assert ccr._keep(record(source="claude"), **filters)


def test_unknown_price_preserves_tokens_and_marks_partial_totals():
    records = [record(), record(message_id="paid", model="claude-opus-4-6", source="claude", cost_usd=2)]
    rows = aggregate.model_rows(records, aggregate.NokCtx())
    assert rows.total.tokens.total == 2040
    assert rows.total.unpriced_count == 1
    assert rows.total.cost == 2
    payload = aggregate.rows_to_json(rows)
    assert aggregate.rows_from_json(payload).total.unpriced_count == 1
    entry = ccr._json_entry(records[0], aggregate.NokCtx())
    assert entry["cost_usd"] is None
    assert entry["total_tokens"] == 1020
    assert entry["source"] == "codex"
    assert entry["reasoning_output_tokens"] == 12


def test_tool_and_model_rows_include_cache_hits_and_calls():
    records = [record(), record(message_id="r2"), record(source="claude", cost_usd=1)]
    tools = aggregate.tool_rows(records, aggregate.NokCtx())
    codex = next(r for r in tools.rows if r.key == "codex")
    assert codex.agg.count == 2
    assert codex.agg.tokens.cache_read == 1800
    assert ccr._fmt_cache_read(codex.agg.tokens).endswith("(90%)")
    assert aggregate.model_rows(records, aggregate.NokCtx()).total.count == 3


def test_rollups_preserve_source_identity_reasoning_and_unpriced_calls():
    cache_db.init_ccreport_meta(ccr.CACHE_VERSION, "test")
    records = [record(), record(message_id="r2")]
    for r in records:
        ccr._keep(r, **keep_filters())
    ccr._build_rollups(records, datetime(2026, 10, 2, tzinfo=UTC).timestamp(), "fp")
    loaded = ccr._rollup_records(cache_db.load_ccreport_rollups())
    assert len(loaded) == 1
    r = loaded[0]
    assert (r.source, r.account_uuid, r.thread_id, r.reasoning_output_tokens) == (
        "codex", "chat-account", "child", 24,
    )
    assert r.count == 2
    assert aggregate.model_rows(loaded, aggregate.NokCtx()).total.unpriced_count == 2


def test_usage_split_renders_cache_and_unknown_marker(monkeypatch):
    out = io.StringIO()
    monkeypatch.setattr(ccr, "console", Console(file=out, width=180, color_system=None))
    rows = aggregate.model_rows([record()], aggregate.NokCtx())
    ccr.render_usage_split(rows, "Models", "Model", nok=aggregate.NokCtx())
    rendered = out.getvalue()
    assert "codex-auto-review" in rendered
    assert "90%" in rendered
    assert "unpriced" in rendered


@pytest.mark.parametrize("command", ["model", "tool"])
def test_new_cli_commands_and_tool_filter(monkeypatch, command):
    monkeypatch.setattr(ccr.sys, "argv", ["ccreport", command, "--tool", "codex"])
    seen = {}

    def load(**kw):
        seen.update(kw)
        return [record()]

    monkeypatch.setattr(ccr, "load_all_records", load)
    monkeypatch.setattr(ccr, "load_rates_for_records", lambda *a, **kw: (aggregate.NokCtx(), True))
    monkeypatch.setattr(ccr, "render_usage_split", lambda *a, **kw: None)
    ccr.main()
    assert seen["tool_filter"] == "codex"


def test_verified_codex_json_retains_price(monkeypatch):
    monkeypatch.setattr(aggregate, "has_pricing", lambda *a, **kw: True)
    entry = ccr._json_entry(record(cost_usd=2), aggregate.NokCtx())
    assert entry["cost_usd"] == 2


@pytest.mark.parametrize("command", ["model", None])
def test_all_models_never_claim_a_model_for_pulled_day_totals(monkeypatch, command):
    local = record()
    remote = record(source="claude", model="<remote>", cost_usd=50, message_id="", session_id="")
    argv = ["ccreport", "--all"] + ([command] if command else [])
    if command:
        argv = ["ccreport", command, "--all"]
    monkeypatch.setattr(ccr.sys, "argv", argv)
    monkeypatch.setattr(ccr, "load_all_records", lambda **kw: [local])
    monkeypatch.setattr(ccr, "_remote_records", lambda *a: ccr.RemoteSpend([remote], {"other"}))
    monkeypatch.setattr(ccr, "load_rates_for_records", lambda *a, **kw: (aggregate.NokCtx(), True))
    for name in ("daily", "monthly", "project", "session", "account"):
        monkeypatch.setattr(ccr, f"report_{name}", lambda *a, **kw: None)
    reports = {}
    monkeypatch.setattr(
        ccr, "render_usage_split", lambda rows, title, label, **kw: reports.update({title: rows}),
    )
    ccr.main()
    assert [row.key for row in reports["Models"].rows] == ["codex-auto-review"]
    assert reports["Models"].total.count == 1


def test_actual_loader_tool_filter_applies_to_full_and_warm_rollup_paths(monkeypatch):
    from ccreport import scan

    codex = record()
    claude = record(source="claude", message_id="claude", cost_usd=1)
    cache_db.init_ccreport_meta(ccr.CACHE_VERSION, ccr._script_hash())
    cache_db.save_ccreport_files([
        ("/purged/codex.jsonl", 1, 10, scan._serialize_records([codex])),
        ("/purged/claude.jsonl", 1, 10, scan._serialize_records([claude])),
    ])
    monkeypatch.setattr(ccr, "discover_jsonl_files", list)
    for rollups in (False, True, True):
        got = ccr.load_all_records(tool_filter="codex", use_rollups=rollups)
        assert sum(rec.count for rec in got) == 1
        assert {rec.source for rec in got} == {"codex"}
        assert got[0].tokens.total == 1020


def test_unknown_codex_provider_name_is_not_assumed_to_be_free():
    rec = record(model="provider:future-model")
    assert not pricing.has_pricing(rec.model, rec.timestamp, source="codex")
    assert pricing.has_pricing(rec.model, rec.timestamp, source="claude")
    assert ccr._json_entry(rec, aggregate.NokCtx())["cost_usd"] is None
    assert aggregate.model_rows([rec], aggregate.NokCtx()).total.unpriced_count == 1
