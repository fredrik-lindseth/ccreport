"""Codex usage comes from calls, never the cumulative counters beside them."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from ccreport import codex_scan, scan


def _event(kind, payload, ordinal=0):
    return {"type": kind, "timestamp": "2026-10-15T10:00:00Z", "ordinal": ordinal, "payload": payload}


def _meta(thread="root", **extra):
    return _event("session_meta", {
        "id": thread, "session_id": "root", "cwd": "/missing/codex-checkout",
        "git": {"repository_url": "git@example.test:org/shared-project.git"},
        **extra,
    })


def _context(turn="turn-1", model="gpt-6-astra", ordinal=1):
    return _event("turn_context", {"turn_id": turn, "model": model}, ordinal)


def _call(response="response-1", turn="turn-1", thread="root", **usage):
    return _event("token_usage_record", {
        "response_id": response, "turn_id": turn, "thread_id": thread, "session_id": "root",
        "usage": {"input_tokens": 1000, "cached_input_tokens": 800,
                  "cache_write_input_tokens": 0, "output_tokens": 100,
                  "reasoning_output_tokens": 40, "total_tokens": 1100, **usage},
        "thread_token_usage": {"total_tokens": 999999},
        "turn_token_usage": {"total_tokens": 999999},
    }, 10)


def _write(tmp_path, events):
    path = tmp_path / "rollout.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    return path


def test_discovers_both_roots_and_respects_codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    active = tmp_path / "sessions/2026/10/15/a.jsonl"
    archived = tmp_path / "archived_sessions/b.jsonl"
    unrelated = tmp_path / "logs/c.jsonl"
    for path in (active, archived, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert codex_scan.discover_codex_files() == sorted([active, archived])
    assert codex_scan.is_codex_file(active)
    assert codex_scan.is_codex_file(archived)
    assert not codex_scan.is_codex_file(unrelated)


def test_default_home_and_absent_roots(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert codex_scan.discover_codex_files() == []
    active = tmp_path / ".codex/sessions/a.jsonl"
    active.parent.mkdir(parents=True)
    active.touch()
    assert codex_scan.discover_codex_files() == [active]


def test_per_call_usage_excludes_counters_cache_and_reasoning_double_counts(tmp_path):
    counter = _event("event_msg", {"type": "token_count", "info": {
        "total_token_usage": {"input_tokens": 999999},
        "last_token_usage": {"input_tokens": 999999},
    }})
    path = _write(tmp_path, [_meta(creator_account_id="account-a"), _context(), _call(), counter, counter])
    [rec] = codex_scan.parse_rollout_file(path)
    assert (rec.tokens.input, rec.tokens.cache_read, rec.tokens.output, rec.tokens.total) == (
        200, 800, 100, 1100,
    )
    assert rec.reasoning_output_tokens == 40
    assert rec.source == "codex"
    assert rec.account_uuid == "account-a"
    assert rec.thread_id == "root"
    assert rec.session_id == "root"
    assert rec.account == "unknown"
    assert rec.cost_usd is None
    assert rec.model == "gpt-6-astra"
    assert rec.dedup_key == "response-1"


def test_model_switch_exact_turn_and_fallback(tmp_path):
    path = _write(tmp_path, [
        _meta(), _context(), _context("turn-2", "gpt-6.1-sol"),
        _call(), _call("response-2", "turn-2"), _call("response-3", "missing"),
    ])
    with pytest.warns(RuntimeWarning, match="lack matching turn_context"):
        records = codex_scan.parse_rollout_file(path)
    assert [r.model for r in records] == [
        "gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-sol",
    ]


def test_fork_metadata_and_inherited_context_do_not_replace_child(tmp_path):
    path = _write(tmp_path, [
        _meta("child", creator_account_id="child-account", forked_from_id="root",
              subagent_history_start_ordinal=9),
        _meta("root", creator_account_id="parent-account"),
        _context("parent-turn", "gpt-5.6-terra", 6),
        _call("inherited", "parent-turn", "root"),
        _call("response-1", "missing", "child"),
        _context("child-turn", "gpt-6-astra", 14),
        _call("response-2", "child-turn", "child"),
    ])
    with pytest.warns(RuntimeWarning, match="lack matching turn_context"):
        records = codex_scan.parse_rollout_file(path)
    assert [r.message_id for r in records] == ["response-1", "response-2"]
    assert [r.model for r in records] == ["unknown", "gpt-6-astra"]
    assert {r.account_uuid for r in records} == {"child-account"}
    assert {r.thread_id for r in records} == {"child"}
    assert {r.session_id for r in records} == {"root"}


def test_guardian_is_counted_without_guessing_underlying_model(tmp_path):
    path = _write(tmp_path, [
        _meta("guardian", thread_source="guardian_review"),
        _context(model="codex-auto-review"), _call(thread="guardian"),
    ])
    [rec] = codex_scan.parse_rollout_file(path)
    assert rec.model == "codex-auto-review"
    assert rec.account_uuid is None


def test_response_ids_deduplicate_copies_without_dropping_distinct_equal_calls(tmp_path):
    path = _write(tmp_path, [_meta(), _context(), _call(), _call(), _call("response-2")])
    records = codex_scan.parse_rollout_file(path)
    assert len(records) == 2
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(path.read_bytes())
    assert [r.dedup_key for r in codex_scan.parse_rollout_file(copy)] == [r.dedup_key for r in records]


def test_legacy_counts_are_explicitly_unsupported(tmp_path):
    path = _write(tmp_path, [_meta(), _context(), _event("event_msg", {
        "type": "token_count", "info": {"total_token_usage": {"total_tokens": 100}},
    })])
    with pytest.warns(RuntimeWarning, match="only legacy token_count"):
        assert codex_scan.parse_rollout_file(path) == []


def test_partial_line_is_read_after_append(tmp_path):
    path = _write(tmp_path, [_meta(), _context(), _call()])
    encoded = json.dumps(_call("response-2"))
    with path.open("a") as stream:
        stream.write(encoded[:80])
    assert len(codex_scan.parse_rollout_file(path)) == 1
    with path.open("a") as stream:
        stream.write(encoded[80:] + "\n")
    assert len(codex_scan.parse_rollout_file(path)) == 2


@pytest.mark.parametrize("usage", [
    {"input_tokens": -1}, {"input_tokens": "1000"}, {"input_tokens": True},
    {"cached_input_tokens": 1001}, {"reasoning_output_tokens": 101}, {"total_tokens": 999},
])
def test_invalid_usage_is_explicitly_skipped(tmp_path, usage):
    path = _write(tmp_path, [_meta(), _context(), _call(**usage)])
    with pytest.warns(RuntimeWarning, match="1 invalid Codex"):
        assert codex_scan.parse_rollout_file(path) == []


def test_unverified_cache_write_accounting_is_explicitly_unsupported(tmp_path):
    path = _write(tmp_path, [_meta(), _context(), _call(cache_write_input_tokens=50)])
    with pytest.warns(RuntimeWarning, match="unsupported cache-write accounting"):
        assert codex_scan.parse_rollout_file(path) == []


def test_filesystem_project_identity_is_resolved_only_once_per_thread(tmp_path, monkeypatch):
    looked_up = []

    def resolve(cwd):
        looked_up.append(cwd)
        return "example.test/org/project"

    monkeypatch.setattr(scan, "_resolve_remote", resolve)
    path = _write(tmp_path, [_meta(git=None), _context(), _call(), _call("response-2")])
    records = codex_scan.parse_rollout_file(path)
    assert len(records) == 2
    assert looked_up == ["/missing/codex-checkout"]
    assert {rec.project for rec in records} == {"project"}


def test_project_identity_matches_claude_and_does_not_read_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "_resolve_remote", lambda cwd: "example.test/org/shared-project")
    codex = _write(tmp_path, [_meta(), _context(), _call()])
    claude = tmp_path / "claude.jsonl"
    claude.write_text(json.dumps({
        "type": "assistant", "timestamp": "2026-10-15T10:00:00Z", "cwd": "/missing/other-checkout",
        "message": {"id": "msg", "model": "claude-opus-4-6", "usage": {"input_tokens": 10}},
    }))
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        assert Path(file).name != "auth.json"
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    [codex_record] = codex_scan.parse_rollout_file(codex)
    [claude_record] = scan.parse_jsonl_file(claude)
    assert codex_record.project == claude_record.project == "shared-project"
    assert codex_record.repo == claude_record.repo


def test_unreadable_file_propagates(tmp_path):
    with pytest.raises(OSError):
        codex_scan.parse_rollout_file(tmp_path / "absent.jsonl")
