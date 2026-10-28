"""A slow refresh must leave useful, clearly provisional output in scrollback."""

import io
import json
from datetime import UTC, datetime

import pytest
from rich.console import Console

from ccreport import aggregate, cache_db, scan, startup
from ccreport import ccreport as ccr


def _record(**kwargs):
    return aggregate.UsageRecord(**{
        "message_id": "one", "model": "claude-opus-4.5",
        "tokens": aggregate.TokenCounts(10, 5, 0, 0),
        "timestamp": datetime(2026, 10, 15, tzinfo=UTC),
        "session_id": "s", "project": "project", "cost_usd": 1.0, **kwargs,
    })


def _seed(records):
    cache_db.init_ccreport_meta(scan.CACHE_VERSION, scan._script_hash())
    cache_db.save_ccreport_file("/missing/log.jsonl", 1, 2, scan._serialize_records(records))


def test_cached_load_never_discovers_parses_or_invalidates(monkeypatch):
    _seed([_record(), _record(), _record(message_id="two", project="other")])

    def forbidden(*args, **kwargs):
        pytest.fail("cached preview touched the logs or invalidated cached history")

    for name in ("discover_jsonl_files", "_ensure_cache_valid", "_refresh_changed_files", "_build_rollups"):
        monkeypatch.setattr(ccr, name, forbidden)
    records = ccr.load_all_records(cached_only=True, project_filter="project")
    assert len(records) == 1
    assert records[0].cost() == 1.0
    assert len(ccr.load_all_records(cached_only=True)) == 2
    assert ccr.load_all_records(cached_only=True, tool_filter="codex") == []
    assert ccr.load_all_records(cached_only=True, since=datetime(2026, 10, 16, tzinfo=UTC)) == []


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("as_json", [True, False])
def test_preview_is_visible_before_refresh_and_pipes_stay_clean(monkeypatch, interactive, as_json):
    output = io.StringIO()
    monkeypatch.setattr(ccr, "console", Console(file=output, force_terminal=interactive, color_system=None))
    monkeypatch.setattr(ccr.sys, "argv", ["ccreport", "--json"] if as_json else ["ccreport"])
    monkeypatch.setattr(ccr.exchange, "start_prefetch", lambda: None)
    monkeypatch.setattr(ccr.exchange, "read_rates_since", lambda since: {})
    monkeypatch.setattr(ccr, "_remote_records", lambda *args: ccr.RemoteSpend([], set()))
    monkeypatch.setattr(ccr, "load_rates_for_records", lambda *a, **kw: (aggregate.NokCtx(), True))
    events = []

    def render(args, records, *rest):
        value = records[0].cost()
        events.append(f"render {value}")
        ccr.console.print(f"SPEND {value}")

    def load(**kwargs):
        if kwargs.get("cached_only"):
            events.append("cached")
            return [_record()]
        events.append("refresh")
        if interactive and not as_json:
            assert "Loading cached usage" in output.getvalue()
            assert "Cached report" in output.getvalue()
            assert "SPEND 1.0" in output.getvalue()
            assert "Fetching updated information" in output.getvalue()
        else:
            assert output.getvalue() == ""
        return [_record(cost_usd=2.0)]

    monkeypatch.setattr(ccr, "load_all_records", load)
    monkeypatch.setattr(ccr, "_render_local_report", render)
    monkeypatch.setattr(ccr, "report_json", lambda *a, **kw: ccr.console.print('[{"cost": 2.0}]'))
    ccr.main()
    if as_json:
        assert json.loads(output.getvalue()) == [{"cost": 2.0}]
        assert events == ["refresh"]
    elif interactive:
        assert events == ["cached", "render 1.0", "refresh", "render 2.0"]
        assert output.getvalue().index("Updated report") > output.getvalue().index("SPEND 1.0")
    else:
        assert output.getvalue() == "SPEND 2.0\n"
        assert events == ["refresh", "render 2.0"]


def test_empty_preview_does_not_prevent_first_index(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(ccr, "console", Console(file=output, force_terminal=True, color_system=None))
    monkeypatch.setattr(ccr.sys, "argv", ["ccreport"])
    monkeypatch.setattr(ccr, "discover_jsonl_files", list)
    monkeypatch.setattr(ccr.exchange, "start_prefetch", lambda: None)
    with pytest.raises(SystemExit) as exc:
        ccr.main()
    assert exc.value.code == 1
    assert "No cached usage yet" in output.getvalue()
    data = json.loads(cache_db.DB_PATH.with_name("startup-timings.json").read_text())
    assert data["last_indexed_at"] is not None
    assert data["counts"]["discovered_files"] == 0
    assert data["phases"]["discovery"] >= 0


def test_diagnostics_keep_age_after_interrupt_and_replace_previous_run(monkeypatch):
    monkeypatch.setattr(startup.time, "time", lambda: 1_000_000.0)
    with startup.ReportRun() as first:
        assert "unknown" in first.age_message()
        first.indexing_completed()
        with startup.phase("discovery"):
            startup.count("discovered_files", 12)
    monkeypatch.setattr(startup.time, "time", lambda: 1_000_000.0 + 7 * 86400)
    with pytest.raises(KeyboardInterrupt), startup.ReportRun() as interrupted:  # noqa: PT012 - inspect live state
        assert "7 days ago" in interrupted.age_message()
        with startup.phase("discovery"):
            # The on-disk state identifies the blocking phase even after SIGKILL.
            live = json.loads(interrupted.path.read_text())
            assert live["active_phases"] == ["discovery"]
            raise KeyboardInterrupt
    data = json.loads(interrupted.path.read_text())
    assert data["status"] == "interrupted"
    assert data["stopped_phase"] == "discovery"
    assert data["last_indexed_at"] == 1_000_000.0
    assert data["counts"] == {}
    assert data["active_phases"] == []
    with startup.ReportRun() as last:
        assert "7 days ago" in last.age_message()
    assert json.loads(last.path.read_text())["phases"] == {}
    assert not list(last.path.parent.glob("startup-timings.*.tmp"))


@pytest.mark.parametrize("contents", ["not json", "[]", '{"last_indexed_at": "broken"}'])
def test_bad_diagnostic_file_is_ignored(contents):
    path = cache_db.DB_PATH.with_name("startup-timings.json")
    path.write_text(contents)
    with startup.ReportRun() as run:
        assert "unknown" in run.age_message()


def test_diagnostics_write_failure_does_not_fail_report(monkeypatch):
    # A parent that is a file cannot hold a diagnostic file.
    cache_db.DB_PATH.parent.joinpath("blocked").write_text("file")
    monkeypatch.setattr(cache_db, "DB_PATH", cache_db.DB_PATH.parent / "blocked" / "cache.db")
    with startup.ReportRun() as run, startup.phase("discovery"):
        startup.count("discovered_files", 3)
        run.indexing_completed()
    assert run.data["status"] == "complete"


def test_debug_durations_go_to_stderr(capsys):
    with startup.ReportRun(debug=True), startup.phase("discovery"):
        pass
    output = capsys.readouterr()
    assert output.out == ""
    assert "Startup timings:" in output.err
    assert "discovery:" in output.err
