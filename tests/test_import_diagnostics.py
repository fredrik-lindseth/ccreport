"""Import notes appear after reports without Python warning boilerplate."""

import json
import warnings
from datetime import UTC, datetime

import pytest

from ccreport import aggregate
from ccreport import ccreport as ccr
from ccreport.sources import SourceImportWarning


@pytest.mark.parametrize(("tool", "expected"), [
    (None, ["Codex: Legacy logs were skipped.", "OpenCode: Keeping cached history."]),
    ("opencode", ["OpenCode: Keeping cached history."]),
    ("codex", ["Codex: Legacy logs were skipped."]),
    ("claude", []),
])
def test_import_notes_are_scoped_and_deduplicated(tool, expected, capsys):
    with ccr._import_diagnostics(tool) as notes:
        for _ in range(2):
            warnings.warn(SourceImportWarning("details", "codex", "Legacy logs were skipped."), stacklevel=2)
        warnings.warn(SourceImportWarning("details", "opencode", "Keeping cached history."), stacklevel=2)
        assert capsys.readouterr().err == ""
    assert notes == expected
    assert capsys.readouterr().err == ""


def test_unrelated_warnings_are_not_silenced():
    with pytest.warns(UserWarning, match="unrelated"), ccr._import_diagnostics("opencode"):
        warnings.warn("unrelated", UserWarning, stacklevel=2)


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_defers_import_note_until_after_report(monkeypatch, capsys, as_json):
    args = ["ccreport", "model", "--tool", "opencode"] + (["--json"] if as_json else [])
    monkeypatch.setattr(ccr.sys, "argv", args)
    rec = aggregate.UsageRecord(
        message_id="one", model="claude-opus-4.5", tokens=aggregate.TokenCounts(10, 5, 0, 0),
        timestamp=datetime(2026, 10, 15, tzinfo=UTC), session_id="s", project="p", source="opencode",
    )

    def load(**kwargs):
        warnings.warn(SourceImportWarning("old rollout", "codex", "Legacy logs were skipped."), stacklevel=2)
        warnings.warn(
            SourceImportWarning("bad database", "opencode", "Keeping cached history."), stacklevel=2,
        )
        assert capsys.readouterr().err == ""
        return [rec]

    def render(*args, **kwargs):
        assert capsys.readouterr().err == ""
        print('{"report": true}' if as_json else "MODEL REPORT")

    monkeypatch.setattr(ccr, "load_all_records", load)
    monkeypatch.setattr(ccr.exchange, "start_prefetch", lambda: None)
    monkeypatch.setattr(ccr, "_remote_records", lambda *a: ccr.RemoteSpend([], set()))
    monkeypatch.setattr(ccr, "load_rates_for_records", lambda *a, **kw: (aggregate.NokCtx(), True))
    monkeypatch.setattr(ccr, "render_usage_split", render)
    monkeypatch.setattr(ccr, "report_json", render)
    ccr.main()
    output = capsys.readouterr()
    assert output.err == "Import note: OpenCode: Keeping cached history.\n"
    assert "RuntimeWarning" not in output.err
    if as_json:
        assert json.loads(output.out) == {"report": True}
    else:
        assert "MODEL REPORT" in output.out
