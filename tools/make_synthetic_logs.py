#!/usr/bin/env python3
"""Write synthetic Claude Code logs, with optional Codex rollouts alongside them.

Run it with `uv run python tools/make_synthetic_logs.py <out_dir>`; it imports
ccreport.pricing for the projects-dir encoding rather than spelling that rule a
second time.

Every field written is one the reader consumes, so a test that asserts on the
returned Expected list is asserting on the reader's whole input surface. The
skip paths get input too: a blank line, a line that is not JSON, a user turn, a
summary, an assistant line with no usage block and one with an unparseable
timestamp. None of those six appear in the returned list.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ccreport.pricing import project_key

# Anchoring the newest record to a fixed instant is what makes the seed the
# whole input: a corpus ending at "now" differs on every run and cannot be
# compared against a stored expectation.
ANCHOR_END = datetime(2026, 9, 1, 17, 0, tzinfo=UTC)

# Priced by pricing.PRICING_HISTORY from 2026-07-24 (opus 5), 2026-06-01
# (sonnet 5) and 2025-01-01 (haiku 4.5), so a corpus ending at ANCHOR_END
# prices in full. Reaching further back than 2026-07-24 leaves opus 5 unpriced.
MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001")

PROJECT_NAMES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")

_ID_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz0123456789"


@dataclass(frozen=True)
class Expected:
    """One assistant record as parse_jsonl_file should hand it back."""

    path: Path
    project: str
    cwd: str
    session_id: str
    model: str
    message_id: str
    request_id: str
    timestamp: datetime
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    cost_usd: float | None

    @property
    def dedup_key(self) -> str:
        return f"{self.message_id}:{self.request_id}"


def _token(rng: random.Random, prefix: str, length: int) -> str:
    return prefix + "".join(rng.choice(_ID_ALPHABET) for _ in range(length))


def _project_names(count: int) -> list[str]:
    names = list(PROJECT_NAMES[:count])
    names += [f"proj{i}" for i in range(len(names), count)]
    return names


def _make_cwd(out_dir: Path, project: str) -> str:
    """Create the working directory a record was logged from.

    The .git stub is what keeps the name deterministic: scan._resolve_remote
    shells out to git, which walks up past a plain directory and would answer
    with the remote of whatever repo out_dir happens to sit inside. A .git
    pointing at nothing makes git refuse instead of ascending.
    """
    cwd = out_dir / "repos" / project
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / ".git").write_text(f"gitdir: {out_dir / '.no-such-gitdir'}\n")
    return str(cwd)


def _user_line(cwd: str, session_id: str, ts: datetime, text: str) -> dict:
    return {
        "type": "user",
        "cwd": cwd,
        "sessionId": session_id,
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _assistant_line(exp: Expected) -> dict:
    line = {
        "type": "assistant",
        "cwd": exp.cwd,
        "sessionId": exp.session_id,
        "requestId": exp.request_id,
        "timestamp": exp.timestamp.isoformat().replace("+00:00", "Z"),
        "message": {
            "id": exp.message_id,
            "role": "assistant",
            "model": exp.model,
            "usage": {
                "input_tokens": exp.input_tokens,
                "output_tokens": exp.output_tokens,
                "cache_creation_input_tokens": exp.cache_creation_tokens,
                "cache_read_input_tokens": exp.cache_read_tokens,
            },
        },
    }
    # Omitted rather than written as null, the way a log without a priced call
    # leaves it out; the reader treats a missing key and a null the same.
    if exp.cost_usd is not None:
        line["costUSD"] = exp.cost_usd
    return line


def _skipped_lines(cwd: str, session_id: str, ts: datetime) -> list[dict]:
    """Assistant-shaped records extract_assistant_fields rejects.

    One has no usage block, the other a timestamp datetime.fromisoformat
    cannot read; both must leave no record behind.
    """
    stamp = ts.isoformat().replace("+00:00", "Z")
    return [
        {
            "type": "assistant",
            "cwd": cwd,
            "sessionId": session_id,
            "requestId": "req_no_usage",
            "timestamp": stamp,
            "message": {"id": "msg_no_usage", "role": "assistant", "model": MODELS[0]},
        },
        {
            "type": "assistant",
            "cwd": cwd,
            "sessionId": session_id,
            "requestId": "req_bad_ts",
            "timestamp": "not-a-timestamp",
            "message": {
                "id": "msg_bad_ts",
                "role": "assistant",
                "model": MODELS[0],
                "usage": {"input_tokens": 5, "output_tokens": 5,
                          "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            },
        },
    ]


def _session_records(
    rng: random.Random, path: Path, project: str, cwd: str, session_id: str, start: datetime,
) -> list[Expected]:
    out: list[Expected] = []
    ts = start
    for _ in range(rng.randint(3, 8)):
        ts += timedelta(seconds=rng.randint(20, 900), microseconds=rng.randrange(1_000_000))
        # Roughly two calls in three carry costUSD, as Claude Code's own logs do;
        # the rest exercise the branch where the reader has to price from tokens.
        cost = round(rng.uniform(0.002, 0.9), 6) if rng.random() < 0.66 else None
        out.append(Expected(
            path=path,
            project=project,
            cwd=cwd,
            session_id=session_id,
            model=rng.choice(MODELS),
            message_id=_token(rng, "msg_01", 22),
            request_id=_token(rng, "req_011", 21),
            timestamp=ts,
            input_tokens=rng.randint(2, 90),
            output_tokens=rng.randint(15, 1800),
            cache_creation_tokens=rng.choice([0, rng.randint(200, 30_000)]),
            cache_read_tokens=rng.choice([0, rng.randint(1_000, 400_000)]),
            cost_usd=cost,
        ))
    return out


def _write_session(path: Path, records: list[Expected], cwd: str, session_id: str) -> None:
    lines: list[str] = []
    first = records[0].timestamp
    lines.append(json.dumps(_user_line(cwd, session_id, first - timedelta(seconds=5), "hello")))
    for i, exp in enumerate(records):
        lines.append(json.dumps(_user_line(cwd, session_id, exp.timestamp, f"turn {i}")))
        lines.append(json.dumps(_assistant_line(exp)))
        if i == 0:
            lines.append("")
            lines.append('{"type": "assistant", "message": {"id": "msg_trunc"')
            lines.extend(json.dumps(d) for d in _skipped_lines(cwd, session_id, exp.timestamp))
    lines.append(json.dumps({
        "type": "summary",
        "summary": f"Session {session_id[:8]}",
        "leafUuid": str(uuid.UUID(int=0)),
    }))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_corpus(
    out_dir: Path,
    *,
    seed: int = 0,
    projects: int = 3,
    sessions: int = 3,
    days: int = 7,
    end: datetime = ANCHOR_END,
) -> list[Expected]:
    """Write a projects tree under *out_dir* and return its assistant records.

    Sessions are dealt to days round-robin, so every one of *days* carries
    records whenever projects * sessions >= days and a report has something to
    group on each of them.
    """
    if min(projects, sessions, days) < 1:
        raise ValueError("projects, sessions and days must each be at least 1")
    rng = random.Random(seed)
    root = out_dir / "projects"
    expected: list[Expected] = []
    slot = 0
    for project in _project_names(projects):
        cwd = _make_cwd(out_dir, project)
        session_dir = root / project_key(cwd)
        for _ in range(sessions):
            session_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
            path = session_dir / f"{session_id}.jsonl"
            day_start = (end - timedelta(days=days - 1 - slot % days)).replace(
                hour=rng.randint(8, 15), minute=rng.randint(0, 59), second=0, microsecond=0)
            records = _session_records(rng, path, project, cwd, session_id, day_start)
            _write_session(path, records, cwd, session_id)
            expected.extend(records)
            slot += 1
    return expected


@dataclass(frozen=True)
class CodexExpected:
    """One independently specified call in the small Codex compatibility corpus."""

    path: Path
    response_id: str
    session_id: str
    thread_id: str
    model: str
    account_uuid: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int


def _codex_event(kind: str, payload: dict, ordinal: int) -> dict:
    return {"type": kind, "payload": payload, "ordinal": ordinal,
            "timestamp": "2026-10-15T12:00:00Z"}


def _codex_call(exp: CodexExpected, turn: str, ordinal: int) -> dict:
    return _codex_event("token_usage_record", {
        "response_id": exp.response_id, "session_id": exp.session_id,
        "thread_id": exp.thread_id, "turn_id": turn,
        "usage": {
            "input_tokens": exp.input_tokens,
            "cached_input_tokens": exp.cached_input_tokens,
            "cache_write_input_tokens": 0,
            "output_tokens": exp.output_tokens,
            "reasoning_output_tokens": exp.reasoning_output_tokens,
            "total_tokens": exp.input_tokens + exp.output_tokens,
        },
    }, ordinal)


def write_codex_corpus(out_dir: Path, *, seed: int = 0) -> list[CodexExpected]:
    """Write deterministic root, fork and guardian rollouts under a Codex home.

    This is separate from write_corpus so existing Claude fixtures retain their
    exact shape. Both token event forms, model changes, inherited parent history,
    missing turn context, archived sessions and unknown accounts are included.
    The returned counts are per call; cumulative events add no expected rows.
    """
    rng = random.Random(seed)
    root, child, guardian = (str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(3))
    active = out_dir / "sessions/2026/10/15"
    root_path, child_path = active / f"rollout-{root}.jsonl", active / f"rollout-{child}.jsonl"
    guardian_path = out_dir / "archived_sessions" / f"rollout-{guardian}.jsonl"
    expected = [
        CodexExpected(root_path, _token(rng, "resp_", 16), root, root,
                      "gpt-6-astra", "synthetic-chatgpt-account", 1000, 800, 100, 40),
        CodexExpected(root_path, _token(rng, "resp_", 16), root, root,
                      "gpt-6.1-sol", "synthetic-chatgpt-account", 2000, 1800, 200, 70),
        CodexExpected(child_path, _token(rng, "resp_", 16), root, child,
                      "gpt-5.6-terra", "synthetic-chatgpt-account", 3000, 2800, 150, 50),
        CodexExpected(child_path, _token(rng, "resp_", 16), root, child,
                      "gpt-5.6-terra", "synthetic-chatgpt-account", 4000, 3800, 250, 90),
        CodexExpected(guardian_path, _token(rng, "resp_", 16), root, guardian,
                      "codex-auto-review", None, 500, 400, 20, 5),
    ]
    cwd = _make_cwd(out_dir, "alpha")
    root_meta = {
        "id": root, "session_id": root, "cwd": cwd, "thread_source": "user",
        "creator_account_id": "synthetic-chatgpt-account",
        "git": {"repository_url": "https://example.test/synthetic/alpha.git"},
    }
    root_lines = [
        _codex_event("session_meta", root_meta, 0),
        _codex_event("turn_context", {"turn_id": "first", "model": "gpt-6-astra"}, 1),
        _codex_call(expected[0], "first", 2),
        _codex_event("turn_context", {"turn_id": "second", "model": "gpt-6.1-sol"}, 3),
        _codex_call(expected[1], "second", 4),
    ]
    child_lines = [
        _codex_event("session_meta", {
            **root_meta, "id": child, "thread_source": "subagent", "forked_from_id": root,
            "parent_thread_id": root, "subagent_history_start_ordinal": 4,
        }, 0),
        _codex_event("session_meta", root_meta, 1),
        _codex_event("turn_context", {"turn_id": "first", "model": "gpt-6-astra"}, 2),
        _codex_call(expected[0], "first", 3),
        _codex_event("turn_context", {"turn_id": "child", "model": "gpt-5.6-terra"}, 4),
        _codex_call(expected[2], "child", 5),
        _codex_call(expected[3], "missing-context", 6),
    ]
    guardian_lines = [
        _codex_event("session_meta", {
            **root_meta, "id": guardian, "thread_source": "guardian_review", "creator_account_id": None,
        }, 0),
        _codex_event("turn_context", {"turn_id": "review", "model": "codex-auto-review"}, 1),
        _codex_call(expected[4], "review", 2),
    ]
    for path, lines in ((root_path, root_lines), (child_path, child_lines), (guardian_path, guardian_lines)):
        total = {"input_tokens": 0, "output_tokens": 0}
        written = []
        for line in lines:
            written.append(line)
            if line["type"] == "token_usage_record":
                usage = line["payload"]["usage"]
                for key in total:
                    total[key] += usage[key]
                counter = _codex_event("event_msg", {
                    "type": "token_count",
                    "info": {"total_token_usage": dict(total), "last_token_usage": usage},
                }, line["ordinal"])
                written.extend([counter, counter])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(line) for line in written) + "\n")
    return expected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="directory to write projects/ and repos/ into")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--projects", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=3, help="sessions per project")
    ap.add_argument("--days", type=int, default=7, help="how many days the records span")
    ap.add_argument("--codex", action="store_true", help="also write a small Codex corpus under codex/")
    ap.add_argument("--end", type=datetime.fromisoformat, default=ANCHOR_END,
                    help=f"instant the newest record sits before (default {ANCHOR_END.isoformat()})")
    args = ap.parse_args()

    end = args.end if args.end.tzinfo else args.end.replace(tzinfo=UTC)
    records = write_corpus(
        args.out_dir, seed=args.seed, projects=args.projects,
        sessions=args.sessions, days=args.days, end=end,
    )
    files = sorted({r.path for r in records})
    print(f"{len(records)} assistant records in {len(files)} sessions under {args.out_dir / 'projects'}")
    if args.codex:
        codex_records = write_codex_corpus(args.out_dir / "codex", seed=args.seed)
        print(f"{len(codex_records)} Codex calls under {args.out_dir / 'codex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
