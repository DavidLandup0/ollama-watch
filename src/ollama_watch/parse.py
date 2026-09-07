from __future__ import annotations

import re
import time
from datetime import datetime

from .events import (
    CacheVerdict,
    DecodeTick,
    Event,
    PeakMemory,
    PrefillDone,
    Progress,
    RequestEnd,
    RunnerReady,
    SlotTiming,
    SpeculativeStats,
    Terminated,
)

RE_KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
RE_GIN = re.compile(
    r'\[GIN\]\s+(\S+)\s+-\s+(\S+)\s+\|\s+(\d{3})\s+\|\s+(\S+)\s+\|\s+\S+\s+\|\s+(\w+)\s+"([^"]+)"'
)
RE_DUR = re.compile(r"([\d.]+)(h|ms|µs|us|m|s)")
RE_SLOT = re.compile(
    r"slot print_timing:.*?task\s+(\d+)\s+\|\s+(prompt eval time|eval time|total time)\s+=\s+"
    r"([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens(?:.*?([\d.]+)\s+tokens per second)?"
)
#: The llama.cpp (GGUF) runner's slot lines, written bare to stderr with no
#: timestamp of their own and so read at `now`. The MLX runner reports the
#: same milestones as slog lines, handled below.
RE_NEW_PROMPT = re.compile(r"new prompt,.*?task\.n_tokens\s*=\s*(\d+)")
RE_PROMPT_PROGRESS = re.compile(
    r"prompt processing,\s*n_tokens\s*=\s*(\d+),\s*progress\s*=\s*([\d.]+)"
)
RE_INIT_SAMPLER = re.compile(r"init sampler,.*?tokens:\s*text\s*=\s*(\d+),\s*total\s*=\s*(\d+)")
RE_N_GEN = re.compile(
    r"n_gen\s*=\s*(\d+),\s*tg\s*=\s*([\d.]+)\s*t/s"
    r"(?:,\s*tg_3s\s*=\s*([\d.]+)\s*t/s)?"
)

_SLOT_KINDS = {"prompt eval time": "prompt_eval", "eval time": "eval", "total time": "total"}

_DUR_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "µs": 1e-6, "us": 1e-6}

#: Paths that carry inference work. A POST completing on one of these ends a
#: request; anything else (/api/ps, /v1/status) is bookkeeping noise.
REQUEST_PATHS = ("/api/generate", "/api/chat", "/v1/completions", "/v1/chat/completions")

RUNNER_READY = ("mlx runner is ready", "llama runner started")

CACHE_MSGS = ("cache hit", "cache miss")


def parse_kv(line: str) -> dict[str, str]:
    """Decode slog `key=value` pairs, unquoting values."""
    return {k: v.strip('"') for k, v in RE_KV.findall(line)}


def parse_ts(raw: str | None) -> float | None:
    """Parse a slog `time=` value (ISO 8601 with offset) into epoch seconds."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.strip('"')).timestamp()
    except ValueError:
        return None


def parse_gin_ts(date: str, clock: str) -> float | None:
    """Parse a Gin access line's own `2026/09/04 - 14:00:21` clock.

    Gin lines carry no timezone, so they are read as local time. They must be
    read from the line itself: matching a completion against the most recent
    `time=` line instead drifts by however long the request ran.
    """
    try:
        return datetime.strptime(f"{date} {clock}", "%Y/%m/%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def dur_seconds(text: str) -> float | None:
    """Parse a Gin duration (`2m58s`, `63.454ms`, `18.666µs`) into seconds."""
    total, found = 0.0, False
    for value, unit in RE_DUR.findall(text):
        found = True
        total += float(value) * _DUR_UNITS[unit]
    return total if found else None


def _int(kv: dict[str, str], key: str) -> int:
    try:
        return int(kv.get(key, 0) or 0)
    except ValueError:
        return 0


def parse_line(line: str, *, now: float | None = None) -> Event | None:
    """Decode one log line, or return None if it carries nothing we track."""
    now = time.time() if now is None else now
    gin = RE_GIN.search(line)
    if gin:
        date, clock, status, raw_duration, method, path = gin.groups()
        if method != "POST" or not path.startswith(REQUEST_PATHS):
            return None
        return RequestEnd(
            ts=parse_gin_ts(date, clock) or now,
            status=status,
            duration_s=dur_seconds(raw_duration),
            raw_duration=raw_duration,
            path=path,
            method=method,
        )

    slot = RE_SLOT.search(line)
    if slot:
        task, label, ms, tokens, tps = slot.groups()
        return SlotTiming(
            ts=now,
            task=int(task),
            kind=_SLOT_KINDS[label],
            ms=float(ms),
            tokens=int(tokens),
            tokens_per_s=float(tps) if tps else None,
        )

    new_prompt = RE_NEW_PROMPT.search(line)
    if new_prompt:
        # A request started; nothing is known cached, so the whole prompt
        # is the work left -- the same shape as a cache-miss verdict.
        total = int(new_prompt.group(1))
        return CacheVerdict(ts=now, total=total, cached=0, remaining=total)

    progress = RE_PROMPT_PROGRESS.search(line)
    if progress:
        processed, fraction = int(progress.group(1)), float(progress.group(2))
        total = int(processed / fraction) if fraction > 0 else processed
        return Progress(ts=now, processed=processed, remaining_total=total)

    sampler = RE_INIT_SAMPLER.search(line)
    if sampler:
        return PrefillDone(ts=now, total=int(sampler.group(2)))

    tick = RE_N_GEN.search(line)
    if tick:
        generated, rate, recent = tick.groups()
        return DecodeTick(
            ts=now,
            generated=int(generated),
            rate=float(rate),
            recent_rate=float(recent) if recent else None,
        )

    kv = parse_kv(line)
    msg = kv.get("msg")
    if not msg:
        return None
    ts = parse_ts(kv.get("time")) or now

    if msg in CACHE_MSGS:
        return CacheVerdict(
            ts=ts,
            total=_int(kv, "total"),
            cached=_int(kv, "cached"),
            remaining=_int(kv, "left"),
        )
    if msg == "Prompt processing progress":
        return Progress(
            ts=ts, processed=_int(kv, "processed"), remaining_total=_int(kv, "total")
        )
    if msg == "peak memory":
        return PeakMemory(ts=ts, size=kv.get("size", ""))
    if msg == "Request terminated":
        return Terminated(ts=ts, error=kv.get("error") or "terminated")
    if msg == "speculative decode stats":
        acceptance = kv.get("acceptance")
        return SpeculativeStats(
            ts=ts,
            iterations=_int(kv, "iterations"),
            drafted=_int(kv, "drafted"),
            accepted=_int(kv, "accepted"),
            acceptance=float(acceptance) if acceptance else None,
        )
    if msg in RUNNER_READY:
        return RunnerReady(ts=ts)
    return None


def line_ts(line: str) -> float | None:
    """The timestamp a line carries of its own, if any.

    Gin access lines and slog lines are dated; the llama.cpp runner's slot
    lines are not. Replaying history reads those at the clock of the last
    dated line before them, rather than at the wall clock of the replay.
    """
    gin = RE_GIN.search(line)
    if gin:
        return parse_gin_ts(gin.group(1), gin.group(2))
    return parse_ts(parse_kv(line).get("time"))


#: Lines that open a request, whichever runner serves it: the MLX runner logs
#: a prefix-cache verdict, the llama.cpp (GGUF) runner a `new prompt`.
BEGIN_MARKERS = ('msg="cache hit"', 'msg="cache miss"', "new prompt,")

#: Lines that close one. The llama.cpp runner releases its slot and reports
#: itself idle; either runner's HTTP completion is logged by Gin, and a client
#: that walks away is logged by the scheduler.
END_MARKERS = ("all slots are idle", "stop processing:", 'msg="Request terminated"')


def begins_request(line: str) -> bool:
    """True when the line opens a request on either runner."""
    return any(marker in line for marker in BEGIN_MARKERS)


def ends_request(line: str) -> bool:
    """True when the line closes a request: the runner finishing with it, or
    the HTTP completion Gin logs once the response is out."""
    if any(marker in line for marker in END_MARKERS):
        return True
    return isinstance(parse_line(line), RequestEnd)
