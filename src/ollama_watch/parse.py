from __future__ import annotations

import re
import time
from datetime import datetime

from .events import (
    CacheVerdict,
    Event,
    PeakMemory,
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
