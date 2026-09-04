from __future__ import annotations

import pytest

from ollama_watch import parse_line
from ollama_watch.events import (
    CacheVerdict,
    PeakMemory,
    Progress,
    RequestEnd,
    RunnerReady,
    SlotTiming,
    SpeculativeStats,
    Terminated,
)
from ollama_watch.parse import dur_seconds, parse_gin_ts, parse_kv

CACHE_MISS = (
    'time=2026-09-04T13:57:32.477+09:00 level=INFO source=prefix_cache.go:125 '
    'msg="cache miss" total=49077 matched=0 cached=0 left=49077'
)
CACHE_HIT = (
    'time=2026-09-04T14:02:51.952+09:00 level=INFO source=prefix_cache.go:125 '
    'msg="cache hit" total=49091 matched=14336 cached=14336 left=34755'
)
PROGRESS = (
    'time=2026-09-04T14:03:21.589+09:00 level=INFO source=pipeline.go:223 '
    'msg="Prompt processing progress" processed=2048 total=34755'
)
GIN_POST = '[GIN] 2026/09/04 - 14:00:21 | 500 |         2m58s |       127.0.0.1 | POST     "/v1/chat/completions"'
GIN_GET = '[GIN] 2026/09/04 - 14:09:08 | 200 |   51.388416ms |       127.0.0.1 | GET      "/api/ps"'
PEAK = (
    'time=2026-09-04T14:00:32.550+09:00 level=INFO source=pipeline.go:105 '
    'msg="peak memory" size="21.80 GiB"'
)
TERMINATED = (
    'time=2026-09-04T14:00:32.550+09:00 level=INFO source=runner.go:276 '
    'msg="Request terminated" error="context canceled"'
)
SPECULATIVE = (
    'time=2026-09-04T14:17:24.482+09:00 level=INFO source=speculate_stats.go:62 '
    'msg="speculative decode stats" iterations=133 drafted=502 accepted=424 '
    'acceptance=0.84 avg_draft=3.77 max_draft=5 avg_accepted=3.19'
)
SLOT_EVAL = (
    "slot print_timing: id  0 | task 122 |        eval time =      34.03 ms /     "
    "3 tokens (   17.01 ms per token,    58.77 tokens per second)"
)
SLOT_PROMPT = (
    "slot print_timing: id  0 | task 122 | prompt eval time =      50.37 ms /     "
    "5 tokens (   10.07 ms per token,    99.27 tokens per second)"
)
SLOT_TOTAL = "slot print_timing: id  0 | task 117 |       total time =     155.97 ms /    38 tokens"
READY = (
    'time=2026-09-04T13:57:32.438+09:00 level=INFO source=client.go:106 '
    'msg="mlx runner is ready" port=53737'
)


def test_cache_miss_reports_whole_prompt():
    event = parse_line(CACHE_MISS)
    assert isinstance(event, CacheVerdict)
    assert (event.total, event.cached, event.remaining) == (49077, 0, 49077)
    assert not event.hit


def test_cache_hit_separates_cached_from_remaining():
    event = parse_line(CACHE_HIT)
    assert isinstance(event, CacheVerdict)
    assert event.hit
    assert event.cached == 14336
    # the remaining count is what progress lines then count toward
    assert event.remaining == 34755
    assert event.cached + event.remaining == event.total


def test_progress_total_is_the_remaining_count():
    event = parse_line(PROGRESS)
    assert isinstance(event, Progress)
    assert (event.processed, event.remaining_total) == (2048, 34755)


def test_request_end_uses_its_own_clock_not_the_last_timestamp():
    event = parse_line(GIN_POST)
    assert isinstance(event, RequestEnd)
    assert event.status == "500"
    assert not event.ok
    assert event.duration_s == pytest.approx(178.0)
    assert event.ts == parse_gin_ts("2026/09/04", "14:00:21")


def test_non_inference_requests_are_ignored():
    assert parse_line(GIN_GET) is None


def test_peak_memory_and_termination():
    assert parse_line(PEAK) == PeakMemory(ts=parse_line(PEAK).ts, size="21.80 GiB")
    terminated = parse_line(TERMINATED)
    assert isinstance(terminated, Terminated)
    assert terminated.error == "context canceled"


def test_speculative_stats_derive_generated_tokens():
    event = parse_line(SPECULATIVE)
    assert isinstance(event, SpeculativeStats)
    assert (event.iterations, event.accepted) == (133, 424)
    assert event.acceptance == pytest.approx(0.84)
    # one target token per iteration, plus every accepted draft token
    assert event.generated_tokens == 424 + 133


@pytest.mark.parametrize(
    ("line", "kind", "tokens", "rate"),
    [
        (SLOT_EVAL, "eval", 3, 58.77),
        (SLOT_PROMPT, "prompt_eval", 5, 99.27),
        (SLOT_TOTAL, "total", 38, None),
    ],
)
def test_slot_timings(line, kind, tokens, rate):
    event = parse_line(line)
    assert isinstance(event, SlotTiming)
    assert event.task in (117, 122)
    assert event.kind == kind
    assert event.tokens == tokens
    if rate is None:
        assert event.tokens_per_s is None
    else:
        assert event.tokens_per_s == pytest.approx(rate)


def test_runner_ready():
    assert isinstance(parse_line(READY), RunnerReady)


def test_unremarkable_lines_are_dropped():
    assert parse_line("") is None
    assert parse_line("some unrelated output") is None
    assert parse_line('time=2026-09-04T13:59:19.217+09:00 msg="ServeHTTP" status="200 OK"') is None


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("2m58s", 178.0), ("5m0s", 300.0), ("41.254599167s", 41.254599167),
     ("63.454ms", 0.063454), ("18.666µs", 1.8666e-05), ("1h02m", 3720.0)],
)
def test_duration_parsing(text, seconds):
    assert dur_seconds(text) == pytest.approx(seconds)


def test_duration_parsing_rejects_junk():
    assert dur_seconds("") is None
    assert dur_seconds("--") is None


def test_parse_kv_unquotes():
    assert parse_kv('msg="peak memory" size="21.80 GiB"') == {
        "msg": "peak memory",
        "size": "21.80 GiB",
    }
