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


# --- llama.cpp (GGUF) runner slot lines: bare stderr, no timestamps ---

SLOT_NEW_PROMPT = (
    "slot   operator(): id  0 | task 7 | new prompt, n_ctx_slot = 4096, "
    "n_keep = 4, task.n_tokens = 2050"
)
SLOT_PROMPT_HALF = (
    "slot print_timing: id  0 | task 7 | prompt processing, n_tokens =   1024, "
    "progress = 0.50, t =   7.78 s / 131.61 tokens per second"
)
SLOT_PROMPT_FULL = (
    "slot print_timing: id  0 | task 7 | prompt processing, n_tokens =   2046, "
    "progress = 1.00, t =  23.05 s / 88.75 tokens per second"
)
SLOT_INIT_SAMPLER = (
    "slot init_sampler: id  0 | task 7 | init sampler, took 0.55 ms, "
    "tokens: text = 2050, total = 2050"
)
SLOT_N_GEN = (
    "slot print_timing: id  0 | task 7 | n_gen =    100, tg =   4.80 t/s, "
    "tg_3s =   4.85 t/s"
)
SLOT_CACHED = (
    "slot   operator(): id  0 | task 7 | cached n_tokens = 1024, "
    "memory_seq_rm [1024, end)"
)
SLOT_RELEASE = (
    "slot      release: id  0 | task 7 | stop processing: n_tokens = 2475, "
    "truncated = 0"
)
SLOTS_IDLE = "srv  update_slots: all slots are idle"


def test_slot_new_prompt_starts_like_a_cache_miss():
    event = parse_line(SLOT_NEW_PROMPT, now=100.0)
    assert isinstance(event, CacheVerdict)
    assert (event.total, event.cached, event.remaining) == (2050, 0, 2050)
    assert event.ts == 100.0


def test_slot_prompt_progress_derives_the_total():
    event = parse_line(SLOT_PROMPT_HALF, now=100.0)
    assert isinstance(event, Progress)
    assert event.processed == 1024
    assert event.remaining_total == 2048  # int(1024 / 0.50)


def test_slot_init_sampler_marks_prefill_done():
    from ollama_watch.events import PrefillDone

    event = parse_line(SLOT_INIT_SAMPLER, now=100.0)
    assert isinstance(event, PrefillDone)
    assert event.total == 2050


def test_slot_n_gen_is_a_live_decode_tick():
    from ollama_watch.events import DecodeTick

    event = parse_line(SLOT_N_GEN, now=100.0)
    assert isinstance(event, DecodeTick)
    assert (event.generated, event.rate) == (100, pytest.approx(4.80))
    # `tg` averages the request, `tg_3s` says what the rate is doing now
    assert event.recent_rate == pytest.approx(4.85)
    assert event.current_rate == pytest.approx(4.85)


def test_a_decode_tick_without_a_recent_rate_falls_back_to_the_average():
    event = parse_line(
        "slot print_timing: id  0 | task 7 | n_gen =    100, tg =   4.80 t/s", now=100.0
    )
    assert event.recent_rate is None
    assert event.current_rate == pytest.approx(4.80)


def test_slot_bookkeeping_lines_are_dropped():
    assert parse_line(SLOT_CACHED, now=100.0) is None
    assert parse_line(SLOT_RELEASE, now=100.0) is None
    assert parse_line(SLOTS_IDLE, now=100.0) is None


def test_slot_summaries_still_parse():
    # the end-of-request `prompt eval time` / `eval time` lines are unchanged
    event = parse_line(SLOT_PROMPT, now=100.0)
    assert isinstance(event, SlotTiming) and event.kind == "prompt_eval"


class TestRequestBoundaries:
    """Both runners open and close a request with lines of their own; seeding
    reads them to tell a request still in flight from one already finished."""

    def test_begins_on_either_runner(self):
        from ollama_watch.parse import begins_request

        assert begins_request(CACHE_MISS)  # mlx
        assert begins_request(CACHE_HIT)
        assert begins_request(SLOT_NEW_PROMPT)  # llama.cpp
        assert not begins_request(PROGRESS)
        assert not begins_request(GIN_GET)

    def test_ends_on_either_runner(self):
        from ollama_watch.parse import ends_request

        assert ends_request(GIN_POST)  # the HTTP completion, either runner
        assert ends_request(TERMINATED)
        assert ends_request(SLOTS_IDLE)  # llama.cpp released its slot
        assert ends_request(SLOT_RELEASE)
        assert not ends_request(GIN_GET)  # bookkeeping, not inference
        assert not ends_request(SLOT_N_GEN)
        assert not ends_request(CACHE_HIT)

    def test_line_ts_reads_the_clock_a_line_carries(self):
        from ollama_watch.parse import line_ts, parse_gin_ts, parse_ts

        assert line_ts(GIN_POST) == parse_gin_ts("2026/09/04", "14:00:21")
        assert line_ts(CACHE_HIT) == parse_ts("2026-09-04T14:02:51.952+09:00")
        assert line_ts(SLOT_N_GEN) is None  # llama.cpp slot lines carry none
