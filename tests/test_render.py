"""Rendering tests: labels, formatting, and colour-free output."""

from __future__ import annotations

import pytest

from ollama_watch import Phase, Tracker, format_receipt, format_status
from ollama_watch.client import ModelInfo
from ollama_watch.events import CacheVerdict, Progress, RequestEnd, SlotTiming
from ollama_watch.render import PHASE_LABELS, Style, bar, human_duration, human_tokens

T0 = 1_000_000.0
PLAIN = Style(enabled=False)
MODEL = ModelInfo(
    name="qwen3.8:27b-mlx",
    size_bytes=22_000_000_000,
    size_vram_bytes=22_000_000_000,
    context_length=32768,
    forever=True,
)


def test_phase_labels_are_user_facing():
    assert PHASE_LABELS[Phase.PREFILL] == "Processing input"
    assert PHASE_LABELS[Phase.DECODE] == "Generating"


def test_status_line_shows_absolute_progress_and_cache():
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=49091, cached=34816, remaining=14275))
    tracker.feed(Progress(ts=T0 + 30, processed=8192, remaining_total=14275))
    line = format_status(tracker.state, MODEL, PLAIN, now=T0 + 30, columns=120)
    assert "Processing input" in line
    assert "qwen3.8:27b-mlx" in line
    assert "cached 34.8k" in line
    assert "43.0k/49.1k" in line
    assert "87.6%" in line


def test_relative_totals_are_labelled_new():
    tracker = Tracker()
    tracker.feed(Progress(ts=T0, processed=8192, remaining_total=14275))
    line = format_status(tracker.state, MODEL, PLAIN, now=T0, columns=120)
    assert "14.3k new" in line


def test_generating_line_reports_elapsed_and_last_rate():
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=50400, cached=50399, remaining=1))
    line = format_status(
        tracker.state, MODEL, PLAIN, now=T0 + 12.4, last_decode_rate=24.0, columns=120
    )
    assert "Generating" in line
    assert "12.4s" in line
    assert "last ~24 tok/s" in line


def test_idle_line():
    line = format_status(Tracker().state, MODEL, PLAIN, now=T0, columns=120)
    assert "Idle" in line and "waiting" in line


def test_no_model_loaded():
    assert "no model loaded" in format_status(Tracker().state, None, PLAIN, columns=120)


def test_receipt_reports_both_directions():
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=50900, cached=50554, remaining=346))
    tracker.feed(Progress(ts=T0 + 6, processed=345, remaining_total=346))
    tracker.feed(
        RequestEnd(ts=T0 + 20, status="200", duration_s=20.0, raw_duration="20s",
                   path="/v1/chat/completions", method="POST")
    )
    tracker.feed(SlotTiming(ts=T0 + 20, task=1, kind="eval", ms=14_000.0, tokens=350, tokens_per_s=25.0))
    text = format_receipt(tracker.flush()[0], PLAIN)
    assert "200" in text
    assert "input 50.9k" in text
    assert "cached 50.6k" in text
    assert "processed 345" in text
    assert "out 350" in text
    assert "@ 25 tok/s" in text


def test_approximate_generation_is_marked():
    from ollama_watch.events import SpeculativeStats

    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=5000, cached=0, remaining=5000))
    tracker.feed(Progress(ts=T0 + 30, processed=4999, remaining_total=5000))
    tracker.feed(
        RequestEnd(ts=T0 + 60, status="200", duration_s=60.0, raw_duration="60s",
                   path="/v1/chat/completions", method="POST")
    )
    tracker.feed(SpeculativeStats(ts=T0 + 60, iterations=100, drafted=300, accepted=250, acceptance=0.83))
    text = format_receipt(tracker.flush()[0], PLAIN)
    assert "out ~350" in text
    assert "mtp_accept_rate 0.83" in text


def test_plain_style_emits_no_escapes():
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=49077, cached=0, remaining=49077))
    tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=49077))
    assert "\033" not in format_status(tracker.state, MODEL, PLAIN, columns=120)


def test_colour_style_emits_escapes():
    coloured = Style(enabled=True)
    assert "\033[" in format_status(Tracker().state, MODEL, coloured, columns=120)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0.015, "15ms"), (41.2, "41.2s"), (178.0, "2m58s"), (3720.0, "1h02m"), (None, "--")],
)
def test_duration_formatting(seconds, text):
    assert human_duration(seconds) == text


@pytest.mark.parametrize(("count", "text"), [(346, "346"), (9999, "9999"), (49091, "49.1k")])
def test_token_formatting(count, text):
    assert human_tokens(count) == text


def test_bar_is_clamped():
    assert bar(0.0, 10) == "░" * 10
    assert bar(1.0, 10) == "█" * 10
    assert bar(5.0, 10) == "█" * 10
    assert bar(-1.0, 10) == "░" * 10
    assert len(bar(0.5, 10)) == 10


def test_context_chip_flags_a_prompt_beyond_the_declared_window():
    """The MLX engine grows its KV cache past the declared context rather than
    truncating, so the display must not imply a limit that is not enforced."""
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=63307, cached=53200, remaining=10107))
    line = format_status(tracker.state, MODEL, PLAIN, now=T0, columns=140)
    assert "ctx32768<63.3k" in line


def test_context_chip_is_plain_when_within_the_window():
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=20000, cached=0, remaining=20000))
    line = format_status(tracker.state, MODEL, PLAIN, now=T0, columns=140)
    assert "ctx32768" in line and "<" not in line


def test_receipt_groups_are_delimited():
    """status | input | output | memory -- so adjacent fields cannot be misread
    as one, e.g. `mtp_accept_rate` running into `peak`."""
    from ollama_watch.events import PeakMemory, SpeculativeStats

    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=6931, cached=0, remaining=6931))
    tracker.feed(Progress(ts=T0 + 40, processed=2048, remaining_total=6931))
    tracker.feed(Progress(ts=T0 + 77, processed=6930, remaining_total=6931))
    tracker.feed(
        RequestEnd(ts=T0 + 151, status="200", duration_s=151.0, raw_duration="2m31s",
                   path="/v1/chat/completions", method="POST")
    )
    tracker.feed(SpeculativeStats(ts=T0 + 151, iterations=300, drafted=1100, accepted=892, acceptance=0.78))
    tracker.feed(PeakMemory(ts=T0 + 151.1, size="32.49 GiB"))
    text = format_receipt(tracker.flush()[0], PLAIN)

    groups = [group.strip() for group in text.split("|")]
    assert len(groups) == 4
    assert groups[1].startswith("input")
    assert groups[2].startswith("out")
    assert groups[3] == "peak 32.49 GiB"
    assert "mtp_accept_rate 0.78" in groups[2]
