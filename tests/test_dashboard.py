"""Dashboard tests: the pure helpers, plus one real frame rendered off-screen."""

from __future__ import annotations

import curses

import pytest

from ollama_watch import Tracker, gauge, spark
from ollama_watch.dashboard import Dashboard, parse_size
from ollama_watch.events import CacheVerdict, PeakMemory, Progress, RequestEnd, SpeculativeStats
from ollama_watch.client import ModelInfo
from ollama_watch.watch import Update

MODEL = ModelInfo(
    name="qwen3.8:27b-mlx",
    size_bytes=32_000_000_000,
    size_vram_bytes=32_000_000_000,
    context_length=32768,
    forever=True,
)

T0 = 1_000_000.0


def test_spark_scales_to_its_own_peak():
    assert spark([0, 50, 100], 8) == "▁▄█"
    assert spark([5, 5, 5], 8) == "███"
    assert spark([], 8) == ""


def test_spark_keeps_only_the_most_recent_values():
    assert len(spark(list(range(100)), 10)) == 10


def test_spark_survives_all_zero_history():
    assert spark([0, 0], 8) == "▁▁"


def test_gauge_is_clamped():
    assert gauge(0.0, 4) == "░░░░"
    assert gauge(1.0, 4) == "████"
    assert gauge(9.9, 4) == "████"
    assert gauge(-1.0, 4) == "░░░░"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("32.49 GiB", 32.49 * 1024**3), ("18 GB", 18e9), ("512 MiB", 512 * 1024**2)],
)
def test_parse_size(text, expected):
    assert parse_size(text) == pytest.approx(expected)


def test_parse_size_rejects_junk():
    assert parse_size(None) is None
    assert parse_size("unknown") is None


def finished_receipt():
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
    return tracker.flush()[0], tracker


class FakeScreen:
    """The four curses calls the dashboard makes, captured as text."""

    def __init__(self, height: int = 24, width: int = 100) -> None:
        self.height, self.width = height, width
        self.rows: dict[int, str] = {}

    def getmaxyx(self):
        return self.height, self.width

    def erase(self):
        self.rows.clear()

    def addnstr(self, row, col, text, n, attr=0):
        if row >= self.height:
            raise curses.error("out of bounds")
        self.rows[row] = text[:n]

    def refresh(self):
        pass

    @property
    def text(self) -> str:
        return "\n".join(self.rows[key] for key in sorted(self.rows))


def test_frame_records_history_without_double_counting():
    receipt, tracker = finished_receipt()
    screen = FakeScreen()
    dashboard = Dashboard(screen)
    dashboard.draw(Update(tracker.state.snapshot(), receipts=[receipt]))

    assert list(dashboard.input_rates) == [pytest.approx(receipt.prefill_rate)]
    assert list(dashboard.output_rates) == [pytest.approx(receipt.decode_rate)]
    assert dashboard.peak_bytes == pytest.approx(32.49 * 1024**3)
    assert len(dashboard.receipts) == 1

    dashboard.draw(Update(tracker.state.snapshot()))  # no new receipts
    assert len(dashboard.receipts) == 1


def test_frame_shows_the_expected_panels():
    receipt, tracker = finished_receipt()
    screen = FakeScreen()
    dashboard = Dashboard(screen)
    dashboard.draw(Update(tracker.state.snapshot(), model=MODEL, receipts=[receipt]))
    text = dashboard.screen.text

    assert "ollama-watch" in text
    assert "qwen3.8:27b-mlx" in text
    assert "idle, waiting for request" in text
    assert "recent" in text
    assert "1 req" in text and "q quit" in text
    assert "GiB peak" in text


def test_live_prefill_frame_reports_progress():
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=49091, cached=34816, remaining=14275))
    tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=14275))
    tracker.feed(Progress(ts=T0 + 60, processed=8192, remaining_total=14275))
    screen = FakeScreen()
    Dashboard(screen).draw(Update(tracker.state.snapshot(), model=MODEL))
    text = screen.text

    assert "Processing input" in text
    assert "43.0k/49.1k" in text
    assert "cached 34.8k" in text
    assert "eta" in text


def test_frame_fits_a_short_terminal():
    """A window shorter than the layout must clip, not raise."""
    receipt, tracker = finished_receipt()
    screen = FakeScreen(height=6, width=40)
    Dashboard(screen).draw(Update(tracker.state.snapshot(), model=MODEL, receipts=[receipt]))
    assert max(screen.rows) < 6


def test_memory_gauge_colours_by_pressure():
    receipt, tracker = finished_receipt()
    screen = FakeScreen()
    dashboard = Dashboard(screen)
    dashboard.total_bytes = 36 * 1024**3  # peak is 32.49 GiB -> over 90%
    dashboard.draw(Update(tracker.state.snapshot(), receipts=[receipt]))
    assert "90% of 36 GiB" in screen.text
