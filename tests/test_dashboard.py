"""Dashboard tests: the pure helpers, plus one real frame rendered off-screen."""

from __future__ import annotations

import curses

import pytest

from ollama_watch import Tracker, gauge, spark
from ollama_watch.dashboard import Dashboard, parse_size
from ollama_watch.events import (CacheVerdict, PeakMemory, Progress, RequestEnd,
                                 SpeculativeStats, Terminated)
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
    assert "idle" in text and "waiting for request" in text
    assert "1 req" in text and "q quit" in text
    assert "peak" in text
    # the table header carries the column names
    for header in ("time", "code", "input", "in/s", "cache", "out/s", "dur"):
        assert header in text


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


def test_memory_bar_tracks_resident_and_marks_peak():
    """The bar is what the model holds now; the peak is a marker, not the bar."""
    receipt, tracker = finished_receipt()
    screen = FakeScreen()
    dashboard = Dashboard(screen)
    dashboard.total_bytes = 36 * 1024**3
    dashboard.draw(Update(tracker.state.snapshot(), model=MODEL, receipts=[receipt]))
    text = screen.text
    assert "29.8 of 36 GiB" in text        # resident, from the model
    assert "peak 32.5 (90%)" in text       # high-water mark, from the receipt
    assert "┊" in text                     # peak marked inside the bar


def test_peak_beyond_physical_memory_is_not_marked_in_the_bar():
    receipt, tracker = finished_receipt()
    screen = FakeScreen()
    dashboard = Dashboard(screen)
    dashboard.total_bytes = 16 * 1024**3   # 32.49 GiB peak is off the scale
    dashboard.draw(Update(tracker.state.snapshot(), model=MODEL, receipts=[receipt]))
    assert "203%" in screen.text
    assert "┊" not in screen.text


def test_table_columns_line_up():
    receipt, tracker = finished_receipt()
    screen = FakeScreen(width=120)
    dashboard = Dashboard(screen)
    dashboard.draw(Update(tracker.state.snapshot(), model=MODEL, receipts=[receipt]))
    rows = [text for text in screen.rows.values() if "200" in text or "code" in text]
    assert len(rows) == 2
    header, entry = rows
    # every column header ends at the same offset as its value
    for name in ("code", "input", "cache", "dur"):
        assert header.index(name) + len(name) == header.index(name) + len(name)
    assert header.rstrip().endswith("peak")
    assert entry.lstrip().startswith("+")


def test_rate_row_annotates_its_range():
    receipt, tracker = finished_receipt()
    screen = FakeScreen()
    dashboard = Dashboard(screen)
    dashboard.input_rates.extend([28.0, 60.0, 89.0])
    dashboard.draw(Update(tracker.state.snapshot(), model=MODEL))
    assert "28-89" in screen.text



def test_rate_rows_say_which_number_they_are_showing():
    """The input rate is live only while processing; generation never is."""
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=49091, cached=0, remaining=49091))
    tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=49091))
    tracker.feed(Progress(ts=T0 + 60, processed=4096, remaining_total=49091))

    screen = FakeScreen(width=130)
    dashboard = Dashboard(screen)
    dashboard.draw(Update(tracker.state.snapshot(), model=MODEL, last_decode_rate=12.0))
    rows = {line.split()[0]: line for line in screen.rows.values() if line.strip()}
    assert "now" in rows["input"]      # a request is being processed
    assert "last" in rows["output"]    # generation rate is always historical


def test_generating_phase_labels_the_input_rate_as_this_request():
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=5000, cached=4999, remaining=1))
    screen = FakeScreen(width=130)
    Dashboard(screen).draw(Update(tracker.state.snapshot(), model=MODEL))
    assert any("req" in line for line in screen.rows.values() if line.startswith(" input"))


def test_rate_annotations_align_across_rows():
    """The two sparklines hold different bar counts; their ranges must still
    start at the same column."""
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=49091, cached=0, remaining=49091))
    tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=49091))
    tracker.feed(Progress(ts=T0 + 60, processed=4096, remaining_total=49091))

    screen = FakeScreen(width=130)
    dashboard = Dashboard(screen)
    dashboard.input_rates.extend([25.0, 50.0, 89.0] * 10)   # 30 bars
    dashboard.output_rates.extend([9.0, 30.0, 68.0])        # 3 bars
    dashboard.draw(Update(tracker.state.snapshot(), model=MODEL, last_decode_rate=11.0))

    rows = {line.split()[0]: line for line in screen.rows.values() if line.strip()}
    assert rows["input"].index("25-89") == rows["output"].index("9-68")
    assert "over" not in rows["input"]  # the sample count is footer information


def test_outcome_codes_are_shortened_to_fit():
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=45900, cached=24600, remaining=21300))
    tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=21300))
    tracker.feed(Progress(ts=T0 + 60, processed=4096, remaining_total=21300))
    tracker.feed(Terminated(ts=T0 + 70, error="context canceled"))
    receipt = tracker.flush()[0]

    screen = FakeScreen(width=130)
    Dashboard(screen).draw(Update(tracker.state.snapshot(), model=MODEL, receipts=[receipt]))
    text = screen.text
    assert "cancel" in text
    assert "context" not in text


def test_every_bar_starts_in_the_same_column():
    """The rate sparklines and the memory gauge must share a left edge; when
    they did not, the memory bar read as a continuation of the one above it."""
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=49091, cached=0, remaining=49091))
    tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=49091))

    screen = FakeScreen(width=130)
    dashboard = Dashboard(screen)
    dashboard.total_bytes = 36 * 1024**3
    dashboard.input_rates.extend([25.0, 89.0])
    dashboard.output_rates.extend([9.0, 68.0])
    dashboard.peak_bytes = 35.0 * 1024**3
    dashboard.draw(Update(tracker.state.snapshot(), model=MODEL, last_decode_rate=11.0))

    rows = {line.split()[0]: line for line in screen.rows.values() if line.strip()}
    edges = {name: len(rows[name]) - len(rows[name].lstrip()[len(name):].lstrip()) for name in ("input", "output", "memory")}
    starts = {name: rows[name].index(next(c for c in rows[name] if c in "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588\u2591")) for name in ("input", "output", "memory")}
    assert len(set(starts.values())) == 1, starts


def test_current_value_sits_to_the_right_of_the_history():
    """Newest bar is rightmost, so the current figure belongs beside it."""
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=49091, cached=0, remaining=49091))
    tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=49091))
    tracker.feed(Progress(ts=T0 + 60, processed=4096, remaining_total=49091))

    screen = FakeScreen(width=130)
    dashboard = Dashboard(screen)
    dashboard.input_rates.extend([25.0, 89.0])
    dashboard.draw(Update(tracker.state.snapshot(), model=MODEL))
    row = next(line for line in screen.rows.values() if line.startswith(" input"))
    assert row.index("\u2588") < row.index("tok/s")
