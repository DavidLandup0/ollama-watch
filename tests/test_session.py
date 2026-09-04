from __future__ import annotations

import pytest

from ollama_watch import Session, Tracker, segmented_bar
from ollama_watch.events import CacheVerdict, Progress, RequestEnd, SlotTiming
from ollama_watch.session import IDLE_GLYPH, INPUT_GLYPH, OUTPUT_GLYPH

T0 = 1_000_000.0


def request(tracker, *, start, prefill_s, decode_s, duration_s, tokens=5000, gen_rate=20.0):
    """Drive one complete request through the tracker."""
    tracker.feed(CacheVerdict(ts=start, total=tokens, cached=0, remaining=tokens))
    tracker.feed(Progress(ts=start + prefill_s / 2, processed=tokens // 2, remaining_total=tokens))
    tracker.feed(Progress(ts=start + prefill_s, processed=tokens - 1, remaining_total=tokens))
    end = start + prefill_s + decode_s
    tracker.feed(
        RequestEnd(ts=end, status="200", duration_s=duration_s, raw_duration=f"{duration_s}s",
                   path="/v1/chat/completions", method="POST")
    )
    tracker.feed(SlotTiming(ts=end, task=1, kind="eval", ms=decode_s * 1000, tokens=int(decode_s * gen_rate),
                            tokens_per_s=gen_rate))
    return tracker.flush()[0]


class TestSession:
    def test_splits_working_time_from_idle(self):
        tracker, session = Tracker(), Session()
        session.add(request(tracker, start=T0, prefill_s=60, decode_s=30, duration_s=90))
        # a two-minute gap while the user reads, then another request
        session.add(request(tracker, start=T0 + 210, prefill_s=30, decode_s=10, duration_s=40))

        assert session.requests == 2
        assert session.input_s == pytest.approx(90)
        assert session.output_s == pytest.approx(40)
        assert session.busy_s == pytest.approx(130)
        assert session.span_s == pytest.approx(250)
        assert session.idle_s == pytest.approx(120)
        assert session.fraction(session.busy_s) == pytest.approx(0.52)

    def test_other_time_is_request_time_that_is_neither(self):
        """Queueing, tokenising and templating are working time too."""
        tracker, session = Tracker(), Session()
        session.add(request(tracker, start=T0, prefill_s=60, decode_s=30, duration_s=100))
        assert session.other_s == pytest.approx(10)
        assert session.busy_s == pytest.approx(100)

    def test_idle_never_goes_negative_when_requests_overlap(self):
        tracker, session = Tracker(), Session()
        session.add(request(tracker, start=T0, prefill_s=60, decode_s=30, duration_s=600))
        assert session.idle_s == 0.0

    def test_empty_session_is_zero_not_an_error(self):
        session = Session()
        assert session.span_s == 0.0
        assert session.idle_s == 0.0
        assert session.fraction(10) == 0.0

    def test_observe_extends_the_span_in_both_directions(self):
        session = Session()
        session.observe(T0)
        session.observe(T0 - 50)
        session.observe(T0 + 100)
        assert session.span_s == pytest.approx(150)

    def test_segments_are_in_bar_order(self):
        tracker, session = Tracker(), Session()
        session.add(request(tracker, start=T0, prefill_s=60, decode_s=30, duration_s=90))
        session.observe(T0 + 200)
        glyphs = [glyph for glyph, _, _ in session.segments()]
        assert glyphs == [INPUT_GLYPH, OUTPUT_GLYPH, "▒", IDLE_GLYPH]
        assert [label for _, label, _ in session.segments()] == ["input", "output", "other", "idle"]


class TestSegmentedBar:
    def test_fills_exactly_the_requested_width(self):
        for width in range(1, 60):
            bar = segmented_bar([("a", 3.0), ("b", 1.0), ("c", 0.5)], width)
            assert len(bar) == width, width

    def test_apportions_roughly_by_weight(self):
        bar = segmented_bar([("a", 3.0), ("b", 1.0)], 40)
        assert bar.count("a") == 30
        assert bar.count("b") == 10

    def test_a_small_nonzero_part_still_shows(self):
        bar = segmented_bar([("a", 999.0), ("b", 0.01)], 20)
        assert "b" in bar
        assert len(bar) == 20

    def test_zero_weight_parts_are_omitted(self):
        assert segmented_bar([("a", 1.0), ("b", 0.0)], 10) == "a" * 10

    def test_no_weight_at_all_is_blank(self):
        assert segmented_bar([("a", 0.0)], 6) == " " * 6
        assert segmented_bar([], 6) == " " * 6

    def test_zero_width_is_empty(self):
        assert segmented_bar([("a", 1.0)], 0) == ""
