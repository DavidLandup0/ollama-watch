from __future__ import annotations

import pytest

from ollama_watch import Phase, Tracker
from ollama_watch.events import (
    CacheVerdict,
    PeakMemory,
    Progress,
    RequestEnd,
    SlotTiming,
    SpeculativeStats,
    Terminated,
)

T0 = 1_000_000.0


def end(ts, status="200", duration=None):
    return RequestEnd(
        ts=ts,
        status=status,
        duration_s=duration,
        raw_duration=f"{duration}s" if duration else "1s",
        path="/v1/chat/completions",
        method="POST",
    )


def release(tracker, ts):
    """Push time forward so a held receipt is released."""
    return tracker.feed(PeakMemory(ts=ts, size="1 GiB")) + tracker.tick(ts) + tracker.flush()


class TestProgressAccounting:
    def test_cache_miss_progress_is_absolute(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=49077, cached=0, remaining=49077))
        tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=49077))
        state = tracker.state
        assert state.phase is Phase.PREFILL
        assert state.prompt_tokens == 49077
        assert state.done_tokens == 2048
        assert state.fraction == pytest.approx(2048 / 49077)

    def test_cache_hit_adds_cached_prefix_back(self):
        """Progress counts only new tokens, so 8192/14275 is really 43k/49k."""
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=49091, cached=34816, remaining=14275))
        tracker.feed(Progress(ts=T0 + 30, processed=8192, remaining_total=14275))
        state = tracker.state
        assert state.prompt_tokens == 49091
        assert state.cached_tokens == 34816
        assert state.done_tokens == 34816 + 8192
        assert state.fraction == pytest.approx(43008 / 49091)
        assert state.cached_known

    def test_joining_mid_flight_marks_totals_relative(self):
        tracker = Tracker()
        tracker.feed(Progress(ts=T0, processed=8192, remaining_total=14275))
        assert tracker.state.cached_known is False
        assert tracker.state.prompt_tokens == 14275

    def test_rate_and_eta_come_from_the_sample_window(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=10000, cached=0, remaining=10000))
        tracker.feed(Progress(ts=T0 + 32, processed=2048, remaining_total=10000))
        tracker.feed(Progress(ts=T0 + 64, processed=4096, remaining_total=10000))
        assert tracker.state.prefill_rate == pytest.approx(4096 / 64)
        assert tracker.state.eta_s == pytest.approx((10000 - 4096) / (4096 / 64))


class TestPhaseTransitions:
    def test_final_token_is_not_prefilled(self):
        """The runner stops one token short, so `processed == total` never happens."""
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=50300, cached=49081, remaining=1219))
        tracker.feed(Progress(ts=T0 + 18, processed=1218, remaining_total=1219))
        assert tracker.state.phase is Phase.DECODE
        assert tracker.state.prefill_done_at == T0 + 18

    def test_fully_cached_prompt_starts_generating_immediately(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=50400, cached=50399, remaining=1))
        assert tracker.state.phase is Phase.DECODE

    def test_sub_batch_prefill_is_inferred_from_silence(self):
        """A prefill smaller than one batch logs no progress line at all."""
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=50000, cached=49700, remaining=300))
        assert tracker.state.phase is Phase.PREFILL
        tracker._last_progress_wall = 0.0  # simulate a long quiet tail
        tracker.tick(now=10_000.0)
        assert tracker.state.phase is Phase.DECODE

    def test_a_long_batch_is_not_mistaken_for_silence(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=49077, cached=0, remaining=49077))
        tracker._last_progress_wall = 0.0
        tracker.tick(now=10_000.0)
        assert tracker.state.phase is Phase.PREFILL


class TestRequestMatching:
    def test_concurrent_keepalive_ping_does_not_end_the_request(self):
        """A 63ms /api/generate ping must not close a 3-minute request."""
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=49077, cached=0, remaining=49077))
        tracker.feed(Progress(ts=T0 + 130, processed=8192, remaining_total=49077))
        receipts = tracker.feed(end(T0 + 131, duration=0.063))
        assert receipts == []
        assert tracker.state.active

    def test_matching_completion_closes_the_request(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=49077, cached=0, remaining=49077))
        tracker.feed(Progress(ts=T0 + 130, processed=8192, remaining_total=49077))
        assert tracker.feed(end(T0 + 200, duration=200.0)) == []  # held for trailers
        receipts = tracker.flush()
        assert len(receipts) == 1
        assert receipts[0].status == "200"
        assert receipts[0].prefilled_tokens == 8192


class TestReceipts:
    def test_trailing_events_attach_before_release(self):
        """Peak memory and decode stats can arrive after the request-end line."""
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=5000, cached=0, remaining=5000))
        tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=5000))
        tracker.feed(Progress(ts=T0 + 60, processed=4999, remaining_total=5000))
        tracker.feed(end(T0 + 90, duration=90.0))
        tracker.feed(PeakMemory(ts=T0 + 90.1, size="28.02 GiB"))
        tracker.feed(SpeculativeStats(ts=T0 + 90.2, iterations=133, drafted=502, accepted=424, acceptance=0.84))
        receipts = tracker.flush()
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt.peak_memory == "28.02 GiB"
        assert receipt.generated_tokens == 557
        assert receipt.generated_exact is False
        assert receipt.acceptance == pytest.approx(0.84)
        # 557 tokens over the 30s between prefill finishing and the response
        assert receipt.decode_s == pytest.approx(30.0)
        assert receipt.decode_rate == pytest.approx(557 / 30.0)

    def test_llama_cpp_timings_are_exact(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=5000, cached=0, remaining=5000))
        tracker.feed(Progress(ts=T0 + 30, processed=4999, remaining_total=5000))
        tracker.feed(end(T0 + 60, duration=60.0))
        tracker.feed(SlotTiming(ts=T0 + 60, task=122, kind="eval", ms=34.03, tokens=3, tokens_per_s=58.77))
        receipt = tracker.flush()[0]
        assert receipt.generated_exact is True
        assert receipt.decode_rate_exact is True
        assert receipt.decode_rate == pytest.approx(58.77)
        assert receipt.generated_tokens == 3

    def test_prompt_eval_timing_does_not_count_as_generation(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=5000, cached=0, remaining=5000))
        tracker.feed(Progress(ts=T0 + 30, processed=4999, remaining_total=5000))
        tracker.feed(end(T0 + 60, duration=60.0))
        tracker.feed(SlotTiming(ts=T0 + 60, task=1, kind="prompt_eval", ms=50.4, tokens=5, tokens_per_s=99.3))
        receipt = tracker.flush()[0]
        assert receipt.generated_tokens is None

    def test_fully_cached_request_reports_no_rate(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=50400, cached=50399, remaining=1))
        tracker.feed(end(T0 + 31, duration=31.0))
        receipt = tracker.flush()[0]
        assert receipt.fully_cached is True
        assert receipt.prefilled_tokens == 0
        assert receipt.cache_fraction == pytest.approx(50399 / 50400)

    def test_termination_is_reported_with_its_reason(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=49077, cached=0, remaining=49077))
        tracker.feed(Progress(ts=T0 + 30, processed=2048, remaining_total=49077))
        tracker.feed(Progress(ts=T0 + 60, processed=4096, remaining_total=49077))
        tracker.feed(Terminated(ts=T0 + 70, error="context canceled"))
        receipt = tracker.flush()[0]
        assert receipt.outcome == "context canceled"
        assert receipt.status is None

    def test_unobserved_fragment_is_discarded(self):
        """A stray progress line then a termination measures nothing usable."""
        tracker = Tracker()
        tracker.feed(Progress(ts=T0, processed=2048, remaining_total=49077))
        tracker.feed(Terminated(ts=T0 + 0.2, error="context canceled"))
        assert tracker.flush() == []

    def test_next_request_releases_the_previous_receipt(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=5000, cached=0, remaining=5000))
        tracker.feed(Progress(ts=T0 + 30, processed=4999, remaining_total=5000))
        tracker.feed(end(T0 + 60, duration=60.0))
        receipts = tracker.feed(CacheVerdict(ts=T0 + 120, total=6000, cached=5000, remaining=1000))
        assert len(receipts) == 1
        assert receipts[0].status == "200"
        assert tracker.state.prompt_tokens == 6000

    def test_last_decode_rate_is_carried_for_the_live_display(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=5000, cached=0, remaining=5000))
        tracker.feed(Progress(ts=T0 + 30, processed=4999, remaining_total=5000))
        tracker.feed(end(T0 + 60, duration=60.0))
        tracker.feed(SlotTiming(ts=T0 + 60, task=1, kind="eval", ms=1000.0, tokens=25, tokens_per_s=25.0))
        tracker.flush()
        assert tracker.last_decode_rate == pytest.approx(25.0)
        assert tracker.last_decode_exact is True


def test_snapshot_is_independent_of_later_mutation():
    """Consumers that buffer updates must not see state change under them."""
    tracker = Tracker()
    tracker.feed(CacheVerdict(ts=T0, total=10000, cached=0, remaining=10000))
    early = tracker.state.snapshot()
    tracker.feed(Progress(ts=T0 + 30, processed=9999, remaining_total=10000))
    assert early.phase is Phase.PREFILL
    assert early.prefilled_tokens == 0
    assert tracker.state.phase is Phase.DECODE


class TestTimeToFirstToken:
    """Arrival is derived as end - duration, since Gin times the whole handler."""

    def test_ttft_and_queue_are_measured_from_arrival(self):
        tracker = Tracker()
        # arrives at T0, the runner starts work 5s later, prefill ends at +65
        tracker.feed(CacheVerdict(ts=T0 + 5, total=5000, cached=0, remaining=5000))
        tracker.feed(Progress(ts=T0 + 35, processed=2500, remaining_total=5000))
        tracker.feed(Progress(ts=T0 + 65, processed=4999, remaining_total=5000))
        tracker.feed(end(T0 + 100, duration=100.0))
        receipt = tracker.flush()[0]

        assert receipt.arrived_at == pytest.approx(T0)
        assert receipt.queued_s == pytest.approx(5.0)
        assert receipt.ttft_s == pytest.approx(65.0)

    def test_unknown_duration_leaves_them_unset(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=5000, cached=0, remaining=5000))
        tracker.feed(Progress(ts=T0 + 30, processed=2500, remaining_total=5000))
        tracker.feed(Progress(ts=T0 + 60, processed=4999, remaining_total=5000))
        tracker.feed(Terminated(ts=T0 + 70, error="context canceled"))
        receipt = tracker.flush()[0]

        assert receipt.arrived_at is None
        assert receipt.queued_s is None
        assert receipt.ttft_s is None

    def test_queue_is_never_negative(self):
        tracker = Tracker()
        tracker.feed(CacheVerdict(ts=T0, total=5000, cached=0, remaining=5000))
        tracker.feed(Progress(ts=T0 + 30, processed=4999, remaining_total=5000))
        tracker.feed(end(T0 + 40, duration=35.0))  # implies arrival after the start
        receipt = tracker.flush()[0]
        assert receipt.queued_s == 0.0


class TestDecodeMetricsArriveOnEitherSide:
    """The runner logs decode stats before or after the request-end line,
    depending on sub-second ordering. Both must be captured."""

    def _drive(self, tracker):
        tracker.feed(CacheVerdict(ts=T0, total=50427, cached=50294, remaining=133))
        tracker.feed(Progress(ts=T0 + 3, processed=132, remaining_total=133))

    def test_stats_before_the_end_line(self):
        tracker = Tracker()
        self._drive(tracker)
        tracker.feed(SpeculativeStats(ts=T0 + 15.6, iterations=55, drafted=165, accepted=144, acceptance=0.87))
        tracker.feed(end(T0 + 15.7, duration=15.6))
        receipt = tracker.flush()[0]
        assert receipt.generated_tokens == 144 + 55
        assert receipt.acceptance == pytest.approx(0.87)
        assert receipt.decode_rate is not None

    def test_stats_after_the_end_line(self):
        tracker = Tracker()
        self._drive(tracker)
        tracker.feed(end(T0 + 15.7, duration=15.6))
        tracker.feed(SpeculativeStats(ts=T0 + 15.8, iterations=55, drafted=165, accepted=144, acceptance=0.87))
        receipt = tracker.flush()[0]
        assert receipt.generated_tokens == 144 + 55
        assert receipt.decode_rate is not None

    def test_slot_timing_before_the_end_line(self):
        tracker = Tracker()
        self._drive(tracker)
        tracker.feed(SlotTiming(ts=T0 + 15, task=7, kind="eval", ms=12_000.0, tokens=199, tokens_per_s=16.6))
        tracker.feed(end(T0 + 15.7, duration=15.6))
        receipt = tracker.flush()[0]
        assert receipt.generated_exact is True
        assert receipt.decode_rate == pytest.approx(16.6)

    def test_both_orderings_agree(self):
        before, after = Tracker(), Tracker()
        stats = SpeculativeStats(ts=T0 + 15.6, iterations=55, drafted=165, accepted=144, acceptance=0.87)
        self._drive(before)
        before.feed(stats)
        before.feed(end(T0 + 15.7, duration=15.6))
        self._drive(after)
        after.feed(end(T0 + 15.7, duration=15.6))
        after.feed(stats)
        first, second = before.flush()[0], after.flush()[0]
        assert first.generated_tokens == second.generated_tokens
        assert first.decode_rate == pytest.approx(second.decode_rate)

    def test_last_decode_rate_is_set_from_either_side(self):
        tracker = Tracker()
        self._drive(tracker)
        tracker.feed(SlotTiming(ts=T0 + 15, task=7, kind="eval", ms=1000.0, tokens=25, tokens_per_s=25.0))
        tracker.feed(end(T0 + 15.7, duration=15.6))
        tracker.flush()
        assert tracker.last_decode_rate == pytest.approx(25.0)
