"""The request state machine.

Everything Ollama does not put on the wire is reconstructed here from log
events. Three facts about the runner's logging shape this code, and each is
covered by a test:

1. `Progress.processed` counts only *new* tokens and restarts at 0 per
   request, so absolute progress needs the cache verdict added back.
2. The runner prefills every token but the last -- that final token is fed as
   decoding begins -- so `processed` stops one short of the total.
3. Trailing events (peak memory, decode stats) arrive on either side of the
   request-end line, so a finished request is held briefly before its receipt
   is released.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum

from .events import (
    PREFILL_BATCH,
    TRAILING_EVENTS,
    CacheVerdict,
    Event,
    PeakMemory,
    Progress,
    RequestEnd,
    SlotTiming,
    SpeculativeStats,
    Terminated,
)

#: Progress samples kept for the rolling prefill rate.
RATE_WINDOW = 6
#: How far a request-end line's implied start may drift from the observed one
#: before we assume it belongs to a different, concurrent request.
MATCH_TOLERANCE_S = 20.0
#: How long a finished request waits for trailing events before its receipt
#: is released, in log time.
PENDING_GRACE_S = 1.0
#: Never infer a stalled prefill faster than this.
STALE_FLOOR_S = 20.0
#: Assumed seconds per batch before any have been observed.
DEFAULT_BATCH_S = 30.0
#: The final prompt token is consumed by decoding, not prefill.
FINAL_TOKEN_SLACK = 1


class Phase(str, Enum):
    IDLE = "idle"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class RequestState:
    """Live view of the request currently being served."""

    phase: Phase = Phase.IDLE
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prefilled_tokens: int = 0
    started_at: float | None = None
    prefill_done_at: float | None = None
    last_event_at: float = 0.0
    peak_memory: str | None = None
    #: False when we joined a request already in flight and so never saw its
    #: cache verdict; the totals are then relative, not absolute.
    cached_known: bool = True
    #: Decode metrics that arrived *before* the request-end line. The runner
    #: logs them on either side of it, so they must be held either way.
    speculative: "SpeculativeStats | None" = None
    slot_eval: "SlotTiming | None" = None
    samples: deque = field(default_factory=lambda: deque(maxlen=RATE_WINDOW))

    @property
    def active(self) -> bool:
        return self.started_at is not None

    @property
    def done_tokens(self) -> int:
        return self.cached_tokens + self.prefilled_tokens

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.done_tokens)

    @property
    def fraction(self) -> float:
        if not self.prompt_tokens:
            return 0.0
        return min(1.0, self.done_tokens / self.prompt_tokens)

    @property
    def prefill_rate(self) -> float | None:
        """Tokens per second over the sample window, or None if too early."""
        if len(self.samples) < 2:
            return None
        (t0, p0), (t1, p1) = self.samples[0], self.samples[-1]
        return (p1 - p0) / (t1 - t0) if t1 > t0 else None

    @property
    def batch_interval_s(self) -> float:
        if len(self.samples) < 2:
            return DEFAULT_BATCH_S
        (t0, _), (t1, _) = self.samples[0], self.samples[-1]
        return max(1.0, (t1 - t0) / (len(self.samples) - 1))

    @property
    def eta_s(self) -> float | None:
        rate = self.prefill_rate
        if not rate or rate <= 0:
            return None
        return self.remaining_tokens / rate

    def snapshot(self) -> "RequestState":
        """An immutable-in-practice copy.

        The tracker mutates its state in place, so consumers that buffer
        updates must be handed a copy or every one of them would read back
        with the final phase.
        """
        clone = replace(self)
        clone.samples = deque(self.samples, maxlen=self.samples.maxlen)
        return clone

    def decode_elapsed_s(self, now: float | None = None) -> float | None:
        if self.phase is not Phase.DECODE or self.prefill_done_at is None:
            return None
        return max(0.0, (now if now is not None else time.time()) - self.prefill_done_at)


@dataclass
class Receipt:
    """A completed request, ready to report."""

    ts: float
    outcome: str
    status: str | None = None
    raw_duration: str | None = None
    duration_s: float | None = None
    started_at: float | None = None
    #: When the HTTP request reached the server, derived as end - duration.
    #: Only known for requests that completed with a reported duration.
    arrived_at: float | None = None
    #: Time between arrival and the runner starting work: scheduler queueing.
    queued_s: float | None = None
    #: Time to first token: arrival until input processing finished.
    ttft_s: float | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prefilled_tokens: int = 0
    prefill_s: float = 0.0
    prefill_rate: float = 0.0
    prefill_trusted: bool = True
    fully_cached: bool = False
    peak_memory: str | None = None
    generated_tokens: int | None = None
    generated_exact: bool = False
    decode_s: float | None = None
    decode_rate: float | None = None
    decode_rate_exact: bool = False
    acceptance: float | None = None

    @property
    def ok(self) -> bool:
        return bool(self.status and self.status.startswith("2"))

    @property
    def cache_fraction(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


class Tracker:
    """Folds log events into a live `RequestState` and finished `Receipt`s."""

    def __init__(
        self,
        *,
        match_tolerance_s: float = MATCH_TOLERANCE_S,
        pending_grace_s: float = PENDING_GRACE_S,
        stale_floor_s: float = STALE_FLOOR_S,
    ) -> None:
        self.state = RequestState()
        self.match_tolerance_s = match_tolerance_s
        self.pending_grace_s = pending_grace_s
        self.stale_floor_s = stale_floor_s
        #: Decode rate of the last request that reported one, as a reference
        #: while the current request is generating.
        self.last_decode_rate: float | None = None
        self.last_decode_exact: bool = False
        self._pending: Receipt | None = None
        self._last_progress_wall: float = time.time()

    # ---------------- public API ----------------

    def feed(self, event: Event) -> list[Receipt]:
        """Apply one event. Returns any receipts it released."""
        out: list[Receipt] = []
        if not isinstance(event, TRAILING_EVENTS):
            out += self._expire_pending(event.ts)

        if isinstance(event, CacheVerdict):
            out += self._begin(event)
        elif isinstance(event, Progress):
            self._advance(event)
        elif isinstance(event, RequestEnd):
            out += self._end(event)
        elif isinstance(event, Terminated):
            out += self._finish(event.ts, event.error)
        elif isinstance(event, PeakMemory):
            self._attach_peak(event)
        elif isinstance(event, SpeculativeStats):
            self._attach_speculative(event)
        elif isinstance(event, SlotTiming):
            self._attach_slot(event)
        return out

    def tick(self, now: float | None = None) -> list[Receipt]:
        """Advance time without an event: detects a prefill that has gone quiet."""
        now = now if now is not None else time.time()
        st = self.state
        if st.phase is not Phase.PREFILL or not st.active:
            return []
        # A prefill shorter than one batch logs no progress line at all, so a
        # quiet tail is the only signal that decoding has begun.
        if st.remaining_tokens >= PREFILL_BATCH:
            return []
        if now - self._last_progress_wall > max(self.stale_floor_s, 1.5 * st.batch_interval_s):
            st.phase = Phase.DECODE
            st.prefill_done_at = st.last_event_at
        return []

    def flush(self) -> list[Receipt]:
        """Release a held receipt, and report any request still in flight."""
        out: list[Receipt] = []
        if self._pending is not None:
            out.append(self._pending)
            self._pending = None
        if self.state.active:
            # _finish defers into _pending, so drain it again afterwards
            self._finish(self.state.last_event_at, "in flight")
            if self._pending is not None:
                out.append(self._pending)
                self._pending = None
        return out

    # ---------------- internals ----------------

    def _expire_pending(self, ts: float) -> list[Receipt]:
        if self._pending is None:
            return []
        if ts >= self._pending.ts + self.pending_grace_s:
            receipt, self._pending = self._pending, None
            return [receipt]
        return []

    def _begin(self, event: CacheVerdict) -> list[Receipt]:
        out: list[Receipt] = []
        if self.state.active:
            out += self._finish(event.ts, "superseded")
        if self._pending is not None:
            out.append(self._pending)
            self._pending = None

        self.state = RequestState(
            phase=Phase.PREFILL,
            prompt_tokens=event.total,
            cached_tokens=event.cached,
            started_at=event.ts,
            last_event_at=event.ts,
        )
        self.state.samples.append((event.ts, 0))
        self._last_progress_wall = time.time()
        if event.total and event.total - event.cached <= FINAL_TOKEN_SLACK:
            self.state.phase = Phase.DECODE
            self.state.prefill_done_at = event.ts
        return out

    def _advance(self, event: Progress) -> None:
        st = self.state
        if not st.active:  # joined mid-flight: the cached prefix is unknown
            st.started_at = event.ts
            st.cached_tokens = 0
            st.cached_known = False
            st.prompt_tokens = event.remaining_total
        st.prefilled_tokens = event.processed
        st.last_event_at = event.ts
        st.phase = Phase.PREFILL
        absolute = st.cached_tokens + event.remaining_total
        if absolute != st.prompt_tokens:
            st.prompt_tokens = absolute
        st.samples.append((event.ts, event.processed))
        self._last_progress_wall = time.time()
        if event.remaining_total - event.processed <= FINAL_TOKEN_SLACK:
            st.phase = Phase.DECODE
            st.prefill_done_at = event.ts

    def _end(self, event: RequestEnd) -> list[Receipt]:
        if not self.state.active:
            return []
        # Requests are served concurrently, so a completion is only ours if
        # its own reported duration lines up with the start we observed.
        if event.duration_s is not None and self.state.started_at is not None:
            implied_start = event.ts - event.duration_s
            if abs(implied_start - self.state.started_at) > self.match_tolerance_s:
                return []
        return self._finish(
            event.ts,
            "completed",
            status=event.status,
            raw_duration=event.raw_duration,
            duration_s=event.duration_s,
        )

    def _finish(
        self,
        ts: float,
        outcome: str,
        *,
        status: str | None = None,
        raw_duration: str | None = None,
        duration_s: float | None = None,
    ) -> list[Receipt]:
        # Gin times the whole handler, so arrival precedes any scheduler wait.
        arrived_at = (ts - duration_s) if duration_s is not None else None
        st = self.state
        if not st.active:
            return []

        prefill_s = max(0.0, (st.prefill_done_at or st.last_event_at or ts) - (st.started_at or ts))
        windowed = st.prefill_rate
        rate = windowed or (st.prefilled_tokens / prefill_s if prefill_s > 0 else 0.0)
        fully_cached = st.cached_known and st.cached_tokens > 0 and st.prefilled_tokens == 0
        trusted = len(st.samples) >= 2 or fully_cached

        # A fragment of a request we only partly observed measures nothing.
        if not trusted and (prefill_s < 0.5 or st.prefilled_tokens == 0):
            self.state = RequestState()
            return []

        decode_s = (ts - st.prefill_done_at) if st.prefill_done_at else None
        receipt = Receipt(
            ts=ts,
            outcome=outcome,
            status=status,
            raw_duration=raw_duration,
            duration_s=duration_s if duration_s is not None else (ts - (st.started_at or ts)),
            started_at=st.started_at,
            arrived_at=arrived_at,
            queued_s=(
                max(0.0, st.started_at - arrived_at)
                if arrived_at is not None and st.started_at is not None
                else None
            ),
            ttft_s=(
                max(0.0, (st.prefill_done_at or ts) - arrived_at)
                if arrived_at is not None
                else None
            ),
            prompt_tokens=st.prompt_tokens,
            cached_tokens=st.cached_tokens,
            prefilled_tokens=st.prefilled_tokens,
            prefill_s=prefill_s,
            prefill_rate=rate,
            prefill_trusted=trusted,
            fully_cached=fully_cached,
            peak_memory=st.peak_memory,
            decode_s=decode_s,
        )
        if st.speculative is not None:
            self._apply_speculative(receipt, st.speculative)
        if st.slot_eval is not None:
            self._apply_slot(receipt, st.slot_eval)
        self._note_decode_rate(receipt)

        self.state = RequestState()
        # Hold it: peak memory and decode stats may still be on their way.
        self._pending = receipt
        return []

    def _target(self) -> Receipt | None:
        return self._pending

    def _attach_peak(self, event: PeakMemory) -> None:
        target = self._target()
        if target is not None:
            target.peak_memory = event.size
        else:
            self.state.peak_memory = event.size

    @staticmethod
    def _apply_speculative(receipt: Receipt, event: SpeculativeStats) -> None:
        receipt.generated_tokens = event.generated_tokens
        receipt.generated_exact = False
        receipt.acceptance = event.acceptance
        if receipt.decode_s and receipt.decode_s > 0:
            receipt.decode_rate = event.generated_tokens / receipt.decode_s
            receipt.decode_rate_exact = False

    @staticmethod
    def _apply_slot(receipt: Receipt, event: SlotTiming) -> None:
        receipt.generated_tokens = event.tokens
        receipt.generated_exact = True
        receipt.decode_s = event.ms / 1000.0
        if event.tokens_per_s:
            receipt.decode_rate = event.tokens_per_s
            receipt.decode_rate_exact = True

    def _note_decode_rate(self, receipt: Receipt) -> None:
        if receipt.decode_rate:
            self.last_decode_rate = receipt.decode_rate
            self.last_decode_exact = receipt.decode_rate_exact

    def _attach_speculative(self, event: SpeculativeStats) -> None:
        target = self._target()
        if target is None:  # arrived before the request-end line
            self.state.speculative = event
            return
        self._apply_speculative(target, event)
        self._note_decode_rate(target)

    def _attach_slot(self, event: SlotTiming) -> None:
        if event.kind != "eval":
            return
        target = self._target()
        if target is None:
            self.state.slot_eval = event
            return
        self._apply_slot(target, event)
        self._note_decode_rate(target)
