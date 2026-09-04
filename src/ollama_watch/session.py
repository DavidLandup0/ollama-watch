"""Session time accounting: where the wall clock actually went.

A session splits into time the model spent working and time it spent waiting
for you. Working time splits again into processing input, generating output,
and the remainder of a request that is neither (queueing, tokenising,
templating). Everything else is idle -- the model sat loaded while you read,
typed, or approved something.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import Receipt

#: Distinct glyphs, not colours: a monochrome terminal theme must still be
#: able to tell the segments of the bar apart.
INPUT_GLYPH, OUTPUT_GLYPH, OTHER_GLYPH, IDLE_GLYPH = "█", "▓", "▒", "░"


@dataclass
class Session:
    """Accumulates elapsed time by category."""

    started_at: float | None = None
    last_seen_at: float | None = None
    requests: int = 0
    input_s: float = 0.0
    output_s: float = 0.0
    other_s: float = 0.0

    def observe(self, ts: float) -> None:
        """Extend the session span to include `ts`."""
        if self.started_at is None or ts < self.started_at:
            self.started_at = ts
        if self.last_seen_at is None or ts > self.last_seen_at:
            self.last_seen_at = ts

    def add(self, receipt: Receipt) -> None:
        """Fold in a finished request."""
        self.requests += 1
        self.input_s += max(0.0, receipt.prefill_s)
        decode_s = max(0.0, receipt.decode_s or 0.0)
        self.output_s += decode_s
        if receipt.duration_s:
            self.other_s += max(0.0, receipt.duration_s - receipt.prefill_s - decode_s)
        if receipt.started_at is not None:
            self.observe(receipt.started_at)
        self.observe(receipt.ts)

    @property
    def span_s(self) -> float:
        if self.started_at is None or self.last_seen_at is None:
            return 0.0
        return max(0.0, self.last_seen_at - self.started_at)

    @property
    def busy_s(self) -> float:
        return self.input_s + self.output_s + self.other_s

    @property
    def idle_s(self) -> float:
        """Span not accounted for by requests.

        Clamped at zero: concurrent requests can overlap, so summed request
        time may exceed the wall clock.
        """
        return max(0.0, self.span_s - self.busy_s)

    def fraction(self, seconds: float) -> float:
        return seconds / self.span_s if self.span_s else 0.0

    def segments(self) -> list[tuple[str, str, float]]:
        """`(glyph, label, seconds)` in bar order."""
        return [
            (INPUT_GLYPH, "input", self.input_s),
            (OUTPUT_GLYPH, "output", self.output_s),
            (OTHER_GLYPH, "other", self.other_s),
            (IDLE_GLYPH, "idle", self.idle_s),
        ]
