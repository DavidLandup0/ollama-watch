"""Events decoded from the Ollama runner log.

The runner logs two kinds of line: structured `key=value` records from Go's
slog, and Gin HTTP access lines. Each is decoded into one of the frozen
dataclasses below, which are the only vocabulary the rest of the library
speaks. Timestamps are always epoch seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

#: The runner processes the prompt in batches of this many tokens, and logs
#: one progress line per batch. It is the granularity of everything we can see.
PREFILL_BATCH = 2048


@dataclass(frozen=True)
class CacheVerdict:
    """A request started; the prefix cache reports how much it can reuse.

    `total` is the absolute prompt length. `cached` was matched from a previous
    request and costs nothing. `remaining` is the work actually left to do, and
    is the denominator that subsequent `Progress.processed` counts toward.
    """

    ts: float
    total: int
    cached: int
    remaining: int

    @property
    def hit(self) -> bool:
        return self.cached > 0


@dataclass(frozen=True)
class Progress:
    """A prefill batch finished.

    `processed` restarts at 0 for every request and counts only *new* tokens,
    so absolute progress is `CacheVerdict.cached + processed`. `remaining_total`
    mirrors the verdict's `remaining`, not the prompt length.
    """

    ts: float
    processed: int
    remaining_total: int


@dataclass(frozen=True)
class RequestEnd:
    """An HTTP request to an inference path completed."""

    ts: float
    status: str
    duration_s: float | None
    raw_duration: str
    path: str
    method: str

    @property
    def ok(self) -> bool:
        return self.status.startswith("2")


@dataclass(frozen=True)
class Terminated:
    """The runner abandoned a request (usually the client disconnected)."""

    ts: float
    error: str


@dataclass(frozen=True)
class PeakMemory:
    ts: float
    size: str


@dataclass(frozen=True)
class SpeculativeStats:
    """Per-request speculative-decoding summary from the MLX engine.

    Logged once per request, on either side of the request-end line.
    """

    ts: float
    iterations: int
    drafted: int
    accepted: int
    acceptance: float | None

    @property
    def generated_tokens(self) -> int:
        """Approximate tokens produced.

        Each speculative iteration emits the draft tokens the target model
        accepted, plus one token the target produces itself, so the generated
        count is `accepted + iterations`. Approximate because a request that
        stops mid-iteration is not reported separately.
        """
        return self.accepted + self.iterations


@dataclass(frozen=True)
class SlotTiming:
    """Exact timing from the llama.cpp engine's `slot print_timing` lines.

    `kind` is one of "prompt_eval", "eval" (decode) or "total". These lines
    carry no timestamp of their own.
    """

    ts: float
    task: int
    kind: str
    ms: float
    tokens: int
    tokens_per_s: float | None


@dataclass(frozen=True)
class RunnerReady:
    """A model was loaded and its runner accepted connections."""

    ts: float


Event = Union[
    CacheVerdict,
    Progress,
    RequestEnd,
    Terminated,
    PeakMemory,
    SpeculativeStats,
    SlotTiming,
    RunnerReady,
]

#: Events that describe a request that has already ended, and so may arrive on
#: either side of its request-end line.
TRAILING_EVENTS = (PeakMemory, SpeculativeStats, SlotTiming)
