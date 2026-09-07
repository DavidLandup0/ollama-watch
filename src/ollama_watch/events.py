from __future__ import annotations

from dataclasses import dataclass
from typing import Union

#: The MLX runner processes the prompt in batches of this many tokens, and
#: logs one progress line per batch. The llama.cpp runner uses a smaller
#: batch, so this is only the assumption to fall back on before any batch has
#: been observed -- see `RequestState.batch_tokens`.
PREFILL_BATCH = 2048


@dataclass(frozen=True)
class CacheVerdict:
    """A request started; the prefix cache reports how much it can reuse.

    `total` is the whole prompt, `cached` came free from a previous request,
    and `remaining` is the work left -- the denominator `Progress.processed`
    counts toward.
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

    `processed` restarts at 0 each request and counts only *new* tokens, so
    absolute progress is `CacheVerdict.cached + processed`. `remaining_total`
    mirrors the verdict's `remaining`, not the prompt length.
    """

    ts: float
    processed: int
    remaining_total: int


@dataclass(frozen=True)
class PrefillDone:
    """Input processing finished and sampling began.

    The llama.cpp runner logs `init sampler` once the whole prompt is
    consumed; `total` is the exact prompt length, correcting the rounded
    estimate derived from progress fractions.
    """

    ts: float
    total: int


@dataclass(frozen=True)
class DecodeTick:
    """One streaming decode sample from the llama.cpp runner.

    `n_gen = 100, tg = 4.80 t/s, tg_3s = 4.85 t/s` lines arrive while tokens
    stream out, so unlike the end-of-request summaries this is live.
    `generated` counts tokens produced so far, `rate` is the average since
    generation began, and `recent_rate` covers only the last few seconds --
    the one that moves when generation speeds up or slows down.
    """

    ts: float
    generated: int
    rate: float
    recent_rate: float | None = None

    @property
    def current_rate(self) -> float:
        """The rate to report live: recent if the runner gave one."""
        return self.recent_rate or self.rate


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
    """MLX speculative-decoding summary, logged once per request."""

    ts: float
    iterations: int
    drafted: int
    accepted: int
    acceptance: float | None

    @property
    def generated_tokens(self) -> int:
        """Approximate tokens produced: each iteration emits its accepted
        draft tokens plus one from the target model. Approximate because a
        request stopping mid-iteration is not reported separately."""
        return self.accepted + self.iterations


@dataclass(frozen=True)
class SlotTiming:
    """Exact timing from llama.cpp's `slot print_timing` lines.

    `kind` is "prompt_eval", "eval" (decode) or "total". These lines carry no
    timestamp of their own.
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
    PrefillDone,
    DecodeTick,
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
