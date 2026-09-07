from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from .client import DEFAULT_HOST, ModelInfo, fetch_loaded_model, ping_server
from .events import Event, Terminated
from .parse import line_ts, parse_line
from .state import Receipt, RequestState, Tracker
from .tail import DEFAULT_LOG, POLL_INTERVAL_S, follow_lines

PS_INTERVAL_S = 10.0
#: Consecutive /api/ps polls reporting no loaded model while a request is
#: tracked before the request is declared dead. One miss could be a hiccup;
#: two is the server (or the model) going away mid-flight.
GONE_LIMIT = 2


@dataclass
class Update:
    """One observable change: new state, plus anything it produced."""

    state: RequestState
    model: ModelInfo | None = None
    event: Event | None = None
    receipts: list[Receipt] = field(default_factory=list)
    #: Generation rate of the most recent request that reported one. The log
    #: emits nothing per token, so this is the only generation figure available
    #: while a request is still decoding.
    last_decode_rate: float | None = None
    last_decode_exact: bool = False
    #: True while replaying history to rebuild state, so callers can stay quiet.
    seeded: bool = False
    rotated: bool = False
    #: The clock this update was made at: wall time when tailing live, and the
    #: log's own clock while replaying history, whose lines were written long
    #: before we read them. Elapsed times must be measured against it.
    now: float = field(default_factory=time.time)


def _update(tracker: Tracker, *, now: float | None = None, **kwargs) -> "Update":
    return Update(
        tracker.state.snapshot(),
        now=now if now is not None else time.time(),
        last_decode_rate=tracker.last_decode_rate,
        last_decode_exact=tracker.last_decode_exact,
        **kwargs,
    )


def watch(
    *,
    log: str = DEFAULT_LOG,
    host: str = DEFAULT_HOST,
    replay: bool = False,
    follow: bool = True,
    poll_interval: float = POLL_INTERVAL_S,
    ps_interval: float = PS_INTERVAL_S,
    query_server: bool = True,
) -> Iterator[Update]:
    """Yield an `Update` whenever the tracked state changes.

    >>> for update in watch():                     # doctest: +SKIP
    ...     if update.state.phase is Phase.PREFILL:
    ...         print(update.state.fraction)
    """
    tracker = Tracker()
    model: ModelInfo | None = None
    next_ps = 0.0
    gone_misses = 0
    #: True while everything tracked came from replayed lines rather than
    #: lines that arrived while we watched.
    from_history = False
    #: Latest timestamp read out of the log itself, used as the clock for
    #: replayed lines that carry none of their own.
    log_clock: float | None = None
    clock = time.time()

    for line in follow_lines(
        log, replay=replay, follow=follow, poll_interval=poll_interval
    ):
        now = time.time()
        if replay and line.text:
            log_clock = line_ts(line.text) or log_clock
        clock = log_clock or now

        if query_server and now >= next_ps:
            next_ps = now + ps_interval
            model = fetch_loaded_model(host)
            if tracker.state.active and model is None:
                # A tracked request implies a loaded model; if the server
                # stops reporting one, the request can never complete and
                # would otherwise sit in Generating forever.
                gone_misses += 1
                if gone_misses >= GONE_LIMIT:
                    gone_misses = 0
                    gone = (
                        "server unreachable"
                        if not ping_server(host)
                        else "model unloaded"
                    )
                    receipts = tracker.feed(Terminated(ts=now, error=gone))
                    # Nothing else is coming, so release the receipt rather
                    # than hold it for trailing lines.
                    receipts += tracker.flush()
                    yield _update(tracker, now=clock, model=model, receipts=receipts)
            else:
                gone_misses = 0

        if from_history and not line.seeded:
            # Seeding is over. A replayed request the log never closed is
            # only real if a model is still loaded; otherwise it died with an
            # earlier server and is a phantom.
            from_history = False
            if tracker.state.active and query_server:
                model = fetch_loaded_model(host)
                next_ps = now + ps_interval
                if model is None:
                    tracker.state = RequestState()

        if line.caught_up:
            # Replay history ends here; live lines rebuild from scratch.
            # Anything still tracked is a phantom of history, not a request,
            # so it is dropped before flushing -- but a completed request
            # still held for trailing lines is real, and is reported.
            tracker.state = RequestState()
            receipts = tracker.flush()
            if receipts:
                yield _update(tracker, now=clock, model=model, receipts=receipts)
            continue

        if line.rotated:
            receipts = []
            if tracker.state.active:
                # The file was replaced (server restart): the tracked
                # request died with the old process.
                receipts = tracker.feed(Terminated(ts=now, error="log rotated"))
                receipts += tracker.flush()
            yield _update(tracker, now=clock, model=model, rotated=True, receipts=receipts)
            continue

        if not line.text:  # idle beat
            receipts = tracker.tick(now)
            yield _update(tracker, now=clock, model=model, receipts=receipts)
            continue

        event = parse_line(line.text, now=clock)
        if event is None:
            continue
        receipts = tracker.feed(event)
        if line.seeded:
            receipts = []  # history, not news
            from_history = True
            # Replayed llama.cpp lines carry no clock, so they are read at
            # `now`: the request began earlier than this state says.
            tracker.state.start_observed = False
        yield _update(
            tracker,
            now=clock,
            model=model,
            event=event if not line.seeded else None,
            receipts=receipts,
            seeded=line.seeded,
        )

    receipts = tracker.flush()
    if receipts:
        yield _update(tracker, now=clock, model=model, receipts=receipts)


__all__ = ["Update", "watch"]
