"""The library's main entry point: a stream of state updates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from .client import DEFAULT_HOST, ModelInfo, fetch_loaded_model
from .events import Event, RunnerReady
from .parse import parse_line
from .state import Receipt, RequestState, Tracker
from .tail import DEFAULT_LOG, POLL_INTERVAL_S, follow_lines

PS_INTERVAL_S = 10.0


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


def _update(tracker: Tracker, **kwargs) -> "Update":
    return Update(
        tracker.state.snapshot(),
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

    for line in follow_lines(
        log, replay=replay, follow=follow, poll_interval=poll_interval
    ):
        now = time.time()
        if query_server and now >= next_ps:
            next_ps = now + ps_interval
            model = fetch_loaded_model(host)

        if line.rotated:
            yield _update(tracker, model=model, rotated=True)
            continue

        if not line.text:  # idle beat
            receipts = tracker.tick(now)
            yield _update(tracker, model=model, receipts=receipts)
            continue

        event = parse_line(line.text, now=now)
        if event is None:
            continue
        receipts = tracker.feed(event)
        if line.seeded:
            receipts = []  # history, not news
        yield _update(
            tracker,
            model=model,
            event=event if not line.seeded else None,
            receipts=receipts,
            seeded=line.seeded,
        )

    receipts = tracker.flush()
    if receipts:
        yield _update(tracker, model=model, receipts=receipts)


__all__ = ["Update", "watch", "RunnerReady"]
