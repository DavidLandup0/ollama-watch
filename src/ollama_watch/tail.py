from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

from .parse import begins_request, ends_request

DEFAULT_LOG = os.path.expanduser("~/.ollama/logs/server.log")
#: How far back to look for the cache verdict of a request already in flight.
SEED_BYTES = 1 << 20
POLL_INTERVAL_S = 0.25


@dataclass(frozen=True)
class LogLine:
    text: str
    #: True for lines replayed to rebuild state; consumers should not report
    #: these as if they had just happened.
    seeded: bool = False
    #: True for the marker emitted when the log is replaced (server restart).
    rotated: bool = False
    #: True once, when a `--replay` bulk read catches up to live tailing.
    #: A request still tracked at that point was left open by history, and
    #: never ended within it, so it is a phantom rather than news.
    caught_up: bool = False


def _seed_lines(handle, window: int = SEED_BYTES) -> list[str]:
    """Lines replaying the request still in flight, if there is one.

    Joining mid-request would otherwise miss the verdict saying how much of
    the prompt was cached, leaving only a relative remaining count to report.
    Scanning back from the end stops at whichever comes first: the opening of
    a request still running, which is replayed, or the close of the last one,
    after which there is nothing to replay. The distinction matters because
    the llama.cpp runner's lines carry no clock of their own and so are read
    at `now`: replaying a request that ended an hour ago would show it
    generating this second.
    """
    end = handle.tell()
    start = max(0, end - window)
    handle.seek(start)
    chunk = handle.read(end - start)
    handle.seek(end)
    if start:  # a partial first line is unparseable
        newline = chunk.find("\n")
        chunk = chunk[newline + 1 :] if newline >= 0 else ""
    lines = chunk.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if ends_request(lines[index]):
            return []
        if begins_request(lines[index]):
            return lines[index:]
    return []


def follow_lines(
    path: str = DEFAULT_LOG,
    *,
    replay: bool = False,
    follow: bool = True,
    poll_interval: float = POLL_INTERVAL_S,
    seed: bool = True,
) -> Iterator[LogLine]:
    """Yield log lines, seeding state and surviving rotation.

    `replay` starts at the beginning of the file; otherwise it starts at the
    end, first replaying the current request's lines as `seeded`. Without
    `follow` the iterator stops at end of file, else it polls and reopens the
    file when the server rotates it.
    """
    while not os.path.exists(path):
        if not follow:
            return
        time.sleep(poll_interval)

    handle = open(path, "r", errors="replace")
    inode = os.fstat(handle.fileno()).st_ino
    try:
        if not replay:
            handle.seek(0, os.SEEK_END)
            if seed:
                for text in _seed_lines(handle):
                    yield LogLine(text, seeded=True)

        pending = ""
        caught_up_yielded = False
        while True:
            chunk = handle.readline()
            if chunk:
                pending += chunk
                if pending.endswith("\n"):
                    yield LogLine(pending.rstrip("\n"))
                    pending = ""
                continue

            if not follow:
                return
            if replay and not caught_up_yielded:
                caught_up_yielded = True
                yield LogLine("", caught_up=True)
            else:
                yield LogLine("", rotated=False)  # idle beat, lets callers tick
            try:
                stat = os.stat(path)
                if stat.st_ino != inode or stat.st_size < handle.tell():
                    handle.close()
                    handle = open(path, "r", errors="replace")
                    inode = os.fstat(handle.fileno()).st_ino
                    yield LogLine("", rotated=True)
            except FileNotFoundError:
                pass
            time.sleep(poll_interval)
    finally:
        handle.close()
